"""
optimize_nsga2.py
Main entrypoint for Phase 2 NSGA-II Multi-Objective Evolutionary Hyperparameter Optimization.
Orchestrates optimization, Pareto front extraction, model selection, untouched test evaluation,
and artifact generation.
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch

from src.utils import set_seed, load_config, compute_metrics, count_parameters, measure_inference_latency, save_json
from src.features import FeaturePipeline
from src.model import build_model
from src.loss import RegularizedHuberLoss
from evaluate import evaluate_model
from src.nsga2.genome import Genome
from src.nsga2.population import Individual
from src.nsga2.evaluation import GenomeEvaluator
from src.nsga2.evolution import NSGA2Optimizer
from src.nsga2.sorting import fast_non_dominated_sort, calculate_crowding_distance

# Setup logger for NSGA-II
log_dir = "results/nsga2"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "nsga2.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def select_pareto_candidates(pareto_front: List[Individual]) -> Tuple[Individual, Dict[str, Individual]]:
    """
    Documented Multi-Criteria Decision Rule for Candidate Selection from Pareto Front:
    1. Candidate A (Best Validation MAE): Minimum validation_mae.
    2. Candidate B (Balanced Compromise): Minimum Euclidean distance to normalized ideal objective vector.
    3. Candidate C (Most Parameter-Efficient): Lowest parameters with MAE within 5% of best.
    
    CRITICAL: Selected purely on VALIDATION objectives. Test set is strictly excluded.
    
    Returns:
        selected_model: Candidate B (Balanced Compromise)
        candidates_dict: Dictionary of representative candidates
    """
    if not pareto_front:
        raise ValueError("Pareto front is empty!")
        
    # Candidate A: Best Validation MAE
    cand_a = min(pareto_front, key=lambda ind: ind.metrics.validation_mae)
    
    # Candidate C: Most Parameter-Efficient within 5% of Best MAE
    mae_thresh = cand_a.metrics.validation_mae * 1.05
    efficient_pool = [ind for ind in pareto_front if ind.metrics.validation_mae <= mae_thresh]
    if efficient_pool:
        cand_c = min(efficient_pool, key=lambda ind: ind.metrics.trainable_parameters)
    else:
        cand_c = cand_a
        
    # Candidate B: Balanced Compromise (Normalized Ideal Distance)
    # Collect min/max bounds across Pareto front for each objective
    num_objs = len(pareto_front[0].objectives)
    obj_matrix = np.array([ind.objectives for ind in pareto_front])
    min_vals = np.min(obj_matrix, axis=0)
    max_vals = np.max(obj_matrix, axis=0)
    ranges = max_vals - min_vals
    ranges[ranges < 1e-12] = 1.0  # avoid division by zero
    
    # Normalized distance to ideal point (0 on all normalized objectives)
    norm_matrix = (obj_matrix - min_vals) / ranges
    distances = np.linalg.norm(norm_matrix, axis=1)
    best_balanced_idx = int(np.argmin(distances))
    cand_b = pareto_front[best_balanced_idx]
    
    candidates = {
        "candidate_a_best_mae": cand_a,
        "candidate_b_balanced_compromise": cand_b,
        "candidate_c_parameter_efficient": cand_c
    }
    
    return cand_b, candidates


def retrain_and_evaluate_test(
    genome: Genome,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_pipeline: FeaturePipeline,
    target_column: str = "next_semester_sgpa",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Retrain the selected candidate genome from scratch using the train cohort
    and evaluate ONLY on the untouched held-out test cohort.
    """
    set_seed(seed, deterministic=True)
    device = torch.device("cpu")
    
    X_train = feature_pipeline.transform(train_df)
    y_train = train_df[target_column].values.astype(np.float32)
    
    X_val = feature_pipeline.transform(val_df)
    y_val = val_df[target_column].values.astype(np.float32)
    
    X_test = feature_pipeline.transform(test_df)
    y_test = test_df[target_column].values.astype(np.float32)
    
    input_dim = X_train.shape[1]
    
    model = build_model(
        {"hidden_dims": genome.hidden_dims, "dropout_rate": genome.dropout_rate, "activation": genome.activation},
        input_dim=input_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=genome.learning_rate, weight_decay=genome.weight_decay)
    criterion = RegularizedHuberLoss(delta=genome.huber_delta, l1_lambda=genome.l1_lambda)
    
    train_ds = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train).view(-1, 1))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=genome.batch_size, shuffle=True)
    
    val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_targets = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    
    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    t_start = time.perf_counter()
    
    for epoch in range(1, genome.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss, _ = criterion(pred, yb, model=model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=genome.gradient_clip_norm)
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            v_pred = model(val_tensor)
            v_loss, v_dict = criterion(v_pred, val_targets, model=None)
            val_loss_val = v_dict["huber_loss"]
            
        if val_loss_val < best_val_loss - 1e-4:
            best_val_loss = val_loss_val
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    training_time = time.perf_counter() - t_start
    
    # Final untouched test evaluation
    metrics, test_preds = evaluate_model(model, X_test, y_test, criterion=criterion)
    sample_tensor = torch.tensor(X_test[:1], dtype=torch.float32)
    latency_info = measure_inference_latency(model, sample_tensor, num_runs=200)
    batch_latency = measure_inference_latency(model, torch.tensor(X_test, dtype=torch.float32), num_runs=100)
    param_info = count_parameters(model)
    
    # Save checkpoint of best NSGA-II model
    os.makedirs("models/nsga2", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "genome": genome.to_dict(),
            "metrics": metrics,
            "test_preds": test_preds
        },
        "models/nsga2/nsga2_selected_model.pt"
    )
    
    return {
        "test_mae": metrics["mae"],
        "test_rmse": metrics["rmse"],
        "test_r2": metrics["r2"],
        "test_huber_loss": metrics.get("loss", 0.0),
        "trainable_parameters": param_info["trainable_parameters"],
        "single_sample_latency_ms": latency_info["single_sample_latency_ms"],
        "batch_latency_ms": batch_latency["batch_latency_ms"],
        "training_time_seconds": round(training_time, 3),
        "best_epoch": best_epoch,
        "genome": genome.to_dict()
    }


def run_nsga2(
    mode: str = "standard",
    seed: int = 42,
    custom_pop: int = None,
    custom_gen: int = None
) -> Dict[str, Any]:
    """Execute complete Phase 2 NSGA-II workflow."""
    start_total_time = time.perf_counter()
    config = load_config("config.yaml")
    set_seed(seed, deterministic=True)
    
    # Mode configurations
    mode_configs = {
        "quick_debug": {"pop_size": 6, "generations": 3},
        "standard": {"pop_size": 16, "generations": 10},
        "final": {"pop_size": 24, "generations": 15}
    }
    
    selected_cfg = mode_configs.get(mode, mode_configs["standard"])
    pop_size = custom_pop or selected_cfg["pop_size"]
    generations = custom_gen or selected_cfg["generations"]
    
    logger.info("=" * 70)
    logger.info(f"STARTING PHASE 2 NSGA-II OPTIMIZATION [Mode: {mode.upper()}]")
    logger.info(f"Population: {pop_size} | Generations: {generations} | Seed: {seed}")
    logger.info("=" * 70)
    
    # Load dataset splits
    processed_dir = config["data"]["processed_dir"]
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    
    feature_pipeline = FeaturePipeline.load(os.path.join(config["logging"]["models_dir"], "feature_pipeline.pkl"))
    target_col = config["data"]["target_column"]
    
    # Initialize Evaluator (TRAIN and VAL only!)
    evaluator = GenomeEvaluator(
        train_df=train_df,
        val_df=val_df,
        feature_pipeline=feature_pipeline,
        target_column=target_col,
        device=torch.device("cpu"),
        early_stopping_patience=15
    )
    
    def log_gen_callback(gen: int, pop: List[Individual], pf: List[Individual], dur: float):
        best_mae = min(ind.metrics.validation_mae for ind in pop if ind.metrics)
        min_param = min(ind.metrics.trainable_parameters for ind in pop if ind.metrics)
        min_lat = min(ind.metrics.inference_latency_ms for ind in pop if ind.metrics)
        logger.info(
            f"Generation {gen:02d}/{generations:02d} | "
            f"PF Size: {len(pf):2d} | "
            f"Best Val MAE: {best_mae:.4f} | "
            f"Min Params: {min_param:5d} | "
            f"Min Lat: {min_lat:.4f} ms | "
            f"Duration: {dur:.2f}s"
        )
        
    optimizer = NSGA2Optimizer(
        evaluator=evaluator,
        population_size=pop_size,
        generations=generations,
        mutation_rate=0.15,
        crossover_rate=0.90,
        seed=seed,
        log_callback=log_gen_callback
    )
    
    # Execute Evolution
    final_population, pareto_front = optimizer.run()
    total_opt_time = time.perf_counter() - start_total_time
    
    # 1. Save results/nsga2/generations.csv & convergence.csv
    conv_df = optimizer.tracker.save_csv(os.path.join(log_dir, "convergence.csv"))
    conv_df.to_csv(os.path.join(log_dir, "generations.csv"), index=False)
    
    # 2. Generate convergence plot
    plot_path = os.path.join(log_dir, "nsga2_convergence.png")
    optimizer.tracker.generate_plot(plot_path)
    logger.info(f"Convergence plot saved to: {plot_path}")
    
    # 3. Save results/nsga2/population_history.csv
    pop_hist_df = pd.DataFrame(optimizer.population_history)
    pop_hist_df.to_csv(os.path.join(log_dir, "population_history.csv"), index=False)
    
    # 4. Save results/nsga2/pareto_front.csv
    pf_rows = []
    for ind in pareto_front:
        g = ind.genome
        m = ind.metrics
        pf_rows.append({
            "individual_id": ind.id,
            "generation": ind.generation,
            "rank": ind.rank,
            "crowding_distance": ind.crowding_distance,
            "validation_mae": m.validation_mae,
            "validation_rmse": m.validation_rmse,
            "generalization_gap": m.generalization_gap,
            "validation_loss_variance": m.val_loss_variance,
            "trainable_parameters": m.trainable_parameters,
            "inference_latency_ms": m.inference_latency_ms,
            "training_time_seconds": m.training_time_seconds,
            "hidden_dims": str(g.hidden_dims),
            "dropout_rate": g.dropout_rate,
            "activation": g.activation,
            "learning_rate": g.learning_rate,
            "weight_decay": g.weight_decay,
            "l1_lambda": g.l1_lambda,
            "huber_delta": g.huber_delta,
            "batch_size": g.batch_size,
            "gradient_clip_norm": g.gradient_clip_norm,
            "epochs": g.epochs
        })
    pf_df = pd.DataFrame(pf_rows)
    pf_df.to_csv(os.path.join(log_dir, "pareto_front.csv"), index=False)
    logger.info(f"Pareto front ({len(pareto_front)} individuals) saved to: {os.path.join(log_dir, 'pareto_front.csv')}")
    
    # 5. Candidate Selection (Documented Rule on Validation Space)
    selected_ind, candidates = select_pareto_candidates(pareto_front)
    
    cand_rows = []
    for c_name, c_ind in candidates.items():
        cand_rows.append({
            "candidate_role": c_name,
            "individual_id": c_ind.id,
            "validation_mae": c_ind.metrics.validation_mae,
            "validation_rmse": c_ind.metrics.validation_rmse,
            "generalization_gap": c_ind.metrics.generalization_gap,
            "val_loss_variance": c_ind.metrics.val_loss_variance,
            "trainable_parameters": c_ind.metrics.trainable_parameters,
            "inference_latency_ms": c_ind.metrics.inference_latency_ms,
            "hidden_dims": str(c_ind.genome.hidden_dims),
            "dropout": c_ind.genome.dropout_rate,
            "activation": c_ind.genome.activation,
            "learning_rate": c_ind.genome.learning_rate
        })
    cand_df = pd.DataFrame(cand_rows)
    cand_df.to_csv(os.path.join(log_dir, "final_candidates.csv"), index=False)
    
    selected_summary = {
        "selection_rule": "Candidate B: Balanced Compromise (Minimum normalized Euclidean distance to ideal point on validation Pareto front)",
        "selected_individual_id": selected_ind.id,
        "genome": selected_ind.genome.to_dict(),
        "validation_objectives": selected_ind.metrics.to_dict()
    }
    save_json(selected_summary, os.path.join(log_dir, "selected_model.json"))
    
    # 6. Final Evaluation on Untouched Held-Out Test Set
    logger.info("Evaluating selected Pareto candidate on untouched TEST set...")
    test_eval_results = retrain_and_evaluate_test(
        genome=selected_ind.genome,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_pipeline=feature_pipeline,
        target_column=target_col,
        seed=seed
    )
    save_json(test_eval_results, os.path.join(log_dir, "test_evaluation.json"))
    
    # 7. Summary Objective Summary
    obj_summary = {
        "optimization_mode": mode,
        "population_size": pop_size,
        "generations": generations,
        "seed": seed,
        "total_evaluations_attempted": evaluator.total_evaluations,
        "cached_evaluations_count": evaluator.cached_evaluations,
        "failed_evaluations_count": evaluator.failed_evaluations,
        "total_wallclock_seconds": round(total_opt_time, 2),
        "pareto_front_size": len(pareto_front),
        "best_validation_mae": float(pf_df["validation_mae"].min()),
        "min_trainable_parameters": int(pf_df["trainable_parameters"].min()),
        "min_inference_latency_ms": float(pf_df["inference_latency_ms"].min()),
        "min_generalization_gap": float(pf_df["generalization_gap"].min()),
        "selected_candidate": {
            "role": "Candidate B (Balanced Compromise)",
            "validation_mae": selected_ind.metrics.validation_mae,
            "validation_rmse": selected_ind.metrics.validation_rmse,
            "test_mae": test_eval_results["test_mae"],
            "test_rmse": test_eval_results["test_rmse"],
            "test_r2": test_eval_results["test_r2"],
            "parameters": test_eval_results["trainable_parameters"],
            "single_sample_latency_ms": test_eval_results["single_sample_latency_ms"]
        }
    }
    save_json(obj_summary, os.path.join(log_dir, "objective_summary.json"))
    
    # Generate Phase 2 Markdown Report
    generate_phase2_report(obj_summary, candidates, test_eval_results)
    
    logger.info("=" * 70)
    logger.info("PHASE 2 NSGA-II OPTIMIZATION COMPLETED")
    logger.info(f"Pareto Front Size: {len(pareto_front)}")
    logger.info(f"Selected Candidate Test MAE: {test_eval_results['test_mae']:.4f} | R²: {test_eval_results['test_r2']:.4f} | Params: {test_eval_results['trainable_parameters']}")
    logger.info("=" * 70)
    
    return obj_summary


def generate_phase2_report(
    summary: Dict[str, Any],
    candidates: Dict[str, Individual],
    test_results: Dict[str, Any]
):
    """Generate comprehensive PHASE2_REPORT.md."""
    report_path = "results/nsga2/PHASE2_REPORT.md"
    
    md = []
    md.append("# PHASE 2 REPORT: Multi-Objective Evolutionary Hyperparameter Optimization (NSGA-II)\n")
    md.append("**Project:** Adaptive Multi-Objective Student Performance Prediction Under Non-Stationary Distribution Drift  ")
    md.append("**Stage:** Phase 2 (NSGA-II Implementation & Pareto Trade-Off Discovery)  \n")
    md.append("---\n")
    md.append("> **What changed and why:**  \n")
    md.append("> NSGA-II was introduced to evolve neural-network hyperparameters across multiple competing objectives instead of optimizing predictive error alone.\n")
    md.append("---\n")
    
    md.append("## 1. Problem Formulation & Multi-Objective Motivation\n")
    md.append("Single-objective hyperparameter optimization (minimizing predictive error alone) produces models that may overfit, require excessive parameter budgets, or exhibit unstable training dynamics under curriculum distribution drift. In high-stakes educational deployment, an ML system must explicitly balance predictive accuracy with computational efficiency, generalization stability, and inference speed. NSGA-II discovers a non-dominated Pareto front of architectural compromises without requiring arbitrary scalarization weights.\n")
    
    md.append("## 2. Genome Representation & Search Space\n")
    md.append("Each individual encodes a 10-gene hyperparameter chromosome:")
    md.append("* `hidden_dims`: Architecture topology `[32]`, `[64]`, `[128]`, `[32, 16]`, `[64, 32]`, `[128, 64]`, `[128, 64, 32]`")
    md.append(r"* `dropout_rate`: Continuous $\in [0.0, 0.5]$")
    md.append(r"* `activation`: Categorical $\in$ `{relu, gelu, tanh}`")
    md.append(r"* `learning_rate`: Log-uniform continuous $\in [10^{-4}, 10^{-2}]$")
    md.append(r"* `weight_decay`: Log-uniform continuous $\in [10^{-6}, 10^{-2}]$")
    md.append(r"* `l1_lambda`: Continuous $\in [0.0, 10^{-3}]$")
    md.append(r"* `huber_delta`: Continuous $\in [0.5, 2.0]$")
    md.append(r"* `batch_size`: Discrete $\in$ `{16, 32, 64}`")
    md.append(r"* `gradient_clip_norm`: Discrete $\in$ `{0.5, 1.0, 2.0}`")
    md.append(r"* `epochs`: Bounded discrete $\in [30, 150]$" + "\n")
    
    md.append("## 3. The 6 Minimization Objectives\n")
    md.append(r"1. **$f_1$ (Predictive Error):** Validation Mean Absolute Error ($\text{MAE}_{\text{val}}$).")
    md.append(r"2. **$f_2$ (Generalization / Overfitting):** Absolute generalization gap $|L_{\text{val}} - L_{\text{train}}|$.")
    md.append(r"3. **$f_3$ (Loss Stability):** Variance of validation loss across training epochs $\text{Var}(L_{\text{val}})$.")
    md.append(r"4. **$f_4$ (Parameter Efficiency):** Total count of trainable weights and biases.")
    md.append(r"5. **$f_5$ (Inference Latency):** Single-sample inference latency benchmarked on CPU (ms).")
    md.append(r"6. **$f_6$ (Computational Cost):** Total wall-clock model training time (seconds)." + "\n")
    
    md.append("## 4. NSGA-II Algorithmic Operators\n")
    md.append(r"* **Non-Dominated Sorting:** Standard fast sorting algorithm partitioning populations into successive Pareto fronts $F_0, F_1, \dots$")
    md.append("* **Crowding Distance:** Density estimation in normalized objective space to prioritize diverse solutions.")
    md.append("* **Tournament Selection:** Binary tournament using crowded comparison operator (rank priority, crowding tie-breaker).")
    md.append(r"* **Crossover:** Discrete uniform exchange for architectures/activations; Simulated Binary Crossover (SBX, $\eta=2$) for continuous hyperparameters.")
    md.append("* **Mutation:** Point mutations with Gaussian perturbations in linear and log domains with automatic genome repair.")
    md.append(r"* **Elitist Environmental Selection:** Combines parent and offspring pools ($2N$) and preserves the top $N$ individuals across non-dominated fronts." + "\n")
    
    md.append("## 5. Strict Test Isolation & Caching\n")
    md.append("* **Zero Test Leakage:** The 100-student held-out test cohort was strictly excluded from `GenomeEvaluator`. No test evaluations occurred during any generation of the evolutionary run.")
    md.append("* **Exact Canonical Caching:** Genomes are mapped to canonical JSON and SHA-256 hashed. Re-sampled or surviving identical configurations are retrieved instantly, saving substantial compute.\n")
    
    md.append("## 6. Evolutionary Optimization Results\n")
    md.append(f"* **Population Size:** {summary['population_size']} | **Generations:** {summary['generations']}")
    md.append(f"* **Total Evaluations Attempted:** {summary['total_evaluations_attempted']}")
    md.append(f"* **Cached Evaluations Count:** {summary['cached_evaluations_count']}")
    md.append(f"* **Failed Evaluations Count:** {summary['failed_evaluations_count']}")
    md.append(f"* **Pareto Front Size:** {summary['pareto_front_size']} non-dominated configurations discovered")
    md.append(f"* **Total Optimization Runtime:** {summary['total_wallclock_seconds']:.2f} seconds\n")
    
    md.append("### Pareto Candidates Discovered\n")
    md.append("| Candidate Role | Architecture | Activation | Params | Val MAE | Gen Gap | Latency (ms) |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for c_role, c_ind in candidates.items():
        g = c_ind.genome
        m = c_ind.metrics
        md.append(f"| **{c_role}** | `{g.hidden_dims}` | `{g.activation}` | {m.trainable_parameters} | {m.validation_mae:.4f} | {m.generalization_gap:.4f} | {m.inference_latency_ms:.4f} |")
        
    md.append("\n## 7. Final Held-Out Test Evaluation\n")
    md.append("The selected candidate (Candidate B: Balanced Compromise) was retrained from scratch on the training cohort and evaluated once on the untouched held-out test set:\n")
    md.append("| Model System | Test MAE | Test RMSE | Test $R^2$ | Trainable Params | Single Latency |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    md.append("| **Naive Mean Baseline** | 0.8159 | 1.0459 | -0.0001 | 0 | — |")
    md.append("| **Phase 1 Deep Baseline** | 0.8773 | 1.0851 | -0.0765 | 3,585 | 0.0494 ms |")
    md.append(f"| **NSGA-II Selected Model** | **{test_results['test_mae']:.4f}** | **{test_results['test_rmse']:.4f}** | **{test_results['test_r2']:.4f}** | **{test_results['trainable_parameters']}** | **{test_results['single_sample_latency_ms']:.4f} ms** |\n")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
        
    logger.info(f"Phase 2 report generated at: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="standard", choices=["quick_debug", "standard", "final"])
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    run_nsga2(mode=args.mode, seed=args.seed, custom_pop=args.pop_size, custom_gen=args.generations)
