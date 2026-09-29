"""
src/adaptation/reoptimizer.py
Adaptive Evolutionary Re-Optimizer for Phase 4.
Warm-starts the NSGA-II multi-objective optimizer directly from the frozen Phase 2 Pareto front
rather than training random architectures from scratch.
Re-optimizes candidate architectures across the 6 competing objectives on drifted student distributions.
Outputs the newly adapted Pareto front and retrains the selected balanced model checkpoint.
"""

import os
import ast
import json
import time
import argparse
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch

from src.utils import set_seed, load_config, compute_metrics, count_parameters, measure_inference_latency, save_json
from src.features import FeaturePipeline
from src.model import build_model
from src.loss import RegularizedHuberLoss
from evaluate import evaluate_model
from src.nsga2.genome import Genome, repair_genome
from src.nsga2.population import Individual
from src.nsga2.sorting import fast_non_dominated_sort, calculate_crowding_distance
from src.nsga2.selection import tournament_selection
from src.nsga2.crossover import crossover
from src.nsga2.mutation import mutate
from src.nsga2.evolution import environmental_selection
from src.nsga2.evaluation import GenomeEvaluator
from src.nsga2.convergence import ConvergenceTracker

logger = logging.getLogger(__name__)


def load_pareto_genomes(pareto_csv_path: str = "results/nsga2/pareto_front.csv") -> List[Genome]:
    """Load genotypes from Phase 2 Pareto front CSV for warm-start initialization."""
    if not os.path.exists(pareto_csv_path):
        raise FileNotFoundError(f"Pareto front CSV not found at: {pareto_csv_path}")
        
    df = pd.read_csv(pareto_csv_path)
    genomes: List[Genome] = []
    
    for _, row in df.iterrows():
        # Parse hidden dims from string representation if needed
        h_dims = row["hidden_dims"]
        if isinstance(h_dims, str):
            h_dims = ast.literal_eval(h_dims)
            
        genome = Genome(
            hidden_dims=list(h_dims),
            dropout_rate=float(row["dropout_rate"]),
            activation=str(row["activation"]),
            learning_rate=float(row["learning_rate"]),
            weight_decay=float(row["weight_decay"]),
            l1_lambda=float(row["l1_lambda"]),
            huber_delta=float(row["huber_delta"]),
            batch_size=int(row["batch_size"]),
            gradient_clip_norm=float(row["gradient_clip_norm"]),
            epochs=int(row["epochs"])
        )
        genome = repair_genome(genome)
        genomes.append(genome)
        
    return genomes


def warm_start_population(
    seed_genomes: List[Genome],
    pop_size: int,
    rng: np.random.Generator,
    crossover_rate: float = 0.90,
    mutation_rate: float = 0.20
) -> List[Individual]:
    """
    Initialize Generation 0 of adaptive re-optimization using Phase 2 Pareto genotypes.
    Fills remaining population slots by breeding and mutating seed Pareto individuals.
    """
    population: List[Individual] = []
    
    # 1. Add existing Pareto candidates directly
    for i, g in enumerate(seed_genomes[:pop_size]):
        ind = Individual(genome=g, generation=0, individual_id=f"adapt_warm_{i:03d}")
        population.append(ind)
        
    # 2. Fill remaining slots by genetic variation of Pareto seeds
    idx = len(population)
    while len(population) < pop_size:
        # Sample two parent seeds
        p1 = rng.choice(seed_genomes)
        p2 = rng.choice(seed_genomes)
        
        if rng.random() < crossover_rate:
            c1_g, c2_g = crossover(p1, p2, rng=rng)
        else:
            c1_g, c2_g = p1, p2
            
        c1_g = mutate(c1_g, mutation_rate=mutation_rate, rng=rng)
        ind = Individual(genome=c1_g, generation=0, individual_id=f"adapt_warm_{idx:03d}")
        population.append(ind)
        idx += 1
        
    return population


class AdaptiveReOptimizer:
    """
    Multi-Objective Evolutionary Re-Optimizer for Online Covariate & Concept Drift Adaptation.
    """
    def __init__(
        self,
        evaluator: GenomeEvaluator,
        seed_genomes: List[Genome],
        population_size: int = 16,
        generations: int = 5,
        mutation_rate: float = 0.20,
        crossover_rate: float = 0.90,
        seed: int = 42
    ):
        self.evaluator = evaluator
        self.seed_genomes = seed_genomes
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        self.tracker = ConvergenceTracker()
        self.all_evaluated_individuals: List[Individual] = []
        self.population_history: List[Dict[str, Any]] = []

    def run(self) -> Tuple[List[Individual], List[Individual]]:
        """
        Execute warm-started NSGA-II evolutionary adaptation loop.
        
        Returns:
            final_population: List[Individual]
            adapted_pareto_front: List[Individual] (Non-dominated front F0)
        """
        set_seed(self.seed, deterministic=True)
        self.rng = np.random.default_rng(self.seed)
        
        print("=" * 75)
        print(f"STARTING ADAPTIVE NSGA-II RE-OPTIMIZATION [Generations: {self.generations}, Pop: {self.population_size}]")
        print(f"Warm-started from {len(self.seed_genomes)} Phase 2 Pareto Genotypes")
        print("=" * 75)
        
        # 1. Warm-start Generation 0
        t0_start = time.perf_counter()
        population = warm_start_population(
            seed_genomes=self.seed_genomes,
            pop_size=self.population_size,
            rng=self.rng,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate
        )
        
        # Evaluate Generation 0 on drifted environment
        for ind in population:
            metrics, cached = self.evaluator.evaluate(ind.genome)
            ind.set_evaluation_result(metrics, cached=cached)
            self.all_evaluated_individuals.append(ind)
            
        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            calculate_crowding_distance(front)
            
        t0_duration = time.perf_counter() - t0_start
        pareto_front_0 = fronts[0]
        self.tracker.record_generation(0, population, pareto_front_0, elapsed_sec=t0_duration)
        
        best_mae_0 = min(ind.metrics.validation_mae for ind in pareto_front_0)
        print(f"Generation 00/{self.generations:02d} (Warm-Start) | PF Size: {len(pareto_front_0):2d} | Best Val MAE: {best_mae_0:.4f} | Time: {t0_duration:.2f}s")
        
        # 2. Evolutionary Adaptation Loop
        curr_population = population
        for gen in range(1, self.generations + 1):
            t_gen_start = time.perf_counter()
            
            # Tournament selection
            mating_pool = tournament_selection(curr_population, self.population_size, self.rng)
            
            # Offspring generation
            offspring_pool: List[Individual] = []
            for i in range(0, self.population_size, 2):
                p1 = mating_pool[i]
                p2 = mating_pool[(i + 1) % self.population_size]
                
                if self.rng.random() < self.crossover_rate:
                    c1_g, c2_g = crossover(p1.genome, p2.genome, rng=self.rng)
                else:
                    c1_g, c2_g = p1.genome, p2.genome
                    
                c1_g = mutate(c1_g, mutation_rate=self.mutation_rate, rng=self.rng)
                c2_g = mutate(c2_g, mutation_rate=self.mutation_rate, rng=self.rng)
                
                off1 = Individual(genome=c1_g, generation=gen, individual_id=f"adapt_g{gen:02d}_{len(offspring_pool):03d}")
                off2 = Individual(genome=c2_g, generation=gen, individual_id=f"adapt_g{gen:02d}_{len(offspring_pool)+1:03d}")
                
                offspring_pool.extend([off1, off2])
                
            offspring_pool = offspring_pool[:self.population_size]
            
            # Evaluate offspring
            for off in offspring_pool:
                metrics, cached = self.evaluator.evaluate(off.genome)
                off.set_evaluation_result(metrics, cached=cached)
                self.all_evaluated_individuals.append(off)
                
            # Elitist Environmental Selection (2N -> N)
            combined_pool = curr_population + offspring_pool
            curr_population = environmental_selection(combined_pool, self.population_size)
            
            t_gen_dur = time.perf_counter() - t_gen_start
            current_pf = [ind for ind in curr_population if ind.rank == 0]
            self.tracker.record_generation(gen, curr_population, current_pf, elapsed_sec=t_gen_dur)
            
            best_mae = min(ind.metrics.validation_mae for ind in current_pf)
            print(f"Generation {gen:02d}/{self.generations:02d} | PF Size: {len(current_pf):2d} | Best Val MAE: {best_mae:.4f} | Time: {t_gen_dur:.2f}s")
            
        final_fronts = fast_non_dominated_sort(curr_population)
        adapted_pareto_front = final_fronts[0]
        for front in final_fronts:
            calculate_crowding_distance(front)
            
        print("=" * 75)
        print(f"ADAPTIVE RE-OPTIMIZATION COMPLETE: {len(adapted_pareto_front)} Non-Dominated Solutions Discovered")
        print("=" * 75)
        
        return curr_population, adapted_pareto_front


def select_best_adapted_candidate(pareto_front: List[Individual]) -> Individual:
    """
    Select Candidate B (Balanced Compromise): Minimum Euclidean distance to normalized ideal point.
    """
    if not pareto_front:
        raise ValueError("Pareto front is empty!")
        
    obj_matrix = np.array([ind.objectives for ind in pareto_front])
    min_vals = np.min(obj_matrix, axis=0)
    max_vals = np.max(obj_matrix, axis=0)
    ranges = max_vals - min_vals
    ranges[ranges < 1e-12] = 1.0
    
    norm_matrix = (obj_matrix - min_vals) / ranges
    distances = np.linalg.norm(norm_matrix, axis=1)
    best_idx = int(np.argmin(distances))
    return pareto_front[best_idx]


def retrain_adapted_model(
    genome: Genome,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_pipeline: FeaturePipeline,
    target_col: str = "next_semester_sgpa",
    device: Optional[torch.device] = None,
    seed: int = 42
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Retrain the selected adapted architecture on the adaptation dataset with early stopping.
    """
    set_seed(seed, deterministic=True)
    device = device or torch.device("cpu")
    
    X_train = feature_pipeline.transform(train_df)
    y_train = train_df[target_col].values.astype(np.float32)
    X_val = feature_pipeline.transform(val_df)
    y_val = val_df[target_col].values.astype(np.float32)
    
    model = build_model(genome.to_dict(), input_dim=X_train.shape[1])
    model.to(device)
    
    criterion = RegularizedHuberLoss(
        delta=genome.huber_delta,
        l1_lambda=genome.l1_lambda
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=genome.learning_rate,
        weight_decay=genome.weight_decay
    )
    
    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=genome.batch_size, shuffle=True)
    
    best_val_mae = float("inf")
    best_state = None
    best_epoch = 0
    
    for epoch in range(1, genome.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss, _ = criterion(pred, yb, model=model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), genome.gradient_clip_norm)
            optimizer.step()
            
        # Validate
        model.eval()
        val_metrics, _ = evaluate_model(model, X_val, y_val, batch_size=64)
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    
    final_val_metrics, _ = evaluate_model(model, X_val, y_val)
    return model, {
        "best_epoch": best_epoch,
        "validation_mae": final_val_metrics["mae"],
        "validation_rmse": final_val_metrics["rmse"],
        "validation_r2": final_val_metrics["r2"]
    }


def split_cohort_student_level(
    df: pd.DataFrame,
    adaptation_ratio: float = 0.50,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split drifted development cohort strictly at the student level into disjoint sets:
    1. Adaptation Cohort (50% of students):
       - Train subset (70% of adaptation students)
       - Calibration subset (30% of adaptation students)
    2. Held-Out Recovery Cohort (50% of students):
       - Never seen during adaptation search or calibration fitting.
    """
    unique_students = np.array(sorted(df["student_id"].unique()))
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(unique_students)
    
    n_total_students = len(unique_students)
    n_adapt_students = int(round(adaptation_ratio * n_total_students))
    
    adapt_student_ids = set(permuted[:n_adapt_students])
    recovery_student_ids = set(permuted[n_adapt_students:])
    
    adapt_df = df[df["student_id"].isin(adapt_student_ids)].copy().reset_index(drop=True)
    recovery_df = df[df["student_id"].isin(recovery_student_ids)].copy().reset_index(drop=True)
    
    # Sub-split adaptation students into train (70%) and calibration (30%)
    adapt_students_list = sorted(list(adapt_student_ids))
    perm_adapt = rng.permutation(adapt_students_list)
    n_train_students = int(round(0.70 * len(adapt_students_list)))
    
    train_ids = set(perm_adapt[:n_train_students])
    calib_ids = set(perm_adapt[n_train_students:])
    
    adapt_train_df = adapt_df[adapt_df["student_id"].isin(train_ids)].copy().reset_index(drop=True)
    adapt_calib_df = adapt_df[adapt_df["student_id"].isin(calib_ids)].copy().reset_index(drop=True)
    
    return adapt_train_df, adapt_calib_df, recovery_df


def run_adaptation_reoptimization(
    drift_scenario_csv: str = "results/drift/datasets/scenario_e_compound_stress.csv",
    adapt_train_df: Optional[pd.DataFrame] = None,
    adapt_val_df: Optional[pd.DataFrame] = None,
    models_output_dir: str = "models/adaptation",
    results_output_dir: str = "results/adaptation",
    generations: int = 5,
    population_size: int = 16,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Execute full warm-started Phase 4 adaptive re-optimization.
    Saves adapted Pareto front, selected checkpoint, and optimization summary.
    Guarantees recovery cohort never participates in optimization.
    """
    os.makedirs(models_output_dir, exist_ok=True)
    os.makedirs(results_output_dir, exist_ok=True)
    
    # 1. Load seed genomes from Phase 2
    pareto_csv = "results/nsga2/pareto_front.csv"
    seed_genomes = load_pareto_genomes(pareto_csv)
    
    # 2. Prepare adaptation train and validation splits from drifted cohort
    if adapt_train_df is None or adapt_val_df is None:
        drifted_df = pd.read_csv(drift_scenario_csv)
        adapt_train_df, adapt_val_df, recovery_df = split_cohort_student_level(drifted_df, seed=seed)
        
        # Save adaptation split
        split_records = []
        for s_id in adapt_train_df["student_id"].unique():
            split_records.append({"student_id": s_id, "scenario": os.path.basename(drift_scenario_csv), "partition": "adaptation_train"})
        for s_id in adapt_val_df["student_id"].unique():
            split_records.append({"student_id": s_id, "scenario": os.path.basename(drift_scenario_csv), "partition": "adaptation_calibration"})
        for s_id in recovery_df["student_id"].unique():
            split_records.append({"student_id": s_id, "scenario": os.path.basename(drift_scenario_csv), "partition": "recovery"})
        pd.DataFrame(split_records).to_csv(os.path.join(results_output_dir, "adaptation_split.csv"), index=False)
        
    feature_pipeline = FeaturePipeline.load("models/feature_pipeline.pkl")
    
    # 3. Setup evaluator
    evaluator = GenomeEvaluator(
        train_df=adapt_train_df,
        val_df=adapt_val_df,
        feature_pipeline=feature_pipeline,
        target_column="next_semester_sgpa",
        device=torch.device("cpu"),
        early_stopping_patience=10
    )
    
    # 4. Run re-optimizer
    optimizer = AdaptiveReOptimizer(
        evaluator=evaluator,
        seed_genomes=seed_genomes,
        population_size=population_size,
        generations=generations,
        seed=seed
    )
    final_pop, adapted_pf = optimizer.run()
    
    # 5. Save adapted Pareto front CSV
    pf_rows = []
    for ind in adapted_pf:
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
            "val_loss_variance": m.val_loss_variance,
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
    pf_path = os.path.join(results_output_dir, "adapted_pareto_front.csv")
    pf_df.to_csv(pf_path, index=False)
    print(f"Adapted Pareto front saved to: {pf_path}")
    
    # 6. Select balanced candidate and retrain
    selected_ind = select_best_adapted_candidate(adapted_pf)
    retrained_model, train_summary = retrain_adapted_model(
        genome=selected_ind.genome,
        train_df=adapt_train_df,
        val_df=adapt_val_df,
        feature_pipeline=feature_pipeline,
        seed=seed
    )
    
    # Save adapted model checkpoint
    model_ckpt_path = os.path.join(models_output_dir, "adapted_model.pt")
    torch.save({
        "model_state_dict": retrained_model.state_dict(),
        "genome": selected_ind.genome.to_dict(),
        "metrics": train_summary,
        "adapted_from_scenario": os.path.basename(drift_scenario_csv)
    }, model_ckpt_path)
    print(f"Retrained adapted model checkpoint saved to: {model_ckpt_path}")
    
    summary = {
        "generations": generations,
        "population_size": population_size,
        "adapted_pareto_front_size": len(adapted_pf),
        "selected_candidate_id": selected_ind.id,
        "selected_candidate_genome": selected_ind.genome.to_dict(),
        "adaptation_validation_metrics": train_summary
    }
    summary_path = os.path.join(results_output_dir, "reoptimization_summary.json")
    save_json(summary, summary_path)
    print(f"Re-optimization summary saved to: {summary_path}\n")
    
    # Save adaptation_results.csv (Part 13)
    adapt_results_row = [{
        "scenario": os.path.basename(drift_scenario_csv),
        "generations": generations,
        "population_size": population_size,
        "adapted_pareto_front_size": len(adapted_pf),
        "selected_genome_hidden_dims": str(selected_ind.genome.hidden_dims),
        "selected_activation": selected_ind.genome.activation,
        "adapt_val_mae": train_summary["validation_mae"],
        "adapt_val_rmse": train_summary["validation_rmse"],
        "adapt_val_r2": train_summary["validation_r2"],
        "best_epoch": train_summary["best_epoch"]
    }]
    pd.DataFrame(adapt_results_row).to_csv(os.path.join(results_output_dir, "adaptation_results.csv"), index=False)
    
    return summary
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="results/drift/datasets/scenario_e_compound_stress.csv")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--pop-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    run_adaptation_reoptimization(
        drift_scenario_csv=args.scenario,
        generations=args.generations,
        population_size=args.pop_size,
        seed=args.seed
    )
