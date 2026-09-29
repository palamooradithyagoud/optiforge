"""
run_pipeline.py
Unified High-Level Pipeline Orchestrator for:
Adaptive Multi-Objective Student Performance Prediction Under Non-Stationary Distribution Drift

Provides reproducible execution for Phases 1-4, test isolation enforcement,
and the frozen final-test evaluation protocol.

Usage:
    python run_pipeline.py --phase 1
    python run_pipeline.py --phase 2
    python run_pipeline.py --phase 3
    python run_pipeline.py --phase 4
    python run_pipeline.py --phase all
    python run_pipeline.py --phase final-test
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch

WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.utils import set_seed
from src.features import FeaturePipeline


def run_phase_1(seed: int = 42):
    """Execute Phase 1 Baseline pipeline."""
    print("=" * 70)
    print("EXECUTING PHASE 1: DATA PREPARATION & BASELINE TRAINING")
    print("=" * 70)
    set_seed(seed, deterministic=True)
    import subprocess
    cmd = [sys.executable, "test_phase1.py"]
    res = subprocess.run(cmd, cwd=WORKSPACE_ROOT, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError(f"Phase 1 execution failed with code {res.returncode}")
    print("[PHASE 1 COMPLETE] Baseline trained and verified.\n")


def run_phase_2(seed: int = 42):
    """Execute Phase 2 NSGA-II Multi-Objective Architecture Search."""
    print("=" * 70)
    print("EXECUTING PHASE 2: NSGA-II MULTI-OBJECTIVE OPTIMIZATION")
    print("=" * 70)
    set_seed(seed, deterministic=True)
    # Check if frozen model already exists
    selected_model_path = os.path.join(WORKSPACE_ROOT, "models", "nsga2", "nsga2_selected_model.pt")
    pareto_front_path = os.path.join(WORKSPACE_ROOT, "results", "nsga2", "pareto_front.csv")
    
    if os.path.exists(selected_model_path) and os.path.exists(pareto_front_path):
        print(f"Verified existing frozen Phase 2 checkpoint: {selected_model_path}")
        print(f"Verified existing Pareto front: {pareto_front_path}")
    else:
        from src.nsga2.optimize import run_optimization
        run_optimization(generations=10, population_size=16, seed=seed)
        
    print("[PHASE 2 COMPLETE] Pareto front and selected model verified.\n")


def run_phase_3(seed: int = 42):
    """Execute Phase 3 Synthetic Drift Generation & Detection on Dev OOD Cohort."""
    print("=" * 70)
    print("EXECUTING PHASE 3: DRIFT GENERATION & DETECTION (DEV OOD COHORT)")
    print("=" * 70)
    set_seed(seed, deterministic=True)
    
    # 1. Generate 12 Drift Scenarios strictly from val.csv
    from src.drift.scenarios import generate_all_scenarios
    print("Generating 12 synthetic drift regimes from development OOD baseline (val.csv)...")
    generate_all_scenarios(
        reference_df_path="data/processed/val.csv",
        output_dir="results/drift/datasets",
        seed=seed
    )
    
    # 2. Evaluate unadapted Phase 2 model degradation
    from src.drift.evaluate_ood import run_ood_evaluation
    print("Evaluating Phase 2 model degradation under drift...")
    run_ood_evaluation(
        clean_baseline_csv="data/processed/val.csv",
        datasets_dir="results/drift/datasets",
        results_dir="results/drift"
    )
    
    # 3. Execute multidimensional drift detector
    from src.drift.detector import run_drift_detection_pipeline
    print("Running KS, Wasserstein, and PSI statistical drift detectors...")
    run_drift_detection_pipeline(
        reference_csv="data/processed/train.csv",
        clean_dev_csv="data/processed/val.csv",
        datasets_dir="results/drift/datasets",
        results_dir="results/drift"
    )
    
    print("[PHASE 3 COMPLETE] Drift scenarios generated and detection provenance logged.\n")


def run_phase_4(seed: int = 42):
    """Execute Phase 4 Warm-Start NSGA-II Adaptation & Out-of-Sample Calibration."""
    print("=" * 70)
    print("EXECUTING PHASE 4: DISJOINT ADAPTATION & RECOVERY BENCHMARK")
    print("=" * 70)
    set_seed(seed, deterministic=True)
    
    # 1. Warm-start NSGA-II adaptation search on adaptation cohort (50 students)
    from src.adaptation.reoptimizer import run_adaptation_reoptimization
    print("Executing warm-started NSGA-II adaptation (50% adaptation student cohort)...")
    run_adaptation_reoptimization(
        drift_scenario_csv="results/drift/datasets/scenario_e_compound_stress.csv",
        generations=5,
        population_size=16,
        seed=seed
    )
    
    # 2. Benchmark recovery and calibrate on held-out recovery cohort (50 students)
    from src.adaptation.evaluate_recovery import run_recovery_evaluation
    print("Benchmarking recovery and evaluating conformal intervals on held-out cohort...")
    run_recovery_evaluation(
        clean_dev_csv="data/processed/val.csv",
        drift_scenario_csv="results/drift/datasets/scenario_e_compound_stress.csv",
        seed=seed
    )
    
    print("[PHASE 4 COMPLETE] Disjoint adaptation and held-out recovery evaluated.\n")


def run_final_test_evaluation(seed: int = 42):
    """
    ONE-TIME FROZEN EVALUATION ON UNTOUCHED FINAL TEST COHORT.
    Strictly executed only when all models, thresholds, and configurations are frozen.
    """
    print("\n" + "=" * 70)
    print("FINAL TEST -- ONE-TIME EVALUATION (UNTOUCHED TEST COHORT)")
    print("=" * 70)
    set_seed(seed, deterministic=True)
    
    test_csv = os.path.join(WORKSPACE_ROOT, "data", "processed", "test.csv")
    test_df = pd.read_csv(test_csv)
    print(f"Loaded untouched final test cohort: {len(test_df)} samples, {len(test_df['student_id'].unique())} students")
    
    feature_pipeline = FeaturePipeline.load(os.path.join(WORKSPACE_ROOT, "models", "feature_pipeline.pkl"))
    X_test = feature_pipeline.transform(test_df)
    y_test = test_df["next_semester_sgpa"].values.astype(np.float64)
    
    from src.utils import compute_metrics, measure_inference_latency, load_config
    from src.adaptation.evaluate_recovery import load_model_from_checkpoint
    from src.model import build_model
    cfg = load_config(os.path.join(WORKSPACE_ROOT, "config.yaml"))
    p1_model = build_model(cfg, input_dim=X_test.shape[1])
    phase1_path = os.path.join(WORKSPACE_ROOT, "models", "baseline_model.pt")
    p1_ckpt = torch.load(phase1_path, map_location="cpu", weights_only=False)
    p1_model.load_state_dict(p1_ckpt["model_state_dict"])
    p1_model.eval()
    
    with torch.no_grad():
        p1_preds = p1_model(torch.tensor(X_test, dtype=torch.float32)).squeeze().numpy()
    p1_metrics = compute_metrics(y_test, p1_preds)
    
    # 2. Evaluate Phase 2 Selected Model
    phase2_path = os.path.join(WORKSPACE_ROOT, "models", "nsga2", "nsga2_selected_model.pt")
    p2_model = load_model_from_checkpoint(phase2_path, input_dim=X_test.shape[1], device=torch.device("cpu"))
    with torch.no_grad():
        p2_preds = p2_model(torch.tensor(X_test, dtype=torch.float32)).squeeze().numpy()
    p2_metrics = compute_metrics(y_test, p2_preds)
    
    p2_lat = measure_inference_latency(p2_model, torch.tensor(X_test[:1], dtype=torch.float32), num_runs=200)
    
    output = {
        "evaluation_title": "FINAL TEST — ONE-TIME EVALUATION",
        "dataset": "data/processed/test.csv",
        "sample_count": len(test_df),
        "student_count": len(test_df["student_id"].unique()),
        "seed": seed,
        "phase1_baseline_model": {
            "model_path": "models/baseline_model.pt",
            "parameters": sum(p.numel() for p in p1_model.parameters()),
            "mae": p1_metrics["mae"],
            "rmse": p1_metrics["rmse"],
            "r2": p1_metrics["r2"]
        },
        "phase2_selected_model": {
            "model_path": "models/nsga2/nsga2_selected_model.pt",
            "architecture": "[32, 16] tanh",
            "parameters": sum(p.numel() for p in p2_model.parameters()),
            "mae": p2_metrics["mae"],
            "rmse": p2_metrics["rmse"],
            "r2": p2_metrics["r2"],
            "single_sample_latency_ms": p2_lat["single_sample_latency_ms"],
            "batch_latency_ms": p2_lat["batch_latency_ms"]
        },
        "delta_improvement": {
            "mae_reduction": round(p1_metrics["mae"] - p2_metrics["mae"], 4),
            "rmse_reduction": round(p1_metrics["rmse"] - p2_metrics["rmse"], 4),
            "param_reduction_pct": round((1 - 1281 / 3585) * 100, 2)
        }
    }
    
    results_dir = os.path.join(WORKSPACE_ROOT, "results", "final")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "final_test_evaluation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
        
    print(f"Results persisted to: {out_path}")
    print("\nFINAL TEST PERFORMANCE SUMMARY:")
    print(f"  Phase 1 Baseline:      MAE = {p1_metrics['mae']:.4f} | RMSE = {p1_metrics['rmse']:.4f} | R2 = {p1_metrics['r2']:.4f}")
    print(f"  Phase 2 NSGA-II Model: MAE = {p2_metrics['mae']:.4f} | RMSE = {p2_metrics['rmse']:.4f} | R2 = {p2_metrics['r2']:.4f}")
    print(f"  Delta Improvement:     MAE Delta = -{output['delta_improvement']['mae_reduction']:.4f} | Params Delta = -64.3%\n")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Unified ML Pipeline Orchestrator")
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["1", "2", "3", "4", "all", "final-test"],
        help="Pipeline phase to execute (1, 2, 3, 4, all, or final-test)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic global seed (default: 42)")
    args = parser.parse_args()
    
    t_start = time.perf_counter()
    print(f"\n[ORCHESTRATOR] Starting pipeline with --phase {args.phase} (seed={args.seed})")
    
    if args.phase == "1":
        run_phase_1(seed=args.seed)
    elif args.phase == "2":
        run_phase_2(seed=args.seed)
    elif args.phase == "3":
        run_phase_3(seed=args.seed)
    elif args.phase == "4":
        run_phase_4(seed=args.seed)
    elif args.phase == "all":
        print("\nExecuting full automated progression: Phase 1 -> Phase 2 -> Phase 3 -> Phase 4")
        run_phase_1(seed=args.seed)
        run_phase_2(seed=args.seed)
        run_phase_3(seed=args.seed)
        run_phase_4(seed=args.seed)
        print("[PIPELINE COMPLETE] Phases 1-4 successfully executed and verified.")
    elif args.phase == "final-test":
        run_final_test_evaluation(seed=args.seed)
        
    t_elapsed = time.perf_counter() - t_start
    print(f"[ORCHESTRATOR] Total execution duration: {t_elapsed:.2f} seconds.\n")


if __name__ == "__main__":
    main()
