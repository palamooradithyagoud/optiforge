"""
src/adaptation/visualize_adaptation.py
Publication-ready visualizer for Phase 4.
Produces 4 balanced, symmetric 300-DPI figures saved to results/adaptation/:
1. pareto_adaptation_shift.png: Pareto front shift (Phase 2 Baseline vs. Phase 4 Adapted).
2. recovery_waterfall.png: 4-Stage Performance Progression (Clean -> Degraded -> Adapted -> Calibrated).
3. conformal_coverage_intervals.png: Sequential 90% Conformal Prediction Intervals & Empirical Coverage.
4. explainability_shift.png: Global Feature Importance Shift & Local Student Explanation Waterfall.
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.features import FeaturePipeline
from src.adaptation.calibration import SplitConformalCalibrator
from src.adaptation.explainability import IntegratedGradientsExplainer
from src.adaptation.evaluate_recovery import load_model_from_checkpoint


def setup_plot_style():
    """Apply clean, symmetric typography and aesthetic styling for publication figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.1,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.7,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#cccccc",
        "figure.autolayout": False,
        "figure.dpi": 300
    })


def plot_pareto_adaptation_shift(
    p2_pareto_csv: str = "results/nsga2/pareto_front.csv",
    p4_pareto_csv: str = "results/adaptation/adapted_pareto_front.csv",
    output_path: str = "results/adaptation/pareto_adaptation_shift.png"
):
    """
    Symmetric 1x2 panel comparing Pareto fronts across accuracy, complexity, and latency:
    - Left: Validation MAE vs. Trainable Parameters
    - Right: Validation MAE vs. Inference Latency
    """
    setup_plot_style()
    df_p2 = pd.read_csv(p2_pareto_csv)
    df_p4 = pd.read_csv(p4_pareto_csv)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle(
        "Multi-Objective Pareto Trade-Off Shift: Baseline vs. Adapted Front",
        fontsize=15,
        fontweight="bold",
        y=1.03
    )
    
    color_p2 = "#1f77b4"  # Blue
    color_p4 = "#d62728"  # Coral Red
    
    # Panel 1: MAE vs Params
    ax1.scatter(df_p2["trainable_parameters"], df_p2["validation_mae"], color=color_p2, s=80, alpha=0.85, label=f"Phase 2 Pareto Front (N={len(df_p2)})", edgecolors="#333333")
    ax1.scatter(df_p4["trainable_parameters"], df_p4["validation_mae"], color=color_p4, s=95, marker="^", alpha=0.9, label=f"Phase 4 Adapted Front (N={len(df_p4)})", edgecolors="#333333")
    ax1.set_title("Validation MAE vs. Parameter Efficiency", fontsize=12.5, fontweight="bold", pad=8)
    ax1.set_xlabel("Trainable Parameters", fontsize=11, fontweight="medium")
    ax1.set_ylabel("Validation MAE (SGPA)", fontsize=11, fontweight="medium")
    ax1.grid(True)
    ax1.legend(loc="upper right")
    
    # Panel 2: MAE vs Latency
    ax2.scatter(df_p2["inference_latency_ms"], df_p2["validation_mae"], color=color_p2, s=80, alpha=0.85, label=f"Phase 2 Pareto Front", edgecolors="#333333")
    ax2.scatter(df_p4["inference_latency_ms"], df_p4["validation_mae"], color=color_p4, s=95, marker="^", alpha=0.9, label=f"Phase 4 Adapted Front", edgecolors="#333333")
    ax2.set_title("Validation MAE vs. Inference Latency", fontsize=12.5, fontweight="bold", pad=8)
    ax2.set_xlabel("Inference Latency (ms)", fontsize=11, fontweight="medium")
    ax2.set_ylabel("Validation MAE (SGPA)", fontsize=11, fontweight="medium")
    ax2.grid(True)
    ax2.legend(loc="upper right")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Pareto adaptation shift plot to: {output_path}")


def plot_recovery_waterfall(
    benchmark_csv: str = "results/adaptation/recovery_benchmark.csv",
    output_path: str = "results/adaptation/recovery_waterfall.png"
):
    """
    Symmetric 1x2 panel visualizing the 4-stage performance recovery:
    - Left: MAE and RMSE across stages
    - Right: Prediction Bias Neutralization across stages
    """
    setup_plot_style()
    df = pd.read_csv(benchmark_csv)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle(
        "Empirical Closed-Loop Recovery Progression Across Pipeline Stages",
        fontsize=15,
        fontweight="bold",
        y=1.03
    )
    
    stage_labels = ["1. Clean\nBaseline", "2. Drift\nDegradation", "3. Re-Optimized\nRecovery", "4. Calibrated\nOutput"]
    x = np.arange(len(stage_labels))
    width = 0.35
    
    # Panel 1: Error metrics
    ax1.bar(x - width/2, df["mae"], width, label="MAE", color="#d62728", alpha=0.85, edgecolor="#333333")
    ax1.bar(x + width/2, df["rmse"], width, label="RMSE", color="#1f77b4", alpha=0.85, edgecolor="#333333")
    ax1.set_title("Predictive Error (MAE & RMSE)", fontsize=12.5, fontweight="bold", pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stage_labels, fontsize=10.5)
    ax1.set_ylabel("Error Metric (SGPA)", fontsize=11, fontweight="medium")
    ax1.set_ylim(0.0, max(df["rmse"]) * 1.25)
    for i in x:
        ax1.text(i - width/2, df["mae"].iloc[i] + 0.02, f"{df['mae'].iloc[i]:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax1.text(i + width/2, df["rmse"].iloc[i] + 0.02, f"{df['rmse'].iloc[i]:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax1.grid(True)
    ax1.legend(loc="upper right")
    
    # Panel 2: Directional Prediction Bias
    colors_bias = ["#2ca02c" if abs(b) < 0.05 else "#ff7f0e" for b in df["mean_bias"]]
    bars = ax2.bar(x, df["mean_bias"], width=0.5, color=colors_bias, alpha=0.85, edgecolor="#333333")
    ax2.axhline(0.0, color="#333333", linestyle="-", linewidth=1.0)
    ax2.axhline(0.05, color="#888888", linestyle=":", linewidth=1.0, label="Bias Tolerance Target (|Bias| < 0.05)")
    ax2.axhline(-0.05, color="#888888", linestyle=":", linewidth=1.0)
    ax2.set_title("Directional Prediction Bias Neutralization", fontsize=12.5, fontweight="bold", pad=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(stage_labels, fontsize=10.5)
    ax2.set_ylabel("Mean Prediction Bias (Predicted - True)", fontsize=11, fontweight="medium")
    for i in x:
        b_val = df["mean_bias"].iloc[i]
        offset = 0.015 if b_val >= 0 else -0.025
        ax2.text(i, b_val + offset, f"{b_val:+.4f}", ha="center", fontsize=9.5, fontweight="bold")
    ax2.grid(True)
    ax2.legend(loc="upper right")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved recovery waterfall plot to: {output_path}")


def plot_conformal_coverage_intervals(
    calibrator_path: str = "models/adaptation/calibrator.pkl",
    adapted_model_path: str = "models/adaptation/adapted_model.pt",
    eval_csv: str = "results/drift/datasets/scenario_e_compound_stress.csv",
    output_path: str = "results/adaptation/conformal_coverage_intervals.png",
    num_samples: int = 50
):
    """
    Symmetric publication chart showing sequential 90% Conformal Prediction Intervals
    with ground-truth observation points and empirical coverage containment.
    """
    setup_plot_style()
    calibrator = SplitConformalCalibrator.load(calibrator_path)
    feature_pipeline = FeaturePipeline.load("models/feature_pipeline.pkl")
    
    df = pd.read_csv(eval_csv).iloc[:num_samples].copy()
    X = feature_pipeline.transform(df)
    y_true = df["next_semester_sgpa"].values.astype(np.float64)
    
    device = torch.device("cpu")
    model = load_model_from_checkpoint(adapted_model_path, input_dim=X.shape[1], device=device)
    with torch.no_grad():
        raw_preds = model(torch.tensor(X, dtype=torch.float32)).cpu().numpy().ravel()
        
    y_cal, lowers, uppers = calibrator.predict_intervals(raw_preds, cohort_df=df)
    covered = (y_true >= lowers) & (y_true <= uppers)
    emp_cov = np.mean(covered) * 100.0
    
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    x_idx = np.arange(len(df))
    
    # Shaded conformal band
    ax.fill_between(x_idx, lowers, uppers, color="#1f77b4", alpha=0.25, label=f"90% Conformal Prediction Interval (Coverage: {emp_cov:.1f}%)")
    ax.plot(x_idx, y_cal, color="#1f77b4", linewidth=1.8, label="Calibrated Prediction Point", linestyle="--")
    
    # Points
    for i in x_idx:
        c = "#2ca02c" if covered[i] else "#d62728"
        ax.scatter(i, y_true[i], color=c, s=40, zorder=4, edgecolors="#333333")
    # Legend proxy points
    ax.scatter([], [], color="#2ca02c", s=40, label="Ground Truth (Covered)", edgecolors="#333333")
    ax.scatter([], [], color="#d62728", s=40, label="Ground Truth (Outside 90% Band)", edgecolors="#333333")
    
    ax.set_title(f"Split Conformal 90% Prediction Intervals on Compound Stress Cohort\n(Empirical Coverage: {emp_cov:.1f}% vs. Target: 90.0%)", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Sample Student Index", fontsize=11, fontweight="medium")
    ax.set_ylabel("Predicted Next-Semester SGPA", fontsize=11, fontweight="medium")
    ax.set_ylim(3.0, 10.5)
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=10)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved conformal coverage plot to: {output_path}")


def plot_explainability_shift(
    baseline_model_path: str = "models/nsga2/nsga2_selected_model.pt",
    adapted_model_path: str = "models/adaptation/adapted_model.pt",
    cohort_csv: str = "results/drift/datasets/scenario_e_compound_stress.csv",
    output_path: str = "results/adaptation/explainability_shift.png"
):
    """
    Symmetric 1x2 panel showing:
    - Left: Global Feature Importance Shift (Pre-Adaptation vs. Post-Adaptation)
    - Right: Local Waterfall Attribution for a High-Risk Student
    """
    setup_plot_style()
    feature_pipeline = FeaturePipeline.load("models/feature_pipeline.pkl")
    feat_names = feature_pipeline.feature_names
    
    df = pd.read_csv(cohort_csv)
    X = feature_pipeline.transform(df)
    device = torch.device("cpu")
    
    base_model = load_model_from_checkpoint(baseline_model_path, input_dim=X.shape[1], device=device)
    adapt_model = load_model_from_checkpoint(adapted_model_path, input_dim=X.shape[1], device=device)
    
    explainer_base = IntegratedGradientsExplainer(base_model, feat_names, steps=30, device=device)
    explainer_adapt = IntegratedGradientsExplainer(adapt_model, feat_names, steps=30, device=device)
    
    # 1. Global attributions on subset of 30 samples for fast clean rendering
    X_sub = X[:30]
    df_imp_base = explainer_base.compute_global_importance(X_sub)
    df_imp_adapt = explainer_adapt.compute_global_importance(X_sub)
    
    # Merge on feature and sort by adapted importance
    merged = pd.merge(df_imp_base[["feature", "mean_abs_attribution"]], df_imp_adapt[["feature", "mean_abs_attribution"]], on="feature", suffixes=("_base", "_adapt"))
    merged = merged.sort_values(by="mean_abs_attribution_adapt", ascending=True).tail(10)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    fig.suptitle(
        "Model Explainability via Integrated Gradients: Global Shift & Local Attribution",
        fontsize=15,
        fontweight="bold",
        y=1.03
    )
    
    y_pos = np.arange(len(merged))
    height = 0.38
    
    # Panel 1: Global Shift
    ax1.barh(y_pos - height/2, merged["mean_abs_attribution_base"], height, label="Phase 2 Baseline Model", color="#1f77b4", alpha=0.85, edgecolor="#333333")
    ax1.barh(y_pos + height/2, merged["mean_abs_attribution_adapt"], height, label="Phase 4 Adapted Model", color="#d62728", alpha=0.85, edgecolor="#333333")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(merged["feature"], fontsize=10)
    ax1.set_xlabel("Mean Absolute Attribution (Integrated Gradients)", fontsize=11, fontweight="medium")
    ax1.set_title("Top 10 Global Feature Importance Shift", fontsize=12.5, fontweight="bold", pad=8)
    ax1.grid(True)
    ax1.legend(loc="lower right")
    
    # Panel 2: Local Student Waterfall (Student 0)
    student_expl = explainer_adapt.get_student_explanation(X[0], student_id=df["student_id"].iloc[0])
    top_contribs = student_expl["top_contributions"][:8]
    c_feats = [c["feature"] for c in top_contribs][::-1]
    c_vals = [c["attribution"] for c in top_contribs][::-1]
    c_colors = ["#2ca02c" if val >= 0 else "#d62728" for val in c_vals]
    
    ax2.barh(c_feats, c_vals, color=c_colors, alpha=0.85, edgecolor="#333333", height=0.55)
    ax2.axvline(0.0, color="#333333", linestyle="-", linewidth=1.0)
    ax2.set_title(f"Local Attribution: {student_expl['student_id']}\n(Pred: {student_expl['predicted_sgpa']:.2f}, Baseline: {student_expl['baseline_sgpa']:.2f})", fontsize=12.5, fontweight="bold", pad=8)
    ax2.set_xlabel("Attribution Contribution to SGPA Prediction", fontsize=11, fontweight="medium")
    ax2.grid(True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved explainability shift plot to: {output_path}")


def generate_adaptation_visualizations(results_dir: str = "results/adaptation"):
    """Generate all 4 publication-quality figures for Phase 4."""
    print("=" * 70)
    print("GENERATING PHASE 4 PUBLICATION-READY VISUALIZATIONS")
    print("=" * 70)
    
    fig1 = os.path.join(results_dir, "pareto_adaptation_shift.png")
    fig2 = os.path.join(results_dir, "recovery_waterfall.png")
    fig3 = os.path.join(results_dir, "conformal_coverage_intervals.png")
    fig4 = os.path.join(results_dir, "explainability_shift.png")
    
    plot_pareto_adaptation_shift(output_path=fig1)
    plot_recovery_waterfall(output_path=fig2)
    plot_conformal_coverage_intervals(output_path=fig3)
    plot_explainability_shift(output_path=fig4)
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    generate_adaptation_visualizations()
