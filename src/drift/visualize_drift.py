"""
src/drift/visualize_drift.py
Publication-ready visualization generator for Phase 3 OOD stress testing and drift analysis.
Produces 3 balanced, symmetric figures saved to results/drift/:
1. drift_vs_degradation.png: Line plots of Delta MAE & Delta RMSE across perturbation severities.
2. feature_distribution_shift.png: Overlaid KDE/histogram distributions (Train vs. Clean Test vs. Drifted).
3. drift_pvalues_heatmap.png: Matrix heatmap of feature-level statistical drift significance (-log10 p-value).
"""

import os
import argparse
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde

from src.utils import set_seed, load_config
from src.drift.scenarios import SCENARIO_CONFIGS


def setup_plot_style():
    """Apply clean, elegant, symmetric typography and aesthetic styling for publication charts."""
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


def plot_drift_vs_degradation(
    ood_results_path: str = "results/drift/ood_results.csv",
    output_path: str = "results/drift/drift_vs_degradation.png"
):
    """
    Generate symmetric 2x2 grid visualizing performance degradation across perturbation severity:
    - Panel (0,0): Attendance Drift (-5% to -20%)
    - Panel (0,1): Academic Drift (-0.5 to -1.5 SGPA)
    - Panel (1,0): Backlog Surge (+1 to +3 Backlogs)
    - Panel (1,1): Comparative Bar Chart across all Scenario Families
    """
    setup_plot_style()
    df = pd.read_csv(ood_results_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    fig.suptitle(
        "Phase 3: Model Performance Degradation Under Controlled OOD Shifts",
        fontsize=16,
        fontweight="bold",
        y=1.02
    )
    
    color_mae = "#d62728"   # Coral Red
    color_rmse = "#1f77b4"  # Deep Blue
    
    # ----------------------------------------------------
    # Panel (0,0): Attendance Drift
    # ----------------------------------------------------
    ax0 = axes[0, 0]
    scen_a = df[df["family"].str.contains("Attendance Drift", case=False)].copy()
    scen_a["pct_val"] = scen_a["severity"].astype(float) * 100.0
    
    ax0.plot(scen_a["pct_val"], scen_a["delta_mae"], marker="o", linewidth=2.4, markersize=8, color=color_mae, label=r"$\Delta$ MAE")
    ax0.plot(scen_a["pct_val"], scen_a["delta_rmse"], marker="s", linewidth=2.4, markersize=8, color=color_rmse, label=r"$\Delta$ RMSE")
    ax0.axhline(0, color="#888888", linestyle=":", linewidth=1.2)
    ax0.set_title("Scenario A: Attendance Drift", fontsize=13, fontweight="bold", pad=10)
    ax0.set_xlabel("Attendance Drop (%)", fontsize=11, fontweight="medium")
    ax0.set_ylabel("Metric Degradation", fontsize=11, fontweight="medium")
    ax0.set_xticks(scen_a["pct_val"])
    ax0.set_xticklabels([f"-{int(x)}%" for x in scen_a["pct_val"]])
    ax0.grid(True)
    ax0.legend(loc="upper left")
    
    # ----------------------------------------------------
    # Panel (0,1): Academic Drift (Syllabus Shock)
    # ----------------------------------------------------
    ax1 = axes[0, 1]
    scen_b = df[df["family"].str.contains("Academic Drift", case=False)].copy()
    scen_b["shift_val"] = scen_b["severity"].astype(float).abs()
    
    ax1.plot(scen_b["shift_val"], scen_b["delta_mae"], marker="o", linewidth=2.4, markersize=8, color=color_mae, label=r"$\Delta$ MAE")
    ax1.plot(scen_b["shift_val"], scen_b["delta_rmse"], marker="s", linewidth=2.4, markersize=8, color=color_rmse, label=r"$\Delta$ RMSE")
    ax1.axhline(0, color="#888888", linestyle=":", linewidth=1.2)
    ax1.set_title("Scenario B: Academic Drift (Syllabus Shock)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Historical SGPA Deflation Magnitude", fontsize=11, fontweight="medium")
    ax1.set_ylabel("Metric Degradation", fontsize=11, fontweight="medium")
    ax1.set_xticks(scen_b["shift_val"])
    ax1.set_xticklabels([f"-{x:.1f}" for x in scen_b["shift_val"]])
    ax1.grid(True)
    ax1.legend(loc="upper left")
    
    # ----------------------------------------------------
    # Panel (1,0): Backlog Surge
    # ----------------------------------------------------
    ax2 = axes[1, 0]
    scen_c = df[df["family"].str.contains("Backlog Surge", case=False)].copy()
    scen_c["surge_val"] = scen_c["severity"].astype(float)
    
    ax2.plot(scen_c["surge_val"], scen_c["delta_mae"], marker="o", linewidth=2.4, markersize=8, color=color_mae, label=r"$\Delta$ MAE")
    ax2.plot(scen_c["surge_val"], scen_c["delta_rmse"], marker="s", linewidth=2.4, markersize=8, color=color_rmse, label=r"$\Delta$ RMSE")
    ax2.axhline(0, color="#888888", linestyle=":", linewidth=1.2)
    ax2.set_title("Scenario C: Backlog Surge", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Backlog Count Increment", fontsize=11, fontweight="medium")
    ax2.set_ylabel("Metric Degradation", fontsize=11, fontweight="medium")
    ax2.set_xticks(scen_c["surge_val"])
    ax2.set_xticklabels([f"+{int(x)}" for x in scen_c["surge_val"]])
    ax2.grid(True)
    ax2.legend(loc="upper left")
    
    # ----------------------------------------------------
    # Panel (1,1): Comprehensive Degradation Across All Families
    # ----------------------------------------------------
    ax3 = axes[1, 1]
    # Pick representative key scenario from each family
    selected_scenarios = [
        ("Clean Baseline", 0.0, 0.0),
        ("Att -20%", float(scen_a[scen_a["severity"] == 0.20]["delta_mae"].iloc[0]), float(scen_a[scen_a["severity"] == 0.20]["delta_rmse"].iloc[0])),
        ("Acad -1.5", float(scen_b[scen_b["severity"] == -1.5]["delta_mae"].iloc[0]), float(scen_b[scen_b["severity"] == -1.5]["delta_rmse"].iloc[0])),
        ("Backlogs +3", float(scen_c[scen_c["severity"] == 3]["delta_mae"].iloc[0]), float(scen_c[scen_c["severity"] == 3]["delta_rmse"].iloc[0])),
        ("Cohort Shift", float(df[df["scenario_id"] == "scenario_d_cohort_shift"]["delta_mae"].iloc[0]), float(df[df["scenario_id"] == "scenario_d_cohort_shift"]["delta_rmse"].iloc[0])),
        ("Compound Shock", float(df[df["scenario_id"] == "scenario_e_compound_stress"]["delta_mae"].iloc[0]), float(df[df["scenario_id"] == "scenario_e_compound_stress"]["delta_rmse"].iloc[0]))
    ]
    
    names = [s[0] for s in selected_scenarios]
    d_maes = [s[1] for s in selected_scenarios]
    d_rmses = [s[2] for s in selected_scenarios]
    x_pos = np.arange(len(names))
    width = 0.38
    
    ax3.bar(x_pos - width/2, d_maes, width, label=r"$\Delta$ MAE", color=color_mae, alpha=0.85, edgecolor="#555555")
    ax3.bar(x_pos + width/2, d_rmses, width, label=r"$\Delta$ RMSE", color=color_rmse, alpha=0.85, edgecolor="#555555")
    ax3.set_title("Cross-Family Severity Comparison", fontsize=13, fontweight="bold", pad=10)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(names, rotation=25, ha="right", fontsize=9.5)
    ax3.set_ylabel("Metric Degradation", fontsize=11, fontweight="medium")
    ax3.grid(True)
    ax3.legend(loc="upper left")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved degradation curves to: {output_path}")


def plot_feature_distribution_shift(
    train_csv_path: str = "data/processed/train.csv",
    dev_csv_path: str = "data/processed/val.csv",
    drifted_csv_path: str = "results/drift/datasets/scenario_e_compound_stress.csv",
    output_path: str = "results/drift/feature_distribution_shift.png"
):
    """
    Generate symmetric 2x2 grid comparing feature probability distributions:
    - Train (Reference Cohort)
    - Clean Development Baseline (Untouched Val Cohort)
    - Scenario E (Compound Extreme Stress Cohort)
    """
    setup_plot_style()
    train_df = pd.read_csv(train_csv_path)
    dev_df = pd.read_csv(dev_csv_path)
    drift_df = pd.read_csv(drifted_csv_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    fig.suptitle(
        "Empirical Feature Distribution Shifts (Train vs. Clean Dev vs. Drifted Cohort)",
        fontsize=16,
        fontweight="bold",
        y=1.02
    )
    
    features_to_plot = [
        ("previous_attendance", "Attendance: previous_attendance (%)", (50, 100)),
        ("previous_sgpa", "Academic: previous_sgpa (Grade Points)", (3.5, 10.0)),
        ("active_backlogs_count", "Backlogs: active_backlogs_count (Count)", (-0.5, 6.5)),
        ("avg_subject_grade_point", "Grades: avg_subject_grade_point (0-10)", (3.5, 10.0))
    ]
    
    colors = {
        "Train": "#1f77b4",       # Blue
        "Clean Dev": "#2ca02c",   # Green
        "Drifted": "#d62728"      # Coral Red
    }
    
    for idx, (feat, title, xlim) in enumerate(features_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        train_vals = train_df[feat].dropna().values
        dev_vals = dev_df[feat].dropna().values
        drift_vals = drift_df[feat].dropna().values
        
        # Histograms with alpha transparency
        bins = np.linspace(xlim[0], xlim[1], 25)
        ax.hist(train_vals, bins=bins, density=True, alpha=0.35, color=colors["Train"], label="Train Cohort (N=600)")
        ax.hist(dev_vals, bins=bins, density=True, alpha=0.35, color=colors["Clean Dev"], label="Clean Dev Cohort (N=200)")
        ax.hist(drift_vals, bins=bins, density=True, alpha=0.35, color=colors["Drifted"], label="Compound Stress (N=200)")
        
        # Smooth KDE curves
        x_eval = np.linspace(xlim[0], xlim[1], 300)
        try:
            kde_train = gaussian_kde(train_vals)
            ax.plot(x_eval, kde_train(x_eval), color=colors["Train"], linewidth=2.2)
        except Exception:
            pass
            
        try:
            kde_dev = gaussian_kde(dev_vals)
            ax.plot(x_eval, kde_dev(x_eval), color=colors["Clean Dev"], linewidth=2.2)
        except Exception:
            pass
            
        try:
            kde_drift = gaussian_kde(drift_vals)
            ax.plot(x_eval, kde_drift(x_eval), color=colors["Drifted"], linewidth=2.2, linestyle="--")
        except Exception:
            pass
            
        # Add vertical mean markers
        ax.axvline(np.mean(train_vals), color=colors["Train"], linestyle=":", linewidth=1.5)
        ax.axvline(np.mean(test_vals), color=colors["Clean Test"], linestyle=":", linewidth=1.5)
        ax.axvline(np.mean(drift_vals), color=colors["Drifted"], linestyle=":", linewidth=1.5)
        
        ax.set_title(title, fontsize=12.5, fontweight="bold", pad=8)
        ax.set_xlabel(feat, fontsize=10.5)
        ax.set_ylabel("Probability Density", fontsize=10.5)
        ax.set_xlim(xlim)
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=9.5)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved feature shift distribution plot to: {output_path}")


def plot_drift_pvalues_heatmap(
    feature_report_path: str = "results/drift/feature_shift_report.json",
    output_path: str = "results/drift/drift_pvalues_heatmap.png"
):
    """
    Generate symmetric statistical significance heatmap:
    - Rows: Monitored Features (21 features)
    - Columns: Scenarios (Clean Baseline + 12 OOD scenarios)
    - Color scale: -log10(p-value) with alpha=0.05 threshold marker
    """
    setup_plot_style()
    import json
    with open(feature_report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
        
    scenario_order = [
        ("clean_test", "Clean Test"),
        ("scenario_a_attendance_drift_5pct", "Att -5%"),
        ("scenario_a_attendance_drift_10pct", "Att -10%"),
        ("scenario_a_attendance_drift_15pct", "Att -15%"),
        ("scenario_a_attendance_drift_20pct", "Att -20%"),
        ("scenario_b_academic_drift_shift_0.5", "Acad -0.5"),
        ("scenario_b_academic_drift_shift_1.0", "Acad -1.0"),
        ("scenario_b_academic_drift_shift_1.5", "Acad -1.5"),
        ("scenario_c_backlog_surge_plus_1", "Backlogs +1"),
        ("scenario_c_backlog_surge_plus_2", "Backlogs +2"),
        ("scenario_c_backlog_surge_plus_3", "Backlogs +3"),
        ("scenario_d_cohort_shift", "Cohort Shift"),
        ("scenario_e_compound_stress", "Compound Stress")
    ]
    
    # Filter scenarios present in report
    active_scenarios = [s for s in scenario_order if s[0] in report]
    scen_keys = [s[0] for s in active_scenarios]
    scen_labels = [s[1] for s in active_scenarios]
    
    # Collect all feature names
    first_key = scen_keys[0]
    features = list(report[first_key]["p_values"].keys())
    
    # Build -log10(p_value) matrix
    matrix = np.zeros((len(features), len(scen_keys)))
    for j, s_key in enumerate(scen_keys):
        p_dict = report[s_key]["p_values"]
        for i, f_name in enumerate(features):
            p_val = max(1e-15, float(p_dict.get(f_name, 1.0)))
            neg_log_p = -np.log10(p_val)
            # Cap at 10 for clean visualization
            matrix[i, j] = min(10.0, neg_log_p)
            
    fig, ax = plt.subplots(figsize=(15, 11), constrained_layout=True)
    
    # Custom colormap: white/light yellow -> orange -> crimson red
    cmap = plt.cm.YlOrRd
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0.0, vmax=10.0)
    
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(r"Statistical Significance: $-\log_{10}(p\text{-value})$", rotation=-90, va="bottom", fontsize=11, fontweight="medium")
    cbar.ax.axhline(-np.log10(0.05), color="#333333", linestyle="--", linewidth=1.5)
    cbar.ax.text(1.2, -np.log10(0.05), r"$\alpha=0.05$", color="#333333", va="center", fontsize=10)
    
    ax.set_xticks(np.arange(len(scen_labels)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(scen_labels, rotation=35, ha="right", fontsize=10.5, fontweight="medium")
    ax.set_yticklabels(features, fontsize=10)
    
    # Annotate significant cells with asterisks
    for i in range(len(features)):
        for j in range(len(scen_labels)):
            val = matrix[i, j]
            if val >= -np.log10(0.001):
                ax.text(j, i, "***", ha="center", va="center", color="#000000" if val < 6 else "#ffffff", fontsize=10, fontweight="bold")
            elif val >= -np.log10(0.01):
                ax.text(j, i, "**", ha="center", va="center", color="#000000" if val < 6 else "#ffffff", fontsize=10, fontweight="bold")
            elif val >= -np.log10(0.05):
                ax.text(j, i, "*", ha="center", va="center", color="#000000", fontsize=10)
                
    ax.set_title("Feature-Level Two-Sample KS Drift Significance Across Scenarios\n(*: p<0.05, **: p<0.01, ***: p<0.001)", fontsize=15, fontweight="bold", pad=12)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved drift p-values heatmap to: {output_path}")


def generate_all_visualizations(results_dir: str = "results/drift"):
    """Generate all three publication-quality figures."""
    print("=" * 70)
    print("GENERATING PHASE 3 PUBLICATION-READY VISUALIZATIONS")
    print("=" * 70)
    
    ood_results = os.path.join(results_dir, "ood_results.csv")
    fig1 = os.path.join(results_dir, "drift_vs_degradation.png")
    fig2 = os.path.join(results_dir, "feature_distribution_shift.png")
    fig3 = os.path.join(results_dir, "drift_pvalues_heatmap.png")
    
    plot_drift_vs_degradation(ood_results_path=ood_results, output_path=fig1)
    plot_feature_distribution_shift(output_path=fig2)
    plot_drift_pvalues_heatmap(output_path=fig3)
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    generate_all_visualizations()
