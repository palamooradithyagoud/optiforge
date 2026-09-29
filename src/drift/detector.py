"""
src/drift/detector.py
Statistical Distribution Drift Detection Engine for Phase 3.
Monitors incoming evaluation batches against the Phase 1 training set using:
1. Kolmogorov-Smirnov (KS) two-sample test (continuous distributional shift)
2. Wasserstein Distance / Earth Mover's Distance (magnitude of transportation cost)
3. Population Stability Index (PSI) (binned stability metric)

Provides explicit decision triggers (TRIGGER_ADAPTATION vs. PROCEED_TO_PREDICT) for Phase 4.
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from src.utils import load_config, save_json


DEFAULT_MONITORED_FEATURES = [
    # Historical performance
    "previous_sgpa",
    "previous_cgpa",
    "sgpa_change",
    "cgpa_change",
    # Attendance metrics
    "previous_attendance",
    "attendance_change",
    "avg_subject_attendance",
    "min_subject_attendance",
    "var_subject_attendance",
    # Academic performance
    "avg_subject_grade_point",
    "min_subject_grade_point",
    "max_subject_grade_point",
    "var_subject_grade_point",
    # Backlogs
    "active_backlogs_count",
    "failed_subjects_count",
    "backlog_change",
    # Credits
    "total_sem_credits_registered",
    "total_sem_credits_earned",
    "credit_completion_ratio",
    # Temporal context
    "current_semester",
    "has_historical_lag"
]

CORE_ACADEMIC_FEATURES = [
    "previous_sgpa",
    "previous_cgpa",
    "avg_subject_attendance",
    "active_backlogs_count"
]


class DriftDetector:
    """
    Statistical Drift Detector comparing incoming student batches against the reference training cohort.
    """
    def __init__(
        self,
        reference_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        alpha: float = 0.05,
        threshold_ratio: float = 0.20,
        psi_threshold: float = 0.15
    ):
        """
        Args:
            reference_df: Phase 1 reference training DataFrame (e.g. train.csv)
            feature_cols: List of numerical features to monitor (defaults to 21 features)
            alpha: Significance level for two-sample KS hypothesis testing (default: 0.05)
            threshold_ratio: Fraction of drifting features required to declare system drift (default: 0.20)
            psi_threshold: Mean PSI threshold across features to declare system drift (default: 0.15)
        """
        self.reference_df = reference_df
        if feature_cols is None:
            # Filter available columns from default list
            self.feature_cols = [c for c in DEFAULT_MONITORED_FEATURES if c in reference_df.columns]
        else:
            self.feature_cols = feature_cols
            
        self.alpha = float(alpha)
        self.threshold_ratio = float(threshold_ratio)
        self.psi_threshold = float(psi_threshold)
        
        # Precompute reference statistics and quantile bins for PSI
        self._precompute_reference_stats()

    def _precompute_reference_stats(self):
        """Precompute reference means, standard deviations, and quantile bin edges for fast evaluation."""
        self.ref_data: Dict[str, np.ndarray] = {}
        self.ref_stds: Dict[str, float] = {}
        self.ref_bins: Dict[str, np.ndarray] = {}
        
        for col in self.feature_cols:
            vals = self.reference_df[col].dropna().values.astype(np.float64)
            self.ref_data[col] = vals
            std = float(np.std(vals))
            self.ref_stds[col] = std if std > 1e-6 else 1.0
            
            # Construct 10 quantile bins for PSI
            quantiles = np.linspace(0.0, 1.0, 11)
            bins = np.quantile(vals, quantiles)
            # Ensure strictly monotonic bin edges
            bins = np.unique(bins)
            if len(bins) < 3:
                # If unique values are few (e.g. constant or discrete), fallback to uniform bounds
                v_min, v_max = float(np.min(vals)), float(np.max(vals))
                if abs(v_max - v_min) < 1e-6:
                    bins = np.array([v_min - 0.5, v_min + 0.5])
                else:
                    bins = np.linspace(v_min, v_max, 5)
            # Expand outer edges slightly to include all future samples
            bins[0] = -np.inf
            bins[-1] = np.inf
            self.ref_bins[col] = bins

    def compute_ks_test(self, feature: str, current_vals: np.ndarray) -> Tuple[float, float]:
        """Compute two-sample Kolmogorov-Smirnov test against reference data."""
        ref_vals = self.ref_data[feature]
        res = ks_2samp(ref_vals, current_vals)
        return float(res.statistic), float(res.pvalue)

    def compute_wasserstein(self, feature: str, current_vals: np.ndarray) -> Tuple[float, float]:
        """
        Compute raw Wasserstein distance and normalized Wasserstein distance (relative to reference std).
        """
        ref_vals = self.ref_data[feature]
        raw_w = float(wasserstein_distance(ref_vals, current_vals))
        norm_w = float(raw_w / self.ref_stds[feature])
        return raw_w, norm_w

    def compute_psi(self, feature: str, current_vals: np.ndarray) -> float:
        """
        Compute Population Stability Index (PSI) using precomputed reference bin edges.
        Uses Laplace smoothing (epsilon=1e-4) to avoid division by zero or log(0).
        """
        ref_vals = self.ref_data[feature]
        bins = self.ref_bins[feature]
        
        ref_counts, _ = np.histogram(ref_vals, bins=bins)
        curr_counts, _ = np.histogram(current_vals, bins=bins)
        
        n_ref = len(ref_vals)
        n_curr = len(current_vals)
        if n_ref == 0 or n_curr == 0:
            return 0.0
            
        eps = 1e-4
        expected_pct = (ref_counts + eps) / (n_ref + eps * len(ref_counts))
        actual_pct = (curr_counts + eps) / (n_curr + eps * len(curr_counts))
        
        psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(max(0.0, psi_val))

    def detect_drift(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform complete statistical drift evaluation on an incoming cohort DataFrame.
        
        Returns:
            Structured decision dictionary with drift_detected, drift_score, action,
            and per-feature statistical tests (KS stat, p-value, Wasserstein, PSI).
        """
        p_values: Dict[str, float] = {}
        ks_stats: Dict[str, float] = {}
        wasserstein_dists: Dict[str, float] = {}
        norm_wasserstein_dists: Dict[str, float] = {}
        psi_values: Dict[str, float] = {}
        significant_features: List[str] = []
        
        for col in self.feature_cols:
            if col not in current_df.columns:
                continue
            curr_vals = current_df[col].dropna().values.astype(np.float64)
            if len(curr_vals) == 0:
                continue
                
            # KS test
            ks_stat, p_val = self.compute_ks_test(col, curr_vals)
            ks_stats[col] = round(ks_stat, 4)
            p_values[col] = float(p_val)
            
            # Wasserstein
            raw_w, norm_w = self.compute_wasserstein(col, curr_vals)
            wasserstein_dists[col] = round(raw_w, 4)
            norm_wasserstein_dists[col] = round(norm_w, 4)
            
            # PSI
            psi_val = self.compute_psi(col, curr_vals)
            psi_values[col] = round(psi_val, 4)
            
            # Check significance
            if p_val < self.alpha:
                significant_features.append(col)
                
        num_evaluated = len(p_values)
        num_drifting = len(significant_features)
        drift_feature_ratio = (num_drifting / num_evaluated) if num_evaluated > 0 else 0.0
        mean_psi = float(np.mean(list(psi_values.values()))) if psi_values else 0.0
        
        # Check core academic features for severe targeted drift
        core_severe_drift = False
        for core_f in CORE_ACADEMIC_FEATURES:
            if core_f in p_values:
                # If a core academic feature shifts with p < 0.001 and normalized Wasserstein > 0.30
                if p_values[core_f] < 0.001 and norm_wasserstein_dists.get(core_f, 0.0) > 0.30:
                    core_severe_drift = True
                    break
                    
        # Decision logic:
        # Declare drift if feature ratio >= threshold_ratio OR mean_psi >= psi_threshold OR core_severe_drift
        drift_detected = (
            drift_feature_ratio >= self.threshold_ratio or
            mean_psi >= self.psi_threshold or
            core_severe_drift
        )
        
        # Normalized continuous drift score in [0.0, 1.0]
        # Combines feature ratio and PSI signal
        psi_score = min(1.0, mean_psi / (2.0 * self.psi_threshold))
        drift_score = float(np.clip(0.6 * drift_feature_ratio + 0.4 * psi_score, 0.0, 1.0))
        
        action = "TRIGGER_ADAPTATION" if drift_detected else "PROCEED_TO_PREDICT"
        
        return {
            "drift_detected": bool(drift_detected),
            "drift_score": round(drift_score, 4),
            "significant_features": significant_features,
            "p_values": p_values,
            "ks_statistics": ks_stats,
            "wasserstein_distances": wasserstein_dists,
            "norm_wasserstein_distances": norm_wasserstein_dists,
            "psi_values": psi_values,
            "mean_psi": round(mean_psi, 4),
            "num_features_evaluated": num_evaluated,
            "num_drifting_features": num_drifting,
            "drift_feature_ratio": round(drift_feature_ratio, 4),
            "threshold_ratio": self.threshold_ratio,
            "alpha": self.alpha,
            "action": action
        }


def run_drift_analysis(
    train_csv_path: str = "data/processed/train.csv",
    datasets: Optional[Dict[str, pd.DataFrame]] = None,
    output_dir: str = "results/drift"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run complete statistical drift evaluation across clean test and all OOD scenarios.
    Saves results/drift/drift_metrics.json and results/drift/feature_shift_report.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_df = pd.read_csv(train_csv_path)
    detector = DriftDetector(reference_df=train_df)
    
    if datasets is None:
        from src.drift.scenarios import generate_all_scenarios
        datasets = {"clean_test": pd.read_csv("data/processed/test.csv")}
        ood_dict = generate_all_scenarios()
        datasets.update(ood_dict)
        
    all_decisions: Dict[str, Any] = {}
    feature_shift_matrix: Dict[str, Any] = {}
    
    print("=" * 70)
    print("RUNNING STATISTICAL DRIFT DETECTOR ON EVALUATION DATASETS")
    print("=" * 70)
    
    for scen_name, scen_df in datasets.items():
        decision = detector.detect_drift(scen_df)
        all_decisions[scen_name] = {
            "drift_detected": decision["drift_detected"],
            "drift_score": decision["drift_score"],
            "action": decision["action"],
            "num_drifting_features": decision["num_drifting_features"],
            "drift_feature_ratio": decision["drift_feature_ratio"],
            "mean_psi": decision["mean_psi"],
            "significant_features": decision["significant_features"]
        }
        
        feature_shift_matrix[scen_name] = {
            "p_values": decision["p_values"],
            "ks_statistics": decision["ks_statistics"],
            "wasserstein_distances": decision["wasserstein_distances"],
            "norm_wasserstein_distances": decision["norm_wasserstein_distances"],
            "psi_values": decision["psi_values"]
        }
        
        status_flag = "[!] DRIFT DETECTED -> TRIGGER_ADAPTATION" if decision["drift_detected"] else "[PASS] STABLE -> PROCEED_TO_PREDICT"
        print(f"{scen_name:38s} | Ratio: {decision['drift_feature_ratio']:.2f} ({decision['num_drifting_features']:2d}/21) | PSI: {decision['mean_psi']:.3f} | {status_flag}")
        
    # Persist JSON artifacts
    metrics_path = os.path.join(output_dir, "drift_metrics.json")
    shift_report_path = os.path.join(output_dir, "feature_shift_report.json")
    
    save_json(all_decisions, metrics_path)
    save_json(feature_shift_matrix, shift_report_path)
    
    print(f"\nSaved drift metrics to: {metrics_path}")
    print(f"Saved feature shift report to: {shift_report_path}")
    print("=" * 70 + "\n")
    
    return all_decisions, feature_shift_matrix


if __name__ == "__main__":
    run_drift_analysis()
