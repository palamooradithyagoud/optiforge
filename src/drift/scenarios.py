"""
src/drift/scenarios.py
Controlled Out-of-Distribution (OOD) perturbation engine for Phase 3 stress-testing.
Generates 5 distinct shift scenarios across multiple severity levels from the frozen test split.
Preserves ground-truth next_semester_sgpa and guarantees deterministic reproducibility.
"""

import os
import argparse
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.utils import set_seed, load_config


# Metadata definitions for all 12 experimental scenarios
SCENARIO_CONFIGS = [
    {
        "id": "scenario_a_attendance_drift_5pct",
        "family": "Scenario A: Attendance Drift",
        "name": "Attendance Drift -5%",
        "parameter": "scale_down",
        "severity": 0.05,
        "filename": "scenario_a_attendance_drift_5pct.csv",
        "description": "Uniform 5% downward scaling of attendance features"
    },
    {
        "id": "scenario_a_attendance_drift_10pct",
        "family": "Scenario A: Attendance Drift",
        "name": "Attendance Drift -10%",
        "parameter": "scale_down",
        "severity": 0.10,
        "filename": "scenario_a_attendance_drift_10pct.csv",
        "description": "Uniform 10% downward scaling of attendance features"
    },
    {
        "id": "scenario_a_attendance_drift_15pct",
        "family": "Scenario A: Attendance Drift",
        "name": "Attendance Drift -15%",
        "parameter": "scale_down",
        "severity": 0.15,
        "filename": "scenario_a_attendance_drift_15pct.csv",
        "description": "Uniform 15% downward scaling of attendance features"
    },
    {
        "id": "scenario_a_attendance_drift_20pct",
        "family": "Scenario A: Attendance Drift",
        "name": "Attendance Drift -20%",
        "parameter": "scale_down",
        "severity": 0.20,
        "filename": "scenario_a_attendance_drift_20pct.csv",
        "description": "Uniform 20% downward scaling of attendance features"
    },
    {
        "id": "scenario_b_academic_drift_shift_0.5",
        "family": "Scenario B: Academic Drift (Syllabus Shock)",
        "name": "Academic Drift -0.5 SGPA",
        "parameter": "grade_shift",
        "severity": -0.5,
        "filename": "scenario_b_academic_drift_shift_0.5.csv",
        "description": "Grade deflation of -0.5 across historical SGPA, CGPA, and subject grade points"
    },
    {
        "id": "scenario_b_academic_drift_shift_1.0",
        "family": "Scenario B: Academic Drift (Syllabus Shock)",
        "name": "Academic Drift -1.0 SGPA",
        "parameter": "grade_shift",
        "severity": -1.0,
        "filename": "scenario_b_academic_drift_shift_1.0.csv",
        "description": "Moderate syllabus shock with -1.0 grade deflation across all academic metrics"
    },
    {
        "id": "scenario_b_academic_drift_shift_1.5",
        "family": "Scenario B: Academic Drift (Syllabus Shock)",
        "name": "Academic Drift -1.5 SGPA",
        "parameter": "grade_shift",
        "severity": -1.5,
        "filename": "scenario_b_academic_drift_shift_1.5.csv",
        "description": "Severe syllabus shock with -1.5 grade deflation across all academic metrics"
    },
    {
        "id": "scenario_c_backlog_surge_plus_1",
        "family": "Scenario C: Backlog Surge",
        "name": "Backlog Surge +1",
        "parameter": "backlog_increment",
        "severity": 1,
        "filename": "scenario_c_backlog_surge_plus_1.csv",
        "description": "Increase backlogs by +1 and proportionally reduce earned credits and completion ratio"
    },
    {
        "id": "scenario_c_backlog_surge_plus_2",
        "family": "Scenario C: Backlog Surge",
        "name": "Backlog Surge +2",
        "parameter": "backlog_increment",
        "severity": 2,
        "filename": "scenario_c_backlog_surge_plus_2.csv",
        "description": "Increase backlogs by +2 with proportional course credit losses"
    },
    {
        "id": "scenario_c_backlog_surge_plus_3",
        "family": "Scenario C: Backlog Surge",
        "name": "Backlog Surge +3",
        "parameter": "backlog_increment",
        "severity": 3,
        "filename": "scenario_c_backlog_surge_plus_3.csv",
        "description": "Severe failure surge: +3 backlogs with substantial credit deficits"
    },
    {
        "id": "scenario_d_cohort_shift",
        "family": "Scenario D: Cohort Demographic Shift",
        "name": "Synthesized Cohort Shift",
        "parameter": "multivariate_shift",
        "severity": 1.0,
        "filename": "scenario_d_cohort_shift.csv",
        "description": "Systematic demographic drift shifting both mean and variance across continuous features"
    },
    {
        "id": "scenario_e_compound_stress",
        "family": "Scenario E: Compound Extreme Stress",
        "name": "Compound Stress (-15% Att, +2 Backlogs, -0.75 SGPA)",
        "parameter": "compound",
        "severity": 2.0,
        "filename": "scenario_e_compound_stress.csv",
        "description": "Simultaneous catastrophic disruption: -15% attendance, +2 backlogs, -0.75 SGPA"
    }
]


def generate_attendance_drift(df: pd.DataFrame, rate: float) -> pd.DataFrame:
    """
    Scenario A: Scale all attendance features down by rate (e.g. 0.05, 0.10, 0.15, 0.20).
    Values are strictly bounded in [0.0, 100.0]%.
    Target label 'next_semester_sgpa' is preserved untouched.
    """
    perturbed = df.copy()
    factor = 1.0 - rate
    
    # Scale attendance features
    for col in ["previous_attendance", "avg_subject_attendance", "min_subject_attendance"]:
        if col in perturbed.columns:
            perturbed[col] = np.clip(perturbed[col] * factor, 0.0, 100.0)
            
    # Adjust attendance change to reflect the downward shift
    if "attendance_change" in perturbed.columns and "previous_attendance" in df.columns:
        perturbed["attendance_change"] = np.clip(
            df["attendance_change"] - (rate * df["previous_attendance"]),
            -100.0, 100.0
        )
        
    # Scale attendance variance consistently (variance scales as factor^2)
    if "var_subject_attendance" in perturbed.columns:
        perturbed["var_subject_attendance"] = np.maximum(
            perturbed["var_subject_attendance"] * (factor ** 2), 0.0
        )
        
    return perturbed


def generate_academic_drift(df: pd.DataFrame, shift: float) -> pd.DataFrame:
    """
    Scenario B: Shift historical SGPA, CGPA, and subject-level grade stats down by shift.
    shift is negative (e.g. -0.5, -1.0, -1.5).
    Values are strictly bounded in [0.0, 10.0].
    Target label 'next_semester_sgpa' is preserved untouched.
    """
    perturbed = df.copy()
    
    # Primary grade features
    grade_cols = [
        "previous_sgpa", "previous_cgpa",
        "avg_subject_grade_point", "min_subject_grade_point", "max_subject_grade_point"
    ]
    for col in grade_cols:
        if col in perturbed.columns:
            perturbed[col] = np.clip(perturbed[col] + shift, 0.0, 10.0)
            
    # Academic changes
    if "sgpa_change" in perturbed.columns:
        perturbed["sgpa_change"] = np.clip(df["sgpa_change"] + shift, -10.0, 10.0)
    if "cgpa_change" in perturbed.columns:
        perturbed["cgpa_change"] = np.clip(df["cgpa_change"] + (shift * 0.5), -10.0, 10.0)
        
    # Ensure variance remains non-negative
    if "var_subject_grade_point" in perturbed.columns:
        perturbed["var_subject_grade_point"] = np.maximum(perturbed["var_subject_grade_point"], 0.0)
        
    return perturbed


def generate_backlog_surge(df: pd.DataFrame, surge: int, credits_per_backlog: float = 3.0) -> pd.DataFrame:
    """
    Scenario C: Shift backlog_count and failed_subjects up by +surge (+1, +2, +3).
    Scale credit completion and earned credits down proportionally.
    Target label 'next_semester_sgpa' is preserved untouched.
    """
    perturbed = df.copy()
    
    if "active_backlogs_count" in perturbed.columns:
        perturbed["active_backlogs_count"] = np.maximum(0.0, df["active_backlogs_count"] + surge)
    if "failed_subjects_count" in perturbed.columns:
        perturbed["failed_subjects_count"] = np.maximum(0.0, df["failed_subjects_count"] + surge)
    if "backlog_change" in perturbed.columns:
        perturbed["backlog_change"] = df["backlog_change"] + surge
        
    # Deduct earned credits proportionally to failed courses
    if "total_sem_credits_earned" in perturbed.columns and "total_sem_credits_registered" in perturbed.columns:
        credits_lost = surge * credits_per_backlog
        perturbed["total_sem_credits_earned"] = np.clip(
            df["total_sem_credits_earned"] - credits_lost,
            0.0, df["total_sem_credits_registered"]
        )
        perturbed["credit_completion_ratio"] = np.clip(
            perturbed["total_sem_credits_earned"] / np.maximum(perturbed["total_sem_credits_registered"], 1.0),
            0.0, 1.0
        )
        
    return perturbed


def generate_cohort_shift(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Scenario D: Synthesize a shifted student cohort exhibiting shifted mean and variance
    across continuous features (attendance, grades, credits, backlogs).
    Guarantees physical plausibility and determinism.
    Target label 'next_semester_sgpa' is preserved untouched.
    """
    rng = np.random.RandomState(seed)
    perturbed = df.copy()
    n_samples = len(perturbed)
    
    # 1. Shift attendance: -8% mean with +25% variance noise
    for col in ["previous_attendance", "avg_subject_attendance"]:
        if col in perturbed.columns:
            noise = rng.normal(loc=-8.0, scale=4.0, size=n_samples)
            perturbed[col] = np.clip(perturbed[col] + noise, 0.0, 100.0)
            
    if "min_subject_attendance" in perturbed.columns:
        noise = rng.normal(loc=-12.0, scale=5.0, size=n_samples)
        perturbed["min_subject_attendance"] = np.clip(perturbed["min_subject_attendance"] + noise, 0.0, 100.0)
        
    if "var_subject_attendance" in perturbed.columns:
        perturbed["var_subject_attendance"] = np.clip(
            perturbed["var_subject_attendance"] * rng.uniform(1.15, 1.45, size=n_samples),
            0.0, 1000.0
        )
        
    # 2. Shift academic grades: -0.6 mean shift with increased spread
    for col in ["previous_sgpa", "previous_cgpa", "avg_subject_grade_point"]:
        if col in perturbed.columns:
            grade_noise = rng.normal(loc=-0.60, scale=0.35, size=n_samples)
            perturbed[col] = np.clip(perturbed[col] + grade_noise, 0.0, 10.0)
            
    if "min_subject_grade_point" in perturbed.columns:
        min_noise = rng.normal(loc=-0.90, scale=0.40, size=n_samples)
        perturbed["min_subject_grade_point"] = np.clip(perturbed["min_subject_grade_point"] + min_noise, 0.0, 10.0)
        
    if "max_subject_grade_point" in perturbed.columns:
        max_noise = rng.normal(loc=-0.40, scale=0.30, size=n_samples)
        perturbed["max_subject_grade_point"] = np.clip(perturbed["max_subject_grade_point"] + max_noise, 0.0, 10.0)
        
    # 3. Increase backlogs probabilistically (+1 backlog for 40% of cohort)
    backlog_mask = rng.binomial(1, 0.40, size=n_samples)
    if "active_backlogs_count" in perturbed.columns:
        perturbed["active_backlogs_count"] = np.maximum(0.0, perturbed["active_backlogs_count"] + backlog_mask)
    if "failed_subjects_count" in perturbed.columns:
        perturbed["failed_subjects_count"] = np.maximum(0.0, perturbed["failed_subjects_count"] + backlog_mask)
        
    # 4. Adjust credits earned accordingly
    if "total_sem_credits_earned" in perturbed.columns and "total_sem_credits_registered" in perturbed.columns:
        credits_lost = backlog_mask * 3.0
        perturbed["total_sem_credits_earned"] = np.clip(
            perturbed["total_sem_credits_earned"] - credits_lost,
            0.0, perturbed["total_sem_credits_registered"]
        )
        perturbed["credit_completion_ratio"] = np.clip(
            perturbed["total_sem_credits_earned"] / np.maximum(perturbed["total_sem_credits_registered"], 1.0),
            0.0, 1.0
        )
        
    return perturbed


def generate_compound_stress(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scenario E: Compound Extreme Stress.
    Simultaneous multi-factor shock:
    - -15% attendance scaling
    - +2 backlogs with proportional credit deduction
    - -0.75 SGPA/CGPA/grade point shift
    Target label 'next_semester_sgpa' is preserved untouched.
    """
    perturbed = df.copy()
    
    # 1. -15% Attendance
    perturbed = generate_attendance_drift(perturbed, rate=0.15)
    
    # 2. +2 Backlogs
    perturbed = generate_backlog_surge(perturbed, surge=2, credits_per_backlog=3.0)
    
    # 3. -0.75 SGPA
    perturbed = generate_academic_drift(perturbed, shift=-0.75)
    
    return perturbed


def generate_drift_scenario(
    reference_df: pd.DataFrame,
    scenario_config: Dict[str, Any],
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate a single synthetic drift scenario from explicitly supplied reference dataframe.
    Guarantees that target labels and student IDs are preserved without data leakage.
    """
    param_type = scenario_config["parameter"]
    sev = scenario_config["severity"]
    
    if param_type == "scale_down":
        scen_df = generate_attendance_drift(reference_df, rate=float(sev))
    elif param_type == "grade_shift":
        scen_df = generate_academic_drift(reference_df, shift=float(sev))
    elif param_type == "backlog_increment":
        scen_df = generate_backlog_surge(reference_df, surge=int(sev))
    elif param_type == "multivariate_shift":
        scen_df = generate_cohort_shift(reference_df, seed=seed)
    elif param_type == "compound":
        scen_df = generate_compound_stress(reference_df)
    else:
        raise ValueError(f"Unknown scenario parameter type: {param_type}")
        
    target_col = "next_semester_sgpa"
    assert len(scen_df) == len(reference_df), "Row count mismatch in scenario generation"
    assert list(scen_df.columns) == list(reference_df.columns), "Column mismatch in scenario generation"
    assert scen_df[target_col].equals(reference_df[target_col]), "Target altered in scenario generation!"
    assert scen_df.isna().sum().sum() == 0, "NaNs detected in scenario generation!"
    return scen_df


def generate_all_scenarios(
    reference_df: Optional[pd.DataFrame] = None,
    reference_csv_path: str = "data/processed/val.csv",
    output_dir: str = "results/drift/datasets",
    seed: int = 42,
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Generate all 12 controlled OOD datasets from the safe development cohort (val.csv).
    The final test set is strictly excluded and preserved for one-time evaluation.
    Saves each dataset to CSV in output_dir and returns a dictionary of dataframes.
    """
    set_seed(seed, deterministic=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle backwards-compatible argument if passed
    if "test_csv_path" in kwargs and reference_df is None:
        csv_path = kwargs["test_csv_path"]
    else:
        csv_path = reference_csv_path
        
    if reference_df is None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Reference development dataset not found at: {csv_path}")
        clean_dev_df = pd.read_csv(csv_path)
    else:
        clean_dev_df = reference_df.copy()
        
    target_col = "next_semester_sgpa"
    if target_col not in clean_dev_df.columns:
        raise ValueError(f"Target column '{target_col}' missing from development dataset.")
        
    generated_datasets: Dict[str, pd.DataFrame] = {}
    
    print(f"Generating Phase 3 OOD Datasets from development cohort (N={len(clean_dev_df)})...")
    
    for cfg in SCENARIO_CONFIGS:
        scen_id = cfg["id"]
        out_filename = cfg["filename"]
        out_path = os.path.join(output_dir, out_filename)
        
        scen_df = generate_drift_scenario(clean_dev_df, cfg, seed=seed)
        scen_df.to_csv(out_path, index=False)
        generated_datasets[scen_id] = scen_df
        print(f"  [OK] [{scen_id}] Saved {len(scen_df)} samples to {out_path}")
        
    print(f"Successfully generated and validated {len(generated_datasets)} development OOD datasets.\n")
    return generated_datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Phase 3 OOD datasets from development cohort")
    parser.add_argument("--reference-csv", type=str, default="data/processed/val.csv", help="Development cohort CSV (default: val.csv)")
    parser.add_argument("--output-dir", type=str, default="results/drift/datasets")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    generate_all_scenarios(reference_csv_path=args.reference_csv, output_dir=args.output_dir, seed=args.seed)

