"""
tests/test_phase3.py
Comprehensive QA & Validation Test Suite for Phase 3:
Out-of-Distribution (OOD) Stress Testing & Distribution Drift Detection.
Verifies dataset integrity, metric degradation calculations, model immutability,
drift detector sensitivity and mathematical validity, and artifact generation.
Compatible with pytest and standard unittest runner.
"""

import os
import sys
import json
import unittest
import numpy as np
import pandas as pd
import torch

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.utils import set_seed, load_config
from src.features import FeaturePipeline
from src.drift.scenarios import (
    SCENARIO_CONFIGS,
    generate_all_scenarios,
    generate_attendance_drift,
    generate_academic_drift,
    generate_backlog_surge,
    generate_cohort_shift,
    generate_compound_stress
)
from src.drift.detector import DriftDetector, DEFAULT_MONITORED_FEATURES
from src.drift.evaluate_ood import OODEvaluator


class TestPhase3QA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_seed(42, deterministic=True)
        cls.test_csv_path = os.path.join(WORKSPACE_ROOT, "data", "processed", "test.csv")
        cls.train_csv_path = os.path.join(WORKSPACE_ROOT, "data", "processed", "train.csv")
        cls.datasets_dir = os.path.join(WORKSPACE_ROOT, "results", "drift", "datasets")
        cls.results_dir = os.path.join(WORKSPACE_ROOT, "results", "drift")
        
        cls.clean_test_df = pd.read_csv(cls.test_csv_path)
        cls.train_df = pd.read_csv(cls.train_csv_path)
        
        # Instantiate evaluator and detector
        cls.evaluator = OODEvaluator()
        cls.detector = DriftDetector(reference_df=cls.train_df)

    # -------------------------------------------------------------
    # 1. Dataset Generation & Structural Integrity
    # -------------------------------------------------------------
    def test_tc01_ood_datasets_count_and_naming(self):
        """Verify all 12 scenario CSV datasets are generated with correct filenames."""
        for cfg in SCENARIO_CONFIGS:
            fpath = os.path.join(self.datasets_dir, cfg["filename"])
            self.assertTrue(os.path.exists(fpath), f"Missing scenario file: {cfg['filename']}")
            self.assertGreater(os.path.getsize(fpath), 1000, f"Scenario file is empty: {cfg['filename']}")

    def test_tc02_ood_dataset_structure_and_nans(self):
        """Verify each scenario dataset has exact row count, matching schema, and zero NaNs."""
        expected_rows = len(self.clean_test_df)
        expected_cols = list(self.clean_test_df.columns)
        
        for cfg in SCENARIO_CONFIGS:
            fpath = os.path.join(self.datasets_dir, cfg["filename"])
            df = pd.read_csv(fpath)
            self.assertEqual(len(df), expected_rows, f"Row count mismatch in {cfg['id']}")
            self.assertEqual(list(df.columns), expected_cols, f"Column mismatch in {cfg['id']}")
            self.assertEqual(df.isna().sum().sum(), 0, f"NaNs detected in {cfg['id']}")
            self.assertFalse(np.isinf(df.select_dtypes(include=[np.number]).values).any(), f"Inf detected in {cfg['id']}")

    def test_tc03_target_and_student_preservation(self):
        """Verify ground-truth next_semester_sgpa and student_id are strictly preserved."""
        for cfg in SCENARIO_CONFIGS:
            fpath = os.path.join(self.datasets_dir, cfg["filename"])
            df = pd.read_csv(fpath)
            # Student IDs
            self.assertTrue(
                df["student_id"].equals(self.clean_test_df["student_id"]),
                f"student_id sequence altered in {cfg['id']}"
            )
            # Target SGPA
            np.testing.assert_array_almost_equal(
                df["next_semester_sgpa"].values,
                self.clean_test_df["next_semester_sgpa"].values,
                decimal=5,
                err_msg=f"Ground-truth target next_semester_sgpa modified in {cfg['id']}"
            )

    def test_tc04_domain_boundary_validity(self):
        """Verify feature values respect physical and institutional domain boundaries."""
        for cfg in SCENARIO_CONFIGS:
            fpath = os.path.join(self.datasets_dir, cfg["filename"])
            df = pd.read_csv(fpath)
            
            # Attendance in [0, 100]%
            for att_col in ["previous_attendance", "avg_subject_attendance", "min_subject_attendance"]:
                if att_col in df.columns:
                    self.assertTrue((df[att_col] >= 0.0).all(), f"{att_col} below 0 in {cfg['id']}")
                    self.assertTrue((df[att_col] <= 100.0).all(), f"{att_col} above 100 in {cfg['id']}")
                    
            # Grades in [0, 10]
            for gr_col in ["previous_sgpa", "previous_cgpa", "avg_subject_grade_point", "min_subject_grade_point", "max_subject_grade_point"]:
                if gr_col in df.columns:
                    self.assertTrue((df[gr_col] >= 0.0).all(), f"{gr_col} below 0 in {cfg['id']}")
                    self.assertTrue((df[gr_col] <= 10.0).all(), f"{gr_col} above 10 in {cfg['id']}")
                    
            # Backlogs >= 0
            for bk_col in ["active_backlogs_count", "failed_subjects_count"]:
                if bk_col in df.columns:
                    self.assertTrue((df[bk_col] >= 0).all(), f"{bk_col} negative in {cfg['id']}")
                    
            # Credit completion ratio in [0, 1]
            if "credit_completion_ratio" in df.columns:
                self.assertTrue((df["credit_completion_ratio"] >= 0.0).all(), f"credit_completion_ratio < 0 in {cfg['id']}")
                self.assertTrue((df["credit_completion_ratio"] <= 1.0).all(), f"credit_completion_ratio > 1 in {cfg['id']}")

    # -------------------------------------------------------------
    # 2. Robustness Evaluator & Metric Logic
    # -------------------------------------------------------------
    def test_tc05_clean_baseline_metric_replication(self):
        """Verify OODEvaluator exactly reproduces Phase 2 clean baseline test metrics."""
        clean_res = self.evaluator.evaluate_cohort(self.clean_test_df)
        self.assertAlmostEqual(clean_res["mae"], 0.8359, places=3, msg="Clean MAE does not match Phase 2 baseline")
        self.assertAlmostEqual(clean_res["rmse"], 1.0692, places=3, msg="Clean RMSE does not match Phase 2 baseline")
        self.assertAlmostEqual(clean_res["r2"], -0.0453, places=3, msg="Clean R2 does not match Phase 2 baseline")

    def test_tc06_frozen_model_immutability(self):
        """Verify model weights remain strictly immutable before and after evaluating OOD cohorts."""
        weights_before = [p.clone().detach() for p in self.evaluator.model.parameters()]
        
        # Run inference on extreme compound scenario
        compound_df = pd.read_csv(os.path.join(self.datasets_dir, "scenario_e_compound_stress.csv"))
        _ = self.evaluator.evaluate_cohort(compound_df)
        
        weights_after = [p.clone().detach() for p in self.evaluator.model.parameters()]
        
        for p_bef, p_aft in zip(weights_before, weights_after):
            self.assertTrue(torch.equal(p_bef, p_aft), "Model parameter modified during evaluation!")

    def test_tc07_feature_scaler_not_refitted(self):
        """Verify the pre-fitted FeaturePipeline preserves original training mean and variance."""
        scaler_mean_before = self.evaluator.pipeline.scaler.mean_.copy()
        scaler_var_before = self.evaluator.pipeline.scaler.var_.copy()
        
        # Transform OOD dataset
        compound_df = pd.read_csv(os.path.join(self.datasets_dir, "scenario_e_compound_stress.csv"))
        _ = self.evaluator.pipeline.transform(compound_df)
        
        scaler_mean_after = self.evaluator.pipeline.scaler.mean_
        scaler_var_after = self.evaluator.pipeline.scaler.var_
        
        np.testing.assert_array_equal(scaler_mean_before, scaler_mean_after)
        np.testing.assert_array_equal(scaler_var_before, scaler_var_after)

    def test_tc08_degradation_calculation_consistency(self):
        """Verify delta_mae and pct_delta_mae formulas in results/drift/ood_results.csv."""
        results_csv = os.path.join(self.results_dir, "ood_results.csv")
        self.assertTrue(os.path.exists(results_csv), "ood_results.csv not found")
        
        df_res = pd.read_csv(results_csv)
        baseline_row = df_res[df_res["scenario_id"] == "clean_baseline"].iloc[0]
        clean_mae = baseline_row["mae"]
        clean_rmse = baseline_row["rmse"]
        
        for _, row in df_res.iterrows():
            if row["scenario_id"] == "clean_baseline":
                continue
            expected_d_mae = round(row["mae"] - clean_mae, 4)
            expected_pct = round((expected_d_mae / clean_mae) * 100.0, 2)
            expected_d_rmse = round(row["rmse"] - clean_rmse, 4)
            
            self.assertAlmostEqual(row["delta_mae"], expected_d_mae, places=3, msg=f"delta_mae error in {row['scenario_id']}")
            self.assertAlmostEqual(row["pct_delta_mae"], expected_pct, places=1, msg=f"pct_delta_mae error in {row['scenario_id']}")
            self.assertAlmostEqual(row["delta_rmse"], expected_d_rmse, places=3, msg=f"delta_rmse error in {row['scenario_id']}")

    def test_tc09_prediction_bias_calculation(self):
        """Verify mean prediction bias matches mean(y_pred - y)."""
        compound_df = pd.read_csv(os.path.join(self.datasets_dir, "scenario_e_compound_stress.csv"))
        res = self.evaluator.evaluate_cohort(compound_df)
        y_true = compound_df["next_semester_sgpa"].values
        expected_bias = float(np.mean(res["predictions"] - y_true))
        self.assertAlmostEqual(res["bias"], round(expected_bias, 4), places=4)

    # -------------------------------------------------------------
    # 3. Statistical Drift Detector Sensitivity & Validity
    # -------------------------------------------------------------
    def test_tc10_drift_detector_clean_test_stability(self):
        """Verify drift detector yields False (PROCEED_TO_PREDICT) on untouched clean test set."""
        decision = self.detector.detect_drift(self.clean_test_df)
        self.assertFalse(decision["drift_detected"], "Drift falsely detected on clean test set!")
        self.assertEqual(decision["action"], "PROCEED_TO_PREDICT")
        self.assertLess(decision["drift_feature_ratio"], self.detector.threshold_ratio)
        self.assertLess(decision["mean_psi"], self.detector.psi_threshold)

    def test_tc11_drift_detector_ood_sensitivity(self):
        """Verify drift detector triggers True (TRIGGER_ADAPTATION) across all shifted scenarios."""
        # Test representative scenarios from each family
        trigger_scenarios = [
            "scenario_a_attendance_drift_10pct.csv",
            "scenario_b_academic_drift_shift_1.0.csv",
            "scenario_c_backlog_surge_plus_2.csv",
            "scenario_d_cohort_shift.csv",
            "scenario_e_compound_stress.csv"
        ]
        
        for sc_file in trigger_scenarios:
            df = pd.read_csv(os.path.join(self.datasets_dir, sc_file))
            decision = self.detector.detect_drift(df)
            self.assertTrue(
                decision["drift_detected"],
                f"Drift detector failed to trigger on shifted scenario: {sc_file}"
            )
            self.assertEqual(
                decision["action"],
                "TRIGGER_ADAPTATION",
                f"Action was not TRIGGER_ADAPTATION on {sc_file}"
            )
            self.assertGreater(decision["drift_score"], 0.0)

    def test_tc12_drift_detector_statistical_validity(self):
        """Verify statistical metrics adhere to bounds (p-value in [0,1], KS in [0,1], Wasserstein >= 0, PSI >= 0)."""
        compound_df = pd.read_csv(os.path.join(self.datasets_dir, "scenario_e_compound_stress.csv"))
        decision = self.detector.detect_drift(compound_df)
        
        for feat in self.detector.feature_cols:
            if feat in decision["p_values"]:
                p_val = decision["p_values"][feat]
                ks_val = decision["ks_statistics"][feat]
                w_val = decision["wasserstein_distances"][feat]
                psi_val = decision["psi_values"][feat]
                
                self.assertGreaterEqual(p_val, 0.0, f"Negative p-value for {feat}")
                self.assertLessEqual(p_val, 1.0, f"p-value > 1 for {feat}")
                self.assertGreaterEqual(ks_val, 0.0, f"Negative KS stat for {feat}")
                self.assertLessEqual(ks_val, 1.0, f"KS stat > 1 for {feat}")
                self.assertGreaterEqual(w_val, 0.0, f"Negative Wasserstein dist for {feat}")
                self.assertGreaterEqual(psi_val, 0.0, f"Negative PSI for {feat}")

    def test_tc13_decision_object_schema(self):
        """Verify structured decision object conforms strictly to required schema."""
        decision = self.detector.detect_drift(self.clean_test_df)
        required_keys = [
            "drift_detected", "drift_score", "significant_features",
            "p_values", "ks_statistics", "wasserstein_distances",
            "norm_wasserstein_distances", "psi_values", "mean_psi",
            "num_features_evaluated", "num_drifting_features", "action"
        ]
        for key in required_keys:
            self.assertIn(key, decision, f"Required key '{key}' missing from decision object")
        self.assertIn(decision["action"], ["TRIGGER_ADAPTATION", "PROCEED_TO_PREDICT"])

    # -------------------------------------------------------------
    # 4. Artifact Integrity
    # -------------------------------------------------------------
    def test_tc14_artifacts_exist_and_non_empty(self):
        """Verify all generated reports, JSON metadata, and publication PNGs exist and are non-empty."""
        expected_artifacts = [
            "ood_results.csv",
            "robustness_report.md",
            "drift_metrics.json",
            "feature_shift_report.json",
            "drift_vs_degradation.png",
            "feature_distribution_shift.png",
            "drift_pvalues_heatmap.png"
        ]
        for art in expected_artifacts:
            path = os.path.join(self.results_dir, art)
            self.assertTrue(os.path.exists(path), f"Artifact missing: {art}")
            self.assertGreater(os.path.getsize(path), 500, f"Artifact empty: {art}")

    def test_tc15_deterministic_perturbations(self):
        """Verify perturbing cohorts twice with seed=42 generates byte-identical outputs."""
        df_a1 = generate_cohort_shift(self.clean_test_df, seed=42)
        df_a2 = generate_cohort_shift(self.clean_test_df, seed=42)
        pd.testing.assert_frame_equal(df_a1, df_a2)


if __name__ == "__main__":
    unittest.main()
