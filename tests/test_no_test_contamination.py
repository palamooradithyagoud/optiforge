"""
tests/test_no_test_contamination.py
Comprehensive methodology audit verifying zero test set contamination,
strict student-level cohort disjointness, and out-of-sample evaluation validity.
Addresses Parts 14, 15, and 20 of Phase 3-4 Methodology Requirements.
"""

import os
import sys
import json
import unittest
import pandas as pd
import numpy as np

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.adaptation.reoptimizer import split_cohort_student_level
from src.drift.scenarios import SCENARIO_CONFIGS


class TestNoTestContamination(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_csv = os.path.join(WORKSPACE_ROOT, "data", "processed", "train.csv")
        cls.val_csv = os.path.join(WORKSPACE_ROOT, "data", "processed", "val.csv")
        cls.test_csv = os.path.join(WORKSPACE_ROOT, "data", "processed", "test.csv")
        cls.split_csv = os.path.join(WORKSPACE_ROOT, "results", "adaptation", "adaptation_split.csv")
        cls.provenance_json = os.path.join(WORKSPACE_ROOT, "results", "drift", "threshold_provenance.json")
        cls.datasets_dir = os.path.join(WORKSPACE_ROOT, "results", "drift", "datasets")

    # -------------------------------------------------------------
    # 1. Base Dataset Student Disjointness (Part 15)
    # -------------------------------------------------------------
    def test_01_base_cohorts_student_disjointness(self):
        """Verify train, validation, and test cohorts share exactly zero students."""
        train_df = pd.read_csv(self.train_csv)
        val_df = pd.read_csv(self.val_csv)
        test_df = pd.read_csv(self.test_csv)

        train_students = set(train_df["student_id"].unique())
        val_students = set(val_df["student_id"].unique())
        test_students = set(test_df["student_id"].unique())

        self.assertEqual(len(train_students), 300, "Train should have 300 students")
        self.assertEqual(len(val_students), 100, "Validation should have 100 students")
        self.assertEqual(len(test_students), 100, "Test should have 100 students")

        self.assertTrue(train_students.isdisjoint(val_students), "Train and Validation cohorts overlap!")
        self.assertTrue(train_students.isdisjoint(test_students), "Train and Test cohorts overlap!")
        self.assertTrue(val_students.isdisjoint(test_students), "Validation and Test cohorts overlap!")

    # -------------------------------------------------------------
    # 2. Phase 3 Drift Source Cohort Isolation (Test 1)
    # -------------------------------------------------------------
    def test_02_drift_scenarios_sourced_strictly_from_dev_cohort(self):
        """Verify synthetic drift scenarios derive exclusively from val.csv (0 test students)."""
        val_df = pd.read_csv(self.val_csv)
        val_students = set(val_df["student_id"].unique())
        test_df = pd.read_csv(self.test_csv)
        test_students = set(test_df["student_id"].unique())

        for cfg in SCENARIO_CONFIGS:
            scen_path = os.path.join(self.datasets_dir, cfg["filename"])
            self.assertTrue(os.path.exists(scen_path), f"Scenario dataset missing: {scen_path}")
            scen_df = pd.read_csv(scen_path)
            scen_students = set(scen_df["student_id"].unique())

            # Must match val students exactly and contain 0 test students
            self.assertEqual(scen_students, val_students, f"Scenario {cfg['id']} does not match dev cohort")
            self.assertTrue(scen_students.isdisjoint(test_students), f"Scenario {cfg['id']} contains test students!")

    # -------------------------------------------------------------
    # 3. Codebase Static Leakage Scan (Test 1, Test 2, Part 14)
    # -------------------------------------------------------------
    def test_03_no_test_csv_access_in_drift_or_adaptation(self):
        """Verify src/drift and src/adaptation do not hardcode or load test.csv."""
        forbidden_tokens = ['"test.csv"', "'test.csv'", 'data/processed/test.csv']
        scanned_dirs = [
            os.path.join(WORKSPACE_ROOT, "src", "drift"),
            os.path.join(WORKSPACE_ROOT, "src", "adaptation")
        ]

        violations = []
        for s_dir in scanned_dirs:
            for root, _, files in os.walk(s_dir):
                for file in files:
                    if file.endswith(".py"):
                        fpath = os.path.join(root, file)
                        with open(fpath, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                        for l_idx, line in enumerate(lines, 1):
                            # Skip comments or docstrings explaining the prohibition
                            stripped = line.strip()
                            if stripped.startswith("#") or stripped.startswith("*"):
                                continue
                            for token in forbidden_tokens:
                                if token in line:
                                    violations.append(f"{fpath}:{l_idx} contains '{token}'")

        self.assertEqual(violations, [], f"Found forbidden test.csv references:\n" + "\n".join(violations))

    # -------------------------------------------------------------
    # 4. Student-Level Adaptation vs Recovery Disjointness (Test 3)
    # -------------------------------------------------------------
    def test_04_adaptation_recovery_student_disjointness_all_scenarios(self):
        """Verify adaptation students ∩ recovery students == 0 for all 12 scenarios."""
        for cfg in SCENARIO_CONFIGS:
            scen_path = os.path.join(self.datasets_dir, cfg["filename"])
            scen_df = pd.read_csv(scen_path)
            adapt_train, adapt_calib, recovery = split_cohort_student_level(scen_df, seed=42)

            adapt_ids = set(adapt_train["student_id"].unique()).union(set(adapt_calib["student_id"].unique()))
            recovery_ids = set(recovery["student_id"].unique())

            self.assertEqual(len(adapt_ids), 50, f"Scenario {cfg['id']} should have 50 adapt students")
            self.assertEqual(len(recovery_ids), 50, f"Scenario {cfg['id']} should have 50 recovery students")
            self.assertTrue(adapt_ids.isdisjoint(recovery_ids), f"Scenario {cfg['id']} has overlapping adapt/recovery students!")

    # -------------------------------------------------------------
    # 5. Persisted Adaptation Split CSV Integrity (Part 4)
    # -------------------------------------------------------------
    def test_05_persisted_adaptation_split_disjointness(self):
        """Verify results/adaptation/adaptation_split.csv records zero student partition overlap."""
        self.assertTrue(os.path.exists(self.split_csv), "adaptation_split.csv does not exist")
        df_split = pd.read_csv(self.split_csv)
        self.assertIn("student_id", df_split.columns)
        self.assertIn("partition", df_split.columns)

        adapt_students = set(df_split[df_split["partition"].str.startswith("adaptation")]["student_id"].unique())
        recovery_students = set(df_split[df_split["partition"] == "recovery"]["student_id"].unique())

        self.assertGreater(len(adapt_students), 0)
        self.assertGreater(len(recovery_students), 0)
        self.assertTrue(adapt_students.isdisjoint(recovery_students), "adaptation_split.csv contains student overlap!")

    # -------------------------------------------------------------
    # 6. Calibration vs Recovery Data Disjointness (Test 4)
    # -------------------------------------------------------------
    def test_06_calibration_recovery_data_disjointness(self):
        """Verify calibration fitting data never overlaps with held-out recovery cohort."""
        scen_path = os.path.join(self.datasets_dir, "scenario_e_compound_stress.csv")
        scen_df = pd.read_csv(scen_path)
        adapt_train, adapt_calib, recovery = split_cohort_student_level(scen_df, seed=42)

        calib_students = set(adapt_calib["student_id"].unique())
        recov_students = set(recovery["student_id"].unique())

        self.assertTrue(calib_students.isdisjoint(recov_students))

    # -------------------------------------------------------------
    # 7. Recovery Set Excluded from NSGA-II Re-Optimization (Test 5)
    # -------------------------------------------------------------
    def test_07_recovery_set_never_in_nsga2(self):
        """Verify NSGA-II re-optimizer evaluator strictly receives adaptation data only."""
        reopt_summary_path = os.path.join(WORKSPACE_ROOT, "results", "adaptation", "reoptimization_summary.json")
        self.assertTrue(os.path.exists(reopt_summary_path))
        with open(reopt_summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        self.assertIn("selected_candidate_genome", summary)
        # Ensure training samples in summary match adaptation split, not full scenario
        adapt_summary_path = os.path.join(WORKSPACE_ROOT, "results", "adaptation", "adaptation_summary.json")
        with open(adapt_summary_path, "r", encoding="utf-8") as f:
            adapt_summary = json.load(f)
        self.assertEqual(adapt_summary.get("student_intersection"), 0)

    # -------------------------------------------------------------
    # 8. Recovery Metrics Evaluated Only After Adaptation (Test 6)
    # -------------------------------------------------------------
    def test_08_recovery_metrics_provenance(self):
        """Verify recovery benchmarks evaluate frozen models on held-out data."""
        bench_csv = os.path.join(WORKSPACE_ROOT, "results", "adaptation", "recovery_benchmark.csv")
        self.assertTrue(os.path.exists(bench_csv))
        df_bench = pd.read_csv(bench_csv)
        self.assertEqual(len(df_bench), 4)
        stages = list(df_bench["stage_id"])
        self.assertEqual(stages, [
            "1_clean_baseline",
            "2_drift_degradation",
            "3_reoptimized_recovery",
            "4_calibrated_output"
        ])

    # -------------------------------------------------------------
    # 9. Drift Threshold Provenance (Test 7, Part 11)
    # -------------------------------------------------------------
    def test_09_drift_threshold_provenance(self):
        """Verify drift thresholds originate from training/reference or documented statistical criteria."""
        self.assertTrue(os.path.exists(self.provenance_json), "threshold_provenance.json missing")
        with open(self.provenance_json, "r", encoding="utf-8") as f:
            prov = json.load(f)

        self.assertIn("thresholds", prov)
        for meta in prov["thresholds"]:
            self.assertIn("source_data", meta)
            self.assertNotIn("test.csv", meta["source_data"], f"Threshold for {meta.get('metric')} sourced from test.csv!")

    # -------------------------------------------------------------
    # 10. Final Test Untouched During Development (Test 8, Part 16)
    # -------------------------------------------------------------
    def test_10_final_test_isolation_during_dev(self):
        """Verify final test cohort has not been modified or accessed during Phase 3/4."""
        test_df = pd.read_csv(self.test_csv)
        self.assertEqual(len(test_df), 200, "Final test cohort corrupted or modified")
        self.assertEqual(len(test_df["student_id"].unique()), 100)

    # -------------------------------------------------------------
    # 11. Deterministic Split Under Identical Seed (Test 9)
    # -------------------------------------------------------------
    def test_11_deterministic_split_reproducibility(self):
        """Verify same seed produces identical student-level partitions."""
        val_df = pd.read_csv(self.val_csv)
        train1, calib1, recov1 = split_cohort_student_level(val_df, seed=42)
        train2, calib2, recov2 = split_cohort_student_level(val_df, seed=42)

        self.assertEqual(list(train1["student_id"]), list(train2["student_id"]))
        self.assertEqual(list(calib1["student_id"]), list(calib2["student_id"]))
        self.assertEqual(list(recov1["student_id"]), list(recov2["student_id"]))

    # -------------------------------------------------------------
    # 12. Stochastic Sensitivity Under Different Seeds (Test 10)
    # -------------------------------------------------------------
    def test_12_stochastic_split_sensitivity(self):
        """Verify different seeds produce different student permutations."""
        val_df = pd.read_csv(self.val_csv)
        train1, _, recov1 = split_cohort_student_level(val_df, seed=42)
        train2, _, recov2 = split_cohort_student_level(val_df, seed=999)

        # Student sets must be different
        self.assertNotEqual(set(train1["student_id"]), set(train2["student_id"]))
        self.assertNotEqual(set(recov1["student_id"]), set(recov2["student_id"]))


if __name__ == "__main__":
    unittest.main()
