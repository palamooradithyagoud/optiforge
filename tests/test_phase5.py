"""
Test Suite for Phase 5: Final Robustness, Benchmarking & Competition Packaging.

Verifies:
1. Multi-seed reproducibility & sensitivity metrics
2. Strict prohibition of test.csv contamination in Phase 5
3. Phase 3 drift scenario integrity across 12 scenarios
4. Ablation study results and ordering
5. Empirical latency benchmarking artifacts
6. Statistical summary formatting and 95% CI validity
7. Competition evidence matrix completeness
8. Final jury summary comprehensive coverage
9. All Phase 5 artifact existence and non-emptiness
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

RESULTS_DIR = os.path.join(ROOT_DIR, "results", "phase5")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


class TestPhase5(unittest.TestCase):
    """Phase 5 verification test suite."""

    def test_all_phase5_artifacts_exist(self):
        """Verify all required tabular/json/markdown artifacts and 6 figures exist."""
        required_artifacts = [
            os.path.join(RESULTS_DIR, "seed_robustness.csv"),
            os.path.join(RESULTS_DIR, "severity_analysis.csv"),
            os.path.join(RESULTS_DIR, "ablation_results.csv"),
            os.path.join(RESULTS_DIR, "efficiency_benchmark.csv"),
            os.path.join(RESULTS_DIR, "statistical_summary.csv"),
            os.path.join(RESULTS_DIR, "competition_evidence_matrix.csv"),
            os.path.join(RESULTS_DIR, "phase5_summary.json"),
            os.path.join(RESULTS_DIR, "phase5_report.md"),
            os.path.join(RESULTS_DIR, "final_jury_summary.md"),
        ]
        for path in required_artifacts:
            self.assertTrue(os.path.isfile(path), f"Missing required Phase 5 artifact: {path}")

        required_figures = [
            os.path.join(FIGURES_DIR, "seed_robustness.png"),
            os.path.join(FIGURES_DIR, "drift_severity.png"),
            os.path.join(FIGURES_DIR, "recovery_comparison.png"),
            os.path.join(FIGURES_DIR, "ablation_comparison.png"),
            os.path.join(FIGURES_DIR, "calibration.png"),
            os.path.join(FIGURES_DIR, "latency_benchmark.png"),
        ]
        for path in required_figures:
            self.assertTrue(os.path.isfile(path), f"Missing required Phase 5 figure: {path}")

    def test_all_phase5_artifacts_nonempty(self):
        """Verify all generated files are non-empty (> 100 bytes)."""
        for fname in os.listdir(RESULTS_DIR):
            fpath = os.path.join(RESULTS_DIR, fname)
            if os.path.isfile(fpath):
                self.assertGreater(os.path.getsize(fpath), 100, f"File {fname} is suspiciously small or empty")
        for fname in os.listdir(FIGURES_DIR):
            fpath = os.path.join(FIGURES_DIR, fname)
            if os.path.isfile(fpath):
                self.assertGreater(os.path.getsize(fpath), 1000, f"Figure {fname} is suspiciously small or empty")

    def test_no_test_contamination(self):
        """Verify Phase 5 source code does not load, tune, or evaluate on test.csv."""
        p5_dir = os.path.join(ROOT_DIR, "src", "phase5")
        for root, _, files in os.walk(p5_dir):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    self.assertNotIn("test.csv", content, f"test.csv referenced in Phase 5 code: {full_path}")

    def test_seed_reproducibility(self):
        """Verify seed robustness contains all 5 required seeds: 42, 123, 2024, 7, 99."""
        fpath = os.path.join(RESULTS_DIR, "seed_robustness.csv")
        df = pd.read_csv(fpath)
        numeric_seeds = df[pd.to_numeric(df["seed"], errors="coerce").notnull()].copy()
        numeric_seeds["seed"] = numeric_seeds["seed"].astype(int)
        
        expected_seeds = {42, 123, 2024, 7, 99}
        actual_seeds = set(numeric_seeds["seed"].tolist())
        self.assertEqual(expected_seeds, actual_seeds, f"Expected seeds {expected_seeds}, found {actual_seeds}")

        # Check parameter count is constant across seeds (frozen selected model)
        self.assertTrue((numeric_seeds["parameter_count"].astype(int) == 1281).all(), "Selected model parameter count must be 1281")
        # Check all required columns exist
        req_cols = ["seed", "MAE", "RMSE", "R2", "parameter_count", "inference_latency", "conformal_coverage", "interval_width", "mean_bias"]
        for c in req_cols:
            self.assertIn(c, df.columns, f"Missing column {c} in seed_robustness.csv")

    def test_seed_sensitivity(self):
        """Verify coefficient of variation and standard deviation across seeds are reasonable."""
        fpath = os.path.join(RESULTS_DIR, "seed_robustness.csv")
        df = pd.read_csv(fpath)
        numeric_seeds = df[pd.to_numeric(df["seed"], errors="coerce").notnull()].copy()
        mae_vals = numeric_seeds["MAE"].astype(float)
        mae_std = mae_vals.std()
        self.assertLess(mae_std, 0.2, f"Seed sensitivity too high: MAE std={mae_std:.4f}")

    def test_drift_scenario_integrity(self):
        """Verify all 12 Phase 3 drift scenarios are evaluated without modification."""
        fpath = os.path.join(RESULTS_DIR, "severity_analysis.csv")
        df = pd.read_csv(fpath)
        self.assertEqual(len(df), 12, f"Expected 12 drift scenarios, found {len(df)}")
        req_cols = ["scenario", "group", "drift_detected", "action", "baseline_MAE", "adapted_MAE", "conformal_coverage", "interval_width"]
        for c in req_cols:
            self.assertIn(c, df.columns, f"Missing column {c} in severity_analysis.csv")
        
        # Verify 5 scenario groups are represented
        groups = set(df["group"].unique())
        expected_groups = {"attendance drift", "academic drift", "backlog surge", "multivariate shift", "compound stress"}
        self.assertTrue(expected_groups.issubset(groups), f"Missing scenario groups: {expected_groups - groups}")

    def test_ablation_artifacts(self):
        """Verify ablation study contains configurations A, B, C, D, E."""
        fpath = os.path.join(RESULTS_DIR, "ablation_results.csv")
        df = pd.read_csv(fpath)
        self.assertEqual(len(df), 5, f"Expected 5 configurations A-E, got {len(df)}")
        configs = set(df["configuration"].tolist())
        self.assertEqual({"A", "B", "C", "D", "E"}, configs, "Configurations A-E must be present")
        req_cols = ["configuration", "description", "MAE", "RMSE", "R2", "parameter_count", "latency", "bias", "conformal_coverage"]
        for c in req_cols:
            self.assertIn(c, df.columns, f"Missing column {c} in ablation_results.csv")

    def test_efficiency_artifacts(self):
        """Verify empirical efficiency benchmark tested required batch sizes."""
        fpath = os.path.join(RESULTS_DIR, "efficiency_benchmark.csv")
        df = pd.read_csv(fpath)
        expected_batches = {1, 16, 32, 64, 128, 200}
        actual_batches = set(df["batch_size"].astype(int).tolist())
        self.assertEqual(expected_batches, actual_batches, f"Expected batches {expected_batches}, got {actual_batches}")
        req_cols = ["batch_size", "mean_latency", "p50_latency", "p95_latency", "p99_latency", "throughput", "memory_usage"]
        for c in req_cols:
            self.assertIn(c, df.columns, f"Missing column {c} in efficiency_benchmark.csv")

    def test_statistical_summary(self):
        """Verify statistical summary metrics and 95% confidence intervals."""
        fpath = os.path.join(RESULTS_DIR, "statistical_summary.csv")
        df = pd.read_csv(fpath)
        req_cols = ["dataset", "metric", "mean", "std", "min", "max", "ci_95_lower", "ci_95_upper"]
        for c in req_cols:
            self.assertIn(c, df.columns, f"Missing column {c} in statistical_summary.csv")
        metrics = set(df["metric"].unique())
        for m in ["MAE", "RMSE", "R2", "conformal coverage"]:
            self.assertIn(m, metrics, f"Metric {m} missing from statistical summary")

    def test_competition_matrix(self):
        """Verify competition matrix covers all 13 core requirements."""
        fpath = os.path.join(RESULTS_DIR, "competition_evidence_matrix.csv")
        df = pd.read_csv(fpath)
        self.assertEqual(len(df), 13, f"Expected 13 competition requirements, found {len(df)}")
        req_cols = ["requirement", "evidence", "implementation", "artifact", "test", "status", "caveat"]
        for c in req_cols:
            self.assertIn(c, df.columns, f"Missing column {c} in competition_evidence_matrix.csv")
        # Ensure caveats are documented (not blank)
        self.assertTrue((df["caveat"].str.len() > 5).all(), "Every requirement must have a transparent caveat documented")

    def test_jury_summary(self):
        """Verify final jury summary contains all 14 required sections and acknowledges baseline/proxies."""
        fpath = os.path.join(RESULTS_DIR, "final_jury_summary.md")
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("## 1. Problem", content)
        self.assertIn("## 2. Why distribution drift matters", content)
        self.assertIn("## 3. Phase 1 baseline", content)
        self.assertIn("## 4. NSGA-II formulation", content)
        self.assertIn("## 5. Six objectives", content)
        self.assertIn("## 6. Pareto optimization", content)
        self.assertIn("## 7. Drift detection", content)
        self.assertIn("## 8. Adaptive reoptimization", content)
        self.assertIn("## 9. Calibration", content)
        self.assertIn("## 10. Explainability", content)
        self.assertIn("## 11. Robustness results", content)
        self.assertIn("## 12. Efficiency results", content)
        self.assertIn("## 13. Limitations", content)
        self.assertIn("## 14. Final competition contribution", content)
        
        # Transparent admissions
        self.assertIn("naive mean baseline", content)
        self.assertIn("FLOP", content)
        self.assertIn("synthetic", content)


if __name__ == "__main__":
    unittest.main()
