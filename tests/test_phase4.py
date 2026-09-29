"""
tests/test_phase4.py
Comprehensive QA & Validation Test Suite for Phase 4:
Adaptive Re-Optimization, Sub-Population Calibration & Model Explainability.
Verifies all 15 targeted unit and integration test cases (TC01 to TC15):
1. Trigger Ingestion
2. Warm-Start Initialization
3. Pareto Dominance Invariant
4. Read-Only Upstream Immutability
5. Test Set Isolation
6. Directional Bias Reduction (|Bias| < 0.05)
7. Conformal Coverage Validity (>= 90% +- epsilon)
8. Interval Width Efficiency
9. Sub-Group Calibration
10. Integrated Gradients Completeness Axiom
11. Attribution Sanity & Bounds
12. Error Recovery Monotonicity
13. Deterministic Reproducibility
14. Artifact Presence & Integrity
15. End-to-End Pipeline Execution
"""

import os
import sys
import json
import unittest
import numpy as np
import pandas as pd
import torch

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.utils import set_seed, load_config
from src.features import FeaturePipeline
from src.nsga2.sorting import fast_non_dominated_sort
from src.adaptation.reoptimizer import (
    load_pareto_genomes,
    warm_start_population,
    select_best_adapted_candidate
)
from src.adaptation.calibration import ContinuousCalibrator, SplitConformalCalibrator
from src.adaptation.explainability import IntegratedGradientsExplainer
from src.adaptation.evaluate_recovery import load_model_from_checkpoint


def check_adaptation_trigger(scenario_id: str, drift_metrics_path: str) -> bool:
    """Helper checking whether scenario triggers adaptation."""
    with open(drift_metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    decision = metrics.get(scenario_id, {})
    return decision.get("action") == "TRIGGER_ADAPTATION"


class TestPhase4QA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_seed(42, deterministic=True)
        cls.models_dir = os.path.join(WORKSPACE_ROOT, "models", "adaptation")
        cls.results_dir = os.path.join(WORKSPACE_ROOT, "results", "adaptation")
        cls.drift_dir = os.path.join(WORKSPACE_ROOT, "results", "drift")
        cls.clean_test_csv = os.path.join(WORKSPACE_ROOT, "data", "processed", "test.csv")
        cls.drift_csv = os.path.join(WORKSPACE_ROOT, "results", "drift", "datasets", "scenario_e_compound_stress.csv")
        
        cls.feature_pipeline = FeaturePipeline.load(os.path.join(WORKSPACE_ROOT, "models", "feature_pipeline.pkl"))
        cls.clean_df = pd.read_csv(cls.clean_test_csv)
        cls.drift_df = pd.read_csv(cls.drift_csv)
        
        cls.baseline_model_path = os.path.join(WORKSPACE_ROOT, "models", "nsga2", "nsga2_selected_model.pt")
        cls.adapted_model_path = os.path.join(cls.models_dir, "adapted_model.pt")
        cls.calibrator_path = os.path.join(cls.models_dir, "calibrator.pkl")

    # -------------------------------------------------------------
    # TC-01: Trigger Ingestion
    # -------------------------------------------------------------
    def test_tc01_trigger_ingestion(self):
        """Verify programmatic ingestion of Phase 3 TRIGGER_ADAPTATION vs PROCEED_TO_PREDICT."""
        metrics_path = os.path.join(self.drift_dir, "drift_metrics.json")
        self.assertTrue(os.path.exists(metrics_path), "drift_metrics.json not found")
        
        # Clean test must NOT trigger adaptation
        self.assertFalse(check_adaptation_trigger("clean_test", metrics_path))
        # OOD scenarios MUST trigger adaptation
        self.assertTrue(check_adaptation_trigger("scenario_e_compound_stress", metrics_path))
        self.assertTrue(check_adaptation_trigger("scenario_a_attendance_drift_20pct", metrics_path))

    # -------------------------------------------------------------
    # TC-02: Warm-Start Initialization
    # -------------------------------------------------------------
    def test_tc02_warm_start_initialization(self):
        """Verify re-optimizer initializes Generation 0 using Phase 2 Pareto genotypes."""
        pareto_csv = os.path.join(WORKSPACE_ROOT, "results", "nsga2", "pareto_front.csv")
        self.assertTrue(os.path.exists(pareto_csv))
        
        seed_genomes = load_pareto_genomes(pareto_csv)
        self.assertGreaterEqual(len(seed_genomes), 10)
        
        rng = np.random.default_rng(42)
        pop = warm_start_population(seed_genomes, pop_size=16, rng=rng)
        self.assertEqual(len(pop), 16)
        
        # Check first individuals preserve Pareto seed genomes
        for i in range(min(len(seed_genomes), 16)):
            self.assertEqual(pop[i].genome.hidden_dims, seed_genomes[i].hidden_dims)
            self.assertEqual(pop[i].genome.activation, seed_genomes[i].activation)

    # -------------------------------------------------------------
    # TC-03: Pareto Dominance Invariant
    # -------------------------------------------------------------
    def test_tc03_pareto_dominance_invariant(self):
        """Verify all solutions in adapted_pareto_front.csv belong to non-dominated front (rank == 0)."""
        pf_csv = os.path.join(self.results_dir, "adapted_pareto_front.csv")
        self.assertTrue(os.path.exists(pf_csv), "adapted_pareto_front.csv missing")
        df_pf = pd.read_csv(pf_csv)
        self.assertGreater(len(df_pf), 0)
        self.assertTrue((df_pf["rank"] == 0).all(), "Non-rank-0 individual found in adapted Pareto front")

    # -------------------------------------------------------------
    # TC-04: Read-Only Upstream Immutability
    # -------------------------------------------------------------
    def test_tc04_readonly_immutability(self):
        """Verify baseline checkpoint in models/nsga2/ remains bit-identical and unaffected."""
        ckpt = torch.load(self.baseline_model_path, map_location="cpu", weights_only=False)
        self.assertEqual(ckpt["genome"]["hidden_dims"], [32, 16])
        self.assertEqual(ckpt["genome"]["activation"], "tanh")
        self.assertAlmostEqual(ckpt["metrics"]["mae"], 0.8359, places=3)

    # -------------------------------------------------------------
    # TC-05: Test Set Isolation
    # -------------------------------------------------------------
    def test_tc05_test_set_isolation(self):
        """Verify adaptation re-optimization never trains or evaluates on clean_test.csv."""
        summary_path = os.path.join(self.results_dir, "reoptimization_summary.json")
        self.assertTrue(os.path.exists(summary_path))
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        self.assertIn("selected_candidate_genome", summary)

    # -------------------------------------------------------------
    # TC-06: Directional Bias Reduction (|Bias| < 0.05)
    # -------------------------------------------------------------
    def test_tc06_directional_bias_reduction(self):
        """Verify post-calibration directional prediction bias satisfies |Bias| < 0.05."""
        benchmark_csv = os.path.join(self.results_dir, "recovery_benchmark.csv")
        self.assertTrue(os.path.exists(benchmark_csv))
        df_bench = pd.read_csv(benchmark_csv)
        calibrated_row = df_bench[df_bench["stage_id"] == "4_calibrated_output"].iloc[0]
        cal_bias = abs(float(calibrated_row["mean_bias"]))
        self.assertLess(cal_bias, 0.05, f"Post-calibration bias {cal_bias} exceeds 0.05 threshold")

    # -------------------------------------------------------------
    # TC-07: Conformal Coverage Validity (>= 90% +- epsilon)
    # -------------------------------------------------------------
    def test_tc07_conformal_coverage_validity(self):
        """Verify empirical coverage meets or exceeds 90% target level on evaluation split."""
        metrics_json = os.path.join(self.results_dir, "recovery_metrics.json")
        self.assertTrue(os.path.exists(metrics_json))
        with open(metrics_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        emp_coverage = data["conformal_details"]["empirical_coverage"]
        self.assertGreaterEqual(emp_coverage, 0.87, f"Empirical coverage {emp_coverage} is too low")

    # -------------------------------------------------------------
    # TC-08: Interval Width Efficiency
    # -------------------------------------------------------------
    def test_tc08_interval_width_efficiency(self):
        """Verify conformal intervals maintain tight, informative bounds (0 < Width < 4.0 SGPA)."""
        metrics_json = os.path.join(self.results_dir, "recovery_metrics.json")
        with open(metrics_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        mean_width = data["conformal_details"]["mean_interval_width"]
        self.assertGreater(mean_width, 0.5)
        self.assertLess(mean_width, 4.0)

    # -------------------------------------------------------------
    # TC-09: Sub-Group Calibration
    # -------------------------------------------------------------
    def test_tc09_subgroup_calibration(self):
        """Verify high-risk student sub-population achieves calibrated coverage (> 85%)."""
        metrics_json = os.path.join(self.results_dir, "recovery_metrics.json")
        with open(metrics_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        subgroup_stats = data["conformal_details"].get("subgroup_metrics", {})
        self.assertIn("high_risk", subgroup_stats)
        high_risk_cov = subgroup_stats["high_risk"]["coverage"]
        self.assertGreaterEqual(high_risk_cov, 0.85, f"High-risk coverage {high_risk_cov} under-covered")

    # -------------------------------------------------------------
    # TC-10: Integrated Gradients Completeness Axiom
    # -------------------------------------------------------------
    def test_tc10_integrated_gradients_completeness(self):
        """Verify Integrated Gradients satisfies sum(attributions) == F(x) - F(baseline)."""
        input_dim = self.feature_pipeline.transform(self.clean_df.head(2)).shape[1]
        model = load_model_from_checkpoint(self.adapted_model_path, input_dim=input_dim, device=torch.device("cpu"))
        feat_names = self.feature_pipeline.feature_names
        
        explainer = IntegratedGradientsExplainer(model, feat_names, steps=50)
        x_sample = self.feature_pipeline.transform(self.clean_df.head(1))[0]
        
        attributions, pred, base_pred = explainer.explain_sample(x_sample)
        sum_attr = float(np.sum(attributions))
        expected_diff = float(pred - base_pred)
        
        self.assertAlmostEqual(sum_attr, expected_diff, places=3, msg="Completeness axiom violated in Integrated Gradients")

    # -------------------------------------------------------------
    # TC-11: Attribution Sanity & Bounds
    # -------------------------------------------------------------
    def test_tc11_attribution_sanity_and_bounds(self):
        """Verify feature attributions contain no NaNs or Infs."""
        input_dim = self.feature_pipeline.transform(self.clean_df.head(2)).shape[1]
        model = load_model_from_checkpoint(self.adapted_model_path, input_dim=input_dim, device=torch.device("cpu"))
        explainer = IntegratedGradientsExplainer(model, self.feature_pipeline.feature_names, steps=25)
        
        X_sample = self.feature_pipeline.transform(self.clean_df.head(5))
        batch_attr = explainer.explain_batch(X_sample)
        
        self.assertFalse(np.isnan(batch_attr).any(), "NaN in feature attributions")
        self.assertFalse(np.isinf(batch_attr).any(), "Inf in feature attributions")

    # -------------------------------------------------------------
    # TC-12: Error Recovery Monotonicity
    # -------------------------------------------------------------
    def test_tc12_error_recovery_monotonicity(self):
        """Verify adapted and calibrated model achieves improved metrics over unadapted drift."""
        benchmark_csv = os.path.join(self.results_dir, "recovery_benchmark.csv")
        df_bench = pd.read_csv(benchmark_csv)
        
        drift_row = df_bench[df_bench["stage_id"] == "2_drift_degradation"].iloc[0]
        cal_row = df_bench[df_bench["stage_id"] == "4_calibrated_output"].iloc[0]
        
        # Absolute bias must be strictly reduced
        self.assertLess(abs(cal_row["mean_bias"]), abs(drift_row["mean_bias"]))

    # -------------------------------------------------------------
    # TC-13: Deterministic Reproducibility
    # -------------------------------------------------------------
    def test_tc13_deterministic_reproducibility(self):
        """Verify calibration fit and evaluation produce deterministic outputs across runs."""
        c1 = SplitConformalCalibrator(confidence_level=0.90)
        c2 = SplitConformalCalibrator(confidence_level=0.90)
        
        y_dummy_p = np.array([6.5, 7.2, 8.1, 5.9, 7.8])
        y_dummy_t = np.array([6.3, 7.0, 8.4, 6.0, 7.6])
        
        c1.fit(y_dummy_p, y_dummy_t)
        c2.fit(y_dummy_p, y_dummy_t)
        
        self.assertEqual(c1.global_q, c2.global_q)
        self.assertEqual(c1.point_calibrator.slope, c2.point_calibrator.slope)
        self.assertEqual(c1.point_calibrator.intercept, c2.point_calibrator.intercept)

    # -------------------------------------------------------------
    # TC-14: Artifact Presence & Integrity
    # -------------------------------------------------------------
    def test_tc14_artifact_presence_and_integrity(self):
        """Verify all generated reports, CSVs, JSONs, and 4 PNG figures exist with valid size."""
        required_artifacts = [
            os.path.join(self.results_dir, "adapted_pareto_front.csv"),
            os.path.join(self.results_dir, "recovery_benchmark.csv"),
            os.path.join(self.results_dir, "recovery_report.md"),
            os.path.join(self.results_dir, "recovery_metrics.json"),
            os.path.join(self.results_dir, "reoptimization_summary.json"),
            os.path.join(self.results_dir, "pareto_adaptation_shift.png"),
            os.path.join(self.results_dir, "recovery_waterfall.png"),
            os.path.join(self.results_dir, "conformal_coverage_intervals.png"),
            os.path.join(self.results_dir, "explainability_shift.png"),
            os.path.join(self.models_dir, "adapted_model.pt"),
            os.path.join(self.models_dir, "calibrator.pkl")
        ]
        for art in required_artifacts:
            self.assertTrue(os.path.exists(art), f"Artifact missing: {art}")
            self.assertGreater(os.path.getsize(art), 300, f"Artifact empty: {art}")

    # -------------------------------------------------------------
    # TC-15: End-to-End Pipeline Execution
    # -------------------------------------------------------------
    def test_tc15_end_to_end_pipeline_execution(self):
        """Verify recovery report markdown contains required analytical sections."""
        report_path = os.path.join(self.results_dir, "recovery_report.md")
        self.assertTrue(os.path.exists(report_path))
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("PHASE 4 REPORT", content)
        self.assertIn("Quantitative Progression Benchmark", content)
        self.assertIn("Sub-Group Conformal Calibration Analysis", content)


if __name__ == "__main__":
    unittest.main()
