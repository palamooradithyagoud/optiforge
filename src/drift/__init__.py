"""
src/drift
Phase 3: Out-of-Distribution (OOD) Stress Testing & Statistical Drift Detection.
"""

from src.drift.scenarios import (
    generate_all_scenarios,
    generate_attendance_drift,
    generate_academic_drift,
    generate_backlog_surge,
    generate_cohort_shift,
    generate_compound_stress,
    SCENARIO_CONFIGS
)
from src.drift.detector import DriftDetector
from src.drift.evaluate_ood import OODEvaluator, run_ood_evaluation

__all__ = [
    "generate_all_scenarios",
    "generate_attendance_drift",
    "generate_academic_drift",
    "generate_backlog_surge",
    "generate_cohort_shift",
    "generate_compound_stress",
    "SCENARIO_CONFIGS",
    "DriftDetector",
    "OODEvaluator",
    "run_ood_evaluation"
]
