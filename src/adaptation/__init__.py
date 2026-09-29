"""
src/adaptation
Phase 4: Adaptive Re-Optimization, Sub-Population Calibration & Explainability.
"""

from src.adaptation.reoptimizer import AdaptiveReOptimizer, run_adaptation_reoptimization
from src.adaptation.calibration import ContinuousCalibrator, SplitConformalCalibrator
from src.adaptation.explainability import IntegratedGradientsExplainer
from src.adaptation.evaluate_recovery import run_recovery_evaluation
from src.adaptation.visualize_adaptation import generate_adaptation_visualizations

__all__ = [
    "AdaptiveReOptimizer",
    "run_adaptation_reoptimization",
    "ContinuousCalibrator",
    "SplitConformalCalibrator",
    "IntegratedGradientsExplainer",
    "run_recovery_evaluation",
    "generate_adaptation_visualizations"
]
