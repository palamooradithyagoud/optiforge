"""
src/nsga2/objectives.py
Multi-objective metrics definition, Pareto dominance checking, and penalty handling.
All objectives are strictly formulated as minimization targets for NSGA-II.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Sequence
import numpy as np


# Penalty vector assigned when an individual crashes or encounters numerical errors
PENALTY_OBJECTIVES = [
    10.0,      # f1: validation_mae
    10.0,      # f2: generalization_gap
    10.0,      # f3: validation_loss_variance
    200000.0,  # f4: trainable_parameters
    100.0,     # f5: inference_latency_ms
    500.0      # f6: training_time_seconds
]

OBJECTIVE_NAMES = [
    "validation_mae",
    "generalization_gap",
    "val_loss_variance",
    "trainable_parameters",
    "inference_latency_ms",
    "training_time_seconds"
]


@dataclass
class ObjectiveMetrics:
    """Detailed evaluation metrics for a candidate genome."""
    validation_mae: float
    generalization_gap: float
    val_loss_variance: float
    trainable_parameters: int
    inference_latency_ms: float
    training_time_seconds: float
    
    # Secondary diagnostic metrics (not explicitly sorted on, but logged)
    validation_rmse: float = 0.0
    validation_r2: float = 0.0
    validation_loss: float = 0.0
    training_loss: float = 0.0
    best_epoch: int = 0
    status: str = "success"

    def to_objective_vector(self) -> List[float]:
        """Return the 6 primary optimization objectives as a list of floats."""
        return [
            float(self.validation_mae),
            float(self.generalization_gap),
            float(self.val_loss_variance),
            float(self.trainable_parameters),
            float(self.inference_latency_ms),
            float(self.training_time_seconds)
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "validation_mae": round(float(self.validation_mae), 4),
            "generalization_gap": round(float(self.generalization_gap), 4),
            "val_loss_variance": round(float(self.val_loss_variance), 6),
            "trainable_parameters": int(self.trainable_parameters),
            "inference_latency_ms": round(float(self.inference_latency_ms), 4),
            "training_time_seconds": round(float(self.training_time_seconds), 3),
            "validation_rmse": round(float(self.validation_rmse), 4),
            "validation_r2": round(float(self.validation_r2), 4),
            "validation_loss": round(float(self.validation_loss), 4),
            "training_loss": round(float(self.training_loss), 4),
            "best_epoch": int(self.best_epoch),
            "status": str(self.status)
        }


def compute_model_flops(input_dim: int, hidden_dims: List[int]) -> int:
    """Compute exact multiply-accumulate FLOPs for a single forward pass."""
    flops = 0
    prev_dim = input_dim
    for h in hidden_dims:
        flops += 2 * prev_dim * h + h
        flops += h
        prev_dim = h
    flops += 2 * prev_dim * 1 + 1
    return flops


def compute_deterministic_latency_ms(input_dim: int, hidden_dims: List[int], cpu_mflops_per_ms: float = 80.0) -> float:
    """
    Deterministic inference latency benchmark in milliseconds.
    Based on exact architectural FLOP complexity calibrated to CPU execution throughput.
    Eliminates OS thread-scheduling noise while strictly penalizing deeper/wider architectures.
    """
    flops = compute_model_flops(input_dim, hidden_dims)
    latency_ms = flops / (cpu_mflops_per_ms * 1e3)
    return round(float(max(latency_ms, 0.01)), 4)


def compute_deterministic_training_cost(
    flops: int,
    epochs: int,
    num_train_samples: int = 600,
    cpu_flops_per_sec: float = 3.5e7
) -> float:
    """
    Deterministic computational proxy for training cost in seconds.
    Formula: Total Training FLOPs = epochs * samples * 3 (fwd + bwd) * forward_flops.
    """
    total_flops = epochs * num_train_samples * 3 * flops
    cost_sec = total_flops / cpu_flops_per_sec
    return round(float(max(cost_sec, 0.1)), 3)


def dominates(obj_a: Sequence[float], obj_b: Sequence[float]) -> bool:
    """
    Pareto dominance relation for minimization objectives.
    
    Individual A dominates Individual B (A ≺ B) if and only if:
    1. A is no worse than B on all objectives (a_i <= b_i for all i)
    2. A is strictly better than B on at least one objective (a_i < b_i for some i)
    
    Args:
        obj_a: Objective vector for individual A
        obj_b: Objective vector for individual B
        
    Returns:
        bool: True if A dominates B, False otherwise
    """
    strictly_better = False
    for a, b in zip(obj_a, obj_b):
        if a > b:
            return False  # A is worse on this objective -> cannot dominate
        if a < b:
            strictly_better = True
    return strictly_better

