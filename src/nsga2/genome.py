"""
src/nsga2/genome.py
Genome definition, validation, repair, sampling, and canonical hashing for NSGA-II.
"""

import json
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Union
import numpy as np


SUPPORTED_ARCHITECTURES = [
    [32],
    [64],
    [128],
    [32, 16],
    [64, 32],
    [128, 64],
    [128, 64, 32]
]

SUPPORTED_ACTIVATIONS = ["relu", "gelu", "tanh"]
SUPPORTED_BATCH_SIZES = [16, 32, 64]
SUPPORTED_CLIP_NORMS = [0.5, 1.0, 2.0]


@dataclass
class Genome:
    """Hyperparameter chromosome for neural network architecture and training configuration."""
    hidden_dims: List[int]
    dropout_rate: float
    activation: str
    learning_rate: float
    weight_decay: float
    l1_lambda: float
    huber_delta: float
    batch_size: int
    gradient_clip_norm: float
    epochs: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to standard dictionary representation."""
        return {
            "hidden_dims": list(self.hidden_dims),
            "dropout_rate": round(float(self.dropout_rate), 4),
            "activation": str(self.activation),
            "learning_rate": round(float(self.learning_rate), 6),
            "weight_decay": round(float(self.weight_decay), 7),
            "l1_lambda": round(float(self.l1_lambda), 7),
            "huber_delta": round(float(self.huber_delta), 3),
            "batch_size": int(self.batch_size),
            "gradient_clip_norm": round(float(self.gradient_clip_norm), 3),
            "epochs": int(self.epochs)
        }

    def canonical_json(self) -> str:
        """Deterministic canonical JSON representation for unambiguous caching."""
        d = self.to_dict()
        return json.dumps(d, sort_keys=True)

    def canonical_hash(self) -> str:
        """SHA-256 hash of canonical JSON."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Genome":
        """Reconstruct Genome from dictionary."""
        return cls(
            hidden_dims=list(d["hidden_dims"]),
            dropout_rate=float(d["dropout_rate"]),
            activation=str(d["activation"]),
            learning_rate=float(d["learning_rate"]),
            weight_decay=float(d["weight_decay"]),
            l1_lambda=float(d["l1_lambda"]),
            huber_delta=float(d["huber_delta"]),
            batch_size=int(d["batch_size"]),
            gradient_clip_norm=float(d["gradient_clip_norm"]),
            epochs=int(d["epochs"])
        )


def validate_genome(genome: Genome) -> bool:
    """
    Validate that all genes fall within valid bounds and types.
    
    Checks:
    - hidden_dims is in SUPPORTED_ARCHITECTURES or valid layer list
    - 0.0 <= dropout_rate <= 0.5
    - activation in SUPPORTED_ACTIVATIONS
    - 1e-4 <= learning_rate <= 1e-2
    - 1e-6 <= weight_decay <= 1e-2
    - 0.0 <= l1_lambda <= 1e-3
    - 0.5 <= huber_delta <= 2.0
    - batch_size in SUPPORTED_BATCH_SIZES
    - gradient_clip_norm in SUPPORTED_CLIP_NORMS
    - 40 <= epochs <= 150
    """
    if not isinstance(genome.hidden_dims, list) or len(genome.hidden_dims) == 0:
        return False
    if any(not isinstance(d, int) or d <= 0 for d in genome.hidden_dims):
        return False
    if not (0.0 <= genome.dropout_rate <= 0.5):
        return False
    if str(genome.activation).lower() not in SUPPORTED_ACTIVATIONS:
        return False
    if not (1e-4 <= genome.learning_rate <= 1e-2):
        return False
    if not (1e-6 <= genome.weight_decay <= 1e-2):
        return False
    if not (0.0 <= genome.l1_lambda <= 1e-3):
        return False
    if not (0.5 <= genome.huber_delta <= 2.0):
        return False
    if genome.batch_size not in SUPPORTED_BATCH_SIZES:
        return False
    if genome.gradient_clip_norm not in SUPPORTED_CLIP_NORMS:
        return False
    if not (30 <= genome.epochs <= 150):
        return False
    return True


def repair_genome(genome: Genome) -> Genome:
    """
    Repair any out-of-bounds or invalid genes into their nearest valid states.
    Ensures the optimizer never crashes or trains on illegal configurations.
    """
    # 1. Architecture repair
    arch = list(genome.hidden_dims) if isinstance(genome.hidden_dims, list) else [64, 32]
    # Check if exact match
    if arch not in SUPPORTED_ARCHITECTURES:
        # Find nearest architecture by total parameter count / capacity
        arch = min(SUPPORTED_ARCHITECTURES, key=lambda a: abs(sum(a) - sum(arch)))
        
    # 2. Dropout
    dropout = float(np.clip(genome.dropout_rate, 0.0, 0.5))
    
    # 3. Activation
    act = str(genome.activation).lower()
    if act not in SUPPORTED_ACTIVATIONS:
        act = "relu"
        
    # 4. Learning rate (log scale clamp)
    lr = float(np.clip(genome.learning_rate, 1e-4, 1e-2))
    
    # 5. Weight decay
    wd = float(np.clip(genome.weight_decay, 1e-6, 1e-2))
    
    # 6. L1 lambda
    l1 = float(np.clip(genome.l1_lambda, 0.0, 1e-3))
    
    # 7. Huber delta
    delta = float(np.clip(genome.huber_delta, 0.5, 2.0))
    
    # 8. Batch size
    bs = int(min(SUPPORTED_BATCH_SIZES, key=lambda b: abs(b - genome.batch_size)))
    
    # 9. Gradient clip norm
    clip = float(min(SUPPORTED_CLIP_NORMS, key=lambda c: abs(c - genome.gradient_clip_norm)))
    
    # 10. Epochs
    eps = int(np.clip(genome.epochs, 30, 150))
    
    return Genome(
        hidden_dims=arch,
        dropout_rate=round(dropout, 4),
        activation=act,
        learning_rate=round(lr, 6),
        weight_decay=round(wd, 7),
        l1_lambda=round(l1, 7),
        huber_delta=round(delta, 3),
        batch_size=bs,
        gradient_clip_norm=round(clip, 3),
        epochs=eps
    )


def sample_random_genome(rng: np.random.Generator) -> Genome:
    """Sample a random valid genome uniformly across the search space."""
    arch_idx = int(rng.integers(0, len(SUPPORTED_ARCHITECTURES)))
    hidden_dims = SUPPORTED_ARCHITECTURES[arch_idx]
    
    dropout_rate = float(rng.uniform(0.0, 0.5))
    
    act_idx = int(rng.integers(0, len(SUPPORTED_ACTIVATIONS)))
    activation = SUPPORTED_ACTIVATIONS[act_idx]
    
    # Log-uniform sampling for learning rate: 10^U(-4, -2)
    lr_exp = rng.uniform(-4.0, -2.0)
    learning_rate = float(10.0 ** lr_exp)
    
    # Log-uniform sampling for weight decay: 10^U(-6, -2)
    wd_exp = rng.uniform(-6.0, -2.0)
    weight_decay = float(10.0 ** wd_exp)
    
    # Uniform sampling for L1: 0 to 1e-3
    l1_lambda = float(rng.uniform(0.0, 1e-3))
    
    # Uniform sampling for Huber delta: 0.5 to 2.0
    huber_delta = float(rng.uniform(0.5, 2.0))
    
    # Categorical sampling
    bs_idx = int(rng.integers(0, len(SUPPORTED_BATCH_SIZES)))
    batch_size = SUPPORTED_BATCH_SIZES[bs_idx]
    
    clip_idx = int(rng.integers(0, len(SUPPORTED_CLIP_NORMS)))
    gradient_clip_norm = SUPPORTED_CLIP_NORMS[clip_idx]
    
    epochs = int(rng.integers(40, 101))
    
    g = Genome(
        hidden_dims=hidden_dims,
        dropout_rate=round(dropout_rate, 4),
        activation=activation,
        learning_rate=round(learning_rate, 6),
        weight_decay=round(weight_decay, 7),
        l1_lambda=round(l1_lambda, 7),
        huber_delta=round(huber_delta, 3),
        batch_size=batch_size,
        gradient_clip_norm=gradient_clip_norm,
        epochs=epochs
    )
    return repair_genome(g)
