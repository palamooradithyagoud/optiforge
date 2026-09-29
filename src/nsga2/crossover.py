"""
src/nsga2/crossover.py
Crossover operators combining continuous blending and categorical exchange.
"""

from typing import Tuple
import numpy as np

from src.nsga2.genome import Genome, repair_genome


def simulated_binary_crossover_val(val1: float, val2: float, eta: float = 2.0, rng: np.random.Generator = None) -> Tuple[float, float]:
    """Simulated Binary Crossover (SBX) for a single continuous gene."""
    if rng is None:
        rng = np.random.default_rng()
    u = float(rng.uniform(0.0, 1.0))
    if u <= 0.5:
        beta = (2.0 * u) ** (1.0 / (eta + 1.0))
    else:
        beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
        
    c1 = 0.5 * ((1.0 + beta) * val1 + (1.0 - beta) * val2)
    c2 = 0.5 * ((1.0 - beta) * val1 + (1.0 + beta) * val2)
    return float(c1), float(c2)


def crossover(parent1: Genome, parent2: Genome, rng: np.random.Generator) -> Tuple[Genome, Genome]:
    """
    Perform genetic crossover between two parent genomes.
    
    Combines:
    - Discrete uniform exchange for architecture, activation, batch size, clip norm.
    - Simulated Binary Crossover (SBX) for continuous numerical parameters (dropout, lr, weight decay, l1, huber delta).
    - Repair on offspring to guarantee legal genomes.
    
    Returns:
        child1, child2
    """
    # 1. Architecture: uniform swap
    if rng.uniform() < 0.5:
        arch1, arch2 = list(parent1.hidden_dims), list(parent2.hidden_dims)
    else:
        arch1, arch2 = list(parent2.hidden_dims), list(parent1.hidden_dims)
        
    # 2. Activation: uniform swap
    if rng.uniform() < 0.5:
        act1, act2 = parent1.activation, parent2.activation
    else:
        act1, act2 = parent2.activation, parent1.activation
        
    # 3. Batch size: uniform swap
    if rng.uniform() < 0.5:
        bs1, bs2 = parent1.batch_size, parent2.batch_size
    else:
        bs1, bs2 = parent2.batch_size, parent1.batch_size
        
    # 4. Gradient clip norm: uniform swap
    if rng.uniform() < 0.5:
        clip1, clip2 = parent1.gradient_clip_norm, parent2.gradient_clip_norm
    else:
        clip1, clip2 = parent2.gradient_clip_norm, parent1.gradient_clip_norm
        
    # 5. Epochs: blend or swap
    if rng.uniform() < 0.5:
        ep1, ep2 = parent1.epochs, parent2.epochs
    else:
        ep1 = int(round(0.5 * (parent1.epochs + parent2.epochs)))
        ep2 = int(round(0.5 * (parent1.epochs + parent2.epochs)))
        
    # Continuous SBX Crossover
    # Dropout
    drop1, drop2 = simulated_binary_crossover_val(parent1.dropout_rate, parent2.dropout_rate, eta=2.0, rng=rng)
    
    # Learning rate (log space SBX)
    log_lr1 = np.log10(max(parent1.learning_rate, 1e-6))
    log_lr2 = np.log10(max(parent2.learning_rate, 1e-6))
    c_log1, c_log2 = simulated_binary_crossover_val(log_lr1, log_lr2, eta=2.0, rng=rng)
    lr1, lr2 = 10.0 ** c_log1, 10.0 ** c_log2
    
    # Weight decay (log space SBX)
    log_wd1 = np.log10(max(parent1.weight_decay, 1e-8))
    log_wd2 = np.log10(max(parent2.weight_decay, 1e-8))
    c_wd1, c_wd2 = simulated_binary_crossover_val(log_wd1, log_wd2, eta=2.0, rng=rng)
    wd1, wd2 = 10.0 ** c_wd1, 10.0 ** c_wd2
    
    # L1 Lambda
    l1_1, l1_2 = simulated_binary_crossover_val(parent1.l1_lambda, parent2.l1_lambda, eta=2.0, rng=rng)
    
    # Huber Delta
    del1, del2 = simulated_binary_crossover_val(parent1.huber_delta, parent2.huber_delta, eta=2.0, rng=rng)
    
    child1 = Genome(
        hidden_dims=arch1,
        dropout_rate=drop1,
        activation=act1,
        learning_rate=lr1,
        weight_decay=wd1,
        l1_lambda=l1_1,
        huber_delta=del1,
        batch_size=bs1,
        gradient_clip_norm=clip1,
        epochs=ep1
    )
    
    child2 = Genome(
        hidden_dims=arch2,
        dropout_rate=drop2,
        activation=act2,
        learning_rate=lr2,
        weight_decay=wd2,
        l1_lambda=l1_2,
        huber_delta=del2,
        batch_size=bs2,
        gradient_clip_norm=clip2,
        epochs=ep2
    )
    
    return repair_genome(child1), repair_genome(child2)
