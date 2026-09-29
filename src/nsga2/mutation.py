"""
src/nsga2/mutation.py
Mutation operators preserving valid hyperparameter bounds and architecture constraints.
"""

import numpy as np

from src.nsga2.genome import (
    Genome, repair_genome,
    SUPPORTED_ARCHITECTURES, SUPPORTED_ACTIVATIONS,
    SUPPORTED_BATCH_SIZES, SUPPORTED_CLIP_NORMS
)


def mutate(genome: Genome, mutation_rate: float, rng: np.random.Generator) -> Genome:
    """
    Perform point mutations across genome attributes with probability mutation_rate.
    
    Mutations:
    - Architecture: randomly sample alternative supported architecture
    - Continuous parameters: Gaussian perturbations in linear/log domain
    - Categorical parameters: uniform random replacement
    - Post-mutation repair to guarantee legal bounds
    """
    arch = list(genome.hidden_dims)
    drop = float(genome.dropout_rate)
    act = str(genome.activation)
    lr = float(genome.learning_rate)
    wd = float(genome.weight_decay)
    l1 = float(genome.l1_lambda)
    delta = float(genome.huber_delta)
    bs = int(genome.batch_size)
    clip = float(genome.gradient_clip_norm)
    eps = int(genome.epochs)
    
    # 1. Architecture mutation
    if rng.uniform() < mutation_rate:
        idx = int(rng.integers(0, len(SUPPORTED_ARCHITECTURES)))
        arch = SUPPORTED_ARCHITECTURES[idx]
        
    # 2. Dropout mutation
    if rng.uniform() < mutation_rate:
        drop += float(rng.normal(0.0, 0.05))
        
    # 3. Activation mutation
    if rng.uniform() < mutation_rate:
        idx = int(rng.integers(0, len(SUPPORTED_ACTIVATIONS)))
        act = SUPPORTED_ACTIVATIONS[idx]
        
    # 4. Learning rate mutation (log-scale perturbation)
    if rng.uniform() < mutation_rate:
        log_lr = np.log10(max(lr, 1e-6)) + float(rng.normal(0.0, 0.3))
        lr = 10.0 ** log_lr
        
    # 5. Weight decay mutation (log-scale perturbation)
    if rng.uniform() < mutation_rate:
        log_wd = np.log10(max(wd, 1e-8)) + float(rng.normal(0.0, 0.5))
        wd = 10.0 ** log_wd
        
    # 6. L1 lambda mutation
    if rng.uniform() < mutation_rate:
        l1 += float(rng.normal(0.0, 1e-4))
        
    # 7. Huber delta mutation
    if rng.uniform() < mutation_rate:
        delta += float(rng.normal(0.0, 0.2))
        
    # 8. Batch size mutation
    if rng.uniform() < mutation_rate:
        idx = int(rng.integers(0, len(SUPPORTED_BATCH_SIZES)))
        bs = SUPPORTED_BATCH_SIZES[idx]
        
    # 9. Clip norm mutation
    if rng.uniform() < mutation_rate:
        idx = int(rng.integers(0, len(SUPPORTED_CLIP_NORMS)))
        clip = SUPPORTED_CLIP_NORMS[idx]
        
    # 10. Epochs mutation
    if rng.uniform() < mutation_rate:
        eps += int(rng.integers(-15, 16))
        
    mutated = Genome(
        hidden_dims=arch,
        dropout_rate=drop,
        activation=act,
        learning_rate=lr,
        weight_decay=wd,
        l1_lambda=l1,
        huber_delta=delta,
        batch_size=bs,
        gradient_clip_norm=clip,
        epochs=eps
    )
    return repair_genome(mutated)
