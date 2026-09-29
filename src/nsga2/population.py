"""
src/nsga2/population.py
Individual and Population classes with deterministic initialization.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from src.nsga2.genome import Genome, sample_random_genome
from src.nsga2.objectives import ObjectiveMetrics, PENALTY_OBJECTIVES


class Individual:
    """An individual in the NSGA-II population."""
    def __init__(
        self,
        genome: Genome,
        generation: int = 0,
        individual_id: Optional[str] = None
    ):
        self.genome = genome
        self.generation = generation
        self.id = individual_id or f"gen{generation}_{genome.canonical_hash()[:8]}"
        
        # NSGA-II parameters
        self.objectives: List[float] = list(PENALTY_OBJECTIVES)
        self.metrics: Optional[ObjectiveMetrics] = None
        self.rank: int = -1
        self.crowding_distance: float = 0.0
        
        # Accounting
        self.evaluated: bool = False
        self.cached: bool = False
        self.status: str = "pending"

    def set_evaluation_result(self, metrics: ObjectiveMetrics, cached: bool = False):
        """Assign evaluation metrics and objective vector."""
        self.metrics = metrics
        self.objectives = metrics.to_objective_vector()
        self.evaluated = True
        self.cached = cached
        self.status = metrics.status

    def to_dict(self) -> Dict[str, Any]:
        """Convert individual attributes to dictionary."""
        d = {
            "individual_id": self.id,
            "generation": self.generation,
            "rank": self.rank,
            "crowding_distance": float(self.crowding_distance),
            "status": self.status,
            "cached": self.cached,
            "objectives": self.objectives,
            "genome": self.genome.to_dict()
        }
        if self.metrics is not None:
            d["metrics"] = self.metrics.to_dict()
        return d


Population = List[Individual]


def initialize_population(
    pop_size: int,
    rng: np.random.Generator,
    include_phase1_baseline: bool = True
) -> Population:
    """
    Initialize a deterministic population of size pop_size.
    Optionally seeds the exact Phase 1 baseline genome into generation 0.
    """
    population: Population = []
    
    # Optionally seed Phase 1 baseline
    if include_phase1_baseline and pop_size > 0:
        phase1_baseline_genome = Genome(
            hidden_dims=[64, 32],
            dropout_rate=0.2,
            activation="relu",
            learning_rate=0.001,
            weight_decay=0.0001,
            l1_lambda=0.00005,
            huber_delta=1.0,
            batch_size=32,
            gradient_clip_norm=1.0,
            epochs=78
        )
        ind_base = Individual(
            genome=phase1_baseline_genome,
            generation=0,
            individual_id="gen0_phase1_baseline"
        )
        population.append(ind_base)
        
    # Sample remaining genomes
    while len(population) < pop_size:
        g = sample_random_genome(rng)
        ind = Individual(
            genome=g,
            generation=0,
            individual_id=f"gen0_ind{len(population):03d}"
        )
        population.append(ind)
        
    return population
