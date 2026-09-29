"""
src/nsga2 module.
Implementation of NSGA-II multi-objective evolutionary hyperparameter optimization
for student performance regression under distribution drift.
"""

from src.nsga2.genome import Genome, validate_genome, repair_genome, sample_random_genome
from src.nsga2.population import Individual, Population, initialize_population
from src.nsga2.objectives import dominates, ObjectiveMetrics, PENALTY_OBJECTIVES
from src.nsga2.sorting import fast_non_dominated_sort, calculate_crowding_distance
from src.nsga2.selection import tournament_selection, binary_tournament
from src.nsga2.crossover import crossover
from src.nsga2.mutation import mutate
from src.nsga2.evaluation import GenomeEvaluator
from src.nsga2.evolution import NSGA2Optimizer
from src.nsga2.convergence import compute_hypervolume_2d, ConvergenceTracker

__all__ = [
    "Genome",
    "validate_genome",
    "repair_genome",
    "sample_random_genome",
    "Individual",
    "Population",
    "initialize_population",
    "dominates",
    "ObjectiveMetrics",
    "PENALTY_OBJECTIVES",
    "fast_non_dominated_sort",
    "calculate_crowding_distance",
    "tournament_selection",
    "binary_tournament",
    "crossover",
    "mutate",
    "GenomeEvaluator",
    "NSGA2Optimizer",
    "compute_hypervolume_2d",
    "ConvergenceTracker"
]
