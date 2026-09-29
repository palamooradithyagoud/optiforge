"""
src/nsga2/selection.py
Binary tournament selection based on Pareto rank and crowding distance.
"""

from typing import List
import numpy as np

from src.nsga2.population import Individual


def binary_tournament(ind1: Individual, ind2: Individual, rng: np.random.Generator) -> Individual:
    """
    Select the better individual between two candidates using NSGA-II crowded-comparison operator:
    1. Lower Pareto rank wins (rank 0 > rank 1).
    2. If ranks are equal, higher crowding distance wins (better diversity).
    3. If ranks and crowding distances are equal, break tie uniformly at random.
    """
    if ind1.rank < ind2.rank:
        return ind1
    elif ind2.rank < ind1.rank:
        return ind2
    else:
        # Same rank: compare crowding distance
        if ind1.crowding_distance > ind2.crowding_distance:
            return ind1
        elif ind2.crowding_distance > ind1.crowding_distance:
            return ind2
        else:
            return ind1 if rng.uniform() < 0.5 else ind2


def tournament_selection(
    population: List[Individual],
    num_selections: int,
    rng: np.random.Generator
) -> List[Individual]:
    """
    Select a mating pool of size num_selections via binary tournament selection.
    """
    selected: List[Individual] = []
    n = len(population)
    
    for _ in range(num_selections):
        idx1 = int(rng.integers(0, n))
        idx2 = int(rng.integers(0, n))
        while idx2 == idx1 and n > 1:
            idx2 = int(rng.integers(0, n))
            
        winner = binary_tournament(population[idx1], population[idx2], rng)
        selected.append(winner)
        
    return selected
