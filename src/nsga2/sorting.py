"""
src/nsga2/sorting.py
Standard NSGA-II non-dominated sorting and crowding distance calculation.
"""

from typing import List
import numpy as np

from src.nsga2.population import Individual
from src.nsga2.objectives import dominates


def fast_non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
    """
    Standard NSGA-II fast non-dominated sorting algorithm.
    
    Partitions the population into Pareto fronts:
    - Front 0: Non-dominated individuals (Pareto optimal)
    - Front 1: Individuals dominated only by Front 0
    - Front k: Individuals dominated only by previous fronts
    
    Assigns each individual's .rank attribute (0, 1, 2, ...).
    
    Returns:
        List of fronts, where each front is a list of Individuals.
    """
    fronts: List[List[Individual]] = [[]]
    n_p = {}  # Domination count: number of individuals dominating p
    s_p = {}  # Dominated set: individuals dominated by p
    
    for p in population:
        s_p[p] = []
        n_p[p] = 0
        
        for q in population:
            if p is q:
                continue
            if dominates(p.objectives, q.objectives):
                s_p[p].append(q)
            elif dominates(q.objectives, p.objectives):
                n_p[p] += 1
                
        if n_p[p] == 0:
            p.rank = 0
            fronts[0].append(p)
            
    i = 0
    while len(fronts[i]) > 0:
        next_front: List[Individual] = []
        for p in fronts[i]:
            for q in s_p[p]:
                n_p[q] -= 1
                if n_p[q] == 0:
                    q.rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
        
    # Remove empty trailing front if present
    if len(fronts) > 0 and len(fronts[-1]) == 0:
        fronts.pop()
        
    return fronts


def calculate_crowding_distance(front: List[Individual]) -> None:
    """
    Standard NSGA-II crowding distance assignment.
    
    Measures the density of solutions surrounding a particular individual in objective space.
    Boundary solutions receive infinite crowding distance to preserve extreme solutions.
    Modifies front individuals in-place.
    """
    n = len(front)
    if n == 0:
        return
        
    if n <= 2:
        for ind in front:
            ind.crowding_distance = float("inf")
        return
        
    for ind in front:
        ind.crowding_distance = 0.0
        
    num_objectives = len(front[0].objectives)
    
    for m in range(num_objectives):
        # Sort front by objective m ascending
        front.sort(key=lambda ind: ind.objectives[m])
        
        # Boundary solutions receive infinite distance
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")
        
        f_min = front[0].objectives[m]
        f_max = front[-1].objectives[m]
        
        norm_range = f_max - f_min
        if norm_range <= 1e-12:
            continue
            
        for i in range(1, n - 1):
            if front[i].crowding_distance != float("inf"):
                dist_contrib = (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / norm_range
                front[i].crowding_distance += dist_contrib
