"""
src/nsga2/evolution.py
Complete NSGA-II evolutionary optimization engine.
Executes non-dominated sorting, crowding distance assignment, binary tournament selection,
genetic operators (SBX crossover, mutation), and elitist environmental selection.
"""

import time
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

from src.utils import set_seed
from src.nsga2.population import Individual, Population, initialize_population
from src.nsga2.sorting import fast_non_dominated_sort, calculate_crowding_distance
from src.nsga2.selection import tournament_selection
from src.nsga2.crossover import crossover
from src.nsga2.mutation import mutate
from src.nsga2.evaluation import GenomeEvaluator
from src.nsga2.convergence import ConvergenceTracker

logger = logging.getLogger(__name__)


def environmental_selection(combined_population: List[Individual], pop_size: int) -> List[Individual]:
    """
    Standard NSGA-II elitist environmental selection.
    
    Given the combined parent and offspring pool R_t of size 2N:
    1. Perform non-dominated sorting.
    2. Add successive non-dominated fronts F_0, F_1, ... into P_{t+1}
       until adding the next front would exceed pop_size.
    3. For the boundary front that does not completely fit, calculate crowding distance
       and select individuals with the highest crowding distance (best diversity).
       
    Guarantees:
    - Exactly pop_size individuals selected.
    - Elitism: Best Pareto solutions are never lost.
    """
    fronts = fast_non_dominated_sort(combined_population)
    new_population: List[Individual] = []
    
    for front in fronts:
        calculate_crowding_distance(front)
        if len(new_population) + len(front) <= pop_size:
            new_population.extend(front)
        else:
            # Front only partially fits: sort by crowding distance descending
            needed = pop_size - len(new_population)
            front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
            new_population.extend(front[:needed])
            break
            
    return new_population


class NSGA2Optimizer:
    """
    NSGA-II Multi-Objective Evolutionary Hyperparameter Optimizer.
    """
    def __init__(
        self,
        evaluator: GenomeEvaluator,
        population_size: int = 20,
        generations: int = 15,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.9,
        seed: int = 42,
        log_callback: Any = None
    ):
        self.evaluator = evaluator
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.log_callback = log_callback
        
        self.tracker = ConvergenceTracker()
        self.population_history: List[Dict[str, Any]] = []
        self.all_evaluated_individuals: List[Individual] = []

    def run(self) -> Tuple[List[Individual], List[Individual]]:
        """
        Execute full NSGA-II optimization.
        
        Returns:
            final_population: List[Individual]
            final_pareto_front: List[Individual] (Front 0 of final population)
        """
        logger.info(
            f"Starting NSGA-II optimization: Population={self.population_size}, "
            f"Generations={self.generations}, Seed={self.seed}"
        )
        set_seed(self.seed, deterministic=True)
        self.rng = np.random.default_rng(self.seed)
        
        # 1. Initialize Population
        population = initialize_population(
            self.population_size,
            self.rng,
            include_phase1_baseline=True
        )
        
        # 2. Evaluate Generation 0
        t0_start = time.perf_counter()
        for ind in population:
            metrics, cached = self.evaluator.evaluate(ind.genome)
            ind.set_evaluation_result(metrics, cached=cached)
            self.all_evaluated_individuals.append(ind)
            
        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            calculate_crowding_distance(front)
            
        gen0_time = time.perf_counter() - t0_start
        pareto_front_0 = fronts[0]
        
        # Record Generation 0
        self.tracker.record_generation(0, population, pareto_front_0, gen0_time)
        self._record_population_history(0, population)
        
        if self.log_callback:
            self.log_callback(0, population, pareto_front_0, gen0_time)
            
        # 3. Evolutionary Loop
        for gen in range(1, self.generations + 1):
            t_gen_start = time.perf_counter()
            
            # Binary Tournament Selection (mating pool)
            mating_pool = tournament_selection(population, self.population_size, self.rng)
            
            # Crossover & Mutation to generate offspring
            offspring_genomes = []
            for i in range(0, self.population_size, 2):
                p1 = mating_pool[i].genome
                p2 = mating_pool[(i + 1) % self.population_size].genome
                
                if self.rng.uniform() < self.crossover_rate:
                    c1, c2 = crossover(p1, p2, self.rng)
                else:
                    c1, c2 = p1, p2
                    
                c1 = mutate(c1, self.mutation_rate, self.rng)
                c2 = mutate(c2, self.mutation_rate, self.rng)
                
                offspring_genomes.extend([c1, c2])
                
            offspring_genomes = offspring_genomes[:self.population_size]
            
            # Evaluate Offspring
            offspring: List[Individual] = []
            for idx, g in enumerate(offspring_genomes):
                ind = Individual(
                    genome=g,
                    generation=gen,
                    individual_id=f"gen{gen}_ind{idx:03d}"
                )
                metrics, cached = self.evaluator.evaluate(ind.genome)
                ind.set_evaluation_result(metrics, cached=cached)
                offspring.append(ind)
                self.all_evaluated_individuals.append(ind)
                
            # Combine R_t = P_t + Q_t
            combined = population + offspring
            
            # Elitist Environmental Selection -> P_{t+1} of exact size N
            population = environmental_selection(combined, self.population_size)
            
            # Compute final Pareto ranks and crowding distances for current population
            final_fronts = fast_non_dominated_sort(population)
            for front in final_fronts:
                calculate_crowding_distance(front)
                
            pareto_front = final_fronts[0]
            gen_time = time.perf_counter() - t_gen_start
            
            # Record generation progress
            self.tracker.record_generation(gen, population, pareto_front, gen_time)
            self._record_population_history(gen, population)
            
            if self.log_callback:
                self.log_callback(gen, population, pareto_front, gen_time)
                
        # Final Pareto Front (Front 0 of final population)
        final_fronts = fast_non_dominated_sort(population)
        for front in final_fronts:
            calculate_crowding_distance(front)
        final_pareto_front = final_fronts[0]
        
        return population, final_pareto_front

    def _record_population_history(self, generation: int, population: List[Individual]):
        """Store snapshots of all population members across generations."""
        for ind in population:
            d = ind.to_dict()
            d["generation_recorded"] = generation
            self.population_history.append(d)
