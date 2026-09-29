"""
tests/test_nsga2.py
Comprehensive unit test suite for NSGA-II implementation covering all 15 required items:
1. genome validation
2. genome repair
3. deterministic initialization
4. dominance relation
5. non-dominated sorting
6. crowding distance
7. tournament selection
8. crossover validity
9. mutation validity
10. environmental selection
11. population-size preservation
12. deterministic evolution
13. no test-set usage during optimization
14. objective calculation
15. Pareto-front extraction
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import unittest
import numpy as np
import pandas as pd
import torch

from src.nsga2.genome import (
    Genome, validate_genome, repair_genome, sample_random_genome,
    SUPPORTED_ARCHITECTURES, SUPPORTED_ACTIVATIONS, SUPPORTED_BATCH_SIZES, SUPPORTED_CLIP_NORMS
)
from src.nsga2.objectives import dominates, ObjectiveMetrics, PENALTY_OBJECTIVES
from src.nsga2.population import Individual, initialize_population
from src.nsga2.sorting import fast_non_dominated_sort, calculate_crowding_distance
from src.nsga2.selection import binary_tournament, tournament_selection
from src.nsga2.crossover import crossover
from src.nsga2.mutation import mutate
from src.nsga2.evolution import environmental_selection, NSGA2Optimizer
from src.nsga2.evaluation import GenomeEvaluator
from src.features import FeaturePipeline
from src.utils import save_json


class TestNSGA2(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.valid_genome = Genome(
            hidden_dims=[64, 32],
            dropout_rate=0.2,
            activation="relu",
            learning_rate=0.001,
            weight_decay=0.0001,
            l1_lambda=0.00005,
            huber_delta=1.0,
            batch_size=32,
            gradient_clip_norm=1.0,
            epochs=50
        )

    # 1. Genome Validation
    def test_01_genome_validation(self):
        self.assertTrue(validate_genome(self.valid_genome))
        
        # Invalid dropout
        inv_drop = Genome(**{**self.valid_genome.to_dict(), "dropout_rate": 0.8})
        self.assertFalse(validate_genome(inv_drop))
        
        # Invalid learning rate
        inv_lr = Genome(**{**self.valid_genome.to_dict(), "learning_rate": -0.01})
        self.assertFalse(validate_genome(inv_lr))
        
        # Invalid activation
        inv_act = Genome(**{**self.valid_genome.to_dict(), "activation": "sigmoid"})
        self.assertFalse(validate_genome(inv_act))

    # 2. Genome Repair
    def test_02_genome_repair(self):
        illegal = Genome(
            hidden_dims=[500, 200],  # unsupported
            dropout_rate=0.9,        # out of range
            activation="invalid_act",# invalid
            learning_rate=0.5,       # too high
            weight_decay=-1e-4,      # negative
            l1_lambda=0.05,          # too high
            huber_delta=10.0,        # too high
            batch_size=128,          # unsupported
            gradient_clip_norm=10.0, # unsupported
            epochs=300               # too high
        )
        repaired = repair_genome(illegal)
        self.assertTrue(validate_genome(repaired))
        self.assertIn(repaired.hidden_dims, SUPPORTED_ARCHITECTURES)
        self.assertIn(repaired.activation, SUPPORTED_ACTIVATIONS)
        self.assertIn(repaired.batch_size, SUPPORTED_BATCH_SIZES)
        self.assertIn(repaired.gradient_clip_norm, SUPPORTED_CLIP_NORMS)
        self.assertLessEqual(repaired.dropout_rate, 0.5)

    # 3. Deterministic Initialization
    def test_03_deterministic_initialization(self):
        rng1 = np.random.default_rng(42)
        pop1 = initialize_population(10, rng1, include_phase1_baseline=True)
        
        rng2 = np.random.default_rng(42)
        pop2 = initialize_population(10, rng2, include_phase1_baseline=True)
        
        for ind1, ind2 in zip(pop1, pop2):
            self.assertEqual(ind1.genome.canonical_hash(), ind2.genome.canonical_hash())

    # 4. Dominance Relation
    def test_04_dominance_relation(self):
        # A dominates B if A is <= B on all, and < B on at least one
        sol_a = [0.8, 0.1, 0.01, 1000, 0.05, 5.0]
        sol_b = [0.9, 0.2, 0.02, 2000, 0.08, 8.0]
        self.assertTrue(dominates(sol_a, sol_b))
        self.assertFalse(dominates(sol_b, sol_a))
        
        # Trade-off: mutually non-dominating
        sol_c = [0.75, 0.3, 0.01, 3000, 0.04, 6.0]
        self.assertFalse(dominates(sol_a, sol_c))
        self.assertFalse(dominates(sol_c, sol_a))
        
        # Identical points do not dominate each other
        self.assertFalse(dominates(sol_a, sol_a))

    # 5. Non-Dominated Sorting
    def test_05_non_dominated_sorting(self):
        inds = []
        # Create 3 individuals: ind1 dominates ind2, ind2 dominates ind3
        for i, obj in enumerate([[0.8, 0.1], [0.9, 0.2], [1.0, 0.3]]):
            ind = Individual(self.valid_genome, individual_id=f"test_{i}")
            ind.objectives = obj
            inds.append(ind)
            
        fronts = fast_non_dominated_sort(inds)
        self.assertEqual(len(fronts), 3)
        self.assertEqual(fronts[0][0].id, "test_0")
        self.assertEqual(fronts[1][0].id, "test_1")
        self.assertEqual(fronts[2][0].id, "test_2")
        self.assertEqual(inds[0].rank, 0)
        self.assertEqual(inds[1].rank, 1)
        self.assertEqual(inds[2].rank, 2)

    # 6. Crowding Distance Calculation
    def test_06_crowding_distance(self):
        front = []
        # Points along Pareto trade-off
        for i, obj in enumerate([[0.5, 2.0], [0.6, 1.5], [0.7, 1.2], [0.9, 1.0]]):
            ind = Individual(self.valid_genome, individual_id=f"front_{i}")
            ind.objectives = obj
            ind.rank = 0
            front.append(ind)
            
        calculate_crowding_distance(front)
        
        # Boundary individuals must have infinite crowding distance
        self.assertEqual(front[0].crowding_distance, float("inf"))
        self.assertEqual(front[-1].crowding_distance, float("inf"))
        
        # Interior individuals must have positive finite crowding distance
        self.assertTrue(0.0 < front[1].crowding_distance < float("inf"))
        self.assertTrue(0.0 < front[2].crowding_distance < float("inf"))

    # 7. Tournament Selection
    def test_07_tournament_selection(self):
        ind_rank0 = Individual(self.valid_genome, individual_id="better_rank")
        ind_rank0.rank = 0
        ind_rank0.crowding_distance = 1.0
        
        ind_rank1 = Individual(self.valid_genome, individual_id="worse_rank")
        ind_rank1.rank = 1
        ind_rank1.crowding_distance = 5.0
        
        winner = binary_tournament(ind_rank0, ind_rank1, self.rng)
        self.assertEqual(winner.id, "better_rank")
        
        # Same rank: higher crowding distance wins
        ind_rank0_diverse = Individual(self.valid_genome, individual_id="more_diverse")
        ind_rank0_diverse.rank = 0
        ind_rank0_diverse.crowding_distance = 10.0
        
        winner2 = binary_tournament(ind_rank0, ind_rank0_diverse, self.rng)
        self.assertEqual(winner2.id, "more_diverse")

    # 8. Crossover Validity
    def test_08_crossover_validity(self):
        parent1 = Genome(
            hidden_dims=[128, 64],
            dropout_rate=0.1,
            activation="relu",
            learning_rate=0.005,
            weight_decay=1e-3,
            l1_lambda=1e-4,
            huber_delta=1.5,
            batch_size=16,
            gradient_clip_norm=0.5,
            epochs=40
        )
        parent2 = Genome(
            hidden_dims=[32],
            dropout_rate=0.4,
            activation="gelu",
            learning_rate=0.0005,
            weight_decay=1e-5,
            l1_lambda=5e-5,
            huber_delta=0.8,
            batch_size=64,
            gradient_clip_norm=2.0,
            epochs=90
        )
        for _ in range(10):
            c1, c2 = crossover(parent1, parent2, self.rng)
            self.assertTrue(validate_genome(c1))
            self.assertTrue(validate_genome(c2))

    # 9. Mutation Validity
    def test_09_mutation_validity(self):
        for _ in range(15):
            mutated = mutate(self.valid_genome, mutation_rate=0.3, rng=self.rng)
            self.assertTrue(validate_genome(mutated))

    # 10. Environmental Selection
    def test_10_environmental_selection(self):
        pool = []
        for i in range(20):
            ind = Individual(self.valid_genome, individual_id=f"ind_{i}")
            # Diverse objective values
            ind.objectives = [float(i), float(20 - i)]
            pool.append(ind)
            
        selected = environmental_selection(pool, pop_size=10)
        self.assertEqual(len(selected), 10)

    # 11. Population-Size Preservation
    def test_11_population_size_preservation(self):
        pop_size = 12
        combined_pool = []
        for i in range(pop_size * 2):
            ind = Individual(self.valid_genome, individual_id=f"cand_{i}")
            ind.objectives = [float(i % 5), float((i * 3) % 7)]
            combined_pool.append(ind)
            
        next_pop = environmental_selection(combined_pool, pop_size=pop_size)
        self.assertEqual(len(next_pop), pop_size)

    # 12. Deterministic Evolution
    def test_12_deterministic_evolution(self):
        train_df = pd.read_csv("data/processed/train.csv")
        val_df = pd.read_csv("data/processed/val.csv")
        pipe = FeaturePipeline.load("models/feature_pipeline.pkl")
        
        def run_mini_opt(seed=42):
            evaluator = GenomeEvaluator(train_df, val_df, pipe, early_stopping_patience=5)
            opt = NSGA2Optimizer(evaluator, population_size=4, generations=2, seed=seed)
            pop, pf = opt.run()
            return [ind.genome.canonical_hash() for ind in pf]
            
        pf_hashes_1 = run_mini_opt(seed=42)
        pf_hashes_2 = run_mini_opt(seed=42)
        self.assertEqual(pf_hashes_1, pf_hashes_2)
        
        # Save determinism test result
        save_json(
            {
                "seed": 42,
                "run1_pareto_front_hashes": pf_hashes_1,
                "run2_pareto_front_hashes": pf_hashes_2,
                "deterministic": (pf_hashes_1 == pf_hashes_2)
            },
            "results/nsga2/determinism_report.json"
        )

    # 13. Test Set Isolation
    def test_13_no_test_set_usage_during_optimization(self):
        train_df = pd.read_csv("data/processed/train.csv")
        val_df = pd.read_csv("data/processed/val.csv")
        pipe = FeaturePipeline.load("models/feature_pipeline.pkl")
        evaluator = GenomeEvaluator(train_df, val_df, pipe)
        
        # Check evaluator attributes: test_df or X_test must NOT exist
        self.assertFalse(hasattr(evaluator, "test_df"))
        self.assertFalse(hasattr(evaluator, "X_test"))
        self.assertFalse(hasattr(evaluator, "y_test"))

    # 14. Objective Calculation
    def test_14_objective_calculation(self):
        train_df = pd.read_csv("data/processed/train.csv")
        val_df = pd.read_csv("data/processed/val.csv")
        pipe = FeaturePipeline.load("models/feature_pipeline.pkl")
        evaluator = GenomeEvaluator(train_df, val_df, pipe, early_stopping_patience=5)
        
        test_g = Genome(
            hidden_dims=[32],
            dropout_rate=0.1,
            activation="relu",
            learning_rate=0.005,
            weight_decay=1e-4,
            l1_lambda=0.0,
            huber_delta=1.0,
            batch_size=32,
            gradient_clip_norm=1.0,
            epochs=5
        )
        metrics, _ = evaluator.evaluate(test_g)
        self.assertEqual(metrics.status, "success")
        self.assertGreater(metrics.validation_mae, 0.0)
        self.assertGreaterEqual(metrics.generalization_gap, 0.0)
        self.assertGreaterEqual(metrics.val_loss_variance, 0.0)
        self.assertGreater(metrics.trainable_parameters, 0)
        self.assertGreater(metrics.inference_latency_ms, 0.0)
        self.assertGreater(metrics.training_time_seconds, 0.0)

    # 15. Pareto Front Mutual Non-Dominance
    def test_15_pareto_front_extraction(self):
        inds = []
        for i in range(10):
            ind = Individual(self.valid_genome, individual_id=f"ind_{i}")
            # Generate points: some dominate others
            ind.objectives = [float(i), float(10 - i), float((i * 2) % 5)]
            inds.append(ind)
            
        fronts = fast_non_dominated_sort(inds)
        pf = fronts[0]
        # Verify that no member of Front 0 dominates another member of Front 0
        for a in pf:
            for b in pf:
                if a is not b:
                    self.assertFalse(dominates(a.objectives, b.objectives), f"{a.id} dominates {b.id} in Front 0!")


def run_unit_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNSGA2)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_unit_tests()
    sys.exit(0 if success else 1)
