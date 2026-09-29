# PHASE 2 REPORT: Multi-Objective Evolutionary Hyperparameter Optimization (NSGA-II)

**Project:** Adaptive Multi-Objective Student Performance Prediction Under Non-Stationary Distribution Drift  
**Stage:** Phase 2 (NSGA-II Implementation & Pareto Trade-Off Discovery)  

---

> **What changed and why:**  

> NSGA-II was introduced to evolve neural-network hyperparameters across multiple competing objectives instead of optimizing predictive error alone.

---

## 1. Problem Formulation & Multi-Objective Motivation

Single-objective hyperparameter optimization (minimizing predictive error alone) produces models that may overfit, require excessive parameter budgets, or exhibit unstable training dynamics under curriculum distribution drift. In high-stakes educational deployment, an ML system must explicitly balance predictive accuracy with computational efficiency, generalization stability, and inference speed. NSGA-II discovers a non-dominated Pareto front of architectural compromises without requiring arbitrary scalarization weights.

## 2. Genome Representation & Search Space

Each individual encodes a 10-gene hyperparameter chromosome:
* `hidden_dims`: Architecture topology `[32]`, `[64]`, `[128]`, `[32, 16]`, `[64, 32]`, `[128, 64]`, `[128, 64, 32]`
* `dropout_rate`: Continuous $\in [0.0, 0.5]$
* `activation`: Categorical $\in$ `{relu, gelu, tanh}`
* `learning_rate`: Log-uniform continuous $\in [10^{-4}, 10^{-2}]$
* `weight_decay`: Log-uniform continuous $\in [10^{-6}, 10^{-2}]$
* `l1_lambda`: Continuous $\in [0.0, 10^{-3}]$
* `huber_delta`: Continuous $\in [0.5, 2.0]$
* `batch_size`: Discrete $\in$ `{16, 32, 64}`
* `gradient_clip_norm`: Discrete $\in$ `{0.5, 1.0, 2.0}`
* `epochs`: Bounded discrete $\in [30, 150]$

## 3. The 6 Minimization Objectives

1. **$f_1$ (Predictive Error):** Validation Mean Absolute Error ($	ext{MAE}_{	ext{val}}$).
2. **$f_2$ (Generalization / Overfitting):** Absolute generalization gap $|L_{	ext{val}} - L_{	ext{train}}|$.
3. **$f_3$ (Loss Stability):** Variance of validation loss across training epochs $	ext{Var}(L_{	ext{val}})$.
4. **$f_4$ (Parameter Efficiency):** Total count of trainable weights and biases.
5. **$f_5$ (Inference Latency):** Single-sample inference latency benchmarked on CPU (ms).
6. **$f_6$ (Computational Cost):** Total wall-clock model training time (seconds).

## 4. NSGA-II Algorithmic Operators

* **Non-Dominated Sorting:** Standard fast sorting algorithm partitioning populations into successive Pareto fronts $F_0, F_1, \dots$
* **Crowding Distance:** Density estimation in normalized objective space to prioritize diverse solutions.
* **Tournament Selection:** Binary tournament using crowded comparison operator (rank priority, crowding tie-breaker).
* **Crossover:** Discrete uniform exchange for architectures/activations; Simulated Binary Crossover (SBX, $\eta=2$) for continuous hyperparameters.
* **Mutation:** Point mutations with Gaussian perturbations in linear and log domains with automatic genome repair.
* **Elitist Environmental Selection:** Combines parent and offspring pools ($2N$) and preserves the top $N$ individuals across non-dominated fronts.

## 5. Strict Test Isolation & Caching

* **Zero Test Leakage:** The 100-student held-out test cohort was strictly excluded from `GenomeEvaluator`. No test evaluations occurred during any generation of the evolutionary run.
* **Exact Canonical Caching:** Genomes are mapped to canonical JSON and SHA-256 hashed. Re-sampled or surviving identical configurations are retrieved instantly, saving substantial compute.

## 6. Evolutionary Optimization Results

* **Population Size:** 16 | **Generations:** 10
* **Total Evaluations Attempted:** 173
* **Cached Evaluations Count:** 3
* **Failed Evaluations Count:** 0
* **Pareto Front Size:** 16 non-dominated configurations discovered
* **Total Optimization Runtime:** 296.41 seconds

### Pareto Candidates Discovered

| Candidate Role | Architecture | Activation | Params | Val MAE | Gen Gap | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **candidate_a_best_mae** | `[128, 64, 32]` | `gelu` | 13313 | 0.7315 | 0.1247 | 0.3328 |
| **candidate_b_balanced_compromise** | `[32, 16]` | `tanh` | 1281 | 0.7494 | 0.0097 | 0.0320 |
| **candidate_c_parameter_efficient** | `[32]` | `tanh` | 769 | 0.7643 | 0.1449 | 0.0192 |

## 7. Final Held-Out Test Evaluation

The selected candidate (Candidate B: Balanced Compromise) was retrained from scratch on the training cohort and evaluated once on the untouched held-out test set:

| Model System | Test MAE | Test RMSE | Test $R^2$ | Trainable Params | Single Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Mean Baseline** | 0.8159 | 1.0459 | -0.0001 | 0 | — |
| **Phase 1 Deep Baseline** | 0.8773 | 1.0851 | -0.0765 | 3,585 | 0.0494 ms |
| **NSGA-II Selected Model** | **0.8359** | **1.0692** | **-0.0453** | **1281** | **0.0498 ms** |
