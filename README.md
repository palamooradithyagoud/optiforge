# OptiForge

**Adaptive Multi-Objective Student Performance Prediction Under Non-Stationary Distribution Drift**

OptiForge is a machine learning system for predicting next-semester student performance (SGPA) under curriculum distribution drift and temporal shifts.

---

## Phase 1 Architecture & Status

- **Status:** Phase 1 Complete & Validated (15/15 QA test cases passing).
- **Dataset:** 1,500 longitudinal student semester records (500 unique students across Semesters 1, 2, 3).
- **Cohort Split:** 300 Train / 100 Validation / 100 Test students with zero student overlap.
- **Model:** Regularized Feed-Forward Neural Network with Huber loss, dropout, L1/L2 penalties, and gradient clipping.
- **Interfaces:** Modular `build_model(config)` and `evaluate_model(model, X, y)` designed for Phase 2 NSGA-II hyperparameter optimization.

---

## Project Structure

```
├── config.yaml               # Pipeline & model hyperparameter configurations
├── requirements.txt          # Python dependencies
├── train.py                  # Training pipeline script
├── evaluate.py               # Model evaluation interface
├── test_phase1.py            # Comprehensive QA validation test suite (TC-01 to TC-15)
├── src/
│   ├── data_pipeline.py      # Data validation, subject parsing & temporal splitting
│   ├── features.py           # Feature engineering & scaling pipelines
│   ├── model.py              # BaselineRegressor & build_model()
│   ├── loss.py               # Regularized Huber loss
│   └── utils.py              # Seeding, metrics & benchmarking utilities
├── data/
│   ├── raw/                  # Raw dataset
│   └── processed/            # Processed train/val/test splits
├── models/
│   ├── baseline_model.pt     # Trained baseline model checkpoint
│   └── feature_pipeline.pkl  # Fitted feature transformer
└── results/
    ├── data_quality_report.json
    ├── baseline_metrics.json
    ├── training_history.csv
    └── tests/                # QA reports, comparisons, and loss curve plots
```

---

## Quickstart

### Installation

```bash
pip install -r requirements.txt
```

### Train Baseline Model

```bash
python train.py
```

### Evaluate Saved Model

```bash
python evaluate.py
```

### Run Full QA Suite

```bash
python test_phase1.py
```
