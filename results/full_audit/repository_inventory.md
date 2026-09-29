# Repository Inventory & Module Audit

**Total Items Scanned:** 105

| File Path | Directory | Status | Size (Bytes) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| `src/data_pipeline.py` | `src` | **IMPLEMENTED** | 22545 | Core functional component |
| `src/features.py` | `src` | **IMPLEMENTED** | 4988 | Core functional component |
| `src/loss.py` | `src` | **IMPLEMENTED** | 1950 | Core functional component |
| `src/model.py` | `src` | **IMPLEMENTED** | 3450 | Core functional component |
| `src/utils.py` | `src` | **IMPLEMENTED** | 4008 | Core functional component |
| `src/adaptation/calibration.py` | `src` | **IMPLEMENTED** | 9750 | Core functional component |
| `src/adaptation/evaluate_recovery.py` | `src` | **IMPLEMENTED** | 14068 | Core functional component |
| `src/adaptation/explainability.py` | `src` | **IMPLEMENTED** | 5889 | Core functional component |
| `src/adaptation/reoptimizer.py` | `src` | **IMPLEMENTED** | 18403 | Core functional component |
| `src/adaptation/visualize_adaptation.py` | `src` | **IMPLEMENTED** | 14697 | Core functional component |
| `src/adaptation/__init__.py` | `src` | **IMPLEMENTED** | 770 | Core functional component |
| `src/drift/detector.py` | `src` | **IMPLEMENTED** | 12933 | Core functional component |
| `src/drift/evaluate_ood.py` | `src` | **IMPLEMENTED** | 15311 | Core functional component |
| `src/drift/scenarios.py` | `src` | **IMPLEMENTED** | 16289 | Core functional component |
| `src/drift/visualize_drift.py` | `src` | **IMPLEMENTED** | 16627 | Core functional component |
| `src/drift/__init__.py` | `src` | **IMPLEMENTED** | 762 | Core functional component |
| `src/nsga2/convergence.py` | `src` | **IMPLEMENTED** | 6589 | Core functional component |
| `src/nsga2/crossover.py` | `src` | **IMPLEMENTED** | 4071 | Core functional component |
| `src/nsga2/evaluation.py` | `src` | **IMPLEMENTED** | 9821 | Core functional component |
| `src/nsga2/evolution.py` | `src` | **IMPLEMENTED** | 7797 | Core functional component |
| `src/nsga2/genome.py` | `src` | **IMPLEMENTED** | 7395 | Core functional component |
| `src/nsga2/mutation.py` | `src` | **IMPLEMENTED** | 3039 | Core functional component |
| `src/nsga2/objectives.py` | `src` | **IMPLEMENTED** | 4880 | Core functional component |
| `src/nsga2/population.py` | `src` | **IMPLEMENTED** | 3230 | Core functional component |
| `src/nsga2/selection.py` | `src` | **IMPLEMENTED** | 1661 | Core functional component |
| `src/nsga2/sorting.py` | `src` | **IMPLEMENTED** | 3262 | Core functional component |
| `src/nsga2/__init__.py` | `src` | **IMPLEMENTED** | 1312 | Core functional component |
| `tests/audit_full_phase1_4.py` | `tests` | **IMPLEMENTED** | 50307 | Core functional component |
| `tests/audit_phase2.py` | `tests` | **IMPLEMENTED** | 50659 | Core functional component |
| `tests/test_nsga2.py` | `tests` | **IMPLEMENTED** | 13227 | Core functional component |
| `tests/test_phase3.py` | `tests` | **IMPLEMENTED** | 15219 | Core functional component |
| `tests/test_phase4.py` | `tests` | **IMPLEMENTED** | 15693 | Core functional component |
| `data/processed/test.csv` | `data` | **IMPLEMENTED** | 25151 | Structured artifact / benchmark data |
| `data/processed/train.csv` | `data` | **IMPLEMENTED** | 74306 | Structured artifact / benchmark data |
| `data/processed/val.csv` | `data` | **IMPLEMENTED** | 25107 | Structured artifact / benchmark data |
| `data/raw/cse_batch500_semesters_1_2_3_summary.csv` | `data` | **IMPLEMENTED** | 1016856 | Structured artifact / benchmark data |
| `models/baseline_model.pt` | `models` | **IMPLEMENTED** | 18601 | Trained model checkpoint / serialized pipeline |
| `models/feature_pipeline.pkl` | `models` | **IMPLEMENTED** | 2021 | Trained model checkpoint / serialized pipeline |
| `models/adaptation/adapted_model.pt` | `models` | **IMPLEMENTED** | 8861 | Trained model checkpoint / serialized pipeline |
| `models/adaptation/calibrator.pkl` | `models` | **IMPLEMENTED** | 350 | Trained model checkpoint / serialized pipeline |
| `models/nsga2/nsga2_selected_model.pt` | `models` | **IMPLEMENTED** | 10289 | Trained model checkpoint / serialized pipeline |
| `results/baseline_metrics.json` | `results` | **IMPLEMENTED** | 941 | Structured artifact / benchmark data |
| `results/data_quality_report.json` | `results` | **IMPLEMENTED** | 767 | Structured artifact / benchmark data |
| `results/training_history.csv` | `results` | **IMPLEMENTED** | 5068 | Structured artifact / benchmark data |
| `results/adaptation/adapted_pareto_front.csv` | `results` | **IMPLEMENTED** | 2900 | Structured artifact / benchmark data |
| `results/adaptation/conformal_coverage_intervals.png` | `results` | **IMPLEMENTED** | 272172 | Generated visual chart / diagnostic figure |
| `results/adaptation/explainability_shift.png` | `results` | **IMPLEMENTED** | 376363 | Generated visual chart / diagnostic figure |
| `results/adaptation/pareto_adaptation_shift.png` | `results` | **IMPLEMENTED** | 265724 | Generated visual chart / diagnostic figure |
| `results/adaptation/recovery_benchmark.csv` | `results` | **IMPLEMENTED** | 718 | Structured artifact / benchmark data |
| `results/adaptation/recovery_metrics.json` | `results` | **IMPLEMENTED** | 2620 | Structured artifact / benchmark data |
| `results/adaptation/recovery_report.md` | `results` | **IMPLEMENTED** | 3063 | Documentation / summary report |
| `results/adaptation/recovery_waterfall.png` | `results` | **IMPLEMENTED** | 295473 | Generated visual chart / diagnostic figure |
| `results/adaptation/reoptimization_summary.json` | `results` | **IMPLEMENTED** | 718 | Structured artifact / benchmark data |
| `results/drift/drift_metrics.json` | `results` | **IMPLEMENTED** | 7333 | Structured artifact / benchmark data |
| `results/drift/drift_pvalues_heatmap.png` | `results` | **IMPLEMENTED** | 486228 | Generated visual chart / diagnostic figure |
| `results/drift/drift_vs_degradation.png` | `results` | **IMPLEMENTED** | 609705 | Generated visual chart / diagnostic figure |
| `results/drift/feature_distribution_shift.png` | `results` | **IMPLEMENTED** | 867388 | Generated visual chart / diagnostic figure |
| `results/drift/feature_shift_report.json` | `results` | **IMPLEMENTED** | 66171 | Structured artifact / benchmark data |
| `results/drift/ood_results.csv` | `results` | **IMPLEMENTED** | 2311 | Structured artifact / benchmark data |
| `results/drift/robustness_report.md` | `results` | **IMPLEMENTED** | 6540 | Documentation / summary report |
| `results/drift/datasets/scenario_a_attendance_drift_10pct.csv` | `results` | **IMPLEMENTED** | 29906 | Structured artifact / benchmark data |
| `results/drift/datasets/scenario_a_attendance_drift_15pct.csv` | `results` | **IMPLEMENTED** | 30721 | Structured artifact / benchmark data |
| `results/drift/datasets/scenario_a_attendance_drift_20pct.csv` | `results` | **IMPLEMENTED** | 31614 | Structured artifact / benchmark data |
| `results/drift/datasets/scenario_a_attendance_drift_5pct.csv` | `results` | **IMPLEMENTED** | 30178 | Structured artifact / benchmark data |
| `results/drift/datasets/scenario_b_academic_drift_shift_0.5.csv` | `results` | **IMPLEMENTED** | 26936 | Structured artifact / benchmark data |
| `results/drift/datasets/scenario_b_academic_drift_shift_1.0.csv` | `results` | **IMPLEMENTED** | 28002 | Structured artifact / benchmark data |
| `results/drift/datasets/scenario_b_academic_drift_shift_1.5.csv` | `results` | **IMPLEMENTED** | 28190 | Structured artifact / benchmark data |
| `results/drift/datasets/scenario_c_backlog_surge_plus_1.csv` | `results` | **IMPLEMENTED** | 25630 | Structured artifact / benchmark data |
| `results/drift/datasets/scenario_c_backlog_surge_plus_2.csv` | `results` | **IMPLEMENTED** | 25438 | Structured artifact / benchmark data |
| `results/drift/datasets/scenario_c_backlog_surge_plus_3.csv` | `results` | **IMPLEMENTED** | 25427 | Structured artifact / benchmark data |
| `results/drift/datasets/scenario_d_cohort_shift.csv` | `results` | **IMPLEMENTED** | 47808 | Structured artifact / benchmark data |
| `results/drift/datasets/scenario_e_compound_stress.csv` | `results` | **IMPLEMENTED** | 34394 | Structured artifact / benchmark data |
| `results/nsga2/artifact_consistency_report.json` | `results` | **IMPLEMENTED** | 671 | Structured artifact / benchmark data |
| `results/nsga2/convergence.csv` | `results` | **IMPLEMENTED** | 773 | Structured artifact / benchmark data |
| `results/nsga2/determinism_audit.json` | `results` | **IMPLEMENTED** | 1178 | Structured artifact / benchmark data |
| `results/nsga2/determinism_report.json` | `results` | **IMPLEMENTED** | 748 | Structured artifact / benchmark data |
| `results/nsga2/final_candidates.csv` | `results` | **IMPLEMENTED** | 601 | Structured artifact / benchmark data |
| `results/nsga2/generations.csv` | `results` | **IMPLEMENTED** | 773 | Structured artifact / benchmark data |
| `results/nsga2/genome_audit.csv` | `results` | **IMPLEMENTED** | 8986 | Structured artifact / benchmark data |
| `results/nsga2/nsga2.log` | `results` | **IMPLEMENTED** | 2940 | Core functional component |
| `results/nsga2/nsga2_convergence.png` | `results` | **IMPLEMENTED** | 12031 | Generated visual chart / diagnostic figure |
| `results/nsga2/objective_recalculation.csv` | `results` | **IMPLEMENTED** | 2954 | Structured artifact / benchmark data |
| `results/nsga2/objective_summary.json` | `results` | **IMPLEMENTED** | 785 | Structured artifact / benchmark data |
| `results/nsga2/pareto_front.csv` | `results` | **IMPLEMENTED** | 2939 | Structured artifact / benchmark data |
| `results/nsga2/pareto_validation.csv` | `results` | **IMPLEMENTED** | 6255 | Structured artifact / benchmark data |
| `results/nsga2/phase2_independent_audit.json` | `results` | **IMPLEMENTED** | 15486 | Structured artifact / benchmark data |
| `results/nsga2/phase2_independent_audit.md` | `results` | **IMPLEMENTED** | 8125 | Documentation / summary report |
| `results/nsga2/PHASE2_REPORT.md` | `results` | **IMPLEMENTED** | 5014 | Documentation / summary report |
| `results/nsga2/population_history.csv` | `results` | **IMPLEMENTED** | 119430 | Structured artifact / benchmark data |
| `results/nsga2/selected_model.json` | `results` | **IMPLEMENTED** | 1042 | Structured artifact / benchmark data |
| `results/nsga2/test_evaluation.json` | `results` | **IMPLEMENTED** | 660 | Structured artifact / benchmark data |
| `results/tests/baseline_comparison.csv` | `results` | **IMPLEMENTED** | 232 | Structured artifact / benchmark data |
| `results/tests/cohort_split_report.json` | `results` | **IMPLEMENTED** | 446 | Structured artifact / benchmark data |
| `results/tests/feature_ablation.csv` | `results` | **IMPLEMENTED** | 317 | Structured artifact / benchmark data |
| `results/tests/loss_curve.png` | `results` | **IMPLEMENTED** | 10857 | Generated visual chart / diagnostic figure |
| `results/tests/test_report.json` | `results` | **IMPLEMENTED** | 9705 | Structured artifact / benchmark data |
| `results/tests/test_report.md` | `results` | **IMPLEMENTED** | 10348 | Documentation / summary report |
| `config.yaml` | `root` | **IMPLEMENTED** | 1452 | Root CLI entry point / configuration |
| `train.py` | `root` | **IMPLEMENTED** | 11835 | Root CLI entry point / configuration |
| `evaluate.py` | `root` | **IMPLEMENTED** | 6171 | Root CLI entry point / configuration |
| `optimize_nsga2.py` | `root` | **IMPLEMENTED** | 23516 | Root CLI entry point / configuration |
| `test_phase1.py` | `root` | **IMPLEMENTED** | 42868 | Root CLI entry point / configuration |
| `requirements.txt` | `root` | **IMPLEMENTED** | 73 | Root CLI entry point / configuration |
| `README.md` | `root` | **IMPLEMENTED** | 2415 | Root CLI entry point / configuration |
| `run_pipeline.py` | `root` | **MISSING** | 0 | Centralized end-to-end pipeline orchestrator script missing; individual CLI scripts used instead. |
