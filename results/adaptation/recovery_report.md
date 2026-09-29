# PHASE 4 REPORT: Empirical Drift Recovery & Calibration (Held-Out Evaluation)

**Methodology:** Strictly Disjoint Student Cohorts (50% Adaptation, 50% Held-Out Recovery)

## Quantitative Progression Benchmark (Scenario E Compound Stress)

| stage_id | pipeline_stage | evaluation_cohort | mae | rmse | r2 | mean_bias | delta_mae_vs_clean | conformal_coverage_90 | mean_interval_width | single_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1_clean_baseline | 1. Clean Baseline (Phase 2 Model on Clean Dev) | Clean Development Baseline (val.csv) | 0.748 | 1.007 | -0.022 | 0.0934 | 0.0 | N/A | N/A | 0.101 |
| 2_drift_degradation | 2. Drift Degradation (Phase 2 Model on Recovery) | Held-Out Recovery Cohort (Unseen Drift) | 0.7706 | 1.1078 | -0.0487 | 0.2501 | 0.0226 | N/A | N/A | 0.0752 |
| 3_reoptimized_recovery | 3. Re-Optimized Recovery (Adapted Model on Recovery) | Held-Out Recovery Cohort (Unseen Drift) | 0.8876 | 1.1376 | -0.1058 | -0.2379 | 0.1396 | N/A | N/A | 0.0514 |
| 4_calibrated_output | 4. Calibrated Output (Adapted + Conformal on Recovery) | Held-Out Recovery Cohort (Calibrated) | 0.8102 | 1.0971 | -0.0286 | 0.0171 | 0.0622 | 87.0% | 3.3249 | 0.0514 |

## Multi-Scenario Recovery Matrix Across All 12 OOD Regimes

| scenario_id | scenario_name | frozen_mae | adapted_mae | calibrated_mae | frozen_rmse | adapted_rmse | calibrated_rmse | recovery_pct | conformal_coverage | mean_interval_width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scenario_a_attendance_drift_5pct | Attendance Drift -5% | 0.7913 | 6.5382 | 4.8683 | 1.101 | 7.0613 | 5.316 | -13272.3% | 12.0% | 3.1246 |
| scenario_a_attendance_drift_10pct | Attendance Drift -10% | 0.7953 | 5.5657 | 4.1296 | 1.1027 | 6.0467 | 4.5418 | -10085.4% | 15.0% | 3.3044 |
| scenario_a_attendance_drift_15pct | Attendance Drift -15% | 0.8043 | 4.7333 | 3.497 | 1.1067 | 5.1594 | 3.8672 | -6978.7% | 18.0% | 3.3249 |
| scenario_a_attendance_drift_20pct | Attendance Drift -20% | 0.828 | 4.0641 | 2.9891 | 1.1199 | 4.4421 | 3.325 | -4045.1% | 23.0% | 3.3249 |
| scenario_b_academic_drift_shift_0.5 | Academic Drift -0.5 SGPA | 0.7948 | 7.2628 | 5.3953 | 1.1027 | 7.7911 | 5.8526 | -13820.5% | 10.0% | 2.8666 |
| scenario_b_academic_drift_shift_1.0 | Academic Drift -1.0 SGPA | 0.8019 | 6.9606 | 5.1767 | 1.1072 | 7.4728 | 5.6249 | -11426.2% | 11.0% | 2.9919 |
| scenario_b_academic_drift_shift_1.5 | Academic Drift -1.5 SGPA | 0.8242 | 6.6872 | 4.9718 | 1.1219 | 7.1799 | 5.4046 | -7694.2% | 11.0% | 3.0978 |
| scenario_c_backlog_surge_plus_1 | Backlog Surge +1 | 0.7723 | 3.6268 | 2.6605 | 1.096 | 4.0447 | 3.0313 | -11746.9% | 34.0% | 3.3249 |
| scenario_c_backlog_surge_plus_2 | Backlog Surge +2 | 0.7725 | 1.7484 | 1.3348 | 1.0898 | 2.051 | 1.5985 | -3983.3% | 67.0% | 3.3249 |
| scenario_c_backlog_surge_plus_3 | Backlog Surge +3 | 0.7689 | 1.1499 | 0.9552 | 1.0915 | 1.3932 | 1.2075 | -1823.0% | 80.0% | 3.3249 |
| scenario_d_cohort_shift | Synthesized Cohort Shift | 0.8127 | 4.8587 | 3.5992 | 1.1448 | 5.4329 | 4.0847 | -6253.5% | 21.0% | 3.2837 |
| scenario_e_compound_stress | Compound Stress (-15% Att, +2 Backlogs, -0.75 SGPA) | 0.7706 | 0.8876 | 0.8102 | 1.1078 | 1.1376 | 1.0971 | -517.7% | 87.0% | 3.3249 |

## Sub-Group Conformal Calibration Analysis

| scenario_id | subgroup | sample_count | coverage | target_coverage | coverage_error | mean_interval_width |
| --- | --- | --- | --- | --- | --- | --- |
| scenario_a_attendance_drift_5pct | high_risk | 26 | 46.2% | 90.0% | 0.4385 | 3.3197 |
| scenario_a_attendance_drift_5pct | standard | 74 | 0.0% | 90.0% | 0.9 | 3.056 |
| scenario_a_attendance_drift_10pct | high_risk | 34 | 44.1% | 90.0% | 0.4588 | 3.3249 |
| scenario_a_attendance_drift_10pct | standard | 66 | 0.0% | 90.0% | 0.9 | 3.2938 |
| scenario_a_attendance_drift_15pct | high_risk | 73 | 23.3% | 90.0% | 0.6671 | 3.3249 |
| scenario_a_attendance_drift_15pct | standard | 27 | 3.7% | 90.0% | 0.863 | 3.3249 |
| scenario_a_attendance_drift_20pct | high_risk | 99 | 23.2% | 90.0% | 0.6677 | 3.3249 |
| scenario_a_attendance_drift_20pct | standard | 1 | 0.0% | 90.0% | 0.9 | 3.3249 |
| scenario_b_academic_drift_shift_0.5 | high_risk | 26 | 38.5% | 90.0% | 0.5154 | 3.294 |
| scenario_b_academic_drift_shift_0.5 | standard | 74 | 0.0% | 90.0% | 0.9 | 2.7164 |
