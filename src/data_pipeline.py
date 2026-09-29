"""
src/data_pipeline.py
Robust validation pipeline, subjects JSON parser, temporal sample generator,
and temporal train/val/test splitting.
"""

import os
import json
import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validate_raw_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform rigorous validation checks on raw student records.
    
    Checks:
    1. Duplicate student-semester records
    2. Missing values
    3. Invalid semester values
    4. Attendance outside 0-100
    5. SGPA outside valid range (0-10)
    6. CGPA outside valid range (0-10)
    7. Negative credits
    8. Credits earned > credits registered
    9. Negative backlog counts
    10. Invalid failed_subjects values
    11. Invalid/malformed subjects_breakdown_json
    12. Inconsistent semester ordering / missing sequence
    
    Returns:
        cleaned_df: Validated DataFrame
        report: Detailed data-quality report dictionary
    """
    total_raw_rows = len(df)
    unique_students_raw = df["student_id"].nunique()
    
    report: Dict[str, Any] = {
        "total_rows": total_raw_rows,
        "unique_students": unique_students_raw,
        "semesters_available": sorted(df["semester"].unique().tolist()),
        "branches_available": sorted(df["branch"].unique().tolist()),
        "missing_value_counts": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
        "duplicate_count": 0,
        "invalid_value_count": 0,
        "anomalies_detected": [],
        "modifications_logged": [],
        "students_with_multiple_semesters": 0,
    }
    
    # 1. Duplicate student-semester check
    duplicates_mask = df.duplicated(subset=["student_id", "semester"], keep=False)
    dup_count = int(duplicates_mask.sum())
    report["duplicate_count"] = dup_count
    if dup_count > 0:
        dup_details = df[duplicates_mask][["student_id", "semester"]].to_dict(orient="records")
        report["anomalies_detected"].append(f"Found {dup_count} duplicate student-semester records: {dup_details}")
        report["modifications_logged"].append("Deduplicating student-semester records by keeping first occurrence.")
        df = df.drop_duplicates(subset=["student_id", "semester"], keep="first")
    
    # 2. Missing values inspection (excluding failed_subjects which is naturally NaN when backlogs == 0)
    critical_cols = [
        "student_id", "semester", "branch", "overall_attendance_pct",
        "overall_sgpa", "overall_cgpa", "total_sem_credits_registered",
        "total_sem_credits_earned", "active_backlogs_count", "subjects_breakdown_json"
    ]
    for col in critical_cols:
        col_missing = int(df[col].isnull().sum())
        if col_missing > 0:
            report["invalid_value_count"] += col_missing
            report["anomalies_detected"].append(f"Column '{col}' has {col_missing} missing values.")
            
    # 3. Invalid semester values (must be integer between 1 and 8)
    invalid_sem_mask = (df["semester"] < 1) | (df["semester"] > 8) | (df["semester"] != df["semester"].astype(int))
    inv_sem_count = int(invalid_sem_mask.sum())
    if inv_sem_count > 0:
        report["invalid_value_count"] += inv_sem_count
        report["anomalies_detected"].append(f"Found {inv_sem_count} records with invalid semester numbers.")
        
    # 4. Attendance outside 0-100
    inv_att_mask = (df["overall_attendance_pct"] < 0.0) | (df["overall_attendance_pct"] > 100.0)
    inv_att_count = int(inv_att_mask.sum())
    if inv_att_count > 0:
        report["invalid_value_count"] += inv_att_count
        report["anomalies_detected"].append(f"Found {inv_att_count} records with overall_attendance_pct outside [0, 100].")
        
    # 5. SGPA outside 0-10
    inv_sgpa_mask = (df["overall_sgpa"] < 0.0) | (df["overall_sgpa"] > 10.0)
    inv_sgpa_count = int(inv_sgpa_mask.sum())
    if inv_sgpa_count > 0:
        report["invalid_value_count"] += inv_sgpa_count
        report["anomalies_detected"].append(f"Found {inv_sgpa_count} records with overall_sgpa outside [0, 10].")
        
    # 6. CGPA outside 0-10
    inv_cgpa_mask = (df["overall_cgpa"] < 0.0) | (df["overall_cgpa"] > 10.0)
    inv_cgpa_count = int(inv_cgpa_mask.sum())
    if inv_cgpa_count > 0:
        report["invalid_value_count"] += inv_cgpa_count
        report["anomalies_detected"].append(f"Found {inv_cgpa_count} records with overall_cgpa outside [0, 10].")
        
    # 7. Negative credits
    neg_cred_mask = (df["total_sem_credits_registered"] < 0) | (df["total_sem_credits_earned"] < 0)
    neg_cred_count = int(neg_cred_mask.sum())
    if neg_cred_count > 0:
        report["invalid_value_count"] += neg_cred_count
        report["anomalies_detected"].append(f"Found {neg_cred_count} records with negative registered or earned credits.")
        
    # 8. Credits earned > credits registered
    cred_overflow_mask = df["total_sem_credits_earned"] > df["total_sem_credits_registered"]
    cred_overflow_count = int(cred_overflow_mask.sum())
    if cred_overflow_count > 0:
        report["invalid_value_count"] += cred_overflow_count
        report["anomalies_detected"].append(f"Found {cred_overflow_count} records where total_sem_credits_earned > total_sem_credits_registered.")
        
    # 9. Negative backlog counts
    neg_backlog_mask = df["active_backlogs_count"] < 0
    neg_backlog_count = int(neg_backlog_mask.sum())
    if neg_backlog_count > 0:
        report["invalid_value_count"] += neg_backlog_count
        report["anomalies_detected"].append(f"Found {neg_backlog_count} records with negative active_backlogs_count.")
        
    # 10. Invalid failed_subjects values
    # Standardize failed_subjects: fillna with "None" string or empty list
    for idx, row in df.iterrows():
        b_count = row["active_backlogs_count"]
        f_sub = row["failed_subjects"]
        if b_count == 0:
            if pd.notnull(f_sub) and str(f_sub).strip().lower() not in ["", "none", "nan"]:
                report["invalid_value_count"] += 1
                report["anomalies_detected"].append(f"Student {row['student_id']} Sem {row['semester']} has active_backlogs=0 but failed_subjects='{f_sub}'.")
        else:
            if pd.isnull(f_sub) or str(f_sub).strip().lower() in ["", "none", "nan"]:
                report["invalid_value_count"] += 1
                report["anomalies_detected"].append(f"Student {row['student_id']} Sem {row['semester']} has active_backlogs={b_count} but failed_subjects is empty.")
    
    # 11. subjects_breakdown_json validation
    malformed_json_count = 0
    for idx, row in df.iterrows():
        raw_json = row["subjects_breakdown_json"]
        try:
            parsed = json.loads(raw_json)
            if not isinstance(parsed, list):
                malformed_json_count += 1
        except Exception:
            malformed_json_count += 1
            
    if malformed_json_count > 0:
        report["invalid_value_count"] += malformed_json_count
        report["anomalies_detected"].append(f"Found {malformed_json_count} records with malformed subjects_breakdown_json.")
        
    # 12. Students with inconsistent semester ordering
    stud_grouped = df.groupby("student_id")["semester"].apply(list)
    mult_sem_count = 0
    inconsistent_seq_count = 0
    for s_id, sems in stud_grouped.items():
        if len(sems) > 1:
            mult_sem_count += 1
        sorted_sems = sorted(sems)
        # Check for duplicate semesters within student
        if len(sorted_sems) != len(set(sorted_sems)):
            inconsistent_seq_count += 1
            report["anomalies_detected"].append(f"Student {s_id} has duplicated semester numbers: {sems}")
        # Check for gap in semesters starting from min
        expected_seq = list(range(sorted_sems[0], sorted_sems[0] + len(sorted_sems)))
        if sorted_sems != expected_seq:
            inconsistent_seq_count += 1
            report["anomalies_detected"].append(f"Student {s_id} has discontinuous semester sequence: {sems}")
            
    report["students_with_multiple_semesters"] = mult_sem_count
    if inconsistent_seq_count > 0:
        report["invalid_value_count"] += inconsistent_seq_count
        
    logger.info(f"Validation completed. Total rows: {len(df)}, Invalid issues: {report['invalid_value_count']}")
    return df, report


def parse_subject_breakdown(json_str: str) -> Dict[str, float]:
    """
    Parse subjects_breakdown_json string and compute aggregated subject-level metrics.
    Supports arbitrary numbers and types of subjects without hardcoding subject names or counts.
    
    Returns:
        dict containing:
        - avg_subject_grade_point
        - min_subject_grade_point
        - max_subject_grade_point
        - var_subject_grade_point
        - avg_subject_attendance
        - min_subject_attendance
        - var_subject_attendance
        - num_failed_subjects
        - total_subject_credits
    """
    try:
        subjects = json.loads(json_str)
    except Exception as e:
        logger.warning(f"Error parsing subjects JSON: {e}")
        subjects = []
        
    if not subjects or not isinstance(subjects, list):
        return {
            "avg_subject_grade_point": 0.0,
            "min_subject_grade_point": 0.0,
            "max_subject_grade_point": 0.0,
            "var_subject_grade_point": 0.0,
            "avg_subject_attendance": 0.0,
            "min_subject_attendance": 0.0,
            "var_subject_attendance": 0.0,
            "num_failed_subjects": 0.0,
            "total_subject_credits": 0.0
        }
        
    grade_points = []
    attendances = []
    credits_list = []
    failed_count = 0
    
    for sub in subjects:
        # Grade point
        gp = float(sub.get("grade_point", 0.0))
        grade_points.append(gp)
        
        # Check fail condition
        grade_str = str(sub.get("grade", "")).strip().upper()
        if gp == 0.0 or grade_str in ["F", "FAIL"]:
            failed_count += 1
            
        # Attendance
        att = float(sub.get("attendance", 0.0))
        attendances.append(att)
        
        # Credits
        cr = float(sub.get("credits", 0.0))
        credits_list.append(cr)
        
    num_subs = len(subjects)
    avg_gp = float(np.mean(grade_points)) if grade_points else 0.0
    min_gp = float(np.min(grade_points)) if grade_points else 0.0
    max_gp = float(np.max(grade_points)) if grade_points else 0.0
    var_gp = float(np.var(grade_points, ddof=0)) if grade_points else 0.0
    
    avg_att = float(np.mean(attendances)) if attendances else 0.0
    min_att = float(np.min(attendances)) if attendances else 0.0
    var_att = float(np.var(attendances, ddof=0)) if attendances else 0.0
    tot_cred = float(np.sum(credits_list)) if credits_list else 0.0
    
    return {
        "avg_subject_grade_point": round(avg_gp, 4),
        "min_subject_grade_point": round(min_gp, 4),
        "max_subject_grade_point": round(max_gp, 4),
        "var_subject_grade_point": round(var_gp, 4),
        "avg_subject_attendance": round(avg_att, 4),
        "min_subject_attendance": round(min_att, 4),
        "var_subject_attendance": round(var_att, 4),
        "num_failed_subjects": float(failed_count),
        "total_subject_credits": round(tot_cred, 2)
    }


def create_temporal_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct temporal training examples from longitudinal student semester records.
    
    STRICT TEMPORAL RULES:
    - Target: next_semester_sgpa (from future semester t)
    - Features: Derived strictly from historical semesters (< t)
    - Zero temporal leakage: Target semester data is NEVER used in feature calculation.
    - name is completely removed.
    - student_id is preserved solely for cohort grouping/tracking, NOT as a feature.
    
    Transitions formed:
    - History: [Sem 1]       -> Target: Sem 2 SGPA
    - History: [Sem 1, Sem 2] -> Target: Sem 3 SGPA
    
    Returns:
        pd.DataFrame containing engineered temporal feature samples and target column.
    """
    # Sort rigorously by student_id and semester
    df_sorted = df.sort_values(by=["student_id", "semester"]).reset_index(drop=True)
    
    # Pre-parse subject breakdown for all rows
    subject_features = []
    for idx, row in df_sorted.iterrows():
        sub_metrics = parse_subject_breakdown(row["subjects_breakdown_json"])
        subject_features.append(sub_metrics)
    sub_df = pd.DataFrame(subject_features)
    df_enriched = pd.concat([df_sorted, sub_df], axis=1)
    
    samples = []
    
    # Group by student
    for student_id, student_df in df_enriched.groupby("student_id"):
        records = student_df.to_dict(orient="records")
        n_sem = len(records)
        if n_sem < 2:
            continue
            
        for t in range(1, n_sem):
            target_record = records[t]
            history = records[:t]
            latest = history[-1]
            
            # Target
            target_sgpa = float(target_record["overall_sgpa"])
            target_semester = int(target_record["semester"])
            
            # Historical features (from latest semester t-1)
            prev_sgpa = float(latest["overall_sgpa"])
            prev_cgpa = float(latest["overall_cgpa"])
            prev_attendance = float(latest["overall_attendance_pct"])
            
            # Differential features (t-1 vs t-2) if available
            if len(history) >= 2:
                prev_prev = history[-2]
                sgpa_change = round(prev_sgpa - float(prev_prev["overall_sgpa"]), 4)
                cgpa_change = round(prev_cgpa - float(prev_prev["overall_cgpa"]), 4)
                attendance_change = round(prev_attendance - float(prev_prev["overall_attendance_pct"]), 4)
                backlog_change = float(latest["active_backlogs_count"]) - float(prev_prev["active_backlogs_count"])
                has_lag = 1.0
            else:
                sgpa_change = 0.0
                cgpa_change = 0.0
                attendance_change = 0.0
                backlog_change = 0.0
                has_lag = 0.0
                
            # Academic & subject features from latest historical semester
            avg_sub_gp = float(latest["avg_subject_grade_point"])
            min_sub_gp = float(latest["min_subject_grade_point"])
            max_sub_gp = float(latest["max_subject_grade_point"])
            var_sub_gp = float(latest["var_subject_grade_point"])
            
            avg_sub_att = float(latest["avg_subject_attendance"])
            min_sub_att = float(latest["min_subject_attendance"])
            var_sub_att = float(latest["var_subject_attendance"])
            
            # Backlog features
            active_backlogs = float(latest["active_backlogs_count"])
            # Count failed subjects from latest record
            failed_count = float(latest["num_failed_subjects"])
            
            # Credit features
            reg_credits = float(latest["total_sem_credits_registered"])
            earned_credits = float(latest["total_sem_credits_earned"])
            completion_ratio = round(earned_credits / max(reg_credits, 1.0), 4)
            
            sample = {
                "student_id": student_id,
                "history_length": len(history),
                "current_semester": int(latest["semester"]),
                "target_semester": target_semester,
                "branch": str(latest["branch"]),
                
                # Performance
                "previous_sgpa": prev_sgpa,
                "previous_cgpa": prev_cgpa,
                "sgpa_change": sgpa_change,
                "cgpa_change": cgpa_change,
                
                # Attendance
                "previous_attendance": prev_attendance,
                "attendance_change": attendance_change,
                "avg_subject_attendance": avg_sub_att,
                "min_subject_attendance": min_sub_att,
                "var_subject_attendance": var_sub_att,
                
                # Academic Performance
                "avg_subject_grade_point": avg_sub_gp,
                "min_subject_grade_point": min_sub_gp,
                "max_subject_grade_point": max_sub_gp,
                "var_subject_grade_point": var_sub_gp,
                
                # Backlogs
                "active_backlogs_count": active_backlogs,
                "failed_subjects_count": failed_count,
                "backlog_change": backlog_change,
                
                # Credits
                "total_sem_credits_registered": reg_credits,
                "total_sem_credits_earned": earned_credits,
                "credit_completion_ratio": completion_ratio,
                
                # Meta flag
                "has_historical_lag": has_lag,
                
                # TARGET
                "next_semester_sgpa": target_sgpa
            }
            samples.append(sample)
            
    samples_df = pd.DataFrame(samples)
    logger.info(f"Created {len(samples_df)} temporal training samples.")
    return samples_df


def split_temporal_data(
    samples_df: pd.DataFrame,
    test_ratio: float = 0.2,
    train_count: int = 300,
    val_count: int = 100,
    test_count: int = 100,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Split temporal dataset strictly by student cohorts so that:
    1. A student's entire longitudinal history (e.g. Sem1->Sem2 and Sem1+2->Sem3)
       remains intact within their assigned cohort.
    2. Zero student overlap between Train, Validation, and Test cohorts:
       train_students.isdisjoint(val_students) == True
       train_students.isdisjoint(test_students) == True
       val_students.isdisjoint(test_students) == True
    3. Deterministic shuffling based on random_seed.
    
    Returns:
        train_df, val_df, test_df, split_info
    """
    # 1. Extract unique student IDs and sort for deterministic initial state
    unique_students = np.array(sorted(samples_df["student_id"].unique()))
    total_unique_students = len(unique_students)
    
    if total_unique_students != (train_count + val_count + test_count):
        logger.warning(
            f"Total unique students ({total_unique_students}) does not match sum of counts "
            f"({train_count} + {val_count} + {test_count})"
        )
        
    # 2. Deterministically shuffle student IDs
    rng = np.random.default_rng(random_seed)
    shuffled_students = unique_students.copy()
    rng.shuffle(shuffled_students)
    
    # 3. Partition student IDs into disjoint cohorts
    train_student_list = list(shuffled_students[:train_count])
    val_student_list = list(shuffled_students[train_count:train_count + val_count])
    test_student_list = list(shuffled_students[train_count + val_count:train_count + val_count + test_count])
    
    train_students = set(train_student_list)
    val_students = set(val_student_list)
    test_students = set(test_student_list)
    
    # Assert disjointness
    assert train_students.isdisjoint(val_students), "Train and Validation cohorts overlap!"
    assert train_students.isdisjoint(test_students), "Train and Test cohorts overlap!"
    assert val_students.isdisjoint(test_students), "Validation and Test cohorts overlap!"
    assert len(train_students) == train_count, f"Expected {train_count} train students, got {len(train_students)}"
    assert len(val_students) == val_count, f"Expected {val_count} val students, got {len(val_students)}"
    assert len(test_students) == test_count, f"Expected {test_count} test students, got {len(test_students)}"
    
    # 4. Assign all temporal samples of each student to their assigned cohort
    train_df = samples_df[samples_df["student_id"].isin(train_students)].copy().reset_index(drop=True)
    val_df = samples_df[samples_df["student_id"].isin(val_students)].copy().reset_index(drop=True)
    test_df = samples_df[samples_df["student_id"].isin(test_students)].copy().reset_index(drop=True)
    
    # Ensure temporal ordering inside each sample
    assert (train_df["target_semester"] > train_df["current_semester"]).all(), "Temporal ordering violated in train!"
    assert (val_df["target_semester"] > val_df["current_semester"]).all(), "Temporal ordering violated in val!"
    assert (test_df["target_semester"] > test_df["current_semester"]).all(), "Temporal ordering violated in test!"
    
    split_info = {
        "strategy": "Disjoint Student Cohort Temporal Split (300 Train / 100 Val / 100 Test)",
        "train": {
            "target_semesters": sorted(train_df["target_semester"].unique().tolist()),
            "num_samples": len(train_df),
            "num_students": len(train_students),
            "mean_target_sgpa": round(float(train_df["next_semester_sgpa"].mean()), 4),
            "std_target_sgpa": round(float(train_df["next_semester_sgpa"].std()), 4)
        },
        "validation": {
            "target_semesters": sorted(val_df["target_semester"].unique().tolist()),
            "num_samples": len(val_df),
            "num_students": len(val_students),
            "mean_target_sgpa": round(float(val_df["next_semester_sgpa"].mean()), 4),
            "std_target_sgpa": round(float(val_df["next_semester_sgpa"].std()), 4)
        },
        "test": {
            "target_semesters": sorted(test_df["target_semester"].unique().tolist()),
            "num_samples": len(test_df),
            "num_students": len(test_students),
            "mean_target_sgpa": round(float(test_df["next_semester_sgpa"].mean()), 4),
            "std_target_sgpa": round(float(test_df["next_semester_sgpa"].std()), 4)
        },
        "student_overlap": {
            "train_and_val_student_overlap": len(train_students.intersection(val_students)),
            "train_and_test_student_overlap": len(train_students.intersection(test_students)),
            "val_and_test_student_overlap": len(val_students.intersection(test_students)),
            "notes": "Strictly disjoint student cohorts across Train, Validation, and Test (0 student overlap)."
        },
        "cohort_assignments": {
            "train_students": sorted(train_student_list),
            "val_students": sorted(val_student_list),
            "test_students": sorted(test_student_list)
        }
    }
    
    return train_df, val_df, test_df, split_info

