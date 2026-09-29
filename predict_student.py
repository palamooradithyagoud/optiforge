"""
predict_student.py
Utility script to predict a student's next semester SGPA using the frozen OptiForge NSGA-II model.

Usage:
    python predict_student.py --student_id CSE20260005
    python predict_student.py --sample 0
    python predict_student.py --custom
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch

WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.features import FeaturePipeline
from src.adaptation.evaluate_recovery import load_model_from_checkpoint


def predict_for_student_row(row: pd.DataFrame, model, pipeline):
    X = pipeline.transform(row)
    with torch.no_grad():
        pred_sgpa = model(torch.tensor(X, dtype=torch.float32)).item()
    return float(np.clip(pred_sgpa, 0.0, 10.0))


def main():
    parser = argparse.ArgumentParser(description="OptiForge Student Performance Predictor")
    parser.add_argument("--student_id", type=str, default="CSE20260005", help="Student ID from validation set")
    parser.add_argument("--dataset", type=str, default="data/processed/val.csv", help="Path to evaluation dataset")
    args = parser.parse_args()

    pipeline = FeaturePipeline.load("models/feature_pipeline.pkl")
    model = load_model_from_checkpoint("models/nsga2/nsga2_selected_model.pt", input_dim=22, device=torch.device("cpu"))
    model.eval()

    df = pd.read_csv(args.dataset)
    matches = df[df["student_id"] == args.student_id]

    if matches.empty:
        print(f"Student ID {args.student_id} not found in {args.dataset}. Showing first available student instead.")
        row = df.iloc[[1]]
    else:
        row = matches.iloc[[-1]]  # latest semester record

    sid = row["student_id"].values[0]
    curr_sem = int(row["current_semester"].values[0])
    target_sem = int(row["target_semester"].values[0])
    prev_sgpa = float(row["previous_sgpa"].values[0])
    prev_cgpa = float(row["previous_cgpa"].values[0])
    att = float(row["previous_attendance"].values[0])
    backlogs = int(row["active_backlogs_count"].values[0])
    actual_sgpa = float(row["next_semester_sgpa"].values[0]) if "next_semester_sgpa" in row.columns else None

    pred_sgpa = predict_for_student_row(row, model, pipeline)

    print("\n" + "=" * 55)
    print("   OPTIFORGE: NEXT SEMESTER PERFORMANCE PREDICTION")
    print("=" * 55)
    print(f" Student ID:              {sid}")
    print(f" Current Semester:        Semester {curr_sem}")
    print(f" Target Prediction Sem:   Semester {target_sem}")
    print(f" Previous Semester SGPA:  {prev_sgpa:.2f} / 10.0")
    print(f" Cumulative GPA (CGPA):   {prev_cgpa:.2f} / 10.0")
    print(f" Attendance Rate:         {att:.1f}%")
    print(f" Active Backlogs:         {backlogs}")
    print("-" * 55)
    print(f" PREDICTED NEXT SEM SGPA: {pred_sgpa:.2f} / 10.0")
    if actual_sgpa is not None:
        print(f" ACTUAL RECORDED SGPA:    {actual_sgpa:.2f} / 10.0")
        error = abs(pred_sgpa - actual_sgpa)
        print(f" Prediction Absolute Error: {error:.2f} SGPA")
    print(" 90% Conformal Interval:  [" + f"{max(0.0, pred_sgpa - 0.98):.2f}, {min(10.0, pred_sgpa + 0.98):.2f}]")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
