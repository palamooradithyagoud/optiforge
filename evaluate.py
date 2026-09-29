"""
evaluate.py
Evaluation module exposing evaluate_model(model, X, y) for Phase 1 and future OOD/drift experiments.
Calculates MAE, RMSE, R2, parameter count, and inference latency.
"""

import os
import argparse
from typing import Tuple, Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.utils import set_seed, load_config, compute_metrics, measure_inference_latency, count_parameters, save_json
from src.features import FeaturePipeline
from src.model import build_model
from src.loss import RegularizedHuberLoss


def evaluate_model(
    model: nn.Module,
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    criterion: Optional[nn.Module] = None,
    batch_size: int = 64
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Modular evaluation function reusable across Phase 1, Phase 2, and future Phase 3 OOD/drift experiments.
    
    Args:
        model: Evaluated PyTorch model
        X: Feature matrix (N, num_features)
        y: Ground truth targets (N,) or (N, 1)
        criterion: Optional loss criterion (e.g. RegularizedHuberLoss)
        batch_size: Evaluation batch size
        
    Returns:
        metrics: dict with MAE, RMSE, R2, and test loss (if criterion provided)
        y_pred: numpy array of predictions (N,)
    """
    model.eval()
    device = next(model.parameters()).device
    
    if isinstance(X, np.ndarray):
        X_tensor = torch.tensor(X, dtype=torch.float32)
    else:
        X_tensor = X.float()
        
    if isinstance(y, np.ndarray):
        y_true = y.ravel().astype(np.float64)
    else:
        y_true = y.cpu().numpy().ravel().astype(np.float64)
        
    n_samples = len(X_tensor)
    preds_list = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            xb = X_tensor[i:i + batch_size].to(device)
            out = model(xb)
            preds_list.append(out.cpu().numpy())
            
            if criterion is not None:
                yb = torch.tensor(y_true[i:i + batch_size], dtype=torch.float32).view(-1, 1).to(device)
                if isinstance(criterion, RegularizedHuberLoss):
                    batch_loss, _ = criterion(out, yb, model=None)
                else:
                    batch_loss = criterion(out, yb)
                total_loss += batch_loss.item()
                num_batches += 1
                
    y_pred = np.vstack(preds_list).ravel()
    metrics = compute_metrics(y_true, y_pred)
    
    if criterion is not None and num_batches > 0:
        metrics["loss"] = round(total_loss / num_batches, 4)
        
    return metrics, y_pred


def run_evaluation(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Execute complete evaluation workflow on the held-out test split."""
    config = load_config(config_path)
    set_seed(config["training"].get("seed", 42), config["training"].get("deterministic", True))
    
    processed_dir = config["data"]["processed_dir"]
    test_csv = os.path.join(processed_dir, "test.csv")
    scaler_path = config["artifacts"].get("scaler_save_path", "models/feature_pipeline.pkl") if "artifacts" in config else "models/feature_pipeline.pkl"
    model_path = config["artifacts"].get("model_save_path", "models/baseline_model.pt") if "artifacts" in config else "models/baseline_model.pt"
    
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test data not found at {test_csv}. Please run train.py first.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Pipeline not found at {scaler_path}. Please run train.py first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run train.py first.")
        
    test_df = pd.read_csv(test_csv)
    pipeline = FeaturePipeline.load(scaler_path)
    
    target_col = config["data"]["target_column"]
    X_test = pipeline.transform(test_df)
    y_test = test_df[target_col].values.astype(np.float32)
    
    # Load model
    device = torch.device("cpu")
    input_dim = X_test.shape[1]
    model = build_model(config, input_dim=input_dim)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    criterion = RegularizedHuberLoss(delta=config["training"].get("huber_delta", 1.0))
    metrics, y_pred = evaluate_model(model, X_test, y_test, criterion=criterion)
    
    # Benchmarking latency
    sample_tensor = torch.tensor(X_test, dtype=torch.float32)
    latency_info = measure_inference_latency(model, sample_tensor, num_runs=200)
    param_info = count_parameters(model)
    
    results = {
        "evaluation_split": "Held-Out Test Set (Target Semester 3)",
        "num_test_samples": len(test_df),
        "test_mae": metrics["mae"],
        "test_rmse": metrics["rmse"],
        "test_r2": metrics["r2"],
        "test_huber_loss": metrics.get("loss", 0.0),
        "model_parameters": param_info,
        "inference_latency": latency_info,
        "training_metadata": checkpoint.get("metadata", {})
    }
    
    results_dir = config["logging"].get("results_dir", "results")
    out_file = os.path.join(results_dir, "baseline_metrics.json")
    save_json(results, out_file)
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION RESULTS (HELD-OUT TEST SET)")
    print("=" * 60)
    print(f"Test MAE:  {metrics['mae']:.4f}")
    print(f"Test RMSE: {metrics['rmse']:.4f}")
    print(f"Test R²:   {metrics['r2']:.4f}")
    print(f"Trainable Parameters: {param_info['trainable_parameters']}")
    print(f"Single-sample Latency: {latency_info['single_sample_latency_ms']:.4f} ms")
    print(f"Batch Latency ({len(test_df)} samples): {latency_info['batch_latency_ms']:.4f} ms")
    print(f"Metrics saved to: {out_file}")
    print("=" * 60 + "\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    run_evaluation(args.config)
