"""
src/utils.py
Utility functions for reproducibility, metrics calculation, configuration loading,
and benchmarking inference latency.
"""

import os
import random
import time
import json
import yaml
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds across python, numpy, and torch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration dictionary from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression metrics: MAE, RMSE, and R2 score.
    
    Args:
        y_true: Ground truth target values
        y_pred: Model predictions
        
    Returns:
        dict: {'mae': float, 'rmse': float, 'r2': float}
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4)
    }


def measure_inference_latency(model: torch.nn.Module, sample_input: torch.Tensor, num_runs: int = 200) -> dict:
    """
    Measure inference latency for single-sample and batch inferences.
    
    Args:
        model: Evaluated PyTorch model
        sample_input: Example input tensor (batch_size, input_dim)
        num_runs: Number of forward passes to average over
        
    Returns:
        dict: latency measurements in milliseconds
    """
    model.eval()
    device = next(model.parameters()).device
    sample_input = sample_input.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(20):
            _ = model(sample_input)
            
    # Measure batch latency
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(sample_input)
    total_batch_time = (time.perf_counter() - start) * 1000.0  # ms
    avg_batch_latency_ms = total_batch_time / num_runs
    
    # Measure single sample latency
    single_sample = sample_input[0:1]
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(single_sample)
    total_single_time = (time.perf_counter() - start) * 1000.0  # ms
    avg_single_latency_ms = total_single_time / num_runs
    
    return {
        "batch_latency_ms": round(avg_batch_latency_ms, 4),
        "single_sample_latency_ms": round(avg_single_latency_ms, 4),
        "batch_size_benchmarked": sample_input.shape[0],
        "benchmark_runs": num_runs
    }


def save_json(data: dict, filepath: str) -> None:
    """Save dictionary to a nicely formatted JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def count_parameters(model: torch.nn.Module) -> dict:
    """Count trainable and total parameters in the PyTorch model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_parameters": trainable,
        "total_parameters": total
    }
