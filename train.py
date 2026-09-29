"""
train.py
Main training script for Phase 1 baseline deep learning model.
Executes data validation, temporal sample generation, temporal splitting,
feature scaling, regularized training with gradient clipping, and epoch logging.
"""

import os
import time
import argparse
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils import set_seed, load_config, save_json, count_parameters
from src.data_pipeline import validate_raw_data, create_temporal_samples, split_temporal_data
from src.features import prepare_datasets, create_dataloaders
from src.model import build_model
from src.loss import RegularizedHuberLoss
from evaluate import run_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_baseline(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Execute complete Phase 1 training pipeline."""
    config = load_config(config_path)
    
    # 1. Deterministic configuration and seed setup
    seed = config["training"].get("seed", 42)
    deterministic = config["training"].get("deterministic", True)
    set_seed(seed, deterministic)
    logger.info(f"Initialized environment with seed={seed}, deterministic={deterministic}")
    
    # Setup directories
    results_dir = config["logging"].get("results_dir", "results")
    models_dir = config["logging"].get("models_dir", "models")
    processed_dir = config["data"].get("processed_dir", "data/processed")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # 2. STEP 1: Load and Validate Raw Data
    raw_path = config["data"]["raw_path"]
    logger.info(f"Loading raw data from: {raw_path}")
    raw_df = pd.read_csv(raw_path)
    
    cleaned_df, quality_report = validate_raw_data(raw_df)
    report_path = os.path.join(results_dir, "data_quality_report.json")
    save_json(quality_report, report_path)
    logger.info(f"Data quality report saved to: {report_path}")
    
    # 3. STEP 2 & 3: Parse Subjects and Create Temporal Training Samples
    samples_df = create_temporal_samples(cleaned_df)
    
    # 4. STEP 5: Temporal Splitting (Student Cohort Level)
    train_count = config["data"].get("train_student_count", 300)
    val_count = config["data"].get("val_student_count", 100)
    test_count = config["data"].get("test_student_count", 100)
    train_df, val_df, test_df, split_info = split_temporal_data(
        samples_df,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        random_seed=seed
    )
    
    # Save processed splits
    train_csv = os.path.join(processed_dir, "train.csv")
    val_csv = os.path.join(processed_dir, "val.csv")
    test_csv = os.path.join(processed_dir, "test.csv")
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print("\n" + "=" * 65)
    print("DATA COHORT TEMPORAL SPLIT CONFIGURATION")
    print("=" * 65)
    print(f"TRAIN COHORT:      Students: {split_info['train']['num_students']} | Samples: {len(train_df)} | Target Semesters: {split_info['train']['target_semesters']}")
    print(f"VALIDATION COHORT: Students: {split_info['validation']['num_students']} | Samples: {len(val_df)} | Target Semesters: {split_info['validation']['target_semesters']}")
    print(f"TEST COHORT:       Students: {split_info['test']['num_students']} | Samples: {len(test_df)} | Target Semesters: {split_info['test']['target_semesters']}")
    print("-" * 65)
    print(f"Student Overlap Train & Val:  {split_info['student_overlap']['train_and_val_student_overlap']}")
    print(f"Student Overlap Train & Test: {split_info['student_overlap']['train_and_test_student_overlap']}")
    print(f"Student Overlap Val & Test:   {split_info['student_overlap']['val_and_test_student_overlap']}")
    print(f"Strategy: {split_info['strategy']}")
    print("=" * 65 + "\n")
    
    # 5. STEP 4: Feature Preparation & Leak-Free Fitting
    X_train, y_train, X_val, y_val, X_test, y_test, pipeline = prepare_datasets(
        train_df, val_df, test_df, config
    )
    
    pipeline_save_path = os.path.join(models_dir, "feature_pipeline.pkl")
    pipeline.save(pipeline_save_path)
    logger.info(f"Fitted FeaturePipeline saved to: {pipeline_save_path} (Features: {len(pipeline.feature_names)})")
    
    batch_size = config["training"].get("batch_size", 32)
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size=batch_size)
    
    # 6. STEP 6: Build Baseline Model
    device = torch.device("cpu")
    input_dim = X_train.shape[1]
    model = build_model(config, input_dim=input_dim)
    model.to(device)
    
    param_info = count_parameters(model)
    logger.info(f"Built BaselineRegressor with {param_info['trainable_parameters']} trainable parameters.")
    
    # Optimizer & Regularization setup
    lr = config["training"].get("learning_rate", 0.001)
    weight_decay = config["training"].get("weight_decay", 1e-4)  # L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Loss setup
    huber_delta = config["training"].get("huber_delta", 1.0)
    l1_lambda = config["training"].get("l1_lambda", 5e-5)        # L1 regularization
    criterion = RegularizedHuberLoss(delta=huber_delta, l1_lambda=l1_lambda)
    
    # Gradient clipping & early stopping setup
    max_grad_norm = config["training"].get("max_grad_norm", 1.0)
    epochs = config["training"].get("epochs", 150)
    patience = config["training"].get("early_stopping_patience", 25)
    
    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None
    best_epoch = 0
    
    history_records: List[Dict[str, Any]] = []
    grad_norms_all: List[float] = []
    start_total_train_time = time.perf_counter()
    
    logger.info(f"Beginning training for {epochs} epochs (Early stopping patience={patience})...")
    
    # 7. STEP 7: Baseline Training Loop
    for epoch in range(1, epochs + 1):
        t_epoch_start = time.perf_counter()
        
        # --- Training ---
        model.train()
        train_loss_accum = 0.0
        train_huber_accum = 0.0
        train_batches = 0
        epoch_grad_norms = []
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            y_pred = model(xb)
            loss, loss_dict = criterion(y_pred, yb, model=model)
            
            # Check for NaN / instability
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"Unstable training detected: Loss is NaN/Inf at epoch {epoch}!")
                
            loss.backward()
            
            # Gradient clipping & gradient norm monitoring
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            grad_norm_val = float(grad_norm.item())
            epoch_grad_norms.append(grad_norm_val)
            grad_norms_all.append(grad_norm_val)
            
            optimizer.step()
            
            train_loss_accum += loss_dict["total_loss"]
            train_huber_accum += loss_dict["huber_loss"]
            train_batches += 1
            
        epoch_train_loss = train_loss_accum / max(train_batches, 1)
        epoch_huber_loss = train_huber_accum / max(train_batches, 1)
        avg_grad_norm = float(np.mean(epoch_grad_norms))
        
        # --- Validation ---
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                y_pred = model(xb)
                val_loss, val_dict = criterion(y_pred, yb, model=None)
                val_loss_accum += val_dict["huber_loss"]
                val_batches += 1
                
        epoch_val_loss = val_loss_accum / max(val_batches, 1)
        epoch_duration = time.perf_counter() - t_epoch_start
        
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Record epoch metrics
        epoch_log = {
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 5),
            "train_huber_loss": round(epoch_huber_loss, 5),
            "val_loss": round(epoch_val_loss, 5),
            "learning_rate": current_lr,
            "gradient_norm": round(avg_grad_norm, 5),
            "epoch_time_sec": round(epoch_duration, 4)
        }
        history_records.append(epoch_log)
        
        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:03d}/{epochs} | Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} | Grad Norm: {avg_grad_norm:.4f} | "
                f"Time: {epoch_duration:.3f}s"
            )
            
        # Early stopping logic
        if epoch_val_loss < best_val_loss - 1e-4:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (Best epoch {best_epoch} with val_loss {best_val_loss:.4f})")
                break
                
    total_training_time = time.perf_counter() - start_total_train_time
    
    # Restore best model weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
        logger.info(f"Restored best weights from epoch {best_epoch} (val_loss: {best_val_loss:.4f})")
        
    # Save training history CSV
    history_df = pd.DataFrame(history_records)
    history_csv = os.path.join(results_dir, "training_history.csv")
    history_df.to_csv(history_csv, index=False)
    logger.info(f"Training history saved to: {history_csv}")
    
    # Save trained model checkpoint
    model_save_path = os.path.join(models_dir, "baseline_model.pt")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "feature_names": pipeline.feature_names,
        "metadata": {
            "best_epoch": best_epoch,
            "total_epochs_trained": len(history_records),
            "best_val_loss": round(best_val_loss, 4),
            "final_train_loss": round(history_records[-1]["train_loss"], 4),
            "final_val_loss": round(history_records[-1]["val_loss"], 4),
            "total_training_time_sec": round(total_training_time, 2),
            "grad_norm_mean": round(float(np.mean(grad_norms_all)), 4),
            "grad_norm_max": round(float(np.max(grad_norms_all)), 4),
            "grad_norm_min": round(float(np.min(grad_norms_all)), 4),
            "convergence_status": f"Converged at epoch {best_epoch} (Early stopping patience: {patience})"
        }
    }
    torch.save(checkpoint, model_save_path)
    logger.info(f"Trained model checkpoint saved to: {model_save_path}")
    
    # 8. STEP 8: Run Evaluation on Held-Out Test Set
    eval_results = run_evaluation(config_path)
    
    return {
        "split_info": split_info,
        "training_time": total_training_time,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "evaluation": eval_results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    train_baseline(args.config)
