"""
src/nsga2/evaluation.py
Candidate genome evaluation with deterministic caching and strict test isolation.
Computes the 6 optimization objectives on TRAIN and VALIDATION splits only.
"""

import time
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.features import FeaturePipeline, StudentDataset
from src.model import build_model
from src.loss import RegularizedHuberLoss
from src.utils import compute_metrics, count_parameters, measure_inference_latency
from src.nsga2.genome import Genome
from src.nsga2.objectives import (
    ObjectiveMetrics, PENALTY_OBJECTIVES,
    compute_model_flops, compute_deterministic_latency_ms, compute_deterministic_training_cost
)

logger = logging.getLogger(__name__)


class GenomeEvaluator:
    """
    Evaluates neural network configurations against validation objectives.
    Includes exact canonical caching to prevent redundant training of identical genomes.
    
    CRITICAL ISOLATION RULE:
    Evaluator receives ONLY train and validation datasets.
    Test set is strictly omitted and never referenced.
    """
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_pipeline: FeaturePipeline,
        target_column: str = "next_semester_sgpa",
        device: Optional[torch.device] = None,
        early_stopping_patience: int = 15
    ):
        self.device = device or torch.device("cpu")
        self.target_column = target_column
        self.early_stopping_patience = early_stopping_patience
        
        # Transform data once
        self.X_train = feature_pipeline.transform(train_df)
        self.y_train = train_df[target_column].values.astype(np.float32)
        
        self.X_val = feature_pipeline.transform(val_df)
        self.y_val = val_df[target_column].values.astype(np.float32)
        
        self.input_dim = self.X_train.shape[1]
        self.val_tensor = torch.tensor(self.X_val, dtype=torch.float32).to(self.device)
        self.val_sample = self.val_tensor[:1]
        
        # Exact Canonical Cache: canonical_hash -> ObjectiveMetrics
        self.cache: Dict[str, ObjectiveMetrics] = {}
        
        # Statistics
        self.total_evaluations = 0
        self.cached_evaluations = 0
        self.failed_evaluations = 0

    def evaluate(self, genome: Genome) -> Tuple[ObjectiveMetrics, bool]:
        """
        Evaluate candidate genome on training and validation splits.
        
        Returns:
            metrics: ObjectiveMetrics instance
            cached: True if retrieved from cache, False if trained from scratch
        """
        chash = genome.canonical_hash()
        if chash in self.cache:
            self.cached_evaluations += 1
            return self.cache[chash], True
            
        self.total_evaluations += 1
        t_start = time.perf_counter()
        
        try:
            # Deterministic candidate initialization based on genome hash
            candidate_seed = int(chash[:8], 16) % (2**31 - 1)
            torch.manual_seed(candidate_seed)
            np.random.seed(candidate_seed)
            
            # 1. Instantiate model
            model_config = {
                "hidden_dims": genome.hidden_dims,
                "dropout_rate": genome.dropout_rate,
                "activation": genome.activation
            }
            model = build_model(model_config, input_dim=self.input_dim)
            model.to(self.device)
            
            # Count parameters
            param_count = count_parameters(model)["trainable_parameters"]
            
            # 2. Setup training components
            optimizer = optim.Adam(
                model.parameters(),
                lr=genome.learning_rate,
                weight_decay=genome.weight_decay
            )
            criterion = RegularizedHuberLoss(
                delta=genome.huber_delta,
                l1_lambda=genome.l1_lambda
            )
            
            train_ds = StudentDataset(self.X_train, self.y_train)
            train_loader = DataLoader(
                train_ds,
                batch_size=genome.batch_size,
                shuffle=True,
                drop_last=False
            )
            
            val_ds = StudentDataset(self.X_val, self.y_val)
            val_loader = DataLoader(
                val_ds,
                batch_size=genome.batch_size,
                shuffle=False,
                drop_last=False
            )
            
            # 3. Training Loop with early stopping and loss variance tracking
            val_losses = []
            train_losses = []
            best_val_loss = float("inf")
            patience_counter = 0
            best_weights = None
            best_epoch = 0
            
            for epoch in range(1, genome.epochs + 1):
                # Train
                model.train()
                epoch_train_loss = 0.0
                train_batches = 0
                
                for xb, yb in train_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss, loss_dict = criterion(pred, yb, model=model)
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        raise FloatingPointError(f"NaN/Inf loss at epoch {epoch}")
                        
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=genome.gradient_clip_norm)
                    optimizer.step()
                    
                    epoch_train_loss += loss_dict["total_loss"]
                    train_batches += 1
                    
                avg_train = epoch_train_loss / max(train_batches, 1)
                train_losses.append(avg_train)
                
                # Validation
                model.eval()
                epoch_val_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        pred = model(xb)
                        _, loss_dict = criterion(pred, yb, model=None)
                        epoch_val_loss += loss_dict["huber_loss"]
                        val_batches += 1
                        
                avg_val = epoch_val_loss / max(val_batches, 1)
                val_losses.append(avg_val)
                
                # Early stopping
                if avg_val < best_val_loss - 1e-4:
                    best_val_loss = avg_val
                    best_epoch = epoch
                    patience_counter = 0
                    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        break
                        
            # Restore best weights
            if best_weights is not None:
                model.load_state_dict(best_weights)
                
            elapsed_time = time.perf_counter() - t_start
            
            # 4. Final Validation Metrics
            model.eval()
            with torch.no_grad():
                val_preds = model(self.val_tensor).cpu().numpy().ravel()
            val_metrics = compute_metrics(self.y_val, val_preds)
            
            # Objective calculations
            f1_mae = float(val_metrics["mae"])
            f2_gen_gap = float(abs(val_losses[-1] - train_losses[-1]))
            f3_loss_var = float(np.var(val_losses)) if len(val_losses) > 1 else 0.0
            f4_params = int(param_count)
            
            # Latency benchmark and training computational cost proxies
            flops = compute_model_flops(self.input_dim, genome.hidden_dims)
            f5_latency = compute_deterministic_latency_ms(self.input_dim, genome.hidden_dims)
            f6_time = compute_deterministic_training_cost(flops, genome.epochs, num_train_samples=len(self.X_train))
            
            metrics = ObjectiveMetrics(
                validation_mae=f1_mae,
                generalization_gap=f2_gen_gap,
                val_loss_variance=f3_loss_var,
                trainable_parameters=f4_params,
                inference_latency_ms=f5_latency,
                training_time_seconds=f6_time,
                validation_rmse=float(val_metrics["rmse"]),
                validation_r2=float(val_metrics["r2"]),
                validation_loss=float(val_losses[-1]),
                training_loss=float(train_losses[-1]),
                best_epoch=best_epoch,
                status="success"
            )
            
        except Exception as e:
            self.failed_evaluations += 1
            logger.warning(f"Candidate evaluation failed with exception: {e}")
            metrics = ObjectiveMetrics(
                validation_mae=PENALTY_OBJECTIVES[0],
                generalization_gap=PENALTY_OBJECTIVES[1],
                val_loss_variance=PENALTY_OBJECTIVES[2],
                trainable_parameters=int(PENALTY_OBJECTIVES[3]),
                inference_latency_ms=PENALTY_OBJECTIVES[4],
                training_time_seconds=PENALTY_OBJECTIVES[5],
                validation_rmse=10.0,
                validation_r2=-10.0,
                validation_loss=10.0,
                training_loss=10.0,
                best_epoch=0,
                status=f"failed: {str(e)[:50]}"
            )
            
        # Store in cache
        self.cache[chash] = metrics
        return metrics, False
