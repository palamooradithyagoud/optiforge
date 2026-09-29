"""
src/adaptation/explainability.py
Model Explainability Engine using Integrated Gradients (Sundararajan et al., 2017).
Provides:
1. Exact axiomatic feature attribution satisfying the Completeness property:
   sum(attributions) = F(x) - F(baseline).
2. Local student-level explanations (waterfall / force contributions).
3. Global feature importance rankings comparing pre-adaptation vs. post-adaptation models.
"""

import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.features import FeaturePipeline


class IntegratedGradientsExplainer:
    """
    Native PyTorch Integrated Gradients explainer for neural regression models.
    """
    def __init__(
        self,
        model: nn.Module,
        feature_names: List[str],
        baseline: Optional[np.ndarray] = None,
        steps: int = 50,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.feature_names = feature_names
        self.steps = steps
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Default baseline: zero vector in standardized feature space
        if baseline is None:
            self.baseline = np.zeros(len(feature_names), dtype=np.float32)
        else:
            self.baseline = np.asarray(baseline, dtype=np.float32)

    def explain_sample(self, x: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Compute Integrated Gradients for a single input sample x.
        
        Returns:
            attributions: (d,) array of feature attributions
            pred_val: F(x) predicted value
            baseline_val: F(x') baseline prediction
        """
        x = np.asarray(x, dtype=np.float32).ravel()
        x_base = self.baseline
        diff = x - x_base
        
        # Generate interpolated inputs along linear path: x' + alpha * (x - x')
        # Trapezoidal quadrature rule for exact Completeness precision
        alphas = np.linspace(0.0, 1.0, self.steps + 1)
        interpolated = np.array([x_base + a * diff for a in alphas], dtype=np.float32)
        
        input_tensor = torch.tensor(interpolated, dtype=torch.float32, requires_grad=True, device=self.device)
        preds = self.model(input_tensor)
        
        # Compute gradients with respect to inputs
        grad_outputs = torch.ones_like(preds)
        grads = torch.autograd.grad(
            outputs=preds,
            inputs=input_tensor,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False
        )[0]
        
        # Trapezoidal weights: [0.5, 1, 1, ..., 1, 0.5] / steps
        weights = np.ones(self.steps + 1, dtype=np.float32)
        weights[0] = 0.5
        weights[-1] = 0.5
        weights = weights / float(self.steps)
        
        avg_grads = np.sum(grads.cpu().numpy() * weights[:, None], axis=0)
        attributions = diff * avg_grads
        
        # Calculate endpoint predictions for completeness verification
        with torch.no_grad():
            x_t = torch.tensor(x.reshape(1, -1), dtype=torch.float32, device=self.device)
            base_t = torch.tensor(x_base.reshape(1, -1), dtype=torch.float32, device=self.device)
            pred_val = float(self.model(x_t).item())
            baseline_val = float(self.model(base_t).item())
            
        return attributions, pred_val, baseline_val

    def explain_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Integrated Gradients for a batch of samples.
        
        Returns:
            attributions: (N, d) numpy array
        """
        X = np.asarray(X, dtype=np.float32)
        N = len(X)
        all_attributions = np.zeros_like(X)
        
        for i in range(N):
            attr, _, _ = self.explain_sample(X[i])
            all_attributions[i] = attr
            
        return all_attributions

    def compute_global_importance(self, X: np.ndarray) -> pd.DataFrame:
        """
        Compute global mean absolute feature attributions across cohort X.
        
        Returns:
            DataFrame sorted by mean absolute attribution descending.
        """
        attributions = self.explain_batch(X)
        mean_abs_attr = np.mean(np.abs(attributions), axis=0)
        
        df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_attribution": mean_abs_attr,
            "mean_attribution": np.mean(attributions, axis=0)
        }).sort_values(by="mean_abs_attribution", ascending=False).reset_index(drop=True)
        
        return df

    def get_student_explanation(self, x: np.ndarray, student_id: str = "Student") -> Dict[str, Any]:
        """
        Get structured local explanation for an individual student.
        """
        attributions, pred, base_pred = self.explain_sample(x)
        
        contribs = [
            {"feature": f, "attribution": round(float(a), 4)}
            for f, a in zip(self.feature_names, attributions)
        ]
        contribs.sort(key=lambda item: abs(item["attribution"]), reverse=True)
        
        sum_attr = float(np.sum(attributions))
        completeness_delta = float(pred - base_pred)
        completeness_error = abs(sum_attr - completeness_delta)
        
        return {
            "student_id": student_id,
            "predicted_sgpa": round(pred, 4),
            "baseline_sgpa": round(base_pred, 4),
            "sum_attributions": round(sum_attr, 4),
            "prediction_delta": round(completeness_delta, 4),
            "completeness_error": round(completeness_error, 6),
            "top_contributions": contribs
        }
