"""
src/loss.py
Loss functions with robust Huber formulation and explicit L1 regularization.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn


class RegularizedHuberLoss(nn.Module):
    """
    Huber Loss (Smooth L1) with explicit L1 weight regularization.
    
    Huber loss is less sensitive to outliers in SGPA targets than MSE,
    behaving quadratically for small errors (|error| <= delta) and
    linearly for large errors (|error| > delta).
    """
    def __init__(self, delta: float = 1.0, l1_lambda: float = 0.0):
        super().__init__()
        self.delta = delta
        self.l1_lambda = l1_lambda
        self.huber = nn.HuberLoss(reduction="mean", delta=delta)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        model: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total regularized loss.
        
        Args:
            y_pred: Predicted values (batch_size, 1)
            y_true: Ground truth target (batch_size, 1)
            model: PyTorch model for computing L1 regularization penalty
            
        Returns:
            total_loss: Scalar torch.Tensor
            loss_components: Dict with 'huber_loss', 'l1_penalty', 'total_loss'
        """
        huber_val = self.huber(y_pred, y_true)
        
        l1_penalty = torch.tensor(0.0, device=y_pred.device)
        if self.l1_lambda > 0.0 and model is not None:
            l1_norm = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
            l1_penalty = self.l1_lambda * l1_norm
            
        total_loss = huber_val + l1_penalty
        
        metrics = {
            "huber_loss": float(huber_val.detach().item()),
            "l1_loss": float(l1_penalty.detach().item()),
            "total_loss": float(total_loss.detach().item())
        }
        
        return total_loss, metrics
