"""
src/model.py
Baseline deep learning neural network regression model for next-semester SGPA prediction.
Modular architecture designed for seamless hyperparameter injection by Phase 2 (NSGA-II).
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn


class BaselineRegressor(nn.Module):
    """
    Lightweight, regularized feed-forward neural network for SGPA regression.
    
    Architecture:
    Input
      ↓
    Dense (e.g. 64)
      ↓
    Activation (e.g. ReLU)
      ↓
    Dropout
      ↓
    Dense (e.g. 32)
      ↓
    Activation (e.g. ReLU)
      ↓
    Dense (1)
      ↓
    Predicted SGPA
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
        activation: str = "relu"
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
            
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Select activation
        act_lower = activation.lower()
        if act_lower == "relu":
            act_cls = nn.ReLU
        elif act_lower == "leaky_relu":
            act_cls = nn.LeakyReLU
        elif act_lower == "elu":
            act_cls = nn.ELU
        elif act_lower == "gelu":
            act_cls = nn.GELU
        else:
            act_cls = nn.ReLU

        layers: List[nn.Module] = []
        prev_dim = input_dim
        
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_cls())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = h_dim
            
        # Final linear output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights with Xavier uniform for stability
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting next-semester SGPA."""
        return self.network(x)


def build_model(config: Dict[str, Any], input_dim: Optional[int] = None) -> BaselineRegressor:
    """
    Factory function to instantiate BaselineRegressor from configuration.
    Exposed cleanly so Phase 2 (NSGA-II) can easily generate models from genome configs.
    
    Args:
        config: Configuration dictionary (can be model section or top-level)
        input_dim: Number of input features
        
    Returns:
        BaselineRegressor instance
    """
    model_cfg = config.get("model", config)
    
    if input_dim is None:
        input_dim = model_cfg.get("input_dim", 22)
        
    hidden_dims = model_cfg.get("hidden_dims", [64, 32])
    dropout_rate = float(model_cfg.get("dropout_rate", 0.2))
    activation = str(model_cfg.get("activation", "relu"))
    
    model = BaselineRegressor(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        activation=activation
    )
    return model
