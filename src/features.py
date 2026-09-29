"""
src/features.py
Feature engineering, scaling, categorical encoding, and PyTorch dataset preparation.
Guarantees strict separation between features and identifiers.
"""

import os
import pickle
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class StudentDataset(Dataset):
    """PyTorch Dataset for Student Next-Semester SGPA Prediction."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class FeaturePipeline:
    """
    Feature transformation pipeline that fits scalers and encoders ONLY on train data
    and securely applies them to validation and test data without leakage.
    """
    def __init__(self, numerical_cols: List[str], categorical_cols: List[str]):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """Fit scaler on numerical columns and encoder on categorical columns using training data only."""
        # Fit numerical
        self.scaler.fit(df[self.numerical_cols].values)
        
        # Fit categorical
        self.encoder.fit(df[self.categorical_cols].values)
        cat_feature_names = self.encoder.get_feature_names_out(self.categorical_cols).tolist()
        
        self.feature_names = self.numerical_cols + cat_feature_names
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform dataframe into a 2D float32 numpy array of features."""
        if not self.is_fitted:
            raise RuntimeError("FeaturePipeline must be fitted before transforming data.")
        
        num_scaled = self.scaler.transform(df[self.numerical_cols].values)
        cat_encoded = self.encoder.transform(df[self.categorical_cols].values)
        
        X = np.hstack([num_scaled, cat_encoded]).astype(np.float32)
        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform training dataframe."""
        self.fit(df)
        return self.transform(df)

    def save(self, filepath: str) -> None:
        """Persist fitted pipeline to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "FeaturePipeline":
        """Load fitted pipeline from disk."""
        with open(filepath, "rb") as f:
            pipeline = pickle.load(f)
        return pipeline


def prepare_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, FeaturePipeline]:
    """
    Extract features and targets, fit FeaturePipeline on train_df, and transform all splits.
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, pipeline
    """
    target_col = config["data"]["target_column"]
    
    # Collect numerical features
    num_features = []
    for category in ["historical_performance", "attendance", "academic_performance", "backlogs", "credits", "temporal_context"]:
        num_features.extend(config["features"].get(category, []))
        
    cat_features = config["features"].get("branch", ["branch"])
    
    pipeline = FeaturePipeline(numerical_cols=num_features, categorical_cols=cat_features)
    
    X_train = pipeline.fit_transform(train_df)
    y_train = train_df[target_col].values.astype(np.float32)
    
    X_val = pipeline.transform(val_df)
    y_val = val_df[target_col].values.astype(np.float32)
    
    X_test = pipeline.transform(test_df)
    y_test = test_df[target_col].values.astype(np.float32)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, pipeline


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for training and validation."""
    train_ds = StudentDataset(X_train, y_train)
    val_ds = StudentDataset(X_val, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader
