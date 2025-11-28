"""
Evaluation metrics for glucose prediction models.

Provides standard metrics computation functions for model evaluation.
"""

from __future__ import annotations
from typing import Dict, Optional
import numpy as np


def compute_glucose_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    split_idx: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute glucose prediction metrics.
    
    Args:
        pred: Predicted glucose values (mg/dL)
        true: True glucose values (mg/dL)
        split_idx: Optional train/test split index
    
    Returns:
        Dictionary with metrics:
            - rmse: Root mean squared error
            - mae: Mean absolute error
            - mape: Mean absolute percentage error
            - rmse_train, rmse_test: Split metrics (if split_idx provided)
    """
    # Overall metrics
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae = float(np.mean(np.abs(pred - true)))
    mape = float(np.mean(np.abs((pred - true) / (true + 1e-8))) * 100)
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }
    
    # Split metrics if requested
    if split_idx is not None:
        train_pred = pred[:split_idx]
        train_true = true[:split_idx]
        test_pred = pred[split_idx:]
        test_true = true[split_idx:]
        
        metrics['rmse_train'] = float(np.sqrt(np.mean((train_pred - train_true) ** 2)))
        metrics['rmse_test'] = float(np.sqrt(np.mean((test_pred - test_true) ** 2)))
        metrics['mae_train'] = float(np.mean(np.abs(train_pred - train_true)))
        metrics['mae_test'] = float(np.mean(np.abs(test_pred - test_true)))
    
    return metrics


def compute_latent_metrics(
    insulin_pred: Optional[np.ndarray],
    insulin_true: Optional[np.ndarray],
    digestion_pred: Optional[np.ndarray],
    digestion_true: Optional[np.ndarray],
    split_idx: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute latent state metrics (insulin, digestion).
    
    Args:
        insulin_pred: Predicted insulin (U/dL)
        insulin_true: True insulin
        digestion_pred: Predicted digestion (mg/dL/min)
        digestion_true: True digestion
        split_idx: Optional split index
    
    Returns:
        Dictionary with latent state metrics
    """
    metrics = {}
    
    # Insulin metrics
    if insulin_pred is not None and insulin_true is not None:
        metrics['insulin_rmse'] = float(np.sqrt(np.mean((insulin_pred - insulin_true) ** 2)))
        metrics['insulin_mae'] = float(np.mean(np.abs(insulin_pred - insulin_true)))
        
        if split_idx:
            test_pred = insulin_pred[split_idx:]
            test_true = insulin_true[split_idx:]
            metrics['insulin_rmse_test'] = float(np.sqrt(np.mean((test_pred - test_true) ** 2)))
    
    # Digestion metrics
    if digestion_pred is not None and digestion_true is not None:
        metrics['digestion_rmse'] = float(np.sqrt(np.mean((digestion_pred - digestion_true) ** 2)))
        metrics['digestion_mae'] = float(np.mean(np.abs(digestion_pred - digestion_true)))
        
        if split_idx:
            test_pred = digestion_pred[split_idx:]
            test_true = digestion_true[split_idx:]
            metrics['digestion_rmse_test'] = float(np.sqrt(np.mean((test_pred - test_true) ** 2)))
    
    return metrics


if __name__ == "__main__":
    print("Metrics module created successfully!")