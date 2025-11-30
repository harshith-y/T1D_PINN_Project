"""
Prediction extraction utilities.

Handles differences between BI-RNN and DeepXDE model prediction formats.
Extracted from train_with_visual.py for reusability.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import tensorflow as tf


def extract_predictions(
    model: Any, data: Any, model_type: str
) -> Dict[str, np.ndarray]:
    """
    Extract predictions from trained model.

    Handles differences between BI-RNN and DeepXDE models:
    - BI-RNN: Returns sequences from model.model()
    - DeepXDE: Requires dde_model.predict() with normalized time

    Args:
        model: Trained model (BI-RNN, PINN, or Modified-MLP)
        data: TrainingWindow data object
        model_type: 'birnn', 'pinn', or 'modified_mlp'

    Returns:
        Dictionary with:
            - time: Time array matching prediction length
            - glucose_pred: Predicted glucose (mg/dL)
            - glucose_true: True glucose (mg/dL)
            - insulin_pred: Predicted insulin (U/dL) if available
            - insulin_true: True insulin if available
            - digestion_pred: Predicted digestion (mg/dL/min) if available
            - digestion_true: True digestion if available
            - split_idx: Train/test split index
    """

    if model_type == "birnn":
        return _extract_birnn_predictions(model, data)
    elif model_type in ["pinn", "modified_mlp"]:
        return _extract_deepxde_predictions(model, data)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _extract_birnn_predictions(model, data) -> Dict[str, np.ndarray]:
    """
    Extract predictions from BI-RNN model.

    BI-RNN processes sequences, so output length may differ from input.
    Uses the actual prediction length for time array.
    """
    # Get full sequences (train + test)
    X_full = tf.concat([model.X_train, model.X_test], axis=1)
    Y_full = tf.concat([model.Y_train, model.Y_test], axis=1)

    # Get predictions
    Y_pred_full = model.model(X_full, training=False)

    # Extract and denormalize glucose
    pred_glucose = Y_pred_full[0, :, 0].numpy() * data.m_g
    true_glucose = Y_full[0, :, 0].numpy() * data.m_g

    # CRITICAL: BI-RNN output is shorter than input (sequence processing)
    # Use the time array that matches the prediction length
    time_array = data.t_min[: len(pred_glucose)]

    # Extract latent states if available
    if data.has_latent_states:
        pred_insulin = Y_pred_full[0, :, 1].numpy() * data.m_i
        pred_digestion = Y_pred_full[0, :, 2].numpy() * data.m_d
        true_insulin = Y_full[0, :, 1].numpy() * data.m_i
        true_digestion = Y_full[0, :, 2].numpy() * data.m_d
    else:
        pred_insulin = None
        pred_digestion = None
        true_insulin = None
        true_digestion = None

    split_idx = model.X_train.shape[1]

    return {
        "time": time_array,
        "glucose_pred": pred_glucose,
        "glucose_true": true_glucose,
        "insulin_pred": pred_insulin,
        "insulin_true": true_insulin,
        "digestion_pred": pred_digestion,
        "digestion_true": true_digestion,
        "split_idx": split_idx,
    }


def _extract_deepxde_predictions(model, data) -> Dict[str, np.ndarray]:
    """
    Extract predictions from DeepXDE models (PINN, Modified-MLP).

    DeepXDE models use normalized time input and return normalized outputs.
    Requires denormalization using data.m_g, m_i, m_d.
    """
    # Create time array for prediction (normalized)
    t_norm = data.time_norm.reshape(-1, 1)  # Shape: (N, 1)

    # Get DeepXDE model
    # Different models may have different attribute names
    if hasattr(model, "dde_model"):
        dde_model = model.dde_model
    elif hasattr(model, "model"):
        dde_model = model.model
    else:
        raise AttributeError(
            f"Cannot find DeepXDE model in {type(model).__name__}. "
            f"Expected 'dde_model' or 'model' attribute."
        )

    # Get predictions
    # This returns [G_norm, I_norm, D_norm] in normalized units
    Y_pred_norm = dde_model.predict(t_norm)

    # Denormalize predictions
    # Y_pred_norm shape: (N, 3) for [G, I, D]
    pred_glucose = Y_pred_norm[:, 0] * data.m_g

    # Get ground truth
    true_glucose = data.glucose

    # Make sure lengths match
    min_len = min(len(pred_glucose), len(true_glucose))
    pred_glucose = pred_glucose[:min_len]
    true_glucose = true_glucose[:min_len]
    time_array = data.t_min[:min_len]

    # Extract latent states if available
    if data.has_latent_states:
        # Extract and denormalize latent state predictions
        pred_insulin = Y_pred_norm[:, 1] * data.m_i
        pred_digestion = Y_pred_norm[:, 2] * data.m_d

        # Get ground truth
        true_insulin = data.insulin[:min_len]
        true_digestion = data.digestion[:min_len]

        # Truncate predictions
        pred_insulin = pred_insulin[:min_len]
        pred_digestion = pred_digestion[:min_len]
    else:
        pred_insulin = None
        pred_digestion = None
        true_insulin = None
        true_digestion = None

    split_idx = int(0.8 * min_len)

    return {
        "time": time_array,
        "glucose_pred": pred_glucose,
        "glucose_true": true_glucose,
        "insulin_pred": pred_insulin,
        "insulin_true": true_insulin,
        "digestion_pred": pred_digestion,
        "digestion_true": true_digestion,
        "split_idx": split_idx,
    }


# Example usage
if __name__ == "__main__":
    print("Prediction extractor created successfully!")
    print("\nUsage example:")
    print(
        """
    from src.training.prediction_extractor import extract_predictions
    from src.datasets.loader import load_synthetic_window
    from src.models.birnn import BIRNN
    from src.training.config import load_config
    
    # After training
    data = load_synthetic_window(patient=3)
    config = load_config('birnn', 'forward')
    model = BIRNN(config)
    model.build(data)
    model.compile()
    model.train()
    
    # Extract predictions (handles model type automatically)
    predictions = extract_predictions(model, data, model_type='birnn')
    
    print(f"Time shape: {predictions['time'].shape}")
    print(f"Glucose pred shape: {predictions['glucose_pred'].shape}")
    print(f"Split idx: {predictions['split_idx']}")
    """
    )
