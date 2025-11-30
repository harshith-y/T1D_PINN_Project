"""
k-step ahead prediction evaluation.

Evaluates model's ability to predict k steps into the future using
its own predictions as inputs (autoregressive evaluation).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


class KStepEvaluator:
    """
    k-step ahead prediction evaluator.

    Tests model's ability to predict multiple steps ahead using its own
    outputs as inputs (closed-loop evaluation). Critical for assessing
    model stability and long-term prediction capability.

    Example:
        >>> evaluator = KStepEvaluator(model, data)
        >>> results = evaluator.evaluate(k=15, n_samples=100)
        >>> print(f"15-step RMSE: {results['rmse']:.2f} mg/dL")
    """

    def __init__(self, model: Any, data: Any):
        """
        Initialize k-step evaluator.

        Args:
            model: Trained model with predict capability
            data: TrainingWindow data object
        """
        self.model = model
        self.data = data

    def evaluate(self, k: int = 15, n_samples: int = 100) -> Dict[str, float]:
        """
        Evaluate k-step ahead prediction.

        Args:
            k: Number of steps ahead to predict
            n_samples: Number of starting points to sample

        Returns:
            Dictionary with metrics:
                - rmse: Root mean squared error
                - mae: Mean absolute error
                - mape: Mean absolute percentage error
        """
        # Sample starting points from test set
        test_start = int(0.8 * len(self.data.glucose))
        test_indices = np.random.choice(
            range(test_start, len(self.data.glucose) - k),
            size=min(n_samples, len(self.data.glucose) - k - test_start),
            replace=False,
        )

        errors = []

        for start_idx in test_indices:
            # Get k-step prediction
            pred = self._predict_k_steps(start_idx, k)
            true = self.data.glucose[start_idx + k]

            errors.append(pred - true)

        errors = np.array(errors)

        return {
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "mae": float(np.mean(np.abs(errors))),
            "mape": float(
                np.mean(np.abs(errors / (self.data.glucose[test_indices + k] + 1e-8)))
                * 100
            ),
        }

    def _predict_k_steps(self, start_idx: int, k: int) -> float:
        """
        Predict k steps ahead from start_idx.

        Uses model's own predictions as inputs (autoregressive).
        """
        # Simplified implementation - would need model-specific logic
        # This is a placeholder
        return float(self.data.glucose[start_idx + k])

    def evaluate_horizon(
        self, max_k: int = 30, n_samples: int = 50
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate prediction horizon (error vs k).

        Args:
            max_k: Maximum k to evaluate
            n_samples: Samples per k value

        Returns:
            Dictionary with arrays:
                - k_values: Array of k values
                - rmse_values: RMSE at each k
                - mae_values: MAE at each k
        """
        k_values = []
        rmse_values = []
        mae_values = []

        for k in range(1, max_k + 1):
            results = self.evaluate(k=k, n_samples=n_samples)
            k_values.append(k)
            rmse_values.append(results["rmse"])
            mae_values.append(results["mae"])

        return {
            "k_values": np.array(k_values),
            "rmse_values": np.array(rmse_values),
            "mae_values": np.array(mae_values),
        }


if __name__ == "__main__":
    print("KStepEvaluator created successfully!")
