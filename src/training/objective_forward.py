"""
Forward optimization objective function for Optuna.

This module implements the objective function that:
1. Takes sampled hyperparameters from Optuna
2. Builds and trains a model with those hyperparameters
3. Returns the glucose RMSE as the optimization target
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna

# Suppress TensorFlow warnings during optimization
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def run_forward_training(
    params: Dict[str, Any],
    patient: int,
    data_root: str = "data/synthetic",
    verbose: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """
    Run forward training with given hyperparameters.

    This function is designed to be called from the Optuna objective.
    It handles all model setup, training, and evaluation.

    Args:
        params: Hyperparameters sampled by Optuna (from SearchSpace.sample())
        patient: Patient number (2-11 for synthetic)
        data_root: Root directory for data
        verbose: Whether to print progress

    Returns:
        Tuple of (rmse, results_dict)
    """
    model_type = params["model_type"]

    # Import here to allow TF mode setup before import
    if model_type in ["pinn", "modified_mlp"]:
        import tensorflow as tf

        if tf.executing_eagerly():
            if verbose:
                print(f"  Warning: TF eager mode active, cannot run {model_type}")
            return 1.0, {"error": "TF mode incompatible"}

    from src.datasets.loader import load_synthetic_window
    from src.training.config import Config
    from src.training.search_space import params_to_config_dict

    try:
        # Convert Optuna params to config
        config_dict = params_to_config_dict(params, mode="forward")
        config = Config.from_dict(config_dict)

        # Load data
        data = load_synthetic_window(patient=patient, root=data_root)

        if verbose:
            print(f"  Loading {model_type.upper()} model...")

        # Build model
        if model_type == "birnn":
            from src.models.birnn import BIRNN

            model = BIRNN(config)
        elif model_type == "pinn":
            from src.models.pinn_feedforward import FeedforwardPINN

            model = FeedforwardPINN(config)
        elif model_type == "modified_mlp":
            from src.models.modified_mlp import ModifiedMLPPINN

            model = ModifiedMLPPINN(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.build(data)
        model.compile()

        if verbose:
            print(
                f"  Training {model_type.upper()} for {params.get('epochs', 10000)} epochs..."
            )

        # Train model
        model.train(display_every=max(1000, params.get("epochs", 10000) // 10))

        # Evaluate
        metrics = model.evaluate() if hasattr(model, "evaluate") else {}

        # Get RMSE (primary metric)
        rmse = metrics.get("rmse_total", metrics.get("rmse", 1.0))

        results = {
            "rmse_total": rmse,
            "rmse_interpolation": metrics.get("rmse_interpolation"),
            "rmse_forecast": metrics.get("rmse_forecast"),
            "model_type": model_type,
            "patient": patient,
            "epochs": params.get("epochs", 10000),
        }

        if verbose:
            print(f"  RMSE: {rmse:.4f}")

        return rmse, results

    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
            traceback.print_exc()
        return 1.0, {"error": str(e)}


class ForwardObjective:
    """
    Optuna objective class for forward optimization.

    This class maintains state across trials and handles:
    - Model type selection
    - Hyperparameter sampling
    - Training execution
    - Result logging

    Example:
        >>> objective = ForwardObjective(patient=5, model_type="birnn")
        >>> study = optuna.create_study(direction="minimize")
        >>> study.optimize(objective, n_trials=100)
    """

    def __init__(
        self,
        patient: int,
        model_type: Optional[str] = None,
        data_root: str = "data/synthetic",
        verbose: bool = True,
    ):
        """
        Initialize the forward objective.

        Args:
            patient: Patient number for training (2-11 for synthetic)
            model_type: If specified, only optimize this model type.
                       If None, model_type is sampled as a hyperparameter.
            data_root: Root directory for data files
            verbose: Whether to print progress
        """
        self.patient = patient
        self.fixed_model_type = model_type
        self.data_root = data_root
        self.verbose = verbose

        # Track best result
        self.best_rmse = float("inf")
        self.best_params = None

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function called by Optuna.

        Args:
            trial: Optuna trial object

        Returns:
            Glucose RMSE - lower is better
        """
        from src.training.search_space import SearchSpace

        # Sample model type (or use fixed)
        if self.fixed_model_type is not None:
            model_type = self.fixed_model_type
        else:
            model_type = trial.suggest_categorical(
                "model_type", ["birnn", "pinn", "modified_mlp"]
            )

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Trial {trial.number}: {model_type.upper()}")
            print(f"{'='*60}")

        # Get search space and sample hyperparameters
        search_space = SearchSpace(model_type=model_type, mode="forward")
        params = search_space.sample(trial)

        if self.verbose:
            print("Sampled hyperparameters:")
            for key in ["epochs", "learning_rate", "lambda_g", "lambda_B"]:
                if key in params:
                    print(f"  {key}: {params[key]}")

        # Run training
        rmse, results = run_forward_training(
            params=params,
            patient=self.patient,
            data_root=self.data_root,
            verbose=self.verbose,
        )

        # Store results as trial attributes
        for key, value in results.items():
            if value is not None and not isinstance(value, dict):
                trial.set_user_attr(key, value)

        # Track best
        if rmse < self.best_rmse:
            self.best_rmse = rmse
            self.best_params = params.copy()
            if self.verbose:
                print(f"  NEW BEST: {rmse:.4f}")

        return rmse


def create_forward_study(
    study_name: str,
    storage: Optional[str] = None,
    direction: str = "minimize",
    pruner: Optional[optuna.pruners.BasePruner] = None,
    sampler: Optional[optuna.samplers.BaseSampler] = None,
    load_if_exists: bool = True,
) -> optuna.Study:
    """
    Create an Optuna study for forward optimization.

    Args:
        study_name: Name of the study
        storage: SQLite URL (e.g., "sqlite:///results/optimization/study.db")
        direction: "minimize" for RMSE minimization
        pruner: Pruner for early stopping (default: MedianPruner)
        sampler: Sampler for hyperparameter selection (default: TPE)
        load_if_exists: Whether to resume existing study

    Returns:
        Optuna Study object
    """
    if pruner is None:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2000,
            interval_steps=500,
        )

    if sampler is None:
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=10,
            seed=42,
        )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        pruner=pruner,
        sampler=sampler,
        load_if_exists=load_if_exists,
    )

    return study


def run_optimization(
    n_trials: int,
    patient: int,
    model_type: Optional[str] = None,
    study_name: str = "forward_optimization",
    storage: Optional[str] = None,
    n_jobs: int = 1,
    verbose: bool = True,
) -> optuna.Study:
    """
    Run forward hyperparameter optimization.

    Args:
        n_trials: Number of trials to run
        patient: Patient number for training
        model_type: If specified, only optimize this model type
        study_name: Name of the study
        storage: SQLite storage URL
        n_jobs: Number of parallel jobs (1 for sequential)
        verbose: Whether to print progress

    Returns:
        Completed Optuna study
    """
    # Create study
    study = create_forward_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    # Create objective
    objective = ForwardObjective(
        patient=patient,
        model_type=model_type,
        verbose=verbose,
    )

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    return study


# Example usage
if __name__ == "__main__":
    # Quick test with 2 trials
    print("Testing forward objective...")

    study = run_optimization(
        n_trials=2,
        patient=5,
        model_type="birnn",  # Fix to BI-RNN for testing
        study_name="test_forward",
        verbose=True,
    )

    print(f"\nBest trial:")
    print(f"  Value: {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")
