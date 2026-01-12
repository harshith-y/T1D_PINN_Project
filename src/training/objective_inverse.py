"""
Inverse optimization objective function for Optuna.

This module implements the objective function that:
1. Takes sampled hyperparameters from Optuna
2. Builds and trains a model with those hyperparameters
3. Returns the ksi estimation error (%) as the optimization target
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna

# Suppress TensorFlow warnings during optimization
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def run_inverse_training(
    params: Dict[str, Any],
    patient: int,
    data_root: str = "data/synthetic",
    verbose: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """
    Run inverse training with given hyperparameters.

    This function is designed to be called from the Optuna objective.
    It handles all model setup, training, and evaluation.

    Args:
        params: Hyperparameters sampled by Optuna (from SearchSpace.sample())
        patient: Patient number (2-11 for synthetic)
        data_root: Root directory for data
        verbose: Whether to print progress

    Returns:
        Tuple of (ksi_error_percent, results_dict)
    """
    model_type = params["model_type"]

    # Import here to allow TF mode setup before import
    if model_type in ["pinn", "modified_mlp"]:
        # DeepXDE requires TF 1.x compatibility mode
        import tensorflow as tf
        if tf.executing_eagerly():
            # Already in eager mode - this will cause issues for DeepXDE
            # Return a high error to penalize this configuration
            if verbose:
                print(f"  Warning: TF eager mode active, cannot run {model_type}")
            return 100.0, {"error": "TF mode incompatible"}

    from src.datasets.loader import load_synthetic_window
    from src.physics.magdelaine import make_params_from_preset
    from src.training.config import Config
    from src.training.inverse_trainer import InverseTrainer
    from src.training.search_space import params_to_config_dict

    try:
        # Convert Optuna params to config
        config_dict = params_to_config_dict(params, mode="inverse")
        config = Config.from_dict(config_dict)
        config.inverse_params = ["ksi"]

        # Load data
        data = load_synthetic_window(patient=patient, root=data_root)

        # Get true parameter value
        true_params = make_params_from_preset(patient)
        true_ksi = getattr(true_params, "ksi")

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
            print(f"  Training {model_type.upper()}...")

        # Train with InverseTrainer
        trainer = InverseTrainer(
            model=model,
            config=config,
            data=data,
            true_param_value=true_ksi,
        )

        history = trainer.train()

        # Extract results
        final_ksi = history["final_params"].get("ksi", 0)
        ksi_error = history["param_errors_percent"].get("ksi", 100.0)

        # Get additional metrics
        metrics = model.evaluate() if hasattr(model, "evaluate") else {}

        results = {
            "ksi_estimated": final_ksi,
            "ksi_true": true_ksi,
            "ksi_error_percent": ksi_error,
            "rmse_total": metrics.get("rmse_total", None),
            "stages_completed": history["stages_completed"],
            "model_type": model_type,
            "patient": patient,
        }

        if verbose:
            print(f"  ksi error: {ksi_error:.2f}%")

        return ksi_error, results

    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
            traceback.print_exc()
        return 100.0, {"error": str(e)}


class InverseObjective:
    """
    Optuna objective class for inverse optimization.

    This class maintains state across trials and handles:
    - Model type selection
    - Hyperparameter sampling
    - Training execution
    - Result logging

    Example:
        >>> objective = InverseObjective(patient=5, model_type="birnn")
        >>> study = optuna.create_study(direction="minimize")
        >>> study.optimize(objective, n_trials=100)
    """

    def __init__(
        self,
        patient: int,
        model_type: Optional[str] = None,
        data_root: str = "data/synthetic",
        verbose: bool = True,
        report_intermediate: bool = True,
    ):
        """
        Initialize the inverse objective.

        Args:
            patient: Patient number for training (2-11 for synthetic)
            model_type: If specified, only optimize this model type.
                       If None, model_type is sampled as a hyperparameter.
            data_root: Root directory for data files
            verbose: Whether to print progress
            report_intermediate: Whether to report intermediate values for pruning
        """
        self.patient = patient
        self.fixed_model_type = model_type
        self.data_root = data_root
        self.verbose = verbose
        self.report_intermediate = report_intermediate

        # Track best result
        self.best_error = float("inf")
        self.best_params = None

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function called by Optuna.

        Args:
            trial: Optuna trial object

        Returns:
            ksi estimation error (%) - lower is better
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
        search_space = SearchSpace(model_type=model_type, mode="inverse")
        params = search_space.sample(trial)

        if self.verbose:
            print("Sampled hyperparameters:")
            for key in ["stage1_epochs", "stage1_lr", "stage2_epochs", "stage2_lr",
                        "stage3_epochs", "stage3_lr"]:
                if key in params:
                    print(f"  {key}: {params[key]}")

        # Run training
        ksi_error, results = run_inverse_training(
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
        if ksi_error < self.best_error:
            self.best_error = ksi_error
            self.best_params = params.copy()
            if self.verbose:
                print(f"  NEW BEST: {ksi_error:.2f}%")

        return ksi_error


def create_inverse_study(
    study_name: str,
    storage: Optional[str] = None,
    direction: str = "minimize",
    pruner: Optional[optuna.pruners.BasePruner] = None,
    sampler: Optional[optuna.samplers.BaseSampler] = None,
    load_if_exists: bool = True,
) -> optuna.Study:
    """
    Create an Optuna study for inverse optimization.

    Args:
        study_name: Name of the study
        storage: SQLite URL (e.g., "sqlite:///results/optimization/study.db")
        direction: "minimize" for error minimization
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
    study_name: str = "inverse_optimization",
    storage: Optional[str] = None,
    n_jobs: int = 1,
    verbose: bool = True,
) -> optuna.Study:
    """
    Run inverse hyperparameter optimization.

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
    study = create_inverse_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    # Create objective
    objective = InverseObjective(
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
    print("Testing inverse objective...")

    study = run_optimization(
        n_trials=2,
        patient=5,
        model_type="birnn",  # Fix to BI-RNN for testing
        study_name="test_inverse",
        verbose=True,
    )

    print(f"\nBest trial:")
    print(f"  Value: {study.best_trial.value:.2f}%")
    print(f"  Params: {study.best_trial.params}")
