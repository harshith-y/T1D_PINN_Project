"""
Search space definitions for hyperparameter optimization.

This module defines the search spaces for Optuna optimization across:
- BI-RNN architecture
- PINN (Feedforward) architecture
- Modified-MLP architecture

Each model has both forward and inverse training search spaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import optuna


@dataclass
class SearchSpace:
    """
    Defines hyperparameter search space for a model.

    Attributes:
        model_type: One of 'birnn', 'pinn', 'modified_mlp'
        mode: 'forward' or 'inverse'
        quick_test: If True, use minimal epochs for local testing
    """

    model_type: str
    mode: str
    quick_test: bool = False

    def sample(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sample hyperparameters from this search space.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of sampled hyperparameters
        """
        params = {"model_type": self.model_type, "mode": self.mode}

        # Sample architecture parameters
        params.update(self._sample_architecture(trial))

        # Sample training parameters
        if self.mode == "forward":
            params.update(self._sample_forward_training(trial))
        else:
            params.update(self._sample_inverse_training(trial))

        return params

    def _sample_architecture(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample architecture-specific parameters."""
        if self.model_type == "birnn":
            return {
                "gru_units": trial.suggest_categorical(
                    "gru_units", [32, 64, 96, 128]
                ),
                "n_dense_layers": trial.suggest_categorical(
                    "n_dense_layers", [1, 2, 3]
                ),
            }
        elif self.model_type == "pinn":
            return {
                "n_layers": trial.suggest_categorical("n_layers", [3, 4, 5, 6]),
                "n_neurons": trial.suggest_categorical("n_neurons", [20, 30, 40, 50]),
                "use_fourier": trial.suggest_categorical("use_fourier", [True, False]),
            }
        elif self.model_type == "modified_mlp":
            return {
                "n_layers": trial.suggest_categorical("n_layers", [4, 5, 6, 7]),
                "n_neurons": trial.suggest_categorical("n_neurons", [20, 30, 40]),
                "n_encoders": trial.suggest_categorical("n_encoders", [2, 3, 4]),
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _sample_forward_training(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample forward training parameters."""
        if self.quick_test:
            # Minimal epochs for local testing
            epoch_choices = [50, 100, 200]
        else:
            epoch_choices = [5000, 10000, 15000, 20000, 25000]

        params = {
            "epochs": trial.suggest_categorical("epochs", epoch_choices),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-5, 1e-2, log=True
            ),
        }

        # Model-specific loss weights
        if self.model_type == "birnn":
            params.update({
                "lambda_g": trial.suggest_float("lambda_g", 1.0, 15.0),
                "lambda_B": trial.suggest_float("lambda_B", 1.0, 12.0),
                "lambda_ic": trial.suggest_float("lambda_ic", 0.1, 3.0),
            })
        else:  # PINN / Modified-MLP
            params.update({
                "w_glucose_obs": trial.suggest_float("w_glucose_obs", 1.0, 10.0),
                "w_ode_g": trial.suggest_float("w_ode_g", 1.0, 12.0),
                "w_ode_i": trial.suggest_float("w_ode_i", 1.0, 12.0),
                "w_ode_d": trial.suggest_float("w_ode_d", 0.1, 3.0),
            })

        # L-BFGS refinement (for DeepXDE models)
        if self.model_type in ["pinn", "modified_mlp"]:
            params["use_lbfgs"] = trial.suggest_categorical("use_lbfgs", [True, False])
            if params["use_lbfgs"]:
                params["lbfgs_epochs"] = trial.suggest_categorical(
                    "lbfgs_epochs", [500, 1000, 2000]
                )

        return params

    def _sample_inverse_training(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample inverse (curriculum) training parameters."""
        params = {}

        if self.quick_test:
            # Minimal epochs for local testing
            stage1_choices = [50, 100]
            stage2_choices = [50, 100]
            stage3_choices = [50, 100]
        else:
            stage1_choices = [2000, 3000, 5000, 8000]
            stage2_choices = [1000, 2000, 3000, 5000]
            stage3_choices = [1000, 2000, 3000]

        # Stage 1: Parameters only
        params["stage1_epochs"] = trial.suggest_categorical(
            "stage1_epochs", stage1_choices
        )
        params["stage1_lr"] = trial.suggest_float(
            "stage1_lr", 5e-4, 2e-3, log=True
        )

        # Stage 2: NN only
        params["stage2_epochs"] = trial.suggest_categorical(
            "stage2_epochs", stage2_choices
        )
        params["stage2_lr"] = trial.suggest_float(
            "stage2_lr", 1e-4, 1e-3, log=True
        )

        # Stage 3: Joint
        params["stage3_epochs"] = trial.suggest_categorical(
            "stage3_epochs", stage3_choices
        )
        params["stage3_lr"] = trial.suggest_float(
            "stage3_lr", 5e-5, 5e-4, log=True
        )

        # Loss weights (vary by model type)
        if self.model_type == "birnn":
            params.update({
                "lambda_g": trial.suggest_float("lambda_g", 1.0, 15.0),
                "lambda_B": trial.suggest_float("lambda_B", 1.0, 12.0),
                "lambda_ic": trial.suggest_float("lambda_ic", 0.1, 3.0),
            })
        else:  # PINN / Modified-MLP
            params.update({
                "w_glucose_obs": trial.suggest_float("w_glucose_obs", 1.0, 10.0),
                "w_ode_g": trial.suggest_float("w_ode_g", 1.0, 12.0),
                "w_ode_i": trial.suggest_float("w_ode_i", 1.0, 12.0),
                "w_ode_d": trial.suggest_float("w_ode_d", 0.1, 3.0),
            })

        # L-BFGS refinement (for DeepXDE models)
        if self.model_type in ["pinn", "modified_mlp"]:
            params["use_lbfgs"] = trial.suggest_categorical("use_lbfgs", [True, False])
            if params["use_lbfgs"]:
                params["lbfgs_epochs"] = trial.suggest_categorical(
                    "lbfgs_epochs", [500, 1000, 2000]
                )

        return params


def get_search_space(model_type: str, mode: str) -> SearchSpace:
    """
    Get search space for a specific model and mode.

    Args:
        model_type: One of 'birnn', 'pinn', 'modified_mlp'
        mode: 'forward' or 'inverse'

    Returns:
        SearchSpace object
    """
    valid_models = ["birnn", "pinn", "modified_mlp"]
    valid_modes = ["forward", "inverse"]

    if model_type not in valid_models:
        raise ValueError(f"model_type must be one of {valid_models}, got {model_type}")
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got {mode}")

    return SearchSpace(model_type=model_type, mode=mode)


def params_to_config_dict(params: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """
    Convert Optuna parameters to a config dictionary.

    Args:
        params: Dictionary of sampled hyperparameters
        mode: 'forward' or 'inverse'

    Returns:
        Config dictionary compatible with Config.from_dict()
    """
    model_type = params["model_type"]

    # Base config
    config = {
        "model_name": model_type,
        "mode": mode,
        "device": "cpu",
        "seed": 42,
    }

    # Architecture config
    if model_type == "birnn":
        config["architecture"] = {
            "rnn_units": params.get("gru_units", 96),
            "rnn_type": "GRU",
            "use_fourier": False,
        }
    elif model_type == "pinn":
        config["architecture"] = {
            "n_layers": params.get("n_layers", 3),
            "n_neurons": params.get("n_neurons", 30),
            "use_fourier": params.get("use_fourier", True),
            "activation": "tanh",
        }
    elif model_type == "modified_mlp":
        config["architecture"] = {
            "n_layers": params.get("n_layers", 5),
            "n_neurons": params.get("n_neurons", 30),
            "n_encoders": params.get("n_encoders", 2),
            "use_fourier": False,
        }

    # Training config
    if mode == "forward":
        config["training"] = {
            "epochs": params.get("epochs", 10000),
            "learning_rate": params.get("learning_rate", 1e-3),
            "optimizer": "adam",
            "use_lbfgs_refinement": params.get("use_lbfgs", False),
        }
    else:  # inverse
        # Build stages for curriculum training
        stages = _build_inverse_stages(params, model_type)
        config["training"] = {
            "stages": stages,
            "use_lbfgs_refinement": params.get("use_lbfgs", False),
        }
        config["inverse_param"] = "ksi"
        config["inverse_init_range"] = [150, 300]

    # Loss weights
    if model_type == "birnn":
        config["loss_weights"] = {
            "glucose": params.get("lambda_g", 8.0),
            "biological_residual": params.get("lambda_B", 4.82),
            "auxiliary": params.get("lambda_ic", 0.53),
        }
    else:
        config["loss_weights"] = {
            "glucose": params.get("w_glucose_obs", 3.32),
            "ode_glucose": params.get("w_ode_g", 5.97),
            "ode_insulin": params.get("w_ode_i", 4.70),
            "ode_digestion": params.get("w_ode_d", 2.71),
        }

    # Output config
    config["output"] = {
        "save_dir": "results/optimization",
        "checkpoint_freq": 1000,
        "log_freq": 500,
        "save_best_only": True,
    }

    return config


def _build_inverse_stages(params: Dict[str, Any], model_type: str) -> List[Dict]:
    """Build curriculum training stages from parameters."""
    if model_type == "birnn":
        loss_weights = [
            params.get("lambda_g", 8.0),
            params.get("lambda_B", 4.82),
            params.get("lambda_ic", 0.53),
        ]
    else:
        loss_weights = [
            params.get("w_glucose_obs", 3.32),
            params.get("w_ode_g", 5.97),
            params.get("w_ode_i", 4.70),
            params.get("w_ode_d", 2.71),
        ]

    stages = [
        {
            "name": "stage1_params_only",
            "epochs": params.get("stage1_epochs", 3000),
            "learning_rate": params.get("stage1_lr", 1e-3),
            "loss_weights": [1.0, 0.0, 0.0] if model_type == "birnn" else [1.0, 0.0, 0.0, 0.0],
            "train_inverse_params": True,
            "train_nn_weights": False,
            "optimizer": "adam",
            "display_every": 500,
        },
        {
            "name": "stage2_nn_only",
            "epochs": params.get("stage2_epochs", 2000),
            "learning_rate": params.get("stage2_lr", 5e-4),
            "loss_weights": loss_weights,
            "train_inverse_params": False,
            "train_nn_weights": True,
            "optimizer": "adam",
            "display_every": 500,
        },
        {
            "name": "stage3_joint",
            "epochs": params.get("stage3_epochs", 2000),
            "learning_rate": params.get("stage3_lr", 1e-4),
            "loss_weights": loss_weights,
            "train_inverse_params": True,
            "train_nn_weights": True,
            "optimizer": "adam",
            "display_every": 500,
        },
    ]

    return stages


def get_model_search_spaces() -> Dict[str, SearchSpace]:
    """
    Get all search spaces indexed by model type.

    Returns:
        Dictionary mapping model_type to SearchSpace
    """
    return {
        "birnn": SearchSpace("birnn", "inverse"),
        "pinn": SearchSpace("pinn", "inverse"),
        "modified_mlp": SearchSpace("modified_mlp", "inverse"),
    }


# Example usage
if __name__ == "__main__":
    import optuna

    # Create a study and sample parameters
    study = optuna.create_study(direction="minimize")

    def objective(trial):
        # Sample model type
        model_type = trial.suggest_categorical(
            "model_type", ["birnn", "pinn", "modified_mlp"]
        )

        # Get search space for this model
        search_space = get_search_space(model_type, mode="inverse")

        # Sample hyperparameters
        params = search_space.sample(trial)

        print(f"\nSampled parameters for {model_type}:")
        for k, v in params.items():
            print(f"  {k}: {v}")

        # Convert to config dict
        config_dict = params_to_config_dict(params, mode="inverse")
        print(f"\nConfig dict stages: {len(config_dict['training']['stages'])}")

        # Return dummy loss
        return 0.5

    # Run a few trials
    study.optimize(objective, n_trials=3, show_progress_bar=False)
