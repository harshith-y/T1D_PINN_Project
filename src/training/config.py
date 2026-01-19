"""
Configuration management for training and model hyperparameters.

This module provides utilities for loading, validating, and managing
configuration files (YAML) for model training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelArchitectureConfig:
    """Model architecture hyperparameters."""

    n_layers: int = 3
    n_neurons: int = 30
    activation: str = "tanh"
    use_fourier: bool = True
    fourier_features: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    # Modified MLP specific
    n_encoders: int = 2  # U and V pathways
    use_hard_ic: bool = True  # Use hard initial conditions (Modified-MLP/PINN)
    # BI-RNN specific
    rnn_units: int = 96
    rnn_type: str = "GRU"  # 'GRU' or 'LSTM'


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    epochs: int = 25000
    learning_rate: float = 5e-5
    optimizer: str = "adam"
    batch_size: Optional[int] = None  # None = full batch

    # Toggle for L-BFGS-B refinement
    use_lbfgs_refinement: bool = True

    # For inverse training
    stages: Optional[List[Dict[str, Any]]] = None


@dataclass
class LossWeightsConfig:
    """Loss function weights."""

    glucose: float = 3.32
    ode_glucose: float = 5.97
    ode_insulin: float = 4.70
    ode_digestion: float = 2.71
    ic: float = 0.0  # Initial condition loss
    # BI-RNN specific
    biological_residual: float = 4.82
    auxiliary: float = 0.53


@dataclass
class DataConfig:
    """Data loading and preprocessing settings."""

    data_dir: str = "data"
    source: str = "synthetic"  # 'synthetic' or 'processed'
    patient: str = "Pat3"
    train_split: float = 0.9
    mask_interval: int = 10  # Mask every Nth point for interpolation test
    normalize: bool = True


@dataclass
class OutputConfig:
    """Output and logging settings."""

    save_dir: str = "results"
    checkpoint_freq: int = 1000
    log_freq: int = 100
    save_best_only: bool = False
    wandb_project: Optional[str] = None  # For Weights & Biases logging


@dataclass
class Config:
    """Complete training configuration."""

    model_name: str  # 'pinn', 'modified_mlp', 'birnn'
    mode: str  # 'forward' or 'inverse'

    architecture: ModelArchitectureConfig = field(
        default_factory=ModelArchitectureConfig
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Inverse-specific
    inverse_param: Optional[str] = None  # 'ksi' or 'kl' or 'ku_Vi'
    inverse_init_range: Optional[List[float]] = None  # [min, max]

    # Device
    device: str = "cpu"  # 'cpu', 'cuda', or 'gpu'
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> Config:
        """Create config from dictionary."""
        # Extract top-level fields
        model_name = config_dict.get("model_name")
        mode = config_dict.get("mode", "forward")

        # Parse nested configs
        architecture = ModelArchitectureConfig(**config_dict.get("architecture", {}))
        training = TrainingConfig(**config_dict.get("training", {}))
        loss_weights = LossWeightsConfig(**config_dict.get("loss_weights", {}))
        data = DataConfig(**config_dict.get("data", {}))
        output = OutputConfig(**config_dict.get("output", {}))

        # Device and seed
        device = config_dict.get("device", "cpu")
        seed = config_dict.get("seed", 42)

        # Inverse-specific
        inverse_param = config_dict.get("inverse_param", None)
        inverse_init_range = config_dict.get("inverse_init_range", None)

        return cls(
            model_name=model_name,
            mode=mode,
            architecture=architecture,
            training=training,
            loss_weights=loss_weights,
            data=data,
            output=output,
            device=device,
            seed=seed,
            inverse_param=inverse_param,
            inverse_init_range=inverse_init_range,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_name": self.model_name,
            "mode": self.mode,
            "architecture": self.architecture.__dict__,
            "training": self.training.__dict__,
            "loss_weights": self.loss_weights.__dict__,
            "data": self.data.__dict__,
            "output": self.output.__dict__,
            "device": self.device,
            "seed": self.seed,
            "inverse_param": self.inverse_param,
            "inverse_init_range": self.inverse_init_range,
        }

    def save(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def validate(self) -> None:
        """Validate configuration settings."""
        # Check model name
        valid_models = ["pinn", "modified_mlp", "birnn"]
        if self.model_name not in valid_models:
            raise ValueError(
                f"Invalid model_name: {self.model_name}. Must be one of {valid_models}"
            )

        # Check mode
        valid_modes = ["forward", "inverse"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode: {self.mode}. Must be one of {valid_modes}")

        # Check epochs
        if self.training.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.training.epochs}")

        # Check learning rate
        if self.training.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.training.learning_rate}"
            )

        # Check Fourier features for non-PINN models
        if self.model_name == "modified_mlp" and self.architecture.use_fourier:
            print(
                "⚠️  Warning: Fourier features not recommended for Modified MLP (may degrade performance)"
            )

        if self.model_name == "birnn" and self.architecture.use_fourier:
            print(
                "⚠️  Warning: Fourier features not applicable to BI-RNN (will be ignored)"
            )

        # Check inverse mode settings
        if self.mode == "inverse":
            if self.inverse_param is None:
                raise ValueError("inverse_param must be specified for inverse mode")
            valid_inverse_params = ["ksi", "kl", "ku_Vi"]
            if self.inverse_param not in valid_inverse_params:
                raise ValueError(
                    f"Invalid inverse_param: {self.inverse_param}. Must be one of {valid_inverse_params}"
                )

        print("✅ Configuration validated successfully")


def get_default_config(model_name: str, mode: str = "forward") -> Config:
    """
    Get default configuration for a specific model and mode.

    Args:
        model_name: 'pinn', 'modified_mlp', or 'birnn'
        mode: 'forward' or 'inverse'

    Returns:
        Config object with default settings
    """
    if model_name == "pinn":
        return Config(
            model_name="pinn",
            mode=mode,
            architecture=ModelArchitectureConfig(
                n_layers=3,
                n_neurons=30,
                use_fourier=True,
                fourier_features=[0, 1, 2, 3, 4, 5],
            ),
            training=TrainingConfig(epochs=25000, learning_rate=5e-5),
            loss_weights=LossWeightsConfig(
                glucose=3.32,
                ode_glucose=5.97,
                ode_insulin=4.70,
                ode_digestion=2.71,
                ic=0.0,
            ),
        )

    elif model_name == "modified_mlp":
        return Config(
            model_name="modified_mlp",
            mode=mode,
            architecture=ModelArchitectureConfig(
                n_layers=5,
                n_neurons=30,
                use_fourier=False,  # Degraded performance in your experiments
                n_encoders=2,
            ),
            training=TrainingConfig(epochs=25000, learning_rate=5e-4),
            loss_weights=LossWeightsConfig(
                glucose=0.89,
                ode_glucose=5.84,
                ode_insulin=8.53,
                ode_digestion=9.91,
                ic=0.0,
            ),
        )

    elif model_name == "birnn":
        return Config(
            model_name="birnn",
            mode=mode,
            architecture=ModelArchitectureConfig(
                rnn_units=96, rnn_type="GRU", use_fourier=False  # Not applicable to RNN
            ),
            training=TrainingConfig(epochs=500, learning_rate=1e-2),
            loss_weights=LossWeightsConfig(
                glucose=8.0, biological_residual=4.82, ic=0.53
            ),
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_config(
    config_path: Optional[str | Path] = None,
    model_name: Optional[str] = None,
    mode: str = "forward",
    **overrides,
) -> Config:
    """
    Load configuration with optional overrides.

    Args:
        config_path: Path to YAML config file (if None, use defaults)
        model_name: Model name (required if config_path is None)
        mode: 'forward' or 'inverse'
        **overrides: Additional overrides as keyword arguments

    Returns:
        Config object

    Examples:
        # Load from file
        config = load_config('configs/pinn_forward.yaml')

        # Load defaults
        config = load_config(model_name='pinn', mode='forward')

        # Load with overrides
        config = load_config(
            'configs/pinn_forward.yaml',
            epochs=10000,
            learning_rate=1e-4
        )
    """
    # Load base config
    if config_path is not None:
        config = Config.from_yaml(config_path)
    elif model_name is not None:
        config = get_default_config(model_name, mode)
    else:
        raise ValueError("Must provide either config_path or model_name")

    # Apply overrides
    for key, value in overrides.items():
        # Handle nested attributes (e.g., 'training.epochs')
        if "." in key:
            parts = key.split(".")
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        else:
            setattr(config, key, value)

    # Validate
    config.validate()

    return config


# Example usage
if __name__ == "__main__":
    # Test loading default configs
    print("Testing default configs...\n")

    for model in ["pinn", "modified_mlp", "birnn"]:
        print(f"--- {model.upper()} ---")
        config = get_default_config(model, mode="forward")
        print(f"Epochs: {config.training.epochs}")
        print(f"Learning rate: {config.training.learning_rate}")
        print(f"Use Fourier: {config.architecture.use_fourier}")
        print()

    # Test saving and loading
    print("Testing save/load...")
    config = get_default_config("pinn")
    config.save("/tmp/test_config.yaml")
    loaded_config = Config.from_yaml("/tmp/test_config.yaml")
    print("✅ Save/load successful")

    # Test overrides
    print("\nTesting overrides...")
    config = load_config(
        model_name="pinn", mode="forward", epochs=10000, learning_rate=1e-4
    )
    print(f"Epochs: {config.training.epochs}")
    print(f"Learning rate: {config.training.learning_rate}")
    print("✅ Overrides successful")
