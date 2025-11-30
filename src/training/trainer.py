"""
Unified training orchestrator for all model architectures.

This module provides a single training interface that works with:
- Feedforward PINN (DeepXDE + TF1.x)
- Modified-MLP PINN (DeepXDE + TF1.x)
- BI-RNN (TensorFlow 2.x)

Supports:
- Forward training (single-stage)
- Inverse training (multi-stage with configurable loss weights and learning rates)
- Flexible stage configuration via YAML
- Progress tracking and checkpoint saving
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import tensorflow as tf

if TYPE_CHECKING:
    from src.models.birnn import BIRNN
    from src.models.modified_mlp import ModifiedMLPPINN
    from src.models.pinn_feedforward import FeedforwardPINN


@dataclass
class TrainingStage:
    """
    Configuration for a single training stage (used in inverse training).

    Attributes:
        name: Human-readable stage name (e.g., "stage1_params_only")
        epochs: Number of training epochs
        learning_rate: Learning rate for this stage
        loss_weights: List of loss weights (length depends on model)
        train_inverse_params: Whether to train inverse parameters (ksi, etc.)
        train_nn_weights: Whether to train neural network weights
        optimizer: Optimizer name (default "adam")
        display_every: Print loss every N epochs (default 500)
    """

    name: str
    epochs: int
    learning_rate: float
    loss_weights: List[float]
    train_inverse_params: bool
    train_nn_weights: bool
    optimizer: str = "adam"
    display_every: int = 500


class UnifiedTrainer:
    """
    Unified training orchestrator for all model architectures.

    This class provides a consistent training interface across:
    - Feedforward PINN (DeepXDE)
    - Modified-MLP PINN (DeepXDE)
    - BI-RNN (TensorFlow 2.x)

    It handles both forward and inverse training modes with multi-stage support.

    Example:
        >>> # Forward training
        >>> trainer = UnifiedTrainer(model, config)
        >>> trainer.train()

        >>> # Inverse training (3-stage)
        >>> config.mode = 'inverse'
        >>> config.training.stages = [...]  # Define stages
        >>> trainer = UnifiedTrainer(model, config)
        >>> trainer.train()
    """

    def __init__(
        self,
        model: FeedforwardPINN | ModifiedMLPPINN | BIRNN,
        config,
        save_dir: Optional[str | Path] = None,
    ):
        """
        Initialize the unified trainer.

        Args:
            model: One of the three model types (must be built and compiled)
            config: Configuration object with training settings
            save_dir: Directory for saving outputs (default from config)
        """
        self.model = model
        self.config = config
        self.save_dir = Path(save_dir) if save_dir else Path(config.output.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Determine model type
        self.model_type = self._infer_model_type()

        # Training history
        self.history: Dict[str, List[Any]] = {"stages": [], "losses": [], "metrics": []}

        print(f"\n{'='*80}")
        print(f"UNIFIED TRAINER INITIALIZED")
        print(f"{'='*80}")
        print(f"Model: {self.model_type}")
        print(f"Mode: {config.mode}")
        print(f"Save dir: {self.save_dir}")
        print(f"{'='*80}\n")

    def _infer_model_type(self) -> str:
        """Infer model type from class name."""
        class_name = self.model.__class__.__name__
        if "Feedforward" in class_name or "FeedforwardPINN" in class_name:
            return "pinn"
        elif "ModifiedMLP" in class_name:
            return "modified_mlp"
        elif "BIRNN" in class_name or "GRU" in class_name:
            return "birnn"
        else:
            raise ValueError(f"Unknown model type: {class_name}")

    def train(self) -> Dict[str, Any]:
        """
        Main training entry point.

        Dispatches to forward or inverse training based on config.mode.

        Returns:
            Training history dictionary
        """
        if self.config.mode == "forward":
            return self._train_forward()
        elif self.config.mode == "inverse":
            return self._train_inverse()
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")

    def _train_forward(self) -> Dict[str, Any]:
        """
        Forward training (single stage).

        This is the standard training mode where all parameters are trained
        together from the start.
        """
        print(f"\n{'='*80}")
        print(f"FORWARD TRAINING")
        print(f"{'='*80}")
        print(f"Epochs: {self.config.training.epochs}")
        print(f"Learning rate: {self.config.training.learning_rate}")
        print(f"{'='*80}\n")

        start_time = time.time()

        # Use the model's built-in train method
        # Models already have compile() called with correct settings
        if self.model_type in ["pinn", "modified_mlp"]:
            # DeepXDE models
            self.model.train(display_every=self.config.output.log_freq)
        else:
            # BI-RNN
            self.model.train(display_every=self.config.output.log_freq)

        elapsed = time.time() - start_time

        # Record results
        self.history["stages"].append(
            {"name": "forward", "epochs": self.config.training.epochs, "time": elapsed}
        )

        print(f"\n✅ Forward training complete ({elapsed:.1f}s)")

        # Save history
        self._save_history()

        return self.history

    def _train_inverse(self) -> Dict[str, Any]:
        """
        Inverse training (multi-stage).

        This implements the 3-stage training strategy:
        1. Train inverse parameters only (freeze NN weights)
        2. Train NN weights only (freeze inverse parameters)
        3. Joint fine-tuning (train both together)

        Stages are configured via config.training.stages list.
        """
        if not self.config.training.stages:
            raise ValueError(
                "Inverse training requires config.training.stages to be defined.\n"
                "See docs/inverse_training.md for configuration examples."
            )

        print(f"\n{'='*80}")
        print(f"INVERSE TRAINING ({len(self.config.training.stages)} stages)")
        print(f"{'='*80}")
        print(f"Inverse parameter: {self.config.inverse_param}")
        print(f"{'='*80}\n")

        # Parse stages from config
        stages = self._parse_stages_from_config()

        # Execute each stage
        total_start = time.time()
        for i, stage in enumerate(stages, 1):
            print(f"\n{'='*80}")
            print(f"STAGE {i}/{len(stages)}: {stage.name}")
            print(f"{'='*80}")
            print(f"Epochs: {stage.epochs}")
            print(f"Learning rate: {stage.learning_rate}")
            print(f"Loss weights: {stage.loss_weights}")
            print(f"Train inverse params: {stage.train_inverse_params}")
            print(f"Train NN weights: {stage.train_nn_weights}")
            print(f"{'='*80}\n")

            stage_start = time.time()

            # Execute stage based on model type
            if self.model_type in ["pinn", "modified_mlp"]:
                self._train_stage_deepxde(stage)
            else:
                self._train_stage_birnn(stage)

            stage_time = time.time() - stage_start

            # Record stage results
            self.history["stages"].append(
                {
                    "name": stage.name,
                    "epochs": stage.epochs,
                    "time": stage_time,
                    "lr": stage.learning_rate,
                    "train_inverse": stage.train_inverse_params,
                    "train_nn": stage.train_nn_weights,
                }
            )

            print(f"\n✅ Stage {i} complete ({stage_time:.1f}s)")

            # Save checkpoint after each stage
            self._save_checkpoint(stage_name=stage.name)

        total_time = time.time() - total_start

        print(f"\n{'='*80}")
        print(f"INVERSE TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Stages completed: {len(stages)}")

        # Get final inverse parameter values
        self._log_inverse_parameters()

        # Save final history
        self._save_history()

        return self.history

    def _parse_stages_from_config(self) -> List[TrainingStage]:
        """Parse stage configurations from config.training.stages."""
        stages = []

        for stage_dict in self.config.training.stages:
            stage = TrainingStage(
                name=stage_dict["name"],
                epochs=stage_dict["epochs"],
                learning_rate=stage_dict["learning_rate"],
                loss_weights=stage_dict["loss_weights"],
                train_inverse_params=stage_dict.get("train_inverse_params", False),
                train_nn_weights=stage_dict.get("train_nn_weights", True),
                optimizer=stage_dict.get("optimizer", "adam"),
                display_every=stage_dict.get("display_every", 500),
            )
            stages.append(stage)

        return stages

    def _train_stage_deepxde(self, stage: TrainingStage) -> None:
        """
        Execute a training stage for DeepXDE models (PINN, Modified-MLP).

        DeepXDE requires recompiling the model for each stage to change:
        - Learning rate
        - Loss weights
        - External trainable variables
        """
        import deepxde as dde

        # Determine external trainable variables
        external_vars = None
        if stage.train_inverse_params:
            if (
                self.model.inverse_params
                and self.model.inverse_params.log_ksi is not None
            ):
                external_vars = [self.model.inverse_params.log_ksi]

        # Set up parameter logging callback
        callbacks = []
        if external_vars:
            param_log_path = self.save_dir / f"{stage.name}_params.dat"
            param_logger = dde.callbacks.VariableValue(
                external_vars, period=stage.display_every, filename=str(param_log_path)
            )
            callbacks.append(param_logger)

        # Recompile model with stage-specific settings
        self.model.model.compile(
            stage.optimizer,
            lr=stage.learning_rate,
            loss_weights=stage.loss_weights,
            external_trainable_variables=external_vars,
        )

        # Train for this stage
        losshistory, train_state = self.model.model.train(
            epochs=stage.epochs,
            display_every=stage.display_every,
            callbacks=callbacks if callbacks else None,
        )

        # Store loss history
        self.history["losses"].append(
            {
                "stage": stage.name,
                "loss_train": losshistory.loss_train,
                "loss_test": losshistory.loss_test,
                "steps": losshistory.steps,
            }
        )

    def _train_stage_birnn(self, stage: TrainingStage) -> None:
        """
        Execute a training stage for BI-RNN model.

        BI-RNN uses TensorFlow 2.x, so we control which variables are trainable
        directly via the optimizer's var_list parameter.
        """
        # Get all trainable variables
        all_vars = self.model.model.trainable_variables

        # Separate inverse parameters from NN weights
        inverse_vars = []
        nn_vars = []

        if hasattr(self.model, "log_ksi") and self.model.log_ksi is not None:
            # Inverse parameters exist
            inverse_vars.append(self.model.log_ksi)
            nn_vars = [v for v in all_vars]  # NN weights are all GRU variables
        else:
            # No inverse parameters
            nn_vars = all_vars

        # Determine which variables to train this stage
        train_vars = []
        if stage.train_inverse_params and inverse_vars:
            train_vars.extend(inverse_vars)
        if stage.train_nn_weights:
            train_vars.extend(nn_vars)

        if not train_vars:
            raise ValueError(f"Stage {stage.name}: No variables to train!")

        # Create optimizer for this stage
        optimizer = tf.keras.optimizers.Adam(learning_rate=stage.learning_rate)

        # Update model's optimizer and loss weights
        self.model.optimizer = optimizer

        # Initialize lambda weights if they don't exist
        if not hasattr(self.model, "lambda_g"):
            self.model.lambda_g = 8.0
        if not hasattr(self.model, "lambda_B"):
            self.model.lambda_B = 4.82
        if not hasattr(self.model, "lambda_ic"):
            self.model.lambda_ic = 0.53

        # Update with stage-specific values
        self.model.lambda_g = (
            stage.loss_weights[0] if len(stage.loss_weights) > 0 else 8.0
        )
        self.model.lambda_B = (
            stage.loss_weights[1] if len(stage.loss_weights) > 1 else 4.82
        )
        self.model.lambda_ic = (
            stage.loss_weights[2] if len(stage.loss_weights) > 2 else 0.53
        )

        # Custom training loop for this stage
        print(f"Training {len(train_vars)} variables:")
        if inverse_vars and stage.train_inverse_params:
            print(f"  - Inverse parameters: {len(inverse_vars)}")
        if stage.train_nn_weights:
            print(f"  - NN weights: {len(nn_vars)}")

        # Run training for this stage
        # We need to modify the BI-RNN's _train_step to accept var_list
        for epoch in range(stage.epochs):
            # Training step with specific var_list
            loss_dict = self._birnn_train_step_with_varlist(train_vars)

            # Log progress
            if epoch % stage.display_every == 0:
                print(
                    f"Epoch {epoch}/{stage.epochs}: "
                    + ", ".join([f"{k}={v:.4f}" for k, v in loss_dict.items()])
                )

                # Log inverse parameter if training it
                if inverse_vars and stage.train_inverse_params:
                    ksi_val = float(tf.exp(self.model.params.log_ksi))
                    print(f"  ksi = {ksi_val:.2f}")

        # Store final losses
        self.history["losses"].append({"stage": stage.name, "final_losses": loss_dict})

    def _birnn_train_step_with_varlist(
        self, var_list: List[tf.Variable]
    ) -> Dict[str, float]:
        """
        Execute one training step for BI-RNN with specific variable list.

        This is a modified version of the BI-RNN's _train_step that only
        updates the specified variables.
        """
        with tf.GradientTape() as tape:
            # CRITICAL: Manually watch the variables we want to train
            for var in var_list:
                tape.watch(var)

            # Forward pass using training data
            Y_pred = self.model.model(self.model.X_train, training=True)

            # Compute losses (using model's existing loss computation)
            loss_g = tf.reduce_mean(
                tf.square(Y_pred[:, :, 0:1] - self.model.Y_train[:, :, 0:1])
            )

            # Biological residual loss (if lambda_B > 0)
            if self.model.lambda_B > 0:
                # Denormalize predictions for physics computation
                G_pred = Y_pred[:, :, 0:1] * self.model.data_window.m_g
                G_seq = tf.squeeze(G_pred, axis=0)  # [T]

                # Get denormalized inputs
                U_seq = tf.squeeze(self.model.U_train, axis=0) * self.model.u_max
                R_seq = tf.squeeze(self.model.R_train, axis=0) * self.model.r_max

                from src.physics.magdelaine import residuals_euler_seq

                # Use current ksi if training it
                if hasattr(self.model, "log_ksi") and self.model.log_ksi is not None:
                    # Create temporary params with updated ksi
                    import copy

                    temp_params = copy.copy(self.model.params)
                    temp_params.ksi = float(tf.exp(self.model.log_ksi))
                    params_to_use = temp_params
                else:
                    params_to_use = self.model.params

                bio_losses = residuals_euler_seq(
                    G_seq, U_seq, R_seq, params_to_use, dt=1.0, use_latent_sim=True
                )
                loss_B = tf.reduce_mean(tf.square(bio_losses["LB"]))
            else:
                loss_B = 0.0

            # Initial condition loss
            if self.model.lambda_ic > 0:
                loss_ic = tf.reduce_mean(
                    tf.square(Y_pred[:, 0, :] - self.model.Y_train[:, 0, :])
                )
            else:
                loss_ic = 0.0

            # Total loss
            total_loss = (
                self.model.lambda_g * loss_g
                + self.model.lambda_B * loss_B
                + self.model.lambda_ic * loss_ic
            )

        # Compute gradients only for specified variables
        gradients = tape.gradient(total_loss, var_list)

        # Apply gradients
        self.model.optimizer.apply_gradients(zip(gradients, var_list))

        return {
            "total": float(total_loss),
            "loss_g": float(loss_g),
            "loss_B": float(loss_B) if isinstance(loss_B, tf.Tensor) else loss_B,
            "loss_ic": float(loss_ic) if isinstance(loss_ic, tf.Tensor) else loss_ic,
        }

    def _log_inverse_parameters(self) -> None:
        """Log final inverse parameter values."""
        print(f"\n{'='*80}")
        print("FINAL INVERSE PARAMETERS")
        print(f"{'='*80}")

        if self.model_type in ["pinn", "modified_mlp"]:
            if (
                self.model.inverse_params
                and self.model.inverse_params.log_ksi is not None
            ):
                import tensorflow as tf

                ksi_val = float(tf.exp(self.model.inverse_params.log_ksi).numpy())
                print(f"ksi = {ksi_val:.2f}")

                # Get true value for comparison
                true_ksi = self.model.params.ksi
                error = abs(ksi_val - true_ksi) / true_ksi * 100
                print(f"True ksi = {true_ksi:.2f}")
                print(f"Error = {error:.2f}%")

        elif self.model_type == "birnn":
            if hasattr(self.model, "log_ksi") and self.model.log_ksi is not None:
                ksi_val = float(tf.exp(self.model.log_ksi).numpy())
                print(f"ksi = {ksi_val:.2f}")

                # Get true value
                true_params = {
                    2: 197,
                    3: 274,
                    4: 191,
                    5: 282,
                    6: 203,
                    7: 267,
                    8: 200,
                    9: 272,
                    10: 191,
                    11: 282,
                }
                # Extract patient number from data_window
                try:
                    patient_num = int(
                        self.model.data_window.patient_id.replace("Pat", "")
                    )
                    if patient_num in true_params:
                        true_ksi = true_params[patient_num]
                        error = abs(ksi_val - true_ksi) / true_ksi * 100
                        print(f"True ksi = {true_ksi:.2f}")
                        print(f"Error = {error:.2f}%")
                except:
                    pass

        print(f"{'='*80}\n")

    def _save_checkpoint(self, stage_name: str) -> None:
        """Save model checkpoint after a stage."""
        checkpoint_dir = self.save_dir / "checkpoints" / stage_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        try:
            if self.model_type in ["pinn", "modified_mlp"]:
                # DeepXDE models
                self.model.save(checkpoint_dir / "model")
            else:
                # BI-RNN
                self.model.save(checkpoint_dir)

            print(f"💾 Checkpoint saved: {checkpoint_dir}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save checkpoint: {e}")

    def _save_history(self) -> None:
        """Save training history to JSON."""
        history_path = self.save_dir / "training_history.json"

        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                history_serializable[key] = []
                for item in value:
                    if isinstance(item, dict):
                        item_copy = {}
                        for k, v in item.items():
                            if isinstance(v, np.ndarray):
                                item_copy[k] = v.tolist()
                            else:
                                item_copy[k] = v
                        history_serializable[key].append(item_copy)
                    else:
                        history_serializable[key].append(item)
            else:
                history_serializable[key] = value

        with open(history_path, "w") as f:
            json.dump(history_serializable, f, indent=2)

        print(f"💾 Training history saved: {history_path}")

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model after training.

        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\n{'='*80}")
        print("EVALUATION")
        print(f"{'='*80}\n")

        metrics = self.model.evaluate()

        # Store in history
        self.history["metrics"].append(metrics)

        return metrics


# Example usage and testing
if __name__ == "__main__":
    print("UnifiedTrainer created successfully!")
    print("\nUsage example:")
    print(
        """
    from src.training.trainer import UnifiedTrainer
    from src.models.pinn_feedforward import FeedforwardPINN
    from src.training.config import load_config
    from src.datasets.loader import load_synthetic_window
    
    # Load config with inverse training stages
    config = load_config('configs/pinn_inverse.yaml')
    
    # Load data
    data = load_synthetic_window(patient=3)
    
    # Build model
    model = FeedforwardPINN(config)
    model.build(data)
    model.compile()
    
    # Train with unified trainer
    trainer = UnifiedTrainer(model, config)
    history = trainer.train()
    
    # Evaluate
    metrics = trainer.evaluate()
    """
    )
