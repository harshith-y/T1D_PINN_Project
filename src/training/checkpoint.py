"""
Enhanced checkpoint management for model training.

Provides CheckpointManager class that handles:
- Saving/loading model weights
- Saving/loading optimizer state (for smooth resume)
- Tracking epoch numbers
- Tracking best metrics
- Multiple checkpoint types (best, final, interrupted)
- Complete metadata preservation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf


class CheckpointManager:
    """
    Manage model checkpoints with full state preservation.

    Supports:
    - Model weights (Keras .h5 format)
    - Optimizer state (for resume capability)
    - Training state (epoch, best loss, etc.)
    - Configuration (for reproducibility)
    - Metadata (timestamps, flags, etc.)

    Example:
        >>> manager = CheckpointManager('results/birnn_pat3', config)
        >>> # After training epoch
        >>> manager.save(model, optimizer, epoch=1000, metrics={'loss': 0.5}, is_best=True)
        >>> # Resume training
        >>> state = manager.load(model, optimizer, checkpoint_type='best')
        >>> start_epoch = state['epoch']
    """

    def __init__(
        self, save_dir: str | Path, config: Any = None, create_dirs: bool = True
    ):
        """
        Initialize checkpoint manager.

        Args:
            save_dir: Base directory for checkpoints
            config: Optional config object to save alongside checkpoints
            create_dirs: Whether to create directories immediately
        """
        self.save_dir = Path(save_dir)
        self.checkpoints_dir = self.save_dir / "checkpoints"
        self.config = config

        if create_dirs:
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: Any,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        is_final: bool = False,
        is_interrupted: bool = False,
        checkpoint_name: Optional[str] = None,
    ) -> Path:
        """
        Save checkpoint with full state.

        Args:
            model: Model with .save_weights() method or .model attribute
            optimizer: Optional optimizer (for resume capability)
            epoch: Current epoch number
            metrics: Dictionary of metrics to save
            is_best: Whether this is the best checkpoint so far
            is_final: Whether this is the final checkpoint
            is_interrupted: Whether this is an interruption checkpoint
            checkpoint_name: Custom checkpoint name (e.g., 'stage1_complete')

        Returns:
            Path to saved checkpoint directory
        """
        # Determine checkpoint directory name
        if checkpoint_name:
            checkpoint_dir = self.checkpoints_dir / checkpoint_name
        elif is_best:
            checkpoint_dir = self.checkpoints_dir / "best"
        elif is_final:
            checkpoint_dir = self.checkpoints_dir / "final"
        elif is_interrupted:
            checkpoint_dir = self.checkpoints_dir / "interrupted"
        else:
            checkpoint_dir = self.checkpoints_dir / f"epoch_{epoch:06d}"

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights (handle both old and new Keras formats)
        try:
            if hasattr(model, "model"):
                # BI-RNN style: model.model is the Keras model
                keras_model = model.model
            else:
                # Direct Keras model
                keras_model = model

            # Try new format first (.weights.h5), fall back to old format (.h5)
            weights_path_new = checkpoint_dir / "model.weights.h5"
            weights_path_old = checkpoint_dir / "model_weights.h5"

            try:
                keras_model.save_weights(str(weights_path_new))
            except:
                # Fall back to old format
                keras_model.save_weights(str(weights_path_old))

        except Exception as e:
            print(f"⚠️  Warning: Could not save model weights: {e}")

        # Save optimizer state (for smooth resume)
        if optimizer is not None:
            optimizer_path = checkpoint_dir / "optimizer_state.npy"
            try:
                # Try new Keras 3.x API first
                if hasattr(optimizer, "variables"):
                    optimizer_weights = [v.numpy() for v in optimizer.variables]
                # Fall back to old API
                elif hasattr(optimizer, "get_weights"):
                    optimizer_weights = optimizer.get_weights()
                else:
                    raise AttributeError("Optimizer has no weights attribute")

                np.save(str(optimizer_path), optimizer_weights, allow_pickle=True)
            except Exception as e:
                print(f"⚠️  Warning: Could not save optimizer state: {e}")

        # Save training state
        training_state = {
            "epoch": epoch,
            "metrics": metrics or {},
            "is_best": is_best,
            "is_final": is_final,
            "is_interrupted": is_interrupted,
        }

        # Convert NumPy types to Python types for JSON serialization
        def convert_to_python_type(obj):
            """Recursively convert NumPy types to Python types."""
            if isinstance(obj, dict):
                return {k: convert_to_python_type(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_type(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        training_state = convert_to_python_type(training_state)

        state_path = checkpoint_dir / "training_state.json"
        with open(state_path, "w") as f:
            json.dump(training_state, f, indent=2)

        # Save config
        if self.config is not None:
            config_path = checkpoint_dir / "config.yaml"
            try:
                if hasattr(self.config, "save"):
                    self.config.save(config_path)
                elif hasattr(self.config, "to_dict"):
                    import yaml

                    with open(config_path, "w") as f:
                        yaml.dump(self.config.to_dict(), f)
            except Exception as e:
                print(f"⚠️  Warning: Could not save config: {e}")

        # Save metadata
        from datetime import datetime

        metadata = {
            "saved_at": datetime.now().isoformat(),
            "epoch": epoch,
            "is_best": is_best,
            "is_final": is_final,
            "is_interrupted": is_interrupted,
            "checkpoint_name": checkpoint_name,
        }

        metadata_path = checkpoint_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return checkpoint_dir

    def load(
        self,
        model: Any,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        checkpoint_type: str = "best",
        checkpoint_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore state.

        Args:
            model: Model to load weights into
            optimizer: Optional optimizer to restore state
            checkpoint_type: Type of checkpoint ('best', 'final', 'interrupted')
            checkpoint_path: Optional explicit path to checkpoint directory

        Returns:
            Dictionary with loaded state:
                - epoch: Epoch number
                - metrics: Saved metrics
                - config: Loaded config (if available)
        """
        # Determine checkpoint directory
        if checkpoint_path is not None:
            checkpoint_dir = Path(checkpoint_path)
        else:
            checkpoint_dir = self.checkpoints_dir / checkpoint_type

        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

        # Load model weights (try both old and new formats)
        weights_path_new = checkpoint_dir / "model.weights.h5"
        weights_path_old = checkpoint_dir / "model_weights.h5"

        weights_path = (
            weights_path_new if weights_path_new.exists() else weights_path_old
        )

        if weights_path.exists():
            try:
                if hasattr(model, "model"):
                    model.model.load_weights(str(weights_path))
                else:
                    model.load_weights(str(weights_path))
            except Exception as e:
                print(f"⚠️  Warning: Could not load model weights: {e}")

        # Load optimizer state
        if optimizer is not None:
            optimizer_path = checkpoint_dir / "optimizer_state.npy"
            if optimizer_path.exists():
                try:
                    optimizer_weights = np.load(str(optimizer_path), allow_pickle=True)

                    # Try new API first, fall back to old
                    if hasattr(optimizer, "variables"):
                        for var, weight in zip(optimizer.variables, optimizer_weights):
                            var.assign(weight)
                    elif hasattr(optimizer, "set_weights"):
                        optimizer.set_weights(optimizer_weights)
                    else:
                        raise AttributeError("Optimizer has no set_weights method")

                except Exception as e:
                    print(f"⚠️  Warning: Could not load optimizer state: {e}")

        # Load training state
        state_path = checkpoint_dir / "training_state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                training_state = json.load(f)
        else:
            training_state = {"epoch": 0, "metrics": {}}

        # Load config
        config_path = checkpoint_dir / "config.yaml"
        if config_path.exists():
            try:
                from src.training.config import Config

                training_state["config"] = Config.from_yaml(config_path)
            except Exception as e:
                print(f"⚠️  Warning: Could not load config: {e}")
                training_state["config"] = None

        return training_state

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.

        Returns:
            List of checkpoint info dictionaries
        """
        if not self.checkpoints_dir.exists():
            return []

        checkpoints = []
        for checkpoint_dir in self.checkpoints_dir.iterdir():
            if not checkpoint_dir.is_dir():
                continue

            metadata_path = checkpoint_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            else:
                metadata = {"checkpoint_name": checkpoint_dir.name}

            checkpoints.append(
                {
                    "name": checkpoint_dir.name,
                    "path": checkpoint_dir,
                    "metadata": metadata,
                }
            )

        return checkpoints

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint if it exists."""
        best_dir = self.checkpoints_dir / "best"
        return best_dir if best_dir.exists() else None

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to most recently saved checkpoint."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None

        # Sort by saved_at timestamp
        checkpoints_with_time = [
            (c, c["metadata"].get("saved_at", ""))
            for c in checkpoints
            if "saved_at" in c["metadata"]
        ]

        if not checkpoints_with_time:
            return None

        latest = max(checkpoints_with_time, key=lambda x: x[1])
        return latest[0]["path"]


# Example usage
if __name__ == "__main__":
    print("CheckpointManager created successfully!")
    print("\nUsage example:")
    print(
        """
    from src.training.checkpoint import CheckpointManager
    from src.training.config import load_config
    
    # Initialize
    config = load_config('configs/birnn.yaml')
    manager = CheckpointManager('results/birnn_pat3', config)
    
    # After training
    manager.save(
        model=model,
        optimizer=optimizer,
        epoch=1000,
        metrics={'loss': 0.5, 'rmse': 25.0},
        is_best=True
    )
    
    # Resume training
    state = manager.load(model, optimizer, checkpoint_type='best')
    start_epoch = state['epoch']
    print(f"Resuming from epoch {start_epoch}")
    """
    )
