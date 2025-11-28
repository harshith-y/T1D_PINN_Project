"""
Prediction save/load manager for fast visualization.

Saves predictions to compressed .npz format for instant reload,
avoiding need to re-run model inference.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import json
import numpy as np


class PredictionManager:
    """
    Manage saving and loading of model predictions.
    
    Stores predictions in compressed .npz format for fast reload
    during visualization. Metadata stored separately in JSON.
    
    Example:
        >>> manager = PredictionManager('results/birnn_pat3')
        >>> # After training
        >>> manager.save(
        ...     time=time_array,
        ...     glucose_pred=pred_glucose,
        ...     glucose_true=true_glucose,
        ...     split_idx=2304,
        ...     metadata={'model': 'birnn', 'patient': 3}
        ... )
        >>> # Later, for visualization
        >>> predictions = manager.load()
        >>> print(predictions['glucose_pred'].shape)
    """
    
    def __init__(self, save_dir: str | Path):
        """
        Initialize prediction manager.
        
        Args:
            save_dir: Directory to save predictions
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.predictions_file = self.save_dir / "predictions.npz"
        self.metadata_file = self.save_dir / "predictions_metadata.json"
    
    def save(
        self,
        time: np.ndarray,
        glucose_pred: np.ndarray,
        glucose_true: np.ndarray,
        insulin_pred: Optional[np.ndarray] = None,
        insulin_true: Optional[np.ndarray] = None,
        digestion_pred: Optional[np.ndarray] = None,
        digestion_true: Optional[np.ndarray] = None,
        split_idx: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save predictions to disk.
        
        Args:
            time: Time array (minutes)
            glucose_pred: Predicted glucose (mg/dL)
            glucose_true: True glucose (mg/dL)
            insulin_pred: Optional predicted insulin (U/dL)
            insulin_true: Optional true insulin (U/dL)
            digestion_pred: Optional predicted digestion (mg/dL/min)
            digestion_true: Optional true digestion (mg/dL/min)
            split_idx: Optional train/test split index
            metadata: Optional metadata dictionary
        """
        # Prepare arrays dictionary
        arrays = {
            'time': time.astype(np.float32),
            'glucose_pred': glucose_pred.astype(np.float32),
            'glucose_true': glucose_true.astype(np.float32)
        }
        
        # Add optional arrays if provided
        if insulin_pred is not None:
            arrays['insulin_pred'] = insulin_pred.astype(np.float32)
        if insulin_true is not None:
            arrays['insulin_true'] = insulin_true.astype(np.float32)
        if digestion_pred is not None:
            arrays['digestion_pred'] = digestion_pred.astype(np.float32)
        if digestion_true is not None:
            arrays['digestion_true'] = digestion_true.astype(np.float32)
        if split_idx is not None:
            arrays['split_idx'] = np.array([split_idx], dtype=np.int32)
        
        # Save arrays (compressed)
        np.savez_compressed(self.predictions_file, **arrays)
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        # Add timestamp
        from datetime import datetime
        metadata['saved_at'] = datetime.now().isoformat()
        metadata['has_latent_states'] = insulin_pred is not None
        
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
        
        metadata = convert_to_python_type(metadata)
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load(self) -> Dict[str, Any]:
        """
        Load predictions from disk.
        
        Returns:
            Dictionary with:
                - time: Time array
                - glucose_pred, glucose_true: Glucose arrays
                - insulin_pred, insulin_true: Insulin arrays (if saved)
                - digestion_pred, digestion_true: Digestion arrays (if saved)
                - split_idx: Train/test split index (if saved)
                - metadata: Metadata dictionary
        
        Raises:
            FileNotFoundError: If predictions file doesn't exist
        """
        if not self.predictions_file.exists():
            raise FileNotFoundError(
                f"Predictions file not found: {self.predictions_file}\n"
                f"Train the model first or run with prediction saving enabled."
            )
        
        # Load arrays
        data = np.load(self.predictions_file)
        
        predictions = {
            'time': data['time'],
            'glucose_pred': data['glucose_pred'],
            'glucose_true': data['glucose_true']
        }
        
        # Add optional arrays if present
        if 'insulin_pred' in data:
            predictions['insulin_pred'] = data['insulin_pred']
        if 'insulin_true' in data:
            predictions['insulin_true'] = data['insulin_true']
        if 'digestion_pred' in data:
            predictions['digestion_pred'] = data['digestion_pred']
        if 'digestion_true' in data:
            predictions['digestion_true'] = data['digestion_true']
        if 'split_idx' in data:
            predictions['split_idx'] = int(data['split_idx'][0])
        
        # Load metadata
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                predictions['metadata'] = json.load(f)
        else:
            predictions['metadata'] = {}
        
        return predictions
    
    def exists(self) -> bool:
        """Check if predictions file exists."""
        return self.predictions_file.exists()
    
    def save_parameter_evolution(
        self,
        epochs: np.ndarray,
        param_values: np.ndarray,
        param_name: str = 'ksi',
        true_value: Optional[float] = None,
        losses: Optional[np.ndarray] = None,
        stages: Optional[np.ndarray] = None
    ) -> None:
        """
        Save parameter evolution for inverse training.
        
        Args:
            epochs: Epoch numbers
            param_values: Parameter values at each epoch
            param_name: Name of parameter (e.g., 'ksi')
            true_value: Optional true parameter value
            losses: Optional loss values at each epoch
            stages: Optional stage labels (e.g., 'stage1', 'stage2', 'stage3')
        """
        param_file = self.save_dir / f"parameter_evolution_{param_name}.npz"
        
        arrays = {
            'epochs': epochs.astype(np.int32),
            'param_values': param_values.astype(np.float32)
        }
        
        if true_value is not None:
            arrays['true_value'] = np.array([true_value], dtype=np.float32)
        if losses is not None:
            arrays['losses'] = losses.astype(np.float32)
        if stages is not None:
            # Store stages as string array
            arrays['stages'] = np.array(stages, dtype='U20')
        
        # Compute errors if true value available
        if true_value is not None:
            errors = np.abs(param_values - true_value) / true_value * 100
            arrays['errors_percent'] = errors.astype(np.float32)
        
        np.savez_compressed(param_file, **arrays)
    
    def load_parameter_evolution(
        self,
        param_name: str = 'ksi'
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Load parameter evolution data.
        
        Args:
            param_name: Name of parameter
        
        Returns:
            Dictionary with evolution data, or None if not found
        """
        param_file = self.save_dir / f"parameter_evolution_{param_name}.npz"
        
        if not param_file.exists():
            return None
        
        data = np.load(param_file, allow_pickle=True)
        
        result = {
            'epochs': data['epochs'],
            'param_values': data['param_values']
        }
        
        if 'true_value' in data:
            result['true_value'] = float(data['true_value'][0])
        if 'losses' in data:
            result['losses'] = data['losses']
        if 'stages' in data:
            result['stages'] = data['stages']
        if 'errors_percent' in data:
            result['errors_percent'] = data['errors_percent']
        
        return result


# Example usage
if __name__ == "__main__":
    print("PredictionManager created successfully!")
    print("\nUsage example:")
    print("""
    from src.training.predictions import PredictionManager
    import numpy as np
    
    # Initialize
    manager = PredictionManager('results/birnn_pat3')
    
    # After training
    manager.save(
        time=np.arange(2880),
        glucose_pred=pred_glucose,
        glucose_true=true_glucose,
        split_idx=2304,
        metadata={'model': 'birnn', 'patient': 3, 'rmse': 25.0}
    )
    
    # Later, for visualization (FAST!)
    predictions = manager.load()
    print(predictions['glucose_pred'].shape)
    print(predictions['metadata']['rmse'])
    
    # For inverse training
    manager.save_parameter_evolution(
        epochs=np.arange(0, 3000, 10),
        param_values=ksi_history,
        param_name='ksi',
        true_value=274.0
    )
    """)