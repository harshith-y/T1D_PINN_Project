"""
MLflow integration utilities for T1D PINN project
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow


class MLflowTracker:
    """Wrapper for MLflow experiment tracking"""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        """
        Initialize MLflow tracker

        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking URI. Options:
                - None: Uses MLFLOW_TRACKING_URI env var, or defaults to ./mlruns
                - "file:///path/to/mlruns": Local file storage
                - "http://localhost:5000": MLflow server
            run_name: Optional run name
        """
        # Default to local file storage (no server required)
        default_uri = os.getenv(
            "MLFLOW_TRACKING_URI", f"file://{Path.cwd() / 'mlruns'}"
        )
        self.tracking_uri = tracking_uri or default_uri
        mlflow.set_tracking_uri(self.tracking_uri)

        # Set or create experiment
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self.run_name = run_name

    def start_run(self, run_name: Optional[str] = None):
        """Start MLflow run"""
        return mlflow.start_run(run_name=run_name or self.run_name)

    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact"""
        mlflow.log_artifact(local_path, artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log directory of artifacts"""
        mlflow.log_artifacts(local_dir, artifact_path)

    def log_model(self, model, artifact_path: str = "model"):
        """Log model"""
        # TODO: Implement based on model type
        pass

    def end_run(self):
        """End MLflow run"""
        mlflow.end_run()


def create_tracker(model_name: str, mode: str, patient: int) -> MLflowTracker:
    """
    Factory function to create MLflow tracker

    Args:
        model_name: Model architecture (birnn, pinn, modified_mlp)
        mode: Training mode (forward, inverse)
        patient: Patient number

    Returns:
        MLflowTracker instance
    """
    experiment_name = f"{model_name.upper()}_{mode.capitalize()}"
    run_name = f"{model_name}_Pat{patient}_{mode}"

    return MLflowTracker(experiment_name=experiment_name, run_name=run_name)
