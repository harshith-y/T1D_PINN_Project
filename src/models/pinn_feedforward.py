"""
Feedforward PINN (Physics-Informed Neural Network) using DeepXDE.

This module implements the standard feedforward PINN architecture from your
PINNandModMLP_SyntheticData.ipynb notebook. It uses DeepXDE's FNN class
with automatic differentiation for computing physics residuals.

Architecture:
    - Standard feedforward neural network
    - TensorFlow 1.x backend (via DeepXDE)
    - Automatic differentiation for PDE residuals
    - Supports Fourier feature embeddings (optional)
    - Supports hard initial condition enforcement (optional)

Example:
    >>> from models.pinn_feedforward import FeedforwardPINN
    >>> from configs.config import load_config
    >>> 
    >>> config = load_config(model_name='pinn', mode='forward')
    >>> model = FeedforwardPINN(config)
    >>> model.build(data_window)
    >>> model.compile()
    >>> model.train()
"""

from __future__ import annotations

# CRITICAL: Disable TensorFlow eager execution BEFORE importing DeepXDE
# DeepXDE requires TF1.x compatibility mode (graph execution)
import tensorflow as tf
if tf.executing_eagerly():
    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.disable_eager_execution()

from typing import Optional, Dict, Any, List
import numpy as np
import deepxde as dde
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.physics.magdelaine import (
    MagdelaineParams,
    make_params_from_preset,
    InverseParams,
    make_inverse_params,
    residuals_dde,
)
from src.datasets.loader import TrainingWindow


class InputLookup:
    """
    Lookup tables for u(t) and r(t) used in physics residuals.
    
    This class wraps the input sequences so they can be efficiently
    accessed during PDE residual computation via tf.gather().
    """
    
    def __init__(self, u: np.ndarray, r: np.ndarray):
        """
        Args:
            u: Insulin inputs (U/min), shape (T,)
            r: Carb inputs (g/min), shape (T,)
        """
        self.u = tf.constant(u.reshape(-1, 1), dtype=tf.float32)
        self.r = tf.constant(r.reshape(-1, 1), dtype=tf.float32)


class ScalingFactors:
    """Container for normalization scales."""
    
    def __init__(self, m_t: float, m_g: float, m_i: float, m_d: float):
        self.m_t = float(m_t)
        self.m_g = float(m_g)
        self.m_i = float(m_i)
        self.m_d = float(m_d)


class FeedforwardPINN:
    """
    Feedforward Physics-Informed Neural Network using DeepXDE.
    
    This implements the vanilla PINN architecture with:
    - Standard FNN layers with tanh activation
    - Physics-informed loss via PDE residuals
    - Optional Fourier feature embeddings
    - Optional hard initial condition enforcement
    - Support for inverse problem (trainable ksi parameter)
    
    The architecture matches your notebook implementation exactly.
    """
    
    def __init__(self, config):
        """
        Initialize the feedforward PINN.
        
        Args:
            config: Configuration object with architecture, training, and data settings
        """
        self.config = config
        
        # Extract patient number from config
        patient_str = config.data.patient  # e.g., "Pat3"
        self.patient = int(patient_str.replace("Pat", ""))
        
        # Physics parameters
        self.params = make_params_from_preset(self.patient)
        
        # Inverse mode setup
        self.inverse_params = None
        if config.mode == 'inverse':
            ksi_init = config.inverse_init_range[0] if config.inverse_init_range else None
            self.inverse_params = make_inverse_params(enable=True, ksi_init=ksi_init)
        
        # Model components (initialized in build())
        self.net = None
        self.model = None
        self.data_window = None
        self.input_lookup = None
        self.scales = None
        
        # Training state
        self.losshistory = None
        self.train_state = None
        
    def build(self, data_window: TrainingWindow) -> None:
        """
        Build the DeepXDE model with data and physics.
        
        Args:
            data_window: TrainingWindow containing training data
        """
        self.data_window = data_window
        
        # Setup input lookup and scaling
        self.input_lookup = InputLookup(data_window.u, data_window.r)
        self.scales = ScalingFactors(
            m_t=data_window.m_t,
            m_g=data_window.m_g,
            m_i=data_window.m_i,
            m_d=data_window.m_d
        )
        
        # Build neural network architecture
        layers = [1] + [self.config.architecture.n_neurons] * self.config.architecture.n_layers + [3]
        self.net = dde.nn.FNN(
            layers,
            self.config.architecture.activation,
            "Glorot uniform"
        )
        
        # Apply feature transform (Fourier features)
        if self.config.architecture.use_fourier:
            self._apply_fourier_transform()
        
        # Apply output transform (hard initial conditions)
        if True:  # Always use hard IC in your notebooks
            self._apply_hard_initial_conditions()
        
        # Define geometry (time domain)
        t0 = 0.0
        t1 = 1.0  # Normalized time
        geomtime = dde.geometry.TimeDomain(t0, t1)
        
        # Define PDE residual function
        def pde(x, y):
            """PDE residuals using your exact formulation."""
            return residuals_dde(
                y, x, self.params, self.input_lookup, self.scales,
                inverse=self.inverse_params
            )
        
        # Prepare training data
        # In your notebook, you use observe_g for glucose observations
        t_train, g_train = self._prepare_training_data()
        observe_g = dde.PointSetBC(t_train, g_train, component=0)
        
        # Create PDE data object
        self.pde_data = dde.data.PDE(
            geomtime,
            pde,
            [observe_g],
            num_domain=3000,  # From your notebook
            anchors=t_train,
            num_test=1000
        )
        
        # Create DeepXDE model
        self.model = dde.Model(self.pde_data, self.net)
        
        print("✅ Feedforward PINN built successfully")
        print(f"   Architecture: {layers}")
        print(f"   Patient: {self.patient}")
        print(f"   Mode: {self.config.mode}")
        print(f"   Fourier features: {self.config.architecture.use_fourier}")
    
    def _apply_fourier_transform(self) -> None:
        """Apply Fourier feature embedding to inputs."""
        m_t = self.scales.m_t
        
        def fourier_transform(t):
            """
            Fourier feature embedding matching your notebook:
            [t*m_t, sin(2^k * pi * t*m_t / 500), cos(2^k * pi * t*m_t / 500)]
            for k in fourier_features
            """
            features = [t * m_t]
            
            for k in self.config.architecture.fourier_features:
                freq = 2**k * np.pi * m_t / 500.0
                features.append(tf.sin(freq * t))
                features.append(tf.cos(freq * t))
            
            return tf.concat(features, axis=1)
        
        self.net.apply_feature_transform(fourier_transform)
        print(f"   Applied Fourier transform with features: {self.config.architecture.fourier_features}")
    
    def _apply_hard_initial_conditions(self) -> None:
        """
        Apply hard initial condition enforcement via output transform.
        
        This matches your notebook's implementation:
        G = G0 + (1 - exp(-10*t)) * y_nn[:, 0:1]
        I = I0 + (1 - exp(-10*t)) * y_nn[:, 1:2]
        D = D0 + (1 - exp(-10*t)) * y_nn[:, 2:3]
        """
        # Get initial conditions from data
        G0 = self.data_window.glucose[0] / self.scales.m_g  # Normalized
        I0 = self.data_window.insulin[0] / self.scales.m_i if self.data_window.insulin is not None else 0.0
        D0 = self.data_window.digestion[0] / self.scales.m_d if self.data_window.digestion is not None else 0.0
        
        def output_transform(t, y):
            """
            Hard initial condition enforcement.
            Ramp function: (1 - exp(-10*t))
            """
            ramp = 1.0 - tf.exp(-10.0 * t)
            
            return tf.concat([
                tf.nn.relu(G0 + ramp * y[:, 0:1]),
                tf.nn.relu(I0 + ramp * y[:, 1:2]),
                tf.nn.relu(D0 + ramp * y[:, 2:3])
            ], axis=1)
        
        self.net.apply_output_transform(output_transform)
        print(f"   Applied hard IC: G0={G0:.4f}, I0={I0:.4f}, D0={D0:.4f}")
    
    def _prepare_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with 90/10 interpolation/forecast split and masking.
        
        Returns:
            t_train: Normalized time points for training, shape (N, 1)
            g_train: Normalized glucose values, shape (N, 1)
        """
        # Get first point (for IC)
        first_t = self.data_window.time_norm[0]
        first_g = self.data_window.glucose[0] / self.scales.m_g
        
        # Remaining data
        remaining_t = self.data_window.time_norm[1:]
        remaining_g = self.data_window.glucose[1:] / self.scales.m_g
        
        # Split: 90% for interpolation, 10% for forecast
        split_index = int(0.9 * len(remaining_t))
        t_interp = remaining_t[:split_index]
        g_interp = remaining_g[:split_index]
        
        # Apply mask: keep every 10th point for testing within interpolation segment
        mask = np.ones(len(t_interp), dtype=bool)
        mask[::self.config.data.mask_interval] = False  # Mask every Nth point
        
        t_train_interp = t_interp[mask]
        g_train_interp = g_interp[mask]
        
        # Add first point explicitly
        t_train = np.concatenate([[first_t], t_train_interp]).reshape(-1, 1)
        g_train = np.concatenate([[first_g], g_train_interp]).reshape(-1, 1)
        
        print(f"   Training points: {len(t_train)} (from {len(self.data_window.time_norm)} total)")
        
        return t_train, g_train
    
    def compile(self) -> None:
        """
        Compile the model with optimizer and loss weights.
        
        Uses loss weights from config:
        - loss_weights[0]: glucose observation loss
        - loss_weights[1]: glucose ODE residual
        - loss_weights[2]: insulin ODE residual
        - loss_weights[3]: digestion ODE residual
        """
        loss_weights = [
            self.config.loss_weights.glucose,
            self.config.loss_weights.ode_glucose,
            self.config.loss_weights.ode_insulin,
            self.config.loss_weights.ode_digestion
        ]
        
        # External trainable variables (inverse parameters)
        external_vars = None
        if self.inverse_params and self.inverse_params.log_ksi is not None:
            external_vars = [self.inverse_params.log_ksi]
        
        self.model.compile(
            self.config.training.optimizer,
            lr=self.config.training.learning_rate,
            loss_weights=loss_weights,
            external_trainable_variables=external_vars
        )
        
        print("✅ Model compiled")
        print(f"   Optimizer: {self.config.training.optimizer}")
        print(f"   Learning rate: {self.config.training.learning_rate}")
        print(f"   Loss weights: {loss_weights}")
    
    def train(self, display_every: int = 1000) -> None:
        """
        Train the model.
        
        Args:
            display_every: Print loss every N iterations
        """
        # Callbacks
        callbacks = []
        if self.inverse_params and self.inverse_params.log_ksi is not None:
            Path(self.config.output.save_dir).mkdir(parents=True, exist_ok=True)
            param_logger = dde.callbacks.VariableValue(
                [self.inverse_params.log_ksi],
                period=500,
                filename=str(Path(self.config.output.save_dir) / "inverse_params.dat")
            )
            callbacks.append(param_logger)
        
        # Train
        print(f"\n{'='*80}")
        print(f"TRAINING FEEDFORWARD PINN")
        print(f"{'='*80}")
        
        self.losshistory, self.train_state = self.model.train(
            epochs=self.config.training.epochs,
            display_every=display_every,
            callbacks=callbacks if callbacks else None
        )
        
        # Optional L-BFGS-B refinement (toggleable via config)
        if self.config.training.use_lbfgs_refinement:
            print("\nApplying L-BFGS-B refinement...")
            self.model.compile("L-BFGS-B")
            self.model.train()
            print("✅ L-BFGS-B refinement complete")
        else:
            print("\n⏭️  Skipping L-BFGS-B refinement (disabled in config)")

        print("✅ Training complete")
    
    def predict(self, t: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Make predictions.
        
        Args:
            t: Time points (normalized [0,1]). If None, uses full data window.
        
        Returns:
            Dictionary with keys: 'G', 'I', 'D' (denormalized values)
        """
        if t is None:
            t = self.data_window.time_norm.reshape(-1, 1)
        else:
            t = np.array(t).reshape(-1, 1)
        
        # Predict (returns normalized values)
        y_pred = self.model.predict(t)
        
        # Denormalize
        predictions = {
            'G': y_pred[:, 0] * self.scales.m_g,
            'I': y_pred[:, 1] * self.scales.m_i,
            'D': y_pred[:, 2] * self.scales.m_d,
        }
        
        return predictions
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on interpolation and forecast segments.
        
        Returns:
            Dictionary with RMSE metrics for each segment
        """
        # Full predictions
        predictions = self.predict()
        
        # Ground truth (denormalized)
        G_true = self.data_window.glucose
        
        # Split indices
        split_idx = int(0.9 * len(G_true))
        
        # Interpolation segment
        G_pred_interp = predictions['G'][:split_idx]
        G_true_interp = G_true[:split_idx]
        rmse_interp = np.sqrt(np.mean((G_pred_interp - G_true_interp)**2))
        
        # Forecast segment
        G_pred_forecast = predictions['G'][split_idx:]
        G_true_forecast = G_true[split_idx:]
        rmse_forecast = np.sqrt(np.mean((G_pred_forecast - G_true_forecast)**2))
        
        metrics = {
            'rmse_interpolation': float(rmse_interp),
            'rmse_forecast': float(rmse_forecast),
            'rmse_total': float(np.sqrt(np.mean((predictions['G'] - G_true)**2)))
        }
        
        print("\n" + "="*80)
        print("EVALUATION METRICS")
        print("="*80)
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        
        return metrics
    
    def save(self, path: str | Path) -> None:
        """Save model weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        print(f"✅ Model saved to {path}")
    
    def load(self, path: str | Path) -> None:
        """Load model weights."""
        self.model.restore(str(path))
        print(f"✅ Model loaded from {path}")


# Example usage / Quick test
if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.training.config import load_config
    from src.datasets.loader import load_synthetic_window
    
    # Load config
    config = load_config(model_name='pinn', mode='forward')
    
    # Load data
    data_window = load_synthetic_window(patient=3)
    
    # Create and train model
    model = FeedforwardPINN(config)
    model.build(data_window)
    model.compile()
    model.train(display_every=1000)
    
    # Evaluate
    metrics = model.evaluate()
    
    # Save
    model.save("runs/pinn_test/model")