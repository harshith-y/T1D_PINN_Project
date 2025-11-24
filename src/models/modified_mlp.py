"""
Modified-MLP PINN with custom gating mechanism and U-V encoding.

This module implements the Modified-MLP architecture with gated mixing layers.
It uses a custom forward pass with U and V encoders that are mixed via learned
gating (Z) layers. The architecture still integrates with DeepXDE for physics-
informed training.

Key differences from standard feedforward PINN:
- Dual pathway (U and V) encoding of inputs
- Gated mixing mechanism: H = (1-Z)*U + Z*V
- Custom NN class extending DeepXDE's base NN
"""

from __future__ import annotations

# CRITICAL: Disable TensorFlow eager execution BEFORE importing DeepXDE
# DeepXDE requires TF1.x compatibility mode (graph execution)
import tensorflow as tf
if tf.executing_eagerly():
    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.disable_eager_execution()

from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
import deepxde as dde

# Phase 1 imports
from src.datasets.loader import TrainingWindow
from src.physics.magdelaine import (
    MagdelaineParams,
    InverseParams,
    make_params_from_preset,
    make_inverse_params,
    residuals_dde,
)


class ModifiedMLP(dde.nn.NN):
    """
    Modified Multi-Layer Perceptron with gated U-V encoding.
    
    This custom architecture uses two encoding pathways (U and V) that are
    mixed via learned gating layers (Z). Matches the notebook implementation
    exactly, using TensorFlow 1.x style with explicit build() method.
    
    The forward pass computes:
    1. U = U_encoder(x)
    2. V = V_encoder(x)
    3. H = inputs (start with raw input)
    4. For each hidden layer i:
       - Z_i = sigmoid(Z_layer_i(H_{i-1}))
       - H_i = tanh(H_layer_i((1-Z_i)*U + Z_i*V))
    5. output = output_layer(H_n)
    
    Attributes:
        layers: List of layer dimensions [input, hidden, ..., output]
        activation: Activation function name (default 'tanh')
        initializer: Weight initializer name (default 'glorot_uniform')
    """
    
    def __init__(
        self,
        layers: List[int],
        activation: str = "tanh",
        initializer: str = "glorot_uniform",
    ):
        """
        Initialize Modified-MLP architecture.
        
        Args:
            layers: [input_dim, hidden_dim, ..., output_dim]
                    Example: [1, 32, 32, 32, 3] for 3 hidden layers of 32 neurons
            activation: Activation function for hidden layers
            initializer: Weight initialization strategy
        """
        super().__init__()
        
        self.layers = layers
        self.activation = getattr(tf.nn, activation)
        self.initializer = tf.keras.initializers.get(initializer)
        
        self._input_transform = None
        self._output_transform = None
        
        # U and V encoders (both use layers[1] dimension)
        self.U_encoder = tf.keras.layers.Dense(
            layers[1],
            activation=self.activation,
            kernel_initializer=self.initializer
        )
        self.V_encoder = tf.keras.layers.Dense(
            layers[1],
            activation=self.activation,
            kernel_initializer=self.initializer
        )
        
        # H and Z layers (one pair for each hidden layer after the first)
        # Note: layers[1:-1] = all hidden layers, but we already have first layer in U/V
        # So we need (len(layers) - 3) additional H/Z pairs
        self.H_layers = [
            tf.keras.layers.Dense(
                layers[1],  # All hidden layers use same dimension
                activation=self.activation,
                kernel_initializer=self.initializer
            )
            for _ in range(len(layers) - 3)
        ]
        self.Z_layers = [
            tf.keras.layers.Dense(
                layers[1],  # All hidden layers use same dimension
                activation="sigmoid",
                kernel_initializer=self.initializer
            )
            for _ in range(len(layers) - 3)
        ]
        
        # Output layer
        self.output_layer = tf.keras.layers.Dense(
            layers[-1],
            activation=None,
            kernel_initializer=self.initializer
        )
        
        self.built = False
    
    def apply_feature_transform(self, transform):
        """Apply input transform (e.g., Fourier features)."""
        self._input_transform = transform
    
    def apply_output_transform(self, transform):
        """Apply output transform (e.g., hard initial conditions)."""
        self._output_transform = transform
    
    @property
    def inputs(self):
        """Input placeholder (TF1.x style)."""
        return self.x
    
    @property
    def outputs(self):
        """Output tensor (TF1.x style)."""
        return self.y
    
    @property
    def targets(self):
        """Target placeholder (TF1.x style)."""
        return self.y_
    
    def build(self):
        """
        Build the computation graph (TF1.x style).
        
        This creates placeholders and builds the forward pass graph,
        matching the notebook's implementation exactly.
        """
        print("Building Modified MLP with tf.compat.v1 backend...")
        
        # Create placeholders
        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.layers[0]])
        self.y_ = tf.compat.v1.placeholder(tf.float32, [None, self.layers[-1]])
        
        inputs = self.x
        
        # Apply input transform if provided (e.g., Fourier features)
        if self._input_transform is not None:
            inputs = self._input_transform(inputs)
        
        # Encoding path
        U = self.U_encoder(inputs)
        V = self.V_encoder(inputs)
        
        # Mixing path - CRITICAL: Start with inputs, not U!
        H = inputs
        for H_layer, Z_layer in zip(self.H_layers, self.Z_layers):
            Z = Z_layer(H)
            H = (1.0 - Z) * U + Z * V
            H = H_layer(H)
        
        outputs = self.output_layer(H)
        
        # Apply output transform if provided (e.g., hard IC)
        if self._output_transform is not None:
            outputs = self._output_transform(inputs, outputs)
        
        self.y = outputs
        self.built = True


class InputLookup:
    """
    Lookup table for insulin (u) and carb (r) inputs.
    
    Used by the physics residuals to fetch u(t) and r(t) at collocation points.
    """
    
    def __init__(self, u: np.ndarray, r: np.ndarray):
        """
        Args:
            u: Insulin inputs [T] in U/min
            r: Carb inputs [T] in g/min
        """
        self.u = tf.constant(u.reshape(-1, 1), dtype=tf.float32)
        self.r = tf.constant(r.reshape(-1, 1), dtype=tf.float32)


class ScalesContainer:
    """Container for scaling factors."""
    
    def __init__(self, m_t: float, m_g: float, m_i: float, m_d: float):
        self.m_t = m_t
        self.m_g = m_g
        self.m_i = m_i
        self.m_d = m_d


class ModifiedMLPPINN:
    """
    Wrapper for Modified-MLP PINN that handles:
    - Model construction with optional Fourier features
    - Hard initial conditions
    - Training with physics-informed loss
    - Inverse parameter estimation
    - Evaluation and prediction
    
    This class provides the same interface as FeedforwardPINN for consistency.
    """
    
    def __init__(self, config):
        """
        Initialize Modified-MLP PINN.
        
        Args:
            config: Configuration object with model, training, data settings
        """
        self.config = config
        self.model = None
        self.data_window = None
        self.params = None
        self.inverse_params = None
        self.lookup = None
        self.scales = None
        self.geom = None
        
    def build(self, data_window: TrainingWindow):
        """
        Build the Modified-MLP PINN model.
        
        Args:
            data_window: Training data from Phase 1 loader
        """
        self.data_window = data_window
        
        # Extract patient number from patient_id (e.g., "Pat3" -> 3)
        try:
            patient_num = int(data_window.patient_id.replace("Pat", ""))
        except:
            patient_num = 3  # Default fallback
        
        # Load physics parameters
        self.params = make_params_from_preset(patient_num)
        
        # Setup inverse parameters if needed
        mode = getattr(self.config, 'mode', 'forward')
        if mode == 'inverse':
            self.inverse_params = make_inverse_params(
                enable=True,
                ksi_init=self.params.ksi
            )
        else:
            self.inverse_params = make_inverse_params(enable=False)
        
        # Create input lookup tables
        self.lookup = InputLookup(data_window.u, data_window.r)
        
        # Create scales container
        self.scales = ScalesContainer(
            m_t=data_window.m_t,
            m_g=data_window.m_g,
            m_i=data_window.m_i,
            m_d=data_window.m_d
        )
        
        # Define geometry (time domain)
        self.geom = dde.geometry.TimeDomain(0.0, 1.0)
        
        # Build network architecture
        layer_sizes = self._build_layer_sizes()
        
        # Create Modified-MLP network (matches notebook exactly)
        net = ModifiedMLP(
            layers=layer_sizes,
            activation=self.config.architecture.activation,
            initializer="glorot_uniform"  # Notebook uses lowercase
        )
        
        # Apply Fourier features if enabled (BEFORE applying transforms)
        if self.config.architecture.use_fourier:
            features = self.config.architecture.fourier_features
            m_t = self.scales.m_t
            # Notebook-style Fourier transform
            def fourier_transform(t):
                return tf.concat(
                    [t * m_t] +
                    [tf.sin(2**k * np.pi * t * m_t / 500) for k in features] +
                    [tf.cos(2**k * np.pi * t * m_t / 500) for k in features],
                    axis=1
                )
            net.apply_feature_transform(fourier_transform)
            print(f"   Applied Fourier transform with features: {features}")
        
        # Apply hard initial conditions if configured
        if hasattr(self.config.architecture, 'use_hard_ic') and self.config.architecture.use_hard_ic:
            net = self._apply_hard_ic_to_net(net)
        
        # Define PDE and training data
        pde_data = self._create_pde_data(net)
        
        # Store model
        self.model = dde.Model(pde_data, net)
        
        print(f"✅ Modified-MLP PINN built successfully")
        print(f"   Architecture: {layer_sizes}")
        print(f"   Patient: {patient_num}")
        print(f"   Mode: {mode}")
        print(f"   Fourier features: {self.config.architecture.use_fourier}")
    
    def _build_layer_sizes(self) -> List[int]:
        """Build network architecture layer sizes."""
        input_dim = 1  # Time only
        output_dim = 3  # G, I, D
        hidden_dim = self.config.architecture.n_neurons
        n_layers = self.config.architecture.n_layers
        
        layer_sizes = [input_dim] + [hidden_dim] * n_layers + [output_dim]
        return layer_sizes
    
    def _apply_hard_ic_to_net(self, net):
        """Apply hard initial condition transform to network output."""
        # Get initial values (normalized)
        G0 = self.data_window.glucose[0] / self.scales.m_g
        I0 = self.data_window.insulin[0] / self.scales.m_i if self.data_window.insulin is not None else 0.0
        D0 = 0.0  # Digestion starts at 0
        
        # Notebook-style hard IC transform with ReLU and exponential
        def hard_ic_transform(t, y):
            """
            Transform: y(t) = y0 + (1 - exp(-10*t)) * NN(t)
            With ReLU to ensure positivity.
            """
            return tf.concat([
                tf.nn.relu(G0 + (1 - tf.exp(-10 * t)) * y[:, 0:1]),
                tf.nn.relu(I0 + (1 - tf.exp(-10 * t)) * y[:, 1:2]),
                tf.nn.relu(D0 + (1 - tf.exp(-10 * t)) * y[:, 2:3]),
            ], axis=1)
        
        net.apply_output_transform(hard_ic_transform)
        print(f"   Applied hard IC: G0={G0:.4f}, I0={I0:.4f}, D0={D0:.4f}")
        
        return net
    
    def _create_pde_data(self, net):
        """Create PDE data for DeepXDE model."""
        # Define PDE residuals
        def pde(t, y):
            """Compute physics residuals using Phase 1 magdelaine.residuals_dde."""
            return residuals_dde(
                y=y,
                x=t,
                params=self.params,
                lookup=self.lookup,
                scales=self.scales,
                inverse=self.inverse_params,
                include_prior=False
            )
        
        # Observational data (glucose measurements)
        observe_t = self.data_window.time_norm.reshape(-1, 1)
        observe_g = (self.data_window.glucose / self.scales.m_g).reshape(-1, 1)
        
        # Create observational constraint
        observe_data = dde.icbc.PointSetBC(observe_t, observe_g, component=0)
        
        # Create PDE data
        n_domain = getattr(self.config.training, 'n_collocation', 3000)
        
        # Split into train/test
        train_frac = getattr(self.config.data, 'train_split', 0.9)
        n_train = int(len(observe_t) * train_frac)
        
        pde_data = dde.data.PDE(
            geometry=self.geom,
            pde=pde,
            bcs=[observe_data],
            num_domain=n_domain,
            num_boundary=0,
            train_distribution="uniform",
            anchors=observe_t[:n_train]  # Use training data as anchors
        )
        
        print(f"   Training points: {n_train} (from {len(observe_t)} total)")
        
        return pde_data
    
    def compile(self):
        """Compile the model with optimizer and loss weights."""
        lr = self.config.training.learning_rate
        optimizer = self.config.training.optimizer
        
        # Loss weights: [glucose_data, ode_G, ode_I, ode_D]
        loss_weights = [
            self.config.loss_weights.glucose,
            self.config.loss_weights.ode_glucose,
            self.config.loss_weights.ode_insulin,
            self.config.loss_weights.ode_digestion
        ]
        
        print("Compiling model...")
        self.model.compile(optimizer, lr=lr, loss_weights=loss_weights)
        
        print(f"✅ Model compiled")
        print(f"   Optimizer: {optimizer}")
        print(f"   Learning rate: {lr}")
        print(f"   Loss weights: {loss_weights}")
    
    def train(self, display_every: int = 1000):
        """
        Train the Modified-MLP PINN.
        
        Args:
            display_every: Print loss every N iterations
        """
        # Optional callbacks
        callbacks = []
        
        # Inverse parameter logging
        if self.inverse_params and self.inverse_params.log_ksi is not None:
            Path(self.config.output.save_dir).mkdir(parents=True, exist_ok=True)
            param_logger = dde.callbacks.VariableValue(
                [self.inverse_params.log_ksi],
                period=display_every,
                filename=str(Path(self.config.output.save_dir) / "inverse_params.dat")
            )
            callbacks.append(param_logger)
        
        # Train
        print(f"\n{'='*80}")
        print(f"TRAINING MODIFIED-MLP PINN")
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
            t = self.data_window.time_norm
        
        t = t.reshape(-1, 1)
        
        # Predict (returns normalized values)
        y_pred = self.model.predict(t)  # Shape: [T, 3]
        
        # Denormalize
        G_pred = y_pred[:, 0] * self.scales.m_g
        I_pred = y_pred[:, 1] * self.scales.m_i
        D_pred = y_pred[:, 2] * self.scales.m_d
        
        return {
            'G': G_pred,
            'I': I_pred,
            'D': D_pred,
            't_min': t.flatten() * self.scales.m_t
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Returns:
            Dictionary with RMSE metrics
        """
        # Predict on full data
        pred = self.predict()
        
        # Ground truth
        G_true = self.data_window.glucose
        
        # Compute RMSE
        rmse_total = np.sqrt(np.mean((pred['G'] - G_true) ** 2))
        
        # Split into interpolation (90%) and forecast (10%)
        n_total = len(G_true)
        n_train = int(n_total * 0.9)
        
        rmse_interp = np.sqrt(np.mean((pred['G'][:n_train] - G_true[:n_train]) ** 2))
        rmse_forecast = np.sqrt(np.mean((pred['G'][n_train:] - G_true[n_train:]) ** 2))
        
        metrics = {
            'rmse_interpolation': rmse_interp,
            'rmse_forecast': rmse_forecast,
            'rmse_total': rmse_total
        }
        
        print(f"\n{'='*80}")
        print(f"EVALUATION METRICS")
        print(f"{'='*80}")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        
        return metrics
    
    def save(self, path: str):
        """Save model checkpoint."""
        self.model.save(path)
        print(f"✅ Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        self.model.restore(path)
        print(f"✅ Model loaded from {path}")


# Convenience function for quick model creation
def create_modified_mlp_pinn(config, data_window: TrainingWindow) -> ModifiedMLPPINN:
    """
    Create and build a Modified-MLP PINN model.
    
    Args:
        config: Configuration object
        data_window: Training data
        
    Returns:
        Built ModifiedMLPPINN ready for training
    """
    model = ModifiedMLPPINN(config)
    model.build(data_window)
    return model