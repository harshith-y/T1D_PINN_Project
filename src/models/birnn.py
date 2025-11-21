"""
Biologically-Informed Recurrent Neural Network (BI-RNN) for glucose prediction.

This module implements a GRU-based RNN that incorporates biological constraints
through a physics-based residual loss. Unlike the PINN models which use DeepXDE
and TensorFlow 1.x, this uses modern TensorFlow 2.x with Keras API.

Key differences from PINN models:
- Sequence-to-sequence architecture (not point-wise prediction)
- Teacher forcing during training (uses previous ground truth, not predictions)
- Biological residual computed via forward Euler integration
- Modern TF2.x eager execution
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import tensorflow as tf

# CRITICAL: Force eager execution for BI-RNN
# DeepXDE may have set backend to TF1.x, but we need TF2.x eager mode
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

# Phase 1 imports
from src.datasets.loader import TrainingWindow
from src.physics.magdelaine import (
    MagdelaineParams,
    make_params_from_preset,
    simulate_latents_euler,
    residuals_euler_seq,
)


class GRUDeCarliModel(tf.keras.Model):
    """
    GRU-based sequence model for glucose-insulin-carbohydrate dynamics.
    
    This model predicts sequences of [G, I, D] states given input sequences
    of [u, r, G_prev]. It uses GRU layers for temporal modeling and can
    optionally enforce hard initial conditions.
    
    Architecture:
        Input: [batch, time, features] where features = [u, r, G]
        → GRU layers (return_sequences=True)
        → Dense output layer → [batch, time, 3] for [G, I, D]
    
    Attributes:
        hidden_units: Number of hidden units in each GRU layer
        num_layers: Number of stacked GRU layers
        hard_ic: Optional hard initial condition [1, 1, 3] for [G0, I0, D0]
    """
    
    def __init__(
        self,
        hidden_units: int = 96,
        num_layers: int = 1,
        hard_ic: Optional[tf.Tensor] = None
    ):
        """
        Initialize BI-RNN model.
        
        Args:
            hidden_units: Hidden dimension for GRU layers (default 96)
            num_layers: Number of stacked GRU layers (default 1)
            hard_ic: Hard initial condition tensor [1, 1, 3] or None
        """
        super(GRUDeCarliModel, self).__init__()
        
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.hard_ic = hard_ic  # Shape: [1, 1, 3] or None
        
        # Build GRU layers
        self.gru_layers = [
            tf.keras.layers.GRU(
                hidden_units,
                return_sequences=True,
                name=f"gru_{i}"
            )
            for i in range(num_layers)
        ]
        
        # Output layer predicts [G, I, D]
        self.output_layer = tf.keras.layers.Dense(
            3,
            activation="linear",
            name="output"
        )
    
    def call(self, inputs, training=False):
        """
        Forward pass through the GRU model.
        
        Args:
            inputs: Tensor of shape [batch, time, features]
                   features = [u, r, G_prev] (3 features)
            training: Boolean flag for training mode
            
        Returns:
            Tensor of shape [batch, time, 3] for [G, I, D] predictions
        """
        x = inputs  # Shape: [B, T, 3]
        
        # Pass through GRU layers
        for gru in self.gru_layers:
            x = gru(x, training=training)
        
        # Output predictions
        outputs = self.output_layer(x)  # Shape: [B, T, 3]
        
        # Apply hard initial conditions if provided
        if self.hard_ic is not None:
            T = tf.shape(outputs)[1]  # Sequence length
            t = tf.linspace(0.0, 1.0, T)  # Shape: [T]
            t = tf.reshape(t, [1, T, 1])  # Shape: [1, T, 1] for broadcasting
            
            # Exponential ramp: 1 - exp(-10*t)
            ramp = 1 - tf.exp(-10 * t)
            
            # Transform: y(t) = y0 + ramp(t) * NN(t)
            outputs = self.hard_ic + ramp * outputs
        
        return outputs


class BIRNN:
    """
    Wrapper for BI-RNN training and evaluation.
    
    This class provides:
    - Model construction with optional hard IC
    - Custom training loop with biological residual loss
    - Teacher-forced training (uses ground truth for inputs)
    - Multi-stage training support
    - Inverse parameter estimation
    - Evaluation and prediction
    """
    
    def __init__(self, config):
        """
        Initialize BI-RNN wrapper.
        
        Args:
            config: Configuration object with architecture, training, loss settings
        """
        self.config = config
        self.model = None
        self.optimizer = None
        self.data_window = None
        self.params = None
        self.inverse_params = None
        
        # Training data (populated during build)
        self.X_train = None
        self.Y_train = None
        self.U_train = None
        self.R_train = None
        self.X_test = None
        self.Y_test = None
        self.U_test = None
        self.R_test = None
        self.y_true = None  # Full sequence for teacher forcing
        
        # Training history
        self.history = {'loss': [], 'loss_g': [], 'loss_i': [], 'loss_d': [], 
                       'loss_ic': [], 'loss_bio': []}
    
    def build(self, data_window: TrainingWindow):
        """
        Build the BI-RNN model and prepare training data.
        
        Args:
            data_window: Training data from Phase 1 loader
        """
        self.data_window = data_window
        
        # Extract patient number
        try:
            patient_num = int(data_window.patient_id.replace("Pat", ""))
        except:
            patient_num = 3
        
        # Load physics parameters
        self.params = make_params_from_preset(patient_num)
        
        # Setup inverse parameters if needed
        mode = getattr(self.config, 'mode', 'forward')
        if mode == 'inverse':
            # Create trainable log_ksi
            self.log_ksi = tf.Variable(
                tf.math.log(self.params.ksi),
                dtype=tf.float32,
                name="log_ksi"
            )
            self.inverse_params = [self.log_ksi]
        else:
            self.log_ksi = None
            self.inverse_params = []
        
        # Prepare sequences for RNN training
        self._prepare_sequences()
        
        # Build model
        # For BI-RNN, use rnn_units (not n_neurons) and default to 1 layer
        if hasattr(self.config.architecture, 'rnn_units'):
            hidden_units = self.config.architecture.rnn_units  # e.g., 96
        else:
            hidden_units = self.config.architecture.n_neurons  # Fallback
        
        # For RNN, n_layers typically means number of RNN layers (usually 1)
        # Not the same as feedforward layers
        if hasattr(self.config.architecture, 'rnn_type'):
            num_layers = 1  # Default for RNN (unless explicitly set higher)
        else:
            num_layers = self.config.architecture.n_layers
        
        # Hard IC: use first timestep of ground truth
        if hasattr(self.config.architecture, 'use_hard_ic') and self.config.architecture.use_hard_ic:
            hard_ic = self.y_true[:, 0:1, :]  # Shape: [1, 1, 3]
        else:
            hard_ic = None
        
        self.model = GRUDeCarliModel(
            hidden_units=hidden_units,
            num_layers=num_layers,
            hard_ic=hard_ic
        )
        
        print(f"✅ BI-RNN built successfully")
        print(f"   Architecture: {num_layers} GRU layers, {hidden_units} hidden units")
        print(f"   Patient: {patient_num}")
        print(f"   Mode: {mode}")
        print(f"   Hard IC: {hard_ic is not None}")
        print(f"   Sequence length: {self.X_train.shape[1]} train, {self.X_test.shape[1]} test")
    
    def _prepare_sequences(self):
        """
        Prepare sequence data for RNN training.
        
        Creates teacher-forced sequences where:
        - Input: [u(t), r(t), G(t)] at time t
        - Target: [G(t+1), I(t+1), D(t+1)] at time t+1
        
        All values are normalized to [0, 1] for training.
        
        Handles both synthetic (with I, D) and real (glucose only) data.
        """
        # Get ground truth sequence in physical units
        G = self.data_window.glucose.reshape(1, -1, 1)  # [1, T, 1] mg/dL
        
        # Normalize glucose to [0, 1]
        G_norm = G / self.data_window.m_g
        
        # Handle latent states (I, D) if available
        if self.data_window.has_latent_states:
            I = self.data_window.insulin.reshape(1, -1, 1)  # [1, T, 1] U/dL
            D = self.data_window.digestion.reshape(1, -1, 1)  # [1, T, 1] mg/dL/min
            
            # Normalize to [0, 1]
            I_norm = I / self.data_window.m_i
            D_norm = D / self.data_window.m_d
            
            y_true = np.concatenate([G_norm, I_norm, D_norm], axis=-1)  # [1, T, 3]
        else:
            # Real patient data - glucose only
            # Still output 3 channels for model consistency, but I/D won't be penalized
            I_norm = np.zeros_like(G_norm)  # Dummy values
            D_norm = np.zeros_like(G_norm)  # Dummy values
            y_true = np.concatenate([G_norm, I_norm, D_norm], axis=-1)  # [1, T, 3]
        
        # Get input sequences [1, T, 1]
        u = self.data_window.u.reshape(1, -1, 1)  # [1, T, 1] U/min
        r = self.data_window.r.reshape(1, -1, 1)  # [1, T, 1] g/min
        
        # Normalize inputs (max normalization, matching notebook)
        u_max = np.max(u) + 1e-8
        r_max = np.max(r) + 1e-8
        U_seq = u / u_max
        R_seq = r / r_max
        
        # Teacher forcing setup:
        # Input at time t: [u(t), r(t), G(t)]
        # Target at time t: [G(t+1), I(t+1), D(t+1)]
        G_only = y_true[:, :-1, 0:1]  # Use glucose up to T-1
        Y_target = y_true[:, 1:, :]    # Predict states from t=1 to T
        U_input = U_seq[:, :-1, :]     # Inputs up to T-1
        R_input = R_seq[:, :-1, :]     # Inputs up to T-1
        
        # Concatenate to form input: [u, r, G]
        X_input = np.concatenate([U_input, R_input, G_only], axis=-1)  # [1, T-1, 3]
        
        # Train/test split (90/10)
        T_total = X_input.shape[1]
        T_train = int(0.9 * T_total)
        
        self.X_train = tf.constant(X_input[:, :T_train, :], dtype=tf.float32)
        self.Y_train = tf.constant(Y_target[:, :T_train, :], dtype=tf.float32)
        self.U_train = tf.constant(U_input[:, :T_train, :], dtype=tf.float32)
        self.R_train = tf.constant(R_input[:, :T_train, :], dtype=tf.float32)
        
        self.X_test = tf.constant(X_input[:, T_train:, :], dtype=tf.float32)
        self.Y_test = tf.constant(Y_target[:, T_train:, :], dtype=tf.float32)
        self.U_test = tf.constant(U_input[:, T_train:, :], dtype=tf.float32)
        self.R_test = tf.constant(R_input[:, T_train:, :], dtype=tf.float32)
        
        # Store full y_true for biological residual computation
        self.y_true = tf.constant(y_true, dtype=tf.float32)
        
        # Store scaling factors for denormalization
        self.u_max = u_max
        self.r_max = r_max
    
    def compile(self):
        """Compile the model with optimizer."""
        lr = self.config.training.learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        print(f"✅ Model compiled")
        print(f"   Optimizer: Adam")
        print(f"   Learning rate: {lr}")

    def _compute_biological_residual_loss(
        self,
        y_pred: tf.Tensor,
        y_in: tf.Tensor,
        u_in: tf.Tensor,
        r_in: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute biological residual loss using forward Euler integration.
        
        This enforces physics constraints by checking if:
        y(t+1) ≈ y(t) + dt * f(y(t), u(t), r(t))
        
        where f is the Magdelaine ODE system in PHYSICAL units.
        
        Key implementation details:
        - dt = 1.0 minute (physical time, matching data resolution)
        - All ODEs computed in physical units (mg/dL/min, U/dL/min, etc.)
        - Inputs denormalized using stored max values
        - States denormalized to physical units
        
        Args:
            y_pred: Predicted states [B, T, 3] for [G, I, D] (normalized)
            y_in: Input states [B, T, 3] from teacher forcing (normalized)
            u_in: Insulin inputs [B, T, 1] (normalized [0,1])
            r_in: Carb inputs [B, T, 1] (normalized [0,1])
            
        Returns:
            Mean squared error of biological residuals (in normalized space)
        """
        # Flatten to [T, features] for easier computation
        y_pred = tf.reshape(y_pred, [-1, 3])
        y_in = tf.reshape(y_in, [-1, 3])
        u_in = tf.reshape(u_in, [-1, 1])
        r_in = tf.reshape(r_in, [-1, 1])
        
        # Get scaling factors
        m_g = self.data_window.m_g
        m_i = self.data_window.m_i
        m_d = self.data_window.m_d
        
        # Denormalize states to physical units
        G = y_in[:, 0:1] * m_g  # mg/dL
        I = y_in[:, 1:2] * m_i  # U/dL
        D = y_in[:, 2:3] * m_d  # mg/dL/min
        
        # Denormalize inputs using stored max values
        # (inputs are normalized to [0,1] during data prep)
        ut = u_in * self.u_max  # U/min
        rt = r_in * self.r_max  # g/min
        
        # Get parameters (use inverse if enabled)
        if self.log_ksi is not None:
            ksi = tf.exp(self.log_ksi)
        else:
            ksi = self.params.ksi
        
        # Magdelaine ODEs (first-order approximation)
        # All derivatives in PHYSICAL units:
        # dG/dt: mg/dL/min
        # dI/dt: U/dL/min  
        # dD/dt: mg/dL/min²
        dG = -ksi * I + self.params.kl - self.params.kb + D
        dI = -I / self.params.Tu + (self.params.ku_Vi / self.params.Tu) * ut
        dD = -D / self.params.Tr + (self.params.kr_Vb / self.params.Tr) * rt
        
        # Forward Euler: y(t+1) = y(t) + dt * dy/dt
        # dt = 1.0 minute (physical time, matching data sampling rate)
        dt = 1.0  # minutes
        
        # Normalize derivatives before adding to normalized states
        # This ensures we're comparing apples-to-apples with NN prediction
        dG_norm = dG / m_g  # (mg/dL/min) / (mg/dL) = 1/min
        dI_norm = dI / m_i  # (U/dL/min) / (U/dL) = 1/min
        dD_norm = dD / m_d  # (mg/dL/min²) / (mg/dL/min) = 1/min
        
        # Euler step in normalized space
        y_next_ode = y_in + dt * tf.concat([dG_norm, dI_norm, dD_norm], axis=1)
        
        # Compute MSE between ODE prediction and NN prediction
        # (both in normalized [0,1] space)
        return tf.reduce_mean(tf.square(y_next_ode - y_pred))
    
    def _train_step(
        self,
        x: tf.Tensor,
        y_true: tf.Tensor,
        y_in: tf.Tensor,
        u: tf.Tensor,
        r: tf.Tensor,
        lambda_g: float,
        lambda_i: float,
        lambda_d: float,
        lambda_ic: float,
        lambda_B: float
    ) -> Tuple[tf.Tensor, ...]:
        """
        Single training step with gradient computation.
        
        Note: No @tf.function decorator - using eager execution for flexibility.
        
        Args:
            x: Input sequence [B, T, 3]
            y_true: Target sequence [B, T, 3]
            y_in: Input states for bio residual [B, T, 3]
            u, r: Input sequences for bio residual
            lambda_*: Loss weights
            
        Returns:
            Tuple of (total_loss, loss_g, loss_i, loss_d, loss_ic, loss_bio)
        """
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self.model(x, training=True)
            
            # Data fitting losses
            loss_g = tf.reduce_mean(tf.square(y_true[:, :, 0] - y_pred[:, :, 0]))
            
            # Only compute I/D losses if latent states are available
            if self.data_window.has_latent_states:
                loss_i = tf.reduce_mean(tf.square(y_true[:, :, 1] - y_pred[:, :, 1]))
                loss_d = tf.reduce_mean(tf.square(y_true[:, :, 2] - y_pred[:, :, 2]))
            else:
                # Real patient data - no I/D measurements
                loss_i = 0.0
                loss_d = 0.0
            
            # Initial condition loss
            if self.model.hard_ic is not None:
                loss_ic = 0.0
            else:
                ic_true = y_true[:, 0:1, :]
                ic_pred = y_pred[:, 0:1, :]
                loss_ic = tf.reduce_mean(tf.square(ic_true - ic_pred))
            
            # Biological residual loss
            loss_bio = self._compute_biological_residual_loss(y_pred, y_in, u, r)
            
            # Total weighted loss
            loss = (
                lambda_g * loss_g +
                lambda_i * loss_i +
                lambda_d * loss_d +
                lambda_ic * loss_ic +
                lambda_B * loss_bio
            )
        
        # Compute gradients
        variables = self.model.trainable_variables
        if self.log_ksi is not None:
            variables = variables + self.inverse_params
        
        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))
        
        return loss, loss_g, loss_i, loss_d, loss_ic, loss_bio
    
    def train(self, display_every: int = 50):
        """
        Train the BI-RNN model.
        
        Supports multi-stage training with different loss weights and learning rates.
        
        Args:
            display_every: Print loss every N epochs
        """
        print(f"\n{'='*80}")
        print(f"TRAINING BI-RNN")
        print(f"{'='*80}")
        
        # Check if multi-stage training is configured
        if hasattr(self.config.training, 'stages') and self.config.training.stages is not None:
            # Multi-stage training
            stages = self.config.training.stages
        else:
            # Single-stage training
            stages = [{
                'learning_rate': self.config.training.learning_rate,
                'epochs': self.config.training.epochs,
                'loss_weights': {
                    'lambda_g': self.config.loss_weights.glucose,
                    'lambda_i': self.config.loss_weights.insulin if hasattr(self.config.loss_weights, 'insulin') else 0.0,
                    'lambda_d': self.config.loss_weights.digestion if hasattr(self.config.loss_weights, 'digestion') else 0.0,
                    'lambda_ic': self.config.loss_weights.ic if hasattr(self.config.loss_weights, 'ic') else 0.0,
                    'lambda_B': self.config.loss_weights.biological_residual,
                }
            }]
        
        # Train through stages
        for stage_idx, stage in enumerate(stages):
            lr = stage.get('learning_rate', self.config.training.learning_rate)
            epochs = stage.get('epochs', 500)
            lambdas = stage.get('loss_weights', {})
            
            lambda_g = lambdas.get('lambda_g', 1.0)
            lambda_i = lambdas.get('lambda_i', 0.0)
            lambda_d = lambdas.get('lambda_d', 0.0)
            lambda_ic = lambdas.get('lambda_ic', 0.0)
            lambda_B = lambdas.get('lambda_B', 0.0)
            
            print(f"\n=== Stage {stage_idx + 1} | LR={lr}, Epochs={epochs} ===")
            print(f"    Loss weights: G={lambda_g:.2f}, I={lambda_i:.2f}, D={lambda_d:.2f}, IC={lambda_ic:.2f}, Bio={lambda_B:.2f}")
            
            self.optimizer.learning_rate.assign(lr)
            
            for epoch in range(epochs):
                # Training step
                loss, lg, li, ld, lic, lb = self._train_step(
                    self.X_train,
                    self.Y_train,
                    self.y_true[:, :self.X_train.shape[1], :],
                    self.U_train,
                    self.R_train,
                    lambda_g, lambda_i, lambda_d, lambda_ic, lambda_B
                )
                
                # Record history (tensors should be eager now)
                self.history['loss'].append(float(loss))
                self.history['loss_g'].append(float(lg))
                self.history['loss_i'].append(float(li))
                self.history['loss_d'].append(float(ld))
                self.history['loss_ic'].append(float(lic))
                self.history['loss_bio'].append(float(lb))
                
                # Display progress
                if epoch % display_every == 0:
                    tf.print(f"[Stage {stage_idx+1} | Epoch {epoch}]",
                           "Total=", loss,
                           ", G=", lg,
                           ", I=", li,
                           ", D=", ld,
                           ", IC=", lic,
                           ", Bio=", lb)
        
        print("\n✅ Training complete")
    
    def predict(self, use_test_set: bool = False) -> Dict[str, np.ndarray]:
        """
        Make predictions.
        
        Args:
            use_test_set: If True, predict on test set; otherwise full sequence
            
        Returns:
            Dictionary with keys: 'G', 'I', 'D', 't_min'
        """
        if use_test_set:
            x = self.X_test
        else:
            # Full sequence prediction
            x = tf.concat([self.X_train, self.X_test], axis=1)
        
        y_pred = self.model(x, training=False).numpy()  # [1, T, 3]
        
        # Denormalize (predictions are already in [0, 1] normalized space)
        G_pred = y_pred[0, :, 0] * self.data_window.m_g
        I_pred = y_pred[0, :, 1] * self.data_window.m_i
        D_pred = y_pred[0, :, 2] * self.data_window.m_d
        
        # Time vector
        if use_test_set:
            T_train = self.X_train.shape[1]
            t_min = np.arange(T_train, T_train + len(G_pred))
        else:
            t_min = np.arange(len(G_pred))
        
        return {
            'G': G_pred,
            'I': I_pred,
            'D': D_pred,
            't_min': t_min
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Returns:
            Dictionary with RMSE metrics (glucose only for compatibility)
        """
        # Predict on train and test
        pred_full = self.predict(use_test_set=False)
        
        # Ground truth (skip first timestep since we predict t+1)
        G_true = self.data_window.glucose[1:]  # Skip t=0
        
        # Split into train/test
        T_train = self.X_train.shape[1]
        G_true_train = G_true[:T_train]
        G_true_test = G_true[T_train:]
        
        G_pred_train = pred_full['G'][:T_train]
        G_pred_test = pred_full['G'][T_train:]
        
        # Compute RMSE for glucose
        rmse_train = np.sqrt(np.mean((G_pred_train - G_true_train) ** 2))
        rmse_test = np.sqrt(np.mean((G_pred_test - G_true_test) ** 2))
        rmse_total = np.sqrt(np.mean((pred_full['G'] - G_true) ** 2))
        
        metrics = {
            'rmse_interpolation': rmse_train,
            'rmse_forecast': rmse_test,
            'rmse_total': rmse_total
        }
        
        print(f"\n{'='*80}")
        print(f"EVALUATION METRICS")
        print(f"{'='*80}")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        
        return metrics
    
    def save(self, path: str):
        """Save model weights."""
        self.model.save_weights(path)
        print(f"✅ Model saved to {path}")
    
    def load(self, path: str):
        """Load model weights."""
        self.model.load_weights(path)
        print(f"✅ Model loaded from {path}")


# Convenience function
def create_birnn(config, data_window: TrainingWindow) -> BIRNN:
    """
    Create and build a BI-RNN model.
    
    Args:
        config: Configuration object
        data_window: Training data
        
    Returns:
        Built BIRNN ready for training
    """
    model = BIRNN(config)
    model.build(data_window)
    return model