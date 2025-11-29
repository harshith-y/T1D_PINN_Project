"""
Inverse training orchestrator with flexible multi-stage support.

Implements configurable inverse training (parameter estimation) with:
- Flexible number of stages (2, 3, 4+)
- Per-stage control of what to train (params vs NN)
- Per-stage learning rates and loss weights
- Parameter evolution tracking
- Support for all model architectures (BI-RNN, PINN, Modified-MLP)

Based on the working 3-stage pattern from test_birnn_inverse_training.py
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import numpy as np
import tensorflow as tf


class InverseTrainer:
    """
    Flexible inverse training orchestrator.
    
    Supports configurable multi-stage training where each stage can:
    - Train inverse parameters only
    - Train NN weights only  
    - Train both (joint)
    - Use custom learning rates and loss weights
    
    Example:
        >>> # 3-stage training
        >>> trainer = InverseTrainer(model, config, data)
        >>> history = trainer.train()
        >>> print(f"Final ksi: {history['final_ksi']:.2f}")
        >>> print(f"Error: {history['ksi_error_percent']:.2f}%")
    """
    
    def __init__(
        self,
        model: Any,
        config: Any,
        data: Any,
        true_param_value: Optional[float] = None
    ):
        """
        Initialize inverse trainer.
        
        Args:
            model: Model with inverse parameters (must have log_ksi or similar)
            config: Config with training.stages defined
            data: TrainingWindow data object
            true_param_value: Optional true parameter value (for error tracking)
        """
        self.model = model
        self.config = config
        self.data = data
        self.true_param_value = true_param_value
        
        # Determine model type
        self.model_type = self._infer_model_type()
        
        # Get list of parameters being estimated
        self.inverse_params_list = getattr(config, 'inverse_params', ['ksi'])
        if isinstance(self.inverse_params_list, str):
            self.inverse_params_list = [self.inverse_params_list]
        
        # Initialize parameter tracking for each parameter
        self.param_history = {
            'epochs': [],
            'stages': []
        }
        
        # Add tracking for each parameter
        for param_name in self.inverse_params_list:
            self.param_history[f'{param_name}_values'] = []
            self.param_history[f'{param_name}_losses'] = []
        
        # Validate config has stages
        if not self.config.training.stages:
            raise ValueError(
                "Inverse training requires config.training.stages to be defined.\n"
                "See configs/birnn_inverse.yaml for example."
            )
    
    def _infer_model_type(self) -> str:
        """Infer model type from class name."""
        class_name = self.model.__class__.__name__
        if 'BIRNN' in class_name or 'GRU' in class_name:
            return 'birnn'
        elif 'Feedforward' in class_name:
            return 'pinn'
        elif 'ModifiedMLP' in class_name:
            return 'modified_mlp'
        else:
            raise ValueError(f"Unknown model type: {class_name}")
    
    def train(self) -> Dict[str, Any]:
        """
        Run all training stages from config.
        
        Returns:
            Dictionary with training history and final results:
                - param_history: Evolution of parameter values
                - final_params: Dict of final parameter values
                - param_errors_percent: Dict of errors vs true values (if provided)
                - stages_completed: Number of stages completed
        """
        print(f"\n{'='*80}")
        print(f"INVERSE TRAINING - {len(self.config.training.stages)} STAGES")
        print(f"{'='*80}")
        print(f"Model: {self.model_type.upper()}")
        print(f"Parameters: {self.inverse_params_list}")
        if self.true_param_value:
            # true_param_value is only for first parameter (backward compatibility)
            print(f"True {self.inverse_params_list[0]}: {self.true_param_value:.2f}")
        print(f"{'='*80}\n")
        
        cumulative_epoch = 0
        
        # Execute each stage
        for i, stage_dict in enumerate(self.config.training.stages, 1):
            print(f"\n{'='*80}")
            print(f"STAGE {i}/{len(self.config.training.stages)}: {stage_dict['name']}")
            print(f"{'='*80}")
            print(f"Epochs: {stage_dict['epochs']}")
            print(f"Learning rate: {stage_dict['learning_rate']}")
            print(f"Train inverse params: {stage_dict['train_inverse_params']}")
            print(f"Train NN weights: {stage_dict['train_nn_weights']}")
            print(f"{'='*80}\n")
            
            # Train this stage
            cumulative_epoch = self._train_stage(
                stage_dict,
                cumulative_epoch,
                stage_name=stage_dict['name']
            )
        
        # Get final parameter values for all estimated parameters
        final_params = {}
        param_errors = {}
        
        for param_name in self.inverse_params_list:
            final_value = self._get_current_param_value(param_name)
            final_params[param_name] = final_value
            
            # Compute error if true value available (only for first param - backward compat)
            if param_name == self.inverse_params_list[0] and self.true_param_value:
                error = abs(final_value - self.true_param_value) / self.true_param_value * 100
                param_errors[param_name] = error
        
        print(f"\n{'='*80}")
        print("INVERSE TRAINING COMPLETE")
        print(f"{'='*80}")
        for param_name, final_value in final_params.items():
            print(f"Final {param_name}: {final_value:.2f}")
            if param_name in param_errors:
                print(f"  True {param_name}: {self.true_param_value:.2f}")
                print(f"  Error: {param_errors[param_name]:.2f}%")
        print(f"{'='*80}\n")
        
        return {
            'param_history': self.param_history,
            'final_params': final_params,
            'param_errors_percent': param_errors,
            'stages_completed': len(self.config.training.stages)
        }
    
    def _train_stage(
        self,
        stage_config: Dict,
        cumulative_epoch: int,
        stage_name: str
    ) -> int:
        """
        Train a single stage.
        
        Args:
            stage_config: Stage configuration dictionary
            cumulative_epoch: Starting epoch number
            stage_name: Stage name for tracking
        
        Returns:
            Updated cumulative epoch count
        """
        # Determine what to train
        train_params = stage_config['train_inverse_params']
        train_nn = stage_config['train_nn_weights']
        
        if not train_params and not train_nn:
            raise ValueError(
                f"Stage '{stage_name}': At least one of train_inverse_params "
                f"or train_nn_weights must be true"
            )
        
        # CRITICAL FIX: Check if this is a DeepXDE model BEFORE getting trainable vars
        # DeepXDE models don't use the same trainable_variables pattern
        is_deepxde = hasattr(self.model, 'model') and hasattr(self.model.model, 'train')
        
        if is_deepxde:
            # DeepXDE models handle training differently - run entire stage at once
            return self._train_stage_deepxde(
                stage_config, cumulative_epoch, stage_name, train_params, train_nn
            )
        
        # For Keras models (BI-RNN), get trainable variables
        var_list = self._get_trainable_vars(train_params, train_nn)
        
        # For Keras models (BI-RNN), use epoch-by-epoch training
        # Create optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=stage_config['learning_rate']
        )
        
        # Training loop
        epochs = stage_config['epochs']
        display_every = stage_config.get('display_every', max(100, epochs // 20))
        
        for epoch in range(epochs):
            # Training step (pass optimizer to training step)
            loss_dict = self._train_step(var_list, train_nn, stage_config, optimizer)
            
            # Track parameters (every 10 epochs)
            if epoch % 10 == 0:
                self.param_history['epochs'].append(cumulative_epoch + epoch)
                self.param_history['stages'].append(stage_name)
                
                # Track each parameter being estimated
                for param_name in self.inverse_params_list:
                    param_value = self._get_current_param_value(param_name)
                    self.param_history[f'{param_name}_values'].append(param_value)
                    self.param_history[f'{param_name}_losses'].append(loss_dict['total'])
            
            # Display progress
            if epoch % display_every == 0 or epoch == epochs - 1:
                # Show all parameters being estimated
                param_strs = []
                for param_name in self.inverse_params_list:
                    param_value = self._get_current_param_value(param_name)
                    param_strs.append(f"{param_name}={param_value:.2f}")
                
                print(f"  Epoch {epoch}/{epochs}: "
                      f"Loss = {loss_dict['total']:.4f}, "
                      f"{', '.join(param_strs)}")
        
        return cumulative_epoch + epochs
    
    def _train_stage_deepxde(
        self,
        stage_config: dict,
        cumulative_epoch: int,
        stage_name: str,
        train_params: bool,
        train_nn: bool
    ) -> int:
        """
        Train a complete stage for DeepXDE models.
        
        Uses callbacks to track parameter values during training.
        
        Args:
            stage_config: Stage configuration
            cumulative_epoch: Cumulative epoch count
            stage_name: Name of stage
            train_params: Whether to train inverse parameters
            train_nn: Whether to train NN weights
        
        Returns:
            Updated cumulative epoch count
        """
        epochs = stage_config['epochs']
        display_every = stage_config.get('display_every', max(100, epochs // 20))
        
        print(f"\n  🔧 DeepXDE Training: Running {epochs} epochs...")
        print(f"     Training params: {train_params}, Training NN: {train_nn}")
        
        # ============================================================================
        # CRITICAL: Manually freeze/unfreeze variables to enforce stage separation
        # ============================================================================
        # DeepXDE doesn't support external_trainable_variables, so we must manually
        # set the trainable flag on TensorFlow variables to control what gets updated.
        
        # Step 1: Collect all inverse parameter variables
        inverse_param_vars = []
        if hasattr(self.model, 'inverse_params') and self.model.inverse_params:
            inverse_param_vars = self.model.inverse_params  # List of tf.Variable
        
        # Step 2: Collect all NN weight variables (excluding inverse params)
        nn_weight_vars = []
        if hasattr(self.model.model, 'trainable_variables'):
            all_trainable = self.model.model.trainable_variables
            for var in all_trainable:
                # Only include if it's NOT an inverse parameter
                if var not in inverse_param_vars:
                    nn_weight_vars.append(var)
        
        # Step 3: Set trainable flags based on stage configuration
        print(f"     Freezing variables for stage separation...")
        
        # Freeze/unfreeze inverse parameters
        for var in inverse_param_vars:
            var._trainable = train_params  # True to train, False to freeze
        
        # Freeze/unfreeze NN weights
        for var in nn_weight_vars:
            var._trainable = train_nn  # True to train, False to freeze
        
        print(f"        Inverse params ({len(inverse_param_vars)} vars): {'TRAINING' if train_params else 'FROZEN'}")
        print(f"        NN weights ({len(nn_weight_vars)} vars): {'TRAINING' if train_nn else 'FROZEN'}")
        
        # ============================================================================
        # CRITICAL FIX: Use callback to track parameters during training
        # ============================================================================
        import deepxde as dde
        
        class ParameterTracker(dde.callbacks.Callback):
            """Track parameter values during DeepXDE training."""
            
            def __init__(self, trainer, param_names, stage_name, cumulative_epoch):
                super().__init__()
                self.trainer = trainer
                self.param_names = param_names
                self.stage_name = stage_name
                self.cumulative_epoch = cumulative_epoch
            
            def on_epoch_end(self):
                """Called after each training epoch."""
                # Get current epoch from DeepXDE's training state
                epoch = self.model.train_state.epoch
                current_epoch = self.cumulative_epoch + epoch
                
                # Track each parameter
                for param_name in self.param_names:
                    # FIXED: Access log_param directly from InverseParams
                    # InverseParams has attributes like log_ksi, log_kl, etc.
                    log_param_var = getattr(self.trainer.model.inverse_params_obj, f'log_{param_name}', None)
                    
                    if log_param_var is None:
                        continue
                    
                    try:
                        # Try to get value via session (works in graph mode)
                        if hasattr(self.model, 'sess'):
                            # Evaluate the log variable
                            log_val = self.model.sess.run(log_param_var)
                            # Convert to actual parameter value (exp(log))
                            import numpy as np
                            param_val = float(np.exp(log_val))
                        else:
                            # Fallback: try to use the get_param_value method
                            param_val = self.trainer.model.inverse_params_obj.get_param_value(
                                param_name, as_float=False
                            )
                            # If it returns a tensor, try to evaluate it
                            if hasattr(param_val, 'numpy'):
                                param_val = float(param_val.numpy())
                            else:
                                param_val = float(param_val)
                    except Exception as e:
                        # If all else fails, skip tracking this epoch
                        continue
                    
                    # Store in history
                    self.trainer.param_history['epochs'].append(current_epoch)
                    self.trainer.param_history['stages'].append(self.stage_name)
                    self.trainer.param_history[f'{param_name}_values'].append(param_val)
                    
                    # Get current loss (it's an array, get the latest value)
                    if hasattr(self.model.train_state, 'loss_train'):
                        loss_array = self.model.train_state.loss_train
                        if isinstance(loss_array, (list, tuple)) or hasattr(loss_array, '__len__'):
                            # It's an array/list - get the last element
                            loss = float(loss_array[-1]) if len(loss_array) > 0 else 0.0
                        else:
                            # It's a scalar
                            loss = float(loss_array)
                    else:
                        loss = 0.0
                    self.trainer.param_history[f'{param_name}_losses'].append(loss)
        
        # Create callback instance
        tracker = ParameterTracker(
            trainer=self,
            param_names=self.inverse_params_list,
            stage_name=stage_name,
            cumulative_epoch=cumulative_epoch
        )
        
        # Run DeepXDE training with callback
        dde_model = self.model.model
        losshistory, train_state = dde_model.train(
            iterations=epochs,
            display_every=display_every,
            callbacks=[tracker]  # Add callback for parameter tracking
        )
        
        # Print final state
        print(f"\n  📊 Stage '{stage_name}' complete:")
        for param_name in self.inverse_params_list:
            # Get final value from history (last tracked value)
            if self.param_history[f'{param_name}_values']:
                param_value = self.param_history[f'{param_name}_values'][-1]
                print(f"     {param_name}: {param_value:.6f}")
                
                # Print error if true value available
                if param_name == self.inverse_params_list[0] and self.true_param_value is not None:
                    error = abs(param_value - self.true_param_value) / self.true_param_value * 100
                    print(f"        Error: {error:.2f}%")
        
        # ============================================================================
        # CLEANUP: Unfreeze all variables for next stage
        # ============================================================================
        # Reset trainable flags to True so next stage can control them fresh
        for var in inverse_param_vars:
            var._trainable = True
        for var in nn_weight_vars:
            var._trainable = True
        
        return cumulative_epoch + epochs
    
    def _get_trainable_vars(
        self,
        train_params: bool,
        train_nn: bool
    ) -> List[tf.Variable]:
        """
        Get list of variables to train based on flags.
        
        Args:
            train_params: Whether to train inverse parameters
            train_nn: Whether to train NN weights
        
        Returns:
            List of TensorFlow variables to train
        """
        var_list = []
        
        # Add inverse parameters if requested
        if train_params:
            # Collect all trainable inverse parameters
            if hasattr(self.model, 'inverse_params') and self.model.inverse_params:
                # inverse_params is already a list of tf.Variable objects
                var_list.extend(self.model.inverse_params)
            else:
                raise AttributeError("Model does not have trainable inverse parameters")
        
        # Add NN weights if requested
        if train_nn:
            if hasattr(self.model, 'model'):
                var_list.extend(self.model.model.trainable_variables)
            elif hasattr(self.model, 'dde_model'):
                var_list.extend(self.model.dde_model.trainable_variables)
            else:
                raise AttributeError("Cannot find model weights")
        
        return var_list
    
    def _train_step(
        self,
        var_list: List[tf.Variable],
        training: bool,
        stage_config: Dict,
        optimizer: tf.keras.optimizers.Optimizer
    ) -> Dict[str, float]:
        """
        Execute one training step.
        
        Args:
            var_list: Variables to update
            training: Whether to run model in training mode (for NN layers)
            stage_config: Stage configuration with loss_weights
            optimizer: Optimizer instance for this stage
        
        Returns:
            Dictionary of loss values
        """
        if self.model_type == 'birnn':
            return self._train_step_birnn(var_list, training, stage_config, optimizer)
        else:
            return self._train_step_deepxde(var_list, training, stage_config, optimizer)
    
    def _train_step_birnn(
        self,
        var_list: List[tf.Variable],
        training: bool,
        stage_config: Dict,
        optimizer: tf.keras.optimizers.Optimizer
    ) -> Dict[str, float]:
        """Training step for BI-RNN model."""
        with tf.GradientTape() as tape:
            # In TF 2.x eager mode, tape automatically watches all tf.Variable objects
            # No need to manually watch variables
            
            # Forward pass
            Y_pred = self.model.model(self.model.X_train, training=training)
            
            # Glucose loss
            loss_g = tf.reduce_mean(tf.square(
                Y_pred[:, :, 0:1] - self.model.Y_train[:, :, 0:1]
            ))
            
            # Biological residual loss
            loss_B = self._compute_biological_residual(Y_pred)
            
            # Initial condition loss
            loss_ic = tf.reduce_mean(tf.square(
                Y_pred[:, 0, :] - self.model.Y_train[:, 0, :]
            ))
            
            # Total loss with weights from stage config
            loss_weights = stage_config.get('loss_weights', [8.0, 4.82, 0.53])
            total_loss = (
                loss_weights[0] * loss_g +
                loss_weights[1] * loss_B +
                loss_weights[2] * loss_ic
            )
        
        # Compute gradients
        gradients = tape.gradient(total_loss, var_list)
        
        # Apply gradients using the stage's optimizer (not model's optimizer!)
        optimizer.apply_gradients(zip(gradients, var_list))
        
        return {
            'total': float(total_loss),
            'loss_g': float(loss_g),
            'loss_B': float(loss_B),
            'loss_ic': float(loss_ic)
        }
    
    def _compute_biological_residual(self, Y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute biological residual for BI-RNN.
        
        Checks if predicted states satisfy the ODE:
        y[t+1] ≈ y[t] + dt * f(y[t], u[t], r[t])
        
        Uses flexible parameter system - any combination of parameters can be trainable.
        """
        from src.physics.magdelaine import get_param_value
        
        # Reshape to [Batch*Time, Features]
        y_pred_flat = tf.reshape(Y_pred, [-1, 3])
        u_flat = tf.reshape(self.model.U_train, [-1, 1])
        r_flat = tf.reshape(self.model.R_train, [-1, 1])
        
        # We need to compute: y[t+1] = y[t] + dt * f(y[t], u[t], r[t])
        # CRITICAL: Use Y_pred (model's predictions) at time t, not Y_train (ground truth)!
        # This ensures gradients flow through ksi → ODE → Y_pred consistency
        
        # For a sequence of length T, we have T states: [0, 1, 2, ..., T-1]
        # We can compute T-1 transitions: [0→1, 1→2, ..., T-2→T-1]
        
        # PREDICTED states at time t (all except last)
        y_pred_t = y_pred_flat[:-1, :]  # Shape: [T-1, 3]
        
        # PREDICTED states at time t+1 (all except first)  
        y_pred_t_plus_1 = y_pred_flat[1:, :]  # Shape: [T-1, 3]
        
        # Inputs at time t (all except last)
        u_t = u_flat[:-1, :]  # Shape: [T-1, 1]
        r_t = r_flat[:-1, :]  # Shape: [T-1, 1]
        
        # Denormalize PREDICTED states at time t to physical units
        G_t = y_pred_t[:, 0:1] * self.data.m_g  # mg/dL
        I_t = y_pred_t[:, 1:2] * self.data.m_i  # U/dL
        D_t = y_pred_t[:, 2:3] * self.data.m_d  # mg/dL/min
        
        # Denormalize inputs (use stored max values)
        ut = u_t * self.model.u_max  # U/min
        rt = r_t * self.model.r_max  # g/min
        
        # Get parameters (trainable if in inverse_params_obj, else use preset)
        ksi = get_param_value(self.model.inverse_params_obj, self.model.params, 'ksi')
        kl = get_param_value(self.model.inverse_params_obj, self.model.params, 'kl')
        ku_Vi = get_param_value(self.model.inverse_params_obj, self.model.params, 'ku_Vi')
        kb = get_param_value(self.model.inverse_params_obj, self.model.params, 'kb')
        Tu = get_param_value(self.model.inverse_params_obj, self.model.params, 'Tu')
        Tr = get_param_value(self.model.inverse_params_obj, self.model.params, 'Tr')
        kr_Vb = get_param_value(self.model.inverse_params_obj, self.model.params, 'kr_Vb')
        
        # Magdelaine ODEs in physical units
        dG = -ksi * I_t + kl - kb + D_t  # mg/dL/min
        dI = -I_t / Tu + (ku_Vi / Tu) * ut  # U/dL/min
        dD = -D_t / Tr + (kr_Vb / Tr) * rt  # mg/dL/min²
        
        # Forward Euler with PHYSICAL timestep (1 minute)
        dt = 1.0  # minutes (data sampling rate)
        
        # Compute next state in physical units
        G_t_plus_1 = G_t + dt * dG
        I_t_plus_1 = I_t + dt * dI
        D_t_plus_1 = D_t + dt * dD
        
        # Normalize predicted next states for comparison with NN output
        y_ode_t_plus_1 = tf.concat([
            G_t_plus_1 / self.data.m_g,
            I_t_plus_1 / self.data.m_i,
            D_t_plus_1 / self.data.m_d
        ], axis=1)
        
        # Compare ODE prediction with NN prediction at time t+1
        return tf.reduce_mean(tf.square(y_ode_t_plus_1 - y_pred_t_plus_1))
    
    def _train_step_deepxde(
        self,
        var_list: list,
        training: bool,
        stage_config: dict,
        optimizer
    ) -> dict:
        """
        Train step for DeepXDE models (PINN, Modified-MLP).
        
        DeepXDE handles training differently - it has an internal training loop.
        For multi-stage training, we need to train for a certain number of epochs
        with specific variables.
        
        Args:
            var_list: List of variables to train (ignored - DeepXDE auto-collects)
            training: Whether in training mode
            stage_config: Stage configuration
            optimizer: Optimizer to use
        
        Returns:
            Dictionary of losses
        """
        # DeepXDE models have a 'model' attribute that is the dde.Model object
        if not hasattr(self.model, 'model'):
            raise AttributeError("DeepXDE model must have 'model' attribute")
        
        dde_model = self.model.model
        
        # Get training configuration
        epochs = stage_config.get('epochs', 1000)
        display_every = stage_config.get('display_every', 100)
        
        # DeepXDE doesn't support per-stage variable selection directly
        # It trains all variables that are marked as trainable
        # So we need to temporarily set trainability
        
        # Train for the specified number of epochs
        # Note: DeepXDE's train() runs the full training loop internally
        # We're calling it once per stage with the stage's epoch count
        
        losshistory, train_state = dde_model.train(
            iterations=epochs,  # FIXED: Use 'iterations' instead of deprecated 'epochs'
            display_every=display_every
        )
        
        # Extract final loss from training state
        # DeepXDE's train_state contains loss history
        if hasattr(train_state, 'loss_train') and len(train_state.loss_train) > 0:
            final_loss = float(train_state.loss_train[-1])
        else:
            final_loss = 0.0
        
        # Return loss dict (DeepXDE doesn't break down losses the same way)
        return {
            'total': final_loss,
            'glucose': 0.0,  # Not separately tracked
            'biological': 0.0,
            'ic': 0.0
        }
    
    def _get_current_param_value(self, param_name: str = 'ksi') -> float:
        """Get current value of inverse parameter."""
        from src.physics.magdelaine import get_param_value
        
        if hasattr(self.model, 'inverse_params_obj') and self.model.inverse_params_obj is not None:
            value = self.model.inverse_params_obj.get_param_value(param_name)
            if value is not None:
                # CRITICAL FIX: Handle TensorFlow tensors in graph mode
                if isinstance(value, tf.Tensor) or isinstance(value, tf.Variable):
                    # In TF 1.x graph mode, we need to evaluate the tensor
                    try:
                        # Try eager execution first (TF 2.x)
                        return float(value.numpy())
                    except:
                        # Graph mode fallback - try to read the variable value
                        if hasattr(value, 'read_value'):
                            try:
                                value = value.read_value()
                                return float(value.numpy())
                            except:
                                pass
                        # If we can't get the value, try to use the initial value
                        if hasattr(value, '_initial_value'):
                            try:
                                return float(value._initial_value.numpy())
                            except:
                                pass
                        # Last resort: check if it's a trainable variable with an assign operation
                        if hasattr(value, 'value'):
                            try:
                                return float(value.value().numpy())
                            except:
                                pass
                        # Really last resort: return NaN to indicate we couldn't get the value
                        # Don't return 0.0 as that's misleading
                        print(f"⚠️  Warning: Could not extract value for {param_name} from TF graph")
                        return float('nan')
                return float(value)
        
        # Fallback to model params (if not trainable)
        return float(getattr(self.model.params, param_name))


# Example usage
if __name__ == "__main__":
    print("InverseTrainer created successfully!")
    print("\nUsage example:")
    print("""
    from src.training.inverse_trainer import InverseTrainer
    from src.models.birnn import BIRNN
    from src.training.config import Config
    from src.datasets.loader import load_synthetic_window
    
    # Load config with stages defined
    config = Config.from_yaml('configs/birnn_inverse.yaml')
    
    # Load data
    data = load_synthetic_window(patient=3)
    
    # Build model
    model = BIRNN(config)
    model.build(data)
    model.compile()
    
    # Train with flexible multi-stage approach
    trainer = InverseTrainer(model, config, data, true_param_value=274.0)
    history = trainer.train()
    
    print(f"Final ksi: {history['final_param']:.2f}")
    print(f"Error: {history['param_error_percent']:.2f}%")
    """)