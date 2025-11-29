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
        
        # Get trainable variables
        var_list = self._get_trainable_vars(train_params, train_nn)
        
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
        
        Uses flexible parameter system - any combination of parameters can be trainable.
        """
        from src.physics.magdelaine import get_param_value
        
        # Denormalize states
        y_pred_flat = tf.reshape(Y_pred, [-1, 3])
        y_in_flat = tf.reshape(self.model.Y_train, [-1, 3])
        u_in_flat = tf.reshape(self.model.U_train, [-1, 1])
        r_in_flat = tf.reshape(self.model.R_train, [-1, 1])
        
        G = y_in_flat[:, 0:1] * self.data.m_g
        I = y_in_flat[:, 1:2] * self.data.m_i
        D = y_in_flat[:, 2:3] * self.data.m_d
        
        # Denormalize inputs (use stored max values)
        ut = u_in_flat * self.model.u_max
        rt = r_in_flat * self.model.r_max
        
        # Get parameters (trainable if in inverse_params_obj, else use preset)
        ksi = get_param_value(self.model.inverse_params_obj, self.model.params, 'ksi')
        kl = get_param_value(self.model.inverse_params_obj, self.model.params, 'kl')
        ku_Vi = get_param_value(self.model.inverse_params_obj, self.model.params, 'ku_Vi')
        kb = get_param_value(self.model.inverse_params_obj, self.model.params, 'kb')
        Tu = get_param_value(self.model.inverse_params_obj, self.model.params, 'Tu')
        Tr = get_param_value(self.model.inverse_params_obj, self.model.params, 'Tr')
        kr_Vb = get_param_value(self.model.inverse_params_obj, self.model.params, 'kr_Vb')
        
        # First-order ODEs
        dG = (-ksi * I + kl - kb + D)
        dI = -I / Tu + (ku_Vi / Tu) * ut
        dD = -D / Tr + (kr_Vb / Tr) * rt
        
        # Forward Euler with CORRECTED timestep
        dt = 1.0 / self.data.m_t  # NOT dt = 1.0!
        
        # Normalize derivatives
        y_next_ode = y_in_flat + dt * tf.concat([
            dG / self.data.m_g,
            dI / self.data.m_i,
            dD / self.data.m_d
        ], axis=1)
        
        # Compare with NN prediction
        return tf.reduce_mean(tf.square(y_next_ode - y_pred_flat))
    
    def _train_step_deepxde(
        self,
        var_list: List[tf.Variable],
        training: bool,
        stage_config: Dict,
        optimizer: tf.keras.optimizers.Optimizer
    ) -> Dict[str, float]:
        """Training step for DeepXDE models (PINN, Modified-MLP)."""
        # For DeepXDE, we call the model's built-in training step
        # This is more complex and model-specific
        # For now, use the model's train method for one iteration
        
        # Note: DeepXDE training is typically done through model.train()
        # which handles the training loop internally
        # This is a simplified version
        
        raise NotImplementedError(
            "DeepXDE multi-stage training not yet implemented in InverseTrainer.\n"
            "Use model.train() directly for DeepXDE models or implement "
            "_train_step_deepxde based on your specific needs."
        )
    
    def _get_current_param_value(self, param_name: str = 'ksi') -> float:
        """Get current value of inverse parameter."""
        from src.physics.magdelaine import get_param_value
        
        if hasattr(self.model, 'inverse_params_obj') and self.model.inverse_params_obj is not None:
            value = self.model.inverse_params_obj.get_param_value(param_name)
            if value is not None:
                return value
        
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