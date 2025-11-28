#!/usr/bin/env python3
"""
Universal Training Script with Integrated Visualization

Trains any model architecture (BI-RNN, PINN, Modified-MLP) on synthetic or real
patient data and automatically generates publication-quality plots.

Usage:
    # BI-RNN on synthetic patient
    python train_with_visualization.py --model birnn --patient 3 --data-type synthetic --epochs 2000
    
    # PINN on real patient
    python train_with_visualization.py --model pinn --patient 5 --data-type real --epochs 1000
    
    # Modified-MLP on synthetic
    python train_with_visualization.py --model modified_mlp --patient 3 --data-type synthetic --epochs 1000
"""

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
from pathlib import Path
# Get project root (parent of scripts directory)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# CRITICAL: Disable eager for DeepXDE models
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

import numpy as np
import argparse
from datetime import datetime

from src.datasets.loader import load_synthetic_window, load_real_patient_csv
from src.training.config import load_config
from src.visualisation.plotter import ExperimentPlotter


def load_model(model_name: str, config):
    """Load the appropriate model class."""
    if model_name == 'birnn':
        from src.models.birnn import BIRNN
        return BIRNN(config)
    elif model_name == 'pinn':
        from src.models.pinn_feedforward import FeedforwardPINN
        return FeedforwardPINN(config)
    elif model_name == 'modified_mlp':
        from src.models.modified_mlp import ModifiedMLPPINN
        return ModifiedMLPPINN(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_with_visualization(
    model_name: str,
    patient: int,
    data_type: str = 'synthetic',
    epochs: int = 2000,
    save_dir: str = None
):
    """
    Universal training function with visualization.
    
    Args:
        model_name: 'birnn', 'pinn', or 'modified_mlp'
        patient: Patient number
        data_type: 'synthetic' or 'real'
        epochs: Number of training epochs
        save_dir: Output directory
    """
    
    model_display = {
        'birnn': 'BI-RNN',
        'pinn': 'PINN',
        'modified_mlp': 'Modified-MLP'
    }[model_name]
    
    print("="*80)
    print(f"{model_display} FORWARD TRAINING WITH VISUALIZATION")
    print("="*80)
    print(f"Model: {model_display}")
    print(f"Patient: {patient}")
    print(f"Data type: {data_type}")
    print(f"Epochs: {epochs}")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Load Configuration
    # ========================================================================
    print("\n[1/7] Loading configuration...")
    config = load_config(model_name=model_name, mode='forward')
    config.training.epochs = epochs
    
    # Set output directory
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        patient_label = f"Pat{patient}" if data_type == 'synthetic' else f"RealPat{patient}"
        save_dir = f"results/{model_name}_forward/{patient_label}_{timestamp}"
    
    config.output.save_dir = save_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Config loaded")
    print(f"   Output: {save_dir}")
    
    # ========================================================================
    # STEP 2: Load Data
    # ========================================================================
    print("\n[2/7] Loading data...")
    
    if data_type == 'synthetic':
        data = load_synthetic_window(patient=patient, root='data/synthetic')
        patient_label = f"Pat{patient}"
    elif data_type == 'real':
        # Load real patient data with 48-hour limit to prevent memory issues
        # Real patient data can be very large (100k-300k points)
        # Limit to first 48 hours (2880 minutes) to match synthetic data length
        data = load_real_patient_csv(
            patient=patient, 
            root='data/processed',
            t_start=0,
            t_end=2880  # 48 hours = 2880 minutes
        )
        patient_label = f"RealPat{patient}"
        print(f"   ⚠️  Limited to first 48 hours (2880 minutes) to prevent memory issues")
    else:
        raise ValueError(f"Invalid data_type: {data_type}")
    
    print(f"✅ Data loaded: {data}")
    print(f"   Length: {len(data.glucose)} points")
    print(f"   Glucose range: {data.glucose.min():.1f} - {data.glucose.max():.1f} mg/dL")
    print(f"   Has latent states: {data.has_latent_states}")
    
    # ========================================================================
    # STEP 3: Build Model
    # ========================================================================
    print(f"\n[3/7] Building {model_display} model...")
    model = load_model(model_name, config)
    model.build(data)
    print("✅ Model built")
    
    # ========================================================================
    # STEP 4: Compile Model
    # ========================================================================
    print("\n[4/7] Compiling model...")
    model.compile()
    print("✅ Model compiled")
    
    # ========================================================================
    # STEP 5: Train Model
    # ========================================================================
    print(f"\n[5/7] Training for {epochs} epochs...")
    print("="*80)
    
    try:
        model.train(display_every=max(100, epochs // 20))
        print("="*80)
        print("✅ Training completed")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
    
    # ========================================================================
    # STEP 6: Evaluate and Get Predictions
    # ========================================================================
    print("\n[6/7] Evaluating model...")
    
    try:
        metrics = model.evaluate()
        print("✅ Evaluation completed")
        
        # Extract RMSE metrics
        rmse_glucose = metrics.get('test_rmse_G', metrics.get('test_loss', 0))
        rmse_total = metrics.get('test_rmse_total', rmse_glucose)
        
        print(f"   RMSE (glucose): {rmse_glucose:.2f} mg/dL")
        if 'test_rmse_I' in metrics:
            print(f"   RMSE (insulin): {metrics['test_rmse_I']:.4f} U/dL")
        if 'test_rmse_D' in metrics:
            print(f"   RMSE (digestion): {metrics['test_rmse_D']:.2f} mg/dL/min")
        print(f"   RMSE (total): {rmse_total:.2f}")
        
    except Exception as e:
        print(f"⚠️  Evaluation warning: {e}")
        # Use placeholder values
        rmse_glucose = 0
        rmse_total = 0
    
    # Get predictions from model
    print("\n   Getting predictions...")
    
    if model_name == 'birnn':
        # BI-RNN predictions
        X_full = tf.concat([model.X_train, model.X_test], axis=1)
        Y_full = tf.concat([model.Y_train, model.Y_test], axis=1)
        Y_pred_full = model.model(X_full, training=False)
        
        pred_glucose = Y_pred_full[0, :, 0].numpy() * data.m_g
        true_glucose = Y_full[0, :, 0].numpy() * data.m_g
        
        # CRITICAL: BI-RNN output is shorter than input (sequence processing)
        # Use the time array that matches the prediction length
        time_array = data.t_min[:len(pred_glucose)]
        
        if data.has_latent_states:
            pred_insulin = Y_pred_full[0, :, 1].numpy() * data.m_i
            pred_digestion = Y_pred_full[0, :, 2].numpy() * data.m_d
            true_insulin = Y_full[0, :, 1].numpy() * data.m_i
            true_digestion = Y_full[0, :, 2].numpy() * data.m_d
        else:
            pred_insulin = pred_digestion = None
            true_insulin = true_digestion = None
        
        split_idx = model.X_train.shape[1]
        
    else:
        # DeepXDE models (PINN, Modified-MLP)
        # Extract real predictions from DeepXDE model
        
        print("   Extracting predictions from DeepXDE model...")
        
        # DeepXDE uses normalized time as input
        # Create time array for prediction
        t_norm = data.time_norm.reshape(-1, 1)  # Shape: (N, 1)
        
        # Get predictions from DeepXDE model
        # Note: Different models may have different attribute names
        if hasattr(model, 'dde_model'):
            dde_model = model.dde_model
        elif hasattr(model, 'model'):
            dde_model = model.model
        else:
            raise AttributeError(f"Cannot find DeepXDE model in {type(model).__name__}")
        
        # This returns [G_norm, I_norm, D_norm] in normalized units
        Y_pred_norm = dde_model.predict(t_norm)
        
        # Denormalize predictions
        # Y_pred_norm shape: (N, 3) for [G, I, D]
        pred_glucose = Y_pred_norm[:, 0] * data.m_g
        
        # Get ground truth
        true_glucose = data.glucose
        
        # Make sure lengths match
        min_len = min(len(pred_glucose), len(true_glucose))
        pred_glucose = pred_glucose[:min_len]
        true_glucose = true_glucose[:min_len]
        time_array = data.t_min[:min_len]
        
        if data.has_latent_states:
            # Extract and denormalize latent state predictions
            pred_insulin = Y_pred_norm[:, 1] * data.m_i
            pred_digestion = Y_pred_norm[:, 2] * data.m_d
            
            # Get ground truth
            true_insulin = data.insulin[:min_len]
            true_digestion = data.digestion[:min_len]
            
            # Truncate predictions
            pred_insulin = pred_insulin[:min_len]
            pred_digestion = pred_digestion[:min_len]
        else:
            pred_insulin = pred_digestion = None
            true_insulin = true_digestion = None
        
        split_idx = int(0.8 * min_len)
        
        print(f"   Prediction shape: {Y_pred_norm.shape}")
        print(f"   Glucose pred range: {pred_glucose.min():.1f} - {pred_glucose.max():.1f} mg/dL")
        print(f"   Glucose true range: {true_glucose.min():.1f} - {true_glucose.max():.1f} mg/dL")
    
    # Recompute RMSE from predictions
    rmse_glucose = float(np.sqrt(np.mean((pred_glucose - true_glucose) ** 2)))
    
    if pred_insulin is not None:
        rmse_insulin = float(np.sqrt(np.mean((pred_insulin - true_insulin) ** 2)))
        rmse_digestion = float(np.sqrt(np.mean((pred_digestion - true_digestion) ** 2)))
        rmse_total = float(np.sqrt(rmse_glucose**2 + rmse_insulin**2 + rmse_digestion**2))
    else:
        rmse_total = rmse_glucose
    
    print(f"   ✅ RMSE (from predictions): {rmse_glucose:.2f} mg/dL")
    
    # ========================================================================
    # STEP 7: Generate Visualizations
    # ========================================================================
    print("\n[7/7] Generating plots...")
    
    plotter = ExperimentPlotter(
        save_dir=save_dir,
        model_name=f"{model_display} Forward"
    )
    
    # Prepare data for plotter
    predictions_dict = {
        'time': time_array,  # Use the time array that matches prediction length
        'glucose': pred_glucose
    }
    
    ground_truth_dict = {
        'glucose': true_glucose
    }
    
    metrics_dict = {
        'rmse_glucose': rmse_glucose,
        'rmse_total': rmse_total
    }
    
    # Add latent states if available
    if pred_insulin is not None:
        predictions_dict['insulin'] = pred_insulin
        predictions_dict['digestion'] = pred_digestion
        ground_truth_dict['insulin'] = true_insulin
        ground_truth_dict['digestion'] = true_digestion
        metrics_dict['rmse_insulin'] = rmse_insulin
        metrics_dict['rmse_digestion'] = rmse_digestion
    
    # Generate plots
    plot_paths = plotter.plot_all(
        predictions=predictions_dict,
        ground_truth=ground_truth_dict,
        metrics=metrics_dict,
        split_idx=split_idx if data_type == 'synthetic' else None
    )
    
    print("✅ Plots generated:")
    for name, path in plot_paths.items():
        print(f"   📈 {name}: {path.name}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📊 Final Metrics:")
    print(f"   Model: {model_display}")
    print(f"   Patient: {patient_label}")
    print(f"   Data type: {data_type}")
    print(f"   Epochs: {epochs}")
    print(f"   RMSE (glucose): {rmse_glucose:.2f} mg/dL")
    if pred_insulin is not None:
        print(f"   RMSE (insulin): {rmse_insulin:.4f} U/dL")
        print(f"   RMSE (digestion): {rmse_digestion:.2f} mg/dL/min")
    print(f"   RMSE (total): {rmse_total:.2f}")
    
    print(f"\n📁 Output:")
    print(f"   {save_dir}")
    
    print(f"\n💡 View plots:")
    print(f"   open {save_dir}")
    
    print("\n" + "="*80)
    
    return {
        'model': model_name,
        'patient': patient,
        'data_type': data_type,
        'rmse_glucose': rmse_glucose,
        'rmse_total': rmse_total,
        'save_dir': save_dir
    }


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Train any model with automatic visualization'
    )
    parser.add_argument('--model', type=str, required=True,
                        choices=['birnn', 'pinn', 'modified_mlp'],
                        help='Model architecture')
    parser.add_argument('--patient', type=int, required=True,
                        help='Patient number (2-11 synthetic, 1-15 real)')
    parser.add_argument('--data-type', type=str, default='synthetic',
                        choices=['synthetic', 'real'],
                        help='Data type')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Validate patient number
    if args.data_type == 'synthetic' and not (2 <= args.patient <= 11):
        print("❌ Error: Synthetic patients must be 2-11")
        sys.exit(1)
    elif args.data_type == 'real' and not (1 <= args.patient <= 15):
        print("❌ Error: Real patients must be 1-15")
        sys.exit(1)
    
    # Run training
    try:
        results = train_with_visualization(
            model_name=args.model,
            patient=args.patient,
            data_type=args.data_type,
            epochs=args.epochs,
            save_dir=args.save_dir
        )
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()