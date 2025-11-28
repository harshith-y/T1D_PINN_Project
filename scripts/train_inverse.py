#!/usr/bin/env python3
"""
Enhanced Inverse Training Script v2

Uses InverseTrainer for flexible multi-stage parameter estimation.

Usage:
    # Basic inverse training (requires config with stages)
    python scripts/train_inverse_v2.py --config configs/birnn_inverse.yaml --patient 3
    
    # Custom patient and parameter
    python scripts/train_inverse_v2.py --config configs/pinn_inverse.yaml --patient 5 --param ksi
"""

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# CRITICAL: Disable eager for DeepXDE models
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

import argparse
from datetime import datetime

from src.datasets.loader import load_synthetic_window, load_real_patient_csv
from src.training.config import Config
from src.training.checkpoint import CheckpointManager
from src.training.predictions import PredictionManager
from src.training.inverse_trainer import InverseTrainer
from src.physics.magdelaine import make_params_from_preset


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


def main():
    """Main inverse training function."""
    parser = argparse.ArgumentParser(description='Enhanced inverse training v2')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to inverse config YAML with stages defined')
    parser.add_argument('--patient', type=int, required=True,
                        help='Patient number (2-11 for synthetic, 1-15 for real)')
    parser.add_argument('--data-type', type=str, default='synthetic',
                        choices=['synthetic', 'real'],
                        help='Data type: synthetic (with ground truth) or real (estimates only)')
    parser.add_argument('--inverse-params', type=str, nargs='+', default=['ksi'],
                        help='Parameters to estimate (default: ksi). '
                             'Options: ksi kl ku_Vi kb Tu Tr kr_Vb M. '
                             'Examples: --inverse-params ksi  OR  --inverse-params ksi kl ku_Vi')
    parser.add_argument('--save-dir', type=str, default=None)
    
    args = parser.parse_args()
    
    # Validate patient number based on data type
    if args.data_type == 'synthetic' and not (2 <= args.patient <= 11):
        print("❌ Error: Synthetic patients must be in range 2-11")
        sys.exit(1)
    elif args.data_type == 'real' and not (1 <= args.patient <= 15):
        print("❌ Error: Real patients must be in range 1-15")
        sys.exit(1)
    
    print("="*80)
    print("INVERSE TRAINING v2 (Enhanced)")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Load Configuration
    # ========================================================================
    print("\n[1/7] Loading configuration...")
    config = Config.from_yaml(args.config)
    
    # Validate config
    if config.mode != 'inverse':
        print("⚠️  Warning: Config mode is not 'inverse', setting to 'inverse'")
        config.mode = 'inverse'
    
    if not config.training.stages:
        print("❌ Error: Config must have training.stages defined for inverse training")
        print("   See configs/birnn_inverse.yaml for example")
        sys.exit(1)
    
    # Set inverse parameters from command line (overrides config)
    # Default is ['ksi'] if not specified
    config.inverse_params = args.inverse_params
    
    model_display = {
        'birnn': 'BI-RNN',
        'pinn': 'PINN',
        'modified_mlp': 'Modified-MLP'
    }[config.model_name]
    
    print(f"✅ Config loaded")
    print(f"   Model: {model_display}")
    print(f"   Parameters: {config.inverse_params}")
    print(f"   Stages: {len(config.training.stages)}")
    
    # Set output directory
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        patient_label = f"Pat{args.patient}" if args.data_type == 'synthetic' else f"RealPat{args.patient}"
        args.save_dir = f"results/{config.model_name}_inverse/{patient_label}_{config.inverse_param}_{timestamp}"
    
    config.output.save_dir = args.save_dir
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"   Output: {args.save_dir}")
    
    # ========================================================================
    # STEP 2: Load Data
    # ========================================================================
    print(f"\n[2/7] Loading {args.data_type} data...")
    
    if args.data_type == 'synthetic':
        data = load_synthetic_window(patient=args.patient, root='data/synthetic')
        
        if not data.has_latent_states:
            print("❌ Error: Inverse training requires synthetic data with ground truth")
            sys.exit(1)
        
        # Get true parameter value (for error calculation)
        true_params = make_params_from_preset(args.patient)
        true_param_value = getattr(true_params, config.inverse_param)
        print(f"   True {config.inverse_param}: {true_param_value:.2f}")
        
    else:  # real patient
        data = load_real_patient_csv(
            patient=args.patient,
            root='data/processed',
            t_start=0,
            t_end=2880  # 48 hours
        )
        
        # No ground truth for real patients
        true_param_value = None
        print(f"   ⚠️  Real patient: No ground truth available")
        print(f"   Will estimate {config.inverse_param} but cannot compute error")
    
    print(f"✅ Data loaded: {data}")
    
    # ========================================================================
    # STEP 3: Build Model
    # ========================================================================
    print(f"\n[3/7] Building {model_display} model...")
    model = load_model(config.model_name, config)
    model.build(data)
    print("✅ Model built")
    
    # ========================================================================
    # STEP 4: Initialize Managers
    # ========================================================================
    print("\n[4/7] Initializing managers...")
    checkpoint_mgr = CheckpointManager(args.save_dir, config)
    pred_mgr = PredictionManager(args.save_dir)
    print("✅ Managers initialized")
    
    # ========================================================================
    # STEP 5: Compile Model
    # ========================================================================
    print("\n[5/7] Compiling model...")
    model.compile()
    print("✅ Model compiled")
    
    # ========================================================================
    # STEP 6: Train with InverseTrainer (NEW!)
    # ========================================================================
    print(f"\n[6/7] Starting inverse training...")
    print("="*80)
    
    trainer = InverseTrainer(
        model=model,
        config=config,
        data=data,
        true_param_value=true_param_value
    )
    
    history = trainer.train()  # Runs all stages!
    
    print("="*80)
    print("✅ Inverse training completed")
    
    # ========================================================================
    # STEP 7: Save Everything
    # ========================================================================
    print("\n[7/7] Saving results...")
    
    # Evaluate
    metrics = model.evaluate() if hasattr(model, 'evaluate') else {}
    metrics.update({
        f'{config.inverse_param}_estimated': history['final_param'],
        f'{config.inverse_param}_true': true_param_value,
        f'{config.inverse_param}_error_percent': history['param_error_percent']
    })
    
    # Save parameter evolution (NEW!)
    if history['param_history']['epochs']:
        import numpy as np
        pred_mgr.save_parameter_evolution(
            epochs=np.array(history['param_history']['epochs']),
            param_values=np.array(history['param_history']['param_values']),
            param_name=config.inverse_param,
            true_value=true_param_value,
            losses=np.array(history['param_history']['losses']),
            stages=np.array(history['param_history']['stages'])
        )
        print("✅ Parameter evolution saved")
    
    # Save final checkpoint
    checkpoint_mgr.save(
        model=model,
        optimizer=getattr(model, 'optimizer', None),
        epoch=sum(s['epochs'] for s in config.training.stages),
        metrics=metrics,
        is_final=True,
        is_best=(history['param_error_percent'] < 10)
    )
    print("✅ Checkpoint saved")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("✅ INVERSE TRAINING COMPLETE!")
    print("="*80)
    
    print(f"\n📊 Parameter Estimation:")
    print(f"   Parameter: {config.inverse_param}")
    print(f"   Estimated: {history['final_param']:.2f}")
    print(f"   True value: {true_param_value:.2f}")
    print(f"   Error: {history['param_error_percent']:.2f}%")
    
    if history['param_error_percent'] < 5:
        print(f"   🎉 Excellent! Error < 5%")
    elif history['param_error_percent'] < 10:
        print(f"   ✅ Good! Error < 10%")
    elif history['param_error_percent'] < 20:
        print(f"   ⚠️  Acceptable. Error < 20%")
    else:
        print(f"   ❌ Poor. Error > 20%")
    
    print(f"\n📁 Output:")
    print(f"   {args.save_dir}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Inverse training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)