#!/usr/bin/env python3
"""
PINN Inverse Training - Log File Version

This version reads ksi values from the parameter log file instead of
trying to access TF sessions directly. This is more reliable with DeepXDE.
"""

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# CRITICAL: Disable eager execution BEFORE any other imports
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

import numpy as np
from src.datasets.loader import load_synthetic_window
from src.models.pinn_feedforward import FeedforwardPINN
from src.training.config import load_config

print("="*80)
print("PINN INVERSE TRAINING - LOG FILE METHOD")
print("="*80)

# Load config
print("\n[1/5] Loading configuration...")
config = load_config(model_name='pinn', mode='forward')
config.mode = 'inverse'
config.inverse_param = 'ksi'
config.training.epochs = 500  # Short test
config.training.use_lbfgs_refinement = False  # Skip for speed
config.output.save_dir = 'results/pinn_inverse_test'  # Custom output dir
print("✅ Config loaded")

# Load data
print("\n[2/5] Loading synthetic data...")
data = load_synthetic_window(patient=3, root='data/synthetic')
print(f"✅ Data loaded: {data}")

# Build model
print("\n[3/5] Building PINN model...")
model = FeedforwardPINN(config)
model.build(data)
print("✅ Model built")

# Check inverse params exist
if model.inverse_params and model.inverse_params.log_ksi is not None:
    print(f"   Inverse parameter initialized: log_ksi")
else:
    print("❌ ERROR: Inverse parameters not initialized!")
    sys.exit(1)

# Compile
print("\n[4/5] Compiling model...")
model.compile()
print("✅ Model compiled")

# Train
print("\n[5/5] Training (500 epochs)...")
try:
    model.train(display_every=100)
    print("✅ Training completed")
except Exception as e:
    print(f"❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Read ksi values from log file
print("\n[6/6] Reading parameter log...")
log_file = Path(config.output.save_dir) / "inverse_params.dat"

if not log_file.exists():
    print(f"❌ Log file not found: {log_file}")
    print("   Inverse parameter may not have been logged!")
    sys.exit(1)

try:
    with open(log_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    if len(lines) == 0:
        print("❌ Log file is empty!")
        sys.exit(1)
    
    # Parse first line: "0 [5.6123]"
    initial_log_ksi = float(lines[0].split()[1].strip('[]'))
    initial_ksi = np.exp(initial_log_ksi)
    
    # Parse last line
    final_log_ksi = float(lines[-1].split()[1].strip('[]'))
    final_ksi = np.exp(final_log_ksi)
    
    true_ksi = 274.0
    error = abs(final_ksi - true_ksi) / true_ksi * 100
    
    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Initial ksi: {initial_ksi:.2f}")
    print(f"Final ksi:   {final_ksi:.2f}")
    print(f"True ksi:    {true_ksi:.2f}")
    print(f"Error:       {error:.2f}%")
    print(f"Logged {len(lines)} parameter values during training")
    print("="*80)
    
except Exception as e:
    print(f"❌ Error reading log file: {e}")
    import traceback
    traceback.print_exc()
    error = 100.0

# Evaluate
print("\n[7/7] Evaluating...")
try:
    metrics = model.evaluate()
    print("✅ Evaluation completed")
    
    # Print all metrics
    for key, value in metrics.items():
        print(f"   {key}: {value:.6f}")
        
except Exception as e:
    print(f"⚠️  Evaluation warning: {e}")

# Final assessment
print("\n" + "="*80)
if error < 10:
    print("🎉 SUCCESS! Error < 10%")
    print("✨ PINN inverse training works!")
elif error < 20:
    print("⚠️  Acceptable: Error < 20%")
    print("   Consider more epochs or tuning hyperparameters")
else:
    print("❌ FAILURE: Error > 20%")
    print("   Something is wrong - check gradients and loss weights")
print("="*80)

sys.exit(0 if error < 20 else 1)