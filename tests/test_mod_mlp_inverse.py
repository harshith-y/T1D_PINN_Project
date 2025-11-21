#!/usr/bin/env python3
"""
Modified-MLP Inverse Training - Fixed Version

This version properly handles TensorFlow 1.x session management.
Key fix: Use DeepXDE's model session, not create a new one.
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
from src.models.modified_mlp import ModifiedMLPPINN
from src.training.config import load_config

print("="*80)
print("MODIFIED-MLP INVERSE TRAINING - CORRECTED SESSION HANDLING")
print("="*80)

# Load config
print("\n[1/5] Loading configuration...")
config = load_config(model_name='modified_mlp', mode='forward')
config.mode = 'inverse'
config.inverse_param = 'ksi'
config.training.epochs = 500  # Short test
config.training.use_lbfgs_refinement = False  # Skip for speed
print("✅ Config loaded")

# Load data
print("\n[2/5] Loading synthetic data...")
data = load_synthetic_window(patient=3, root='data/synthetic')
print(f"✅ Data loaded: {data}")

# Build model
print("\n[3/5] Building Modified-MLP model...")
model = ModifiedMLPPINN(config)
model.build(data)
print("✅ Model built")

# Compile - this initializes the session
print("\n[4/5] Compiling model...")
model.compile()
print("✅ Model compiled")

# Get initial ksi AFTER compilation (session is initialized)
# Use the model's dde_model session
try:
    sess = model.dde_model.sess
    initial_ksi = float(sess.run(tf.exp(model.inverse_params.log_ksi)))
    print(f"   Initial ksi: {initial_ksi:.2f}")
except Exception as e:
    print(f"⚠️  Could not read initial ksi: {e}")
    initial_ksi = None

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

# Get final ksi using model's session
try:
    sess = model.dde_model.sess
    final_ksi = float(sess.run(tf.exp(model.inverse_params.log_ksi)))
    true_ksi = 274.0
    error = abs(final_ksi - true_ksi) / true_ksi * 100
    
    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    if initial_ksi is not None:
        print(f"Initial ksi: {initial_ksi:.2f}")
    print(f"Final ksi:   {final_ksi:.2f}")
    print(f"True ksi:    {true_ksi:.2f}")
    print(f"Error:       {error:.2f}%")
    print("="*80)
    
except Exception as e:
    print(f"❌ Could not read final ksi: {e}")
    error = 100.0  # Assume failure

# Evaluate
print("\n[6/6] Evaluating...")
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
    print("✨ Modified-MLP inverse training works!")
elif error < 20:
    print("⚠️  Acceptable: Error < 20%")
    print("   Consider more epochs or tuning hyperparameters")
else:
    print("❌ FAILURE: Error > 20%")
    print("   Something is wrong - check gradients and loss weights")
print("="*80)

sys.exit(0 if error < 20 else 1)