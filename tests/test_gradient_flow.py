#!/usr/bin/env python3
"""
Minimal test to verify gradient flow through log_ksi.

This script isolates the gradient computation problem to help debug.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import tensorflow as tf
import numpy as np
from src.datasets.loader import load_synthetic_window
from src.models.birnn import BIRNN
from src.training.config import load_config

print("=" * 80)
print("GRADIENT FLOW DIAGNOSTIC")
print("=" * 80)

# Setup
print("\nSetting up model...")
config = load_config(model_name="birnn", mode="forward")
config.mode = "inverse"
data = load_synthetic_window(patient=3, root="data/synthetic")
model = BIRNN(config)
model.build(data)
model.compile()

print(f"✅ Model built")
print(f"   log_ksi exists: {model.log_ksi is not None}")
print(f"   log_ksi value: {float(model.log_ksi):.4f}")
print(f"   ksi value: {float(tf.exp(model.log_ksi)):.2f}")

# Test 1: Simple gradient computation
print("\n" + "=" * 80)
print("TEST 1: Can we compute gradients for a simple loss?")
print("=" * 80)

with tf.GradientTape() as tape:
    tape.watch(model.log_ksi)
    # Simple loss: just the variable itself
    simple_loss = model.log_ksi**2

grad_simple = tape.gradient(simple_loss, model.log_ksi)
print(f"Simple loss: {float(simple_loss):.4f}")
print(f"Gradient: {float(grad_simple) if grad_simple is not None else 'None'}")

if grad_simple is not None:
    print("✅ Basic gradient computation works!")
else:
    print("❌ Basic gradient computation FAILED!")
    sys.exit(1)

# Test 2: Gradient through model forward pass
print("\n" + "=" * 80)
print("TEST 2: Gradients through model forward pass?")
print("=" * 80)

with tf.GradientTape() as tape:
    tape.watch(model.log_ksi)
    Y_pred = model.model(model.X_train, training=False)
    loss_g = tf.reduce_mean(tf.square(Y_pred[:, :, 0:1] - model.Y_train[:, :, 0:1]))

grad_forward = tape.gradient(loss_g, model.log_ksi)
print(f"Glucose loss: {float(loss_g):.6f}")
print(f"Gradient: {float(grad_forward) if grad_forward is not None else 'None'}")

if grad_forward is None:
    print("❌ Gradient is None - log_ksi not connected to model output")
    print("   This is expected! The NN doesn't use log_ksi internally.")
else:
    print("✅ Gradient exists (surprising!)")

# Test 3: Gradient through biological residual
print("\n" + "=" * 80)
print("TEST 3: Gradients through biological residual?")
print("=" * 80)

with tf.GradientTape() as tape:
    tape.watch(model.log_ksi)

    # Forward pass
    Y_pred = model.model(model.X_train, training=False)

    # Biological residual computation
    G_pred = Y_pred[:, :, 0:1] * model.data_window.m_g
    G_seq = tf.squeeze(G_pred, axis=0)
    U_seq = tf.squeeze(model.U_train, axis=0) * model.u_max
    R_seq = tf.squeeze(model.R_train, axis=0) * model.r_max

    # Use log_ksi in physics
    from src.physics.magdelaine import residuals_euler_seq
    import copy

    temp_params = copy.copy(model.params)
    temp_params.ksi = tf.exp(model.log_ksi)  # ← Use trainable parameter

    bio_losses = residuals_euler_seq(
        G_seq, U_seq, R_seq, temp_params, dt=1.0, use_latent_sim=True
    )
    loss_B = tf.reduce_mean(tf.square(bio_losses["LB"]))

grad_bio = tape.gradient(loss_B, model.log_ksi)
print(f"Biological loss: {float(loss_B):.6f}")
print(f"Gradient: {float(grad_bio) if grad_bio is not None else 'None'}")

if grad_bio is not None:
    print("✅ Gradient flows through biological residual!")
    print(f"   Gradient magnitude: {float(tf.abs(grad_bio)):.6f}")
else:
    print("❌ Gradient is None - log_ksi not connected to biological residual")
    print("   Check if residuals_euler_seq properly uses temp_params.ksi")

# Test 4: Combined loss
print("\n" + "=" * 80)
print("TEST 4: Gradients through combined loss?")
print("=" * 80)

with tf.GradientTape() as tape:
    tape.watch(model.log_ksi)

    # Forward pass
    Y_pred = model.model(model.X_train, training=False)

    # Glucose loss
    loss_g = tf.reduce_mean(tf.square(Y_pred[:, :, 0:1] - model.Y_train[:, :, 0:1]))

    # Biological residual
    G_pred = Y_pred[:, :, 0:1] * model.data_window.m_g
    G_seq = tf.squeeze(G_pred, axis=0)
    U_seq = tf.squeeze(model.U_train, axis=0) * model.u_max
    R_seq = tf.squeeze(model.R_train, axis=0) * model.r_max

    temp_params = copy.copy(model.params)
    temp_params.ksi = tf.exp(model.log_ksi)

    bio_losses = residuals_euler_seq(
        G_seq, U_seq, R_seq, temp_params, dt=1.0, use_latent_sim=True
    )
    loss_B = tf.reduce_mean(tf.square(bio_losses["LB"]))

    # Combined loss
    total_loss = loss_g + 0.1 * loss_B

grad_combined = tape.gradient(total_loss, model.log_ksi)
print(f"Total loss: {float(total_loss):.6f}")
print(f"  - Glucose: {float(loss_g):.6f}")
print(f"  - Bio (0.1x): {float(0.1*loss_B):.6f}")
print(f"Gradient: {float(grad_combined) if grad_combined is not None else 'None'}")

if grad_combined is not None:
    print("✅ Gradient flows through combined loss!")
    print(f"   Gradient magnitude: {float(tf.abs(grad_combined)):.6f}")
    print("\n🎉 SUCCESS! Gradients work correctly.")
    print("   The standalone training script should work now.")
else:
    print("❌ Gradient is None even for combined loss")
    print("   There's a fundamental issue with gradient flow.")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
