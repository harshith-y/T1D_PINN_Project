#!/usr/bin/env python3
"""
BI-RNN Inverse Training - CORRECTED VERSION

This version has BOTH critical fixes:
1. Input denormalization: Use stored u_max and r_max (not recompute from normalized data)
2. Time step: Use dt = 1.0 / m_t to match notebook (not dt = 1.0)

The notebook has a bug where inputs stay normalized but dt is tiny, which compensates.
Our fix properly denormalizes inputs AND uses the tiny dt.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import tensorflow as tf

from src.datasets.loader import load_synthetic_window
from src.models.birnn import BIRNN
from src.training.config import load_config

print("=" * 80)
print("BI-RNN INVERSE TRAINING - CORRECTED (dt AND INPUT SCALING FIXED)")
print("=" * 80)

# Load config
print("\n[1/6] Loading configuration...")
config = load_config(model_name="birnn", mode="forward")
config.mode = "inverse"
config.inverse_param = "ksi"
print("✅ Config loaded")

# Load data
print("\n[2/6] Loading synthetic data...")
data = load_synthetic_window(patient=3, root="data/synthetic")
print(f"✅ Data loaded: {data}")
print(f"   m_t = {data.m_t:.1f} minutes")
print(f"   dt (should be) = 1.0 / {data.m_t:.1f} = {1.0/data.m_t:.10f}")

# Build model
print("\n[3/6] Building BI-RNN model...")
model = BIRNN(config)
model.build(data)
print("✅ Model built")
print(f"   u_max = {float(model.u_max):.6f} U/min")
print(f"   r_max = {float(model.r_max):.6f} g/min")

# Compile
print("\n[4/6] Compiling model...")
model.compile()
print("✅ Model compiled")

# Get true ksi for comparison
true_ksi = 274.0


# Helper function: Compute biological residual (TRULY FIXED!)
def compute_bio_residual_corrected(
    Y_pred, Y_in, U_in, R_in, log_ksi, params, u_max, r_max, m_g, m_i, m_d, m_t
):
    """
    Biological residual with BOTH fixes:

    FIX #1: Input denormalization uses STORED max values
    - U_in and R_in are normalized to [0,1]
    - Multiply by stored u_max, r_max to get physical units

    FIX #2: Time step uses dt = 1.0 / m_t
    - Notebook uses dt = 1.0 / m_t ≈ 0.000347 for Pat3
    - NOT dt = 1.0 (which would be 2879x too large!)
    """
    # Flatten
    y_pred_flat = tf.reshape(Y_pred, [-1, 3])
    y_in_flat = tf.reshape(Y_in, [-1, 3])
    u_in_flat = tf.reshape(U_in, [-1, 1])
    r_in_flat = tf.reshape(R_in, [-1, 1])

    # Denormalize states to physical units
    G = y_in_flat[:, 0:1] * m_g  # mg/dL
    I = y_in_flat[:, 1:2] * m_i  # U/dL
    D = y_in_flat[:, 2:3] * m_d  # mg/dL/min

    # Denormalize inputs - FIX #1: Use STORED max values
    ut = u_in_flat * u_max  # [0,1] * 19.03 → U/min
    rt = r_in_flat * r_max  # [0,1] * 149.0 → g/min

    # Use trainable ksi
    ksi = tf.exp(log_ksi)

    # FIRST-ORDER ODEs (simplified Magdelaine)
    dG = -ksi * I + params.kl - params.kb + D  # mg/dL/min
    dI = -I / params.Tu + (params.ku_Vi / params.Tu) * ut  # U/dL/min
    dD = -D / params.Tr + (params.kr_Vb / params.Tr) * rt  # mg/dL/min²

    # Forward Euler: y(t+1) = y(t) + dt * dy/dt
    # FIX #2: Use dt = 1.0 / m_t (NOT dt = 1.0!)
    dt = 1.0 / m_t  # ≈ 0.000347 for Pat3

    # Normalize derivatives before adding to normalized states
    y_next_ode = y_in_flat + dt * tf.concat(
        [
            dG / m_g,  # Normalized glucose increment
            dI / m_i,  # Normalized insulin increment
            dD / m_d,  # Normalized digestion increment
        ],
        axis=1,
    )

    # Compare ODE prediction with NN prediction
    return tf.reduce_mean(tf.square(y_next_ode - y_pred_flat))


print("\n[5/6] Starting 3-stage inverse training...")
print("=" * 80)

# ============================================================================
# STAGE 1: Train ksi only (freeze NN weights)
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 1: Train ksi Only")
print("=" * 80)
print("Epochs: 100")
print("Learning rate: 0.01")
print("Training: log_ksi only")
print("Frozen: All NN weights")
print("=" * 80 + "\n")

optimizer_stage1 = tf.keras.optimizers.Adam(learning_rate=0.01)

for epoch in range(50):
    with tf.GradientTape() as tape:
        tape.watch(model.log_ksi)

        # Forward pass (freeze NN)
        Y_pred = model.model(model.X_train, training=False)

        # Glucose loss
        loss_g = tf.reduce_mean(tf.square(Y_pred[:, :, 0:1] - model.Y_train[:, :, 0:1]))

        # Biological residual (CORRECTED VERSION!)
        loss_B = compute_bio_residual_corrected(
            Y_pred,
            model.Y_train,
            model.U_train,
            model.R_train,
            model.log_ksi,
            model.params,
            model.u_max,
            model.r_max,  # Stored max values
            model.data_window.m_g,
            model.data_window.m_i,
            model.data_window.m_d,
            model.data_window.m_t,  # ← CRITICAL: Pass m_t for proper dt
        )

        # Combined loss
        total_loss = loss_g + 0.1 * loss_B

    # Gradient for ksi only
    gradients = tape.gradient(total_loss, [model.log_ksi])

    if epoch == 0:
        if gradients[0] is None:
            print("❌ ERROR: Gradients are None!")
            break
        else:
            print(
                f"✅ Gradient check: grad magnitude = {float(tf.abs(gradients[0])):.6f}"
            )

    optimizer_stage1.apply_gradients(zip(gradients, [model.log_ksi]))

    if epoch % 10 == 0:
        ksi_val = float(tf.exp(model.log_ksi))
        error = abs(ksi_val - true_ksi) / true_ksi * 100
        print(
            f"Epoch {epoch:3d}: loss_g={float(loss_g):.6f}, loss_B={float(loss_B):.6f}, ksi={ksi_val:.2f} (error={error:.2f}%)"
        )

print("\n✅ Stage 1 complete!")
ksi_after_stage1 = float(tf.exp(model.log_ksi))
print(f"   ksi after Stage 1: {ksi_after_stage1:.2f}")
print(f"   Error: {abs(ksi_after_stage1 - true_ksi) / true_ksi * 100:.2f}%")

# ============================================================================
# STAGE 2: Train NN only (freeze ksi)
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 2: Train NN Only")
print("=" * 80)
print("Epochs: 200")
print("Learning rate: 0.005")
print("Training: All NN weights")
print("Frozen: log_ksi")
print("=" * 80 + "\n")

optimizer_stage2 = tf.keras.optimizers.Adam(learning_rate=0.005)
nn_vars = model.model.trainable_variables

for epoch in range(50):
    with tf.GradientTape() as tape:
        # Forward pass (train NN)
        Y_pred = model.model(model.X_train, training=True)

        # Losses
        loss_g = tf.reduce_mean(tf.square(Y_pred[:, :, 0:1] - model.Y_train[:, :, 0:1]))
        loss_ic = tf.reduce_mean(tf.square(Y_pred[:, 0, :] - model.Y_train[:, 0, :]))

        # Biological residual (with FIXED ksi from Stage 1)
        loss_B = compute_bio_residual_corrected(
            Y_pred,
            model.Y_train,
            model.U_train,
            model.R_train,
            model.log_ksi,
            model.params,
            model.u_max,
            model.r_max,
            model.data_window.m_g,
            model.data_window.m_i,
            model.data_window.m_d,
            model.data_window.m_t,
        )

        # Total loss (balanced weights from notebook)
        total_loss = 8.0 * loss_g + 4.82 * loss_B + 0.53 * loss_ic

    # Gradient for NN only
    gradients = tape.gradient(total_loss, nn_vars)
    optimizer_stage2.apply_gradients(zip(gradients, nn_vars))

    if epoch % 20 == 0:
        print(
            f"Epoch {epoch:3d}: total={float(total_loss):.4f}, loss_g={float(loss_g):.4f}, loss_B={float(loss_B):.4f}"
        )

print("\n✅ Stage 2 complete!")

# ============================================================================
# STAGE 3: Joint fine-tuning (train both ksi and NN)
# ============================================================================
print("\n" + "=" * 80)
print("STAGE 3: Joint Fine-Tuning")
print("=" * 80)
print("Epochs: 100")
print("Learning rate: 0.002")
print("Training: Both log_ksi AND all NN weights")
print("=" * 80 + "\n")

optimizer_stage3 = tf.keras.optimizers.Adam(learning_rate=0.002)
all_vars = [model.log_ksi] + nn_vars

for epoch in range(50):
    with tf.GradientTape() as tape:
        tape.watch(model.log_ksi)

        # Forward pass
        Y_pred = model.model(model.X_train, training=True)

        # Losses (same as Stage 2)
        loss_g = tf.reduce_mean(tf.square(Y_pred[:, :, 0:1] - model.Y_train[:, :, 0:1]))
        loss_ic = tf.reduce_mean(tf.square(Y_pred[:, 0, :] - model.Y_train[:, 0, :]))

        # Biological residual (with trainable ksi)
        loss_B = compute_bio_residual_corrected(
            Y_pred,
            model.Y_train,
            model.U_train,
            model.R_train,
            model.log_ksi,
            model.params,
            model.u_max,
            model.r_max,
            model.data_window.m_g,
            model.data_window.m_i,
            model.data_window.m_d,
            model.data_window.m_t,
        )

        total_loss = 8.0 * loss_g + 4.82 * loss_B + 0.53 * loss_ic

    # Gradients for ALL variables
    gradients = tape.gradient(total_loss, all_vars)
    optimizer_stage3.apply_gradients(zip(gradients, all_vars))

    if epoch % 10 == 0:
        ksi_val = float(tf.exp(model.log_ksi))
        error = abs(ksi_val - true_ksi) / true_ksi * 100
        print(
            f"Epoch {epoch:3d}: total={float(total_loss):.4f}, ksi={ksi_val:.2f} (error={error:.2f}%)"
        )

print("\n✅ Stage 3 complete!")

# ============================================================================
# FINAL RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

final_ksi = float(tf.exp(model.log_ksi))
error = abs(final_ksi - true_ksi) / true_ksi * 100

print(f"Estimated ksi: {final_ksi:.2f}")
print(f"True ksi:      {true_ksi:.2f}")
print(f"Error:         {error:.2f}%")
print("=" * 80)

# Evaluate on test set
print("\n[6/6] Evaluating on test data...")
Y_pred_test = model.model(model.X_test, training=False)
test_loss_g = tf.reduce_mean(
    tf.square(Y_pred_test[:, :, 0:1] - model.Y_test[:, :, 0:1])
)
print(f"✅ Test loss (glucose): {float(test_loss_g):.6f}")

print("\n" + "=" * 80)
if error < 10:
    print("🎉 SUCCESS! Error < 10%")
    print("✨ Inverse training works correctly!")
else:
    print("⚠️  Warning: Error > 10%")
    print("   Try more epochs or adjust learning rates")
print("=" * 80)
