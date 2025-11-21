#!/usr/bin/env python3
"""
Comprehensive NaN debugging for simulate_latents_euler.

This script checks EVERY step of the computation to find where NaN appears.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import tensorflow as tf
import numpy as np
import copy

# Import after path setup
from src.datasets.loader import load_synthetic_window
from src.models.birnn import BIRNN
from src.training.config import load_config

print("="*80)
print("COMPREHENSIVE NaN DEBUGGING")
print("="*80)

# Setup
config = load_config(model_name='birnn', mode='inverse')
data = load_synthetic_window(patient=3, root='data/synthetic')
model = BIRNN(config)
model.build(data)

print("\n[1] Model Setup")
print(f"✅ Model built successfully")
print(f"   log_ksi: {float(model.log_ksi):.4f}")
print(f"   ksi: {float(tf.exp(model.log_ksi)):.2f}")

# Prepare inputs
U_seq = tf.squeeze(model.U_train, axis=0) * model.u_max
R_seq = tf.squeeze(model.R_train, axis=0) * model.r_max

print(f"\n[2] Input Sequences")
print(f"   U_seq shape: {U_seq.shape}")
print(f"   R_seq shape: {R_seq.shape}")
print(f"   U_seq has NaN: {tf.reduce_any(tf.math.is_nan(U_seq))}")
print(f"   R_seq has NaN: {tf.reduce_any(tf.math.is_nan(R_seq))}")

# Create params with tensor ksi
params = copy.copy(model.params)
params.ksi = tf.exp(model.log_ksi)

print(f"\n[3] Parameters")
print(f"   ksi type: {type(params.ksi)}")
print(f"   ksi value: {float(params.ksi):.2f}")
print(f"   kl: {params.kl}")
print(f"   kb: {params.kb}")
print(f"   Tu: {params.Tu}")
print(f"   ku_Vi: {params.ku_Vi}")

# Manually simulate the first few steps to find where NaN appears
print(f"\n[4] Manual Simulation (checking each step)")
print("="*80)

# Convert to tensors
U = tf.convert_to_tensor(U_seq, dtype=tf.float32)
R = tf.convert_to_tensor(R_seq, dtype=tf.float32)
dt = tf.constant(1.0, dtype=tf.float32)

# Add batch dimension
U = U[tf.newaxis, :]  # [1, T]
R = R[tf.newaxis, :]  # [1, T]
B = tf.shape(U)[0]
T = tf.shape(U)[1]

print(f"   Batch mode shapes: U={U.shape}, R={R.shape}")

# Initial conditions
print("\n[4.1] Computing Initial Conditions")
ksi_tf = tf.convert_to_tensor(params.ksi, dtype=tf.float32)
print(f"   ksi_tf: {float(ksi_tf):.2f}")
print(f"   ksi_tf is NaN: {tf.math.is_nan(ksi_tf)}")

I0_numerator = params.kl - params.kb
print(f"   kl - kb = {I0_numerator:.6f}")

I0_val = I0_numerator / ksi_tf
print(f"   I0_val = (kl - kb) / ksi = {float(I0_val):.6f}")
print(f"   I0_val is NaN: {tf.math.is_nan(I0_val)}")
print(f"   I0_val is finite: {tf.math.is_finite(I0_val)}")

# Try to create I0_tf
try:
    I0_tf = tf.ones([B, 1], dtype=tf.float32) * I0_val
    print(f"   I0_tf: {float(I0_tf[0, 0]):.6f}")
    print(f"   I0_tf is NaN: {tf.math.is_nan(I0_tf[0, 0])}")
except Exception as e:
    print(f"   ❌ Failed to create I0_tf: {e}")
    sys.exit(1)

vI0_tf = tf.fill([B, 1], 0.0)
D0_tf = tf.fill([B, 1], 0.0)
vD0_tf = tf.fill([B, 1], 0.0)

print(f"   Initial states created successfully")
print(f"   I0: {float(I0_tf[0, 0]):.6f}")
print(f"   vI0: {float(vI0_tf[0, 0]):.6f}")
print(f"   D0: {float(D0_tf[0, 0]):.6f}")
print(f"   vD0: {float(vD0_tf[0, 0]):.6f}")

# Compute coefficients
print("\n[4.2] Computing Coefficients")
cI = tf.cast(params.ku_Vi / (params.Tu ** 2), tf.float32)
cD = tf.cast(params.kr_Vb / (params.Tr ** 2), tf.float32)
aI = tf.cast(2.0 / params.Tu, tf.float32)
bI = tf.cast(1.0 / (params.Tu ** 2), tf.float32)
aD = tf.cast(2.0 / params.Tr, tf.float32)
bD = tf.cast(1.0 / (params.Tr ** 2), tf.float32)

print(f"   cI: {float(cI):.6e}")
print(f"   cD: {float(cD):.6e}")
print(f"   aI: {float(aI):.6e}")
print(f"   bI: {float(bI):.6e}")
print(f"   aD: {float(aD):.6e}")
print(f"   bD: {float(bD):.6e}")

for coef, name in [(cI, 'cI'), (cD, 'cD'), (aI, 'aI'), (bI, 'bI'), (aD, 'aD'), (bD, 'bD')]:
    if tf.math.is_nan(coef):
        print(f"   ❌ {name} is NaN!")

# Simulate first 5 steps manually
print("\n[4.3] Simulating First 5 Steps")
I = I0_tf
vI = vI0_tf
D = D0_tf
vD = vD0_tf

for t in range(min(5, int(T) - 1)):
    print(f"\n   Step {t}:")
    ut = U[:, t:t+1]
    rt = R[:, t:t+1]
    
    print(f"     u(t): {float(ut[0, 0]):.2f}")
    print(f"     r(t): {float(rt[0, 0]):.2f}")
    
    # I system
    dI_dt = vI
    dvI_dt = -aI * vI - bI * I + cI * ut
    
    print(f"     dI_dt: {float(dI_dt[0, 0]):.6f}")
    print(f"     dvI_dt: {float(dvI_dt[0, 0]):.6f}")
    
    if tf.math.is_nan(dI_dt[0, 0]):
        print(f"     ❌ dI_dt is NaN at step {t}!")
        break
    if tf.math.is_nan(dvI_dt[0, 0]):
        print(f"     ❌ dvI_dt is NaN at step {t}!")
        break
    
    I_next = I + dt * dI_dt
    vI_next = vI + dt * dvI_dt
    
    print(f"     I(t+1): {float(I_next[0, 0]):.6f}")
    print(f"     vI(t+1): {float(vI_next[0, 0]):.6f}")
    
    if tf.math.is_nan(I_next[0, 0]):
        print(f"     ❌ I_next is NaN at step {t}!")
        break
    if tf.math.is_nan(vI_next[0, 0]):
        print(f"     ❌ vI_next is NaN at step {t}!")
        break
    
    # D system
    dD_dt = vD
    dvD_dt = -aD * vD - bD * D + cD * rt
    D_next = D + dt * dD_dt
    vD_next = vD + dt * dvD_dt
    
    print(f"     D(t+1): {float(D_next[0, 0]):.6f}")
    
    if tf.math.is_nan(D_next[0, 0]):
        print(f"     ❌ D_next is NaN at step {t}!")
        break
    
    I, vI = I_next, vI_next
    D, vD = D_next, vD_next

print("\n[5] Now calling the actual simulate_latents_euler function")
print("="*80)

from src.physics.magdelaine import simulate_latents_euler

try:
    I_seq, D_seq = simulate_latents_euler(U_seq, R_seq, params, dt=1.0)
    print(f"✅ Function completed")
    print(f"   I_seq has NaN: {tf.reduce_any(tf.math.is_nan(I_seq))}")
    print(f"   D_seq has NaN: {tf.reduce_any(tf.math.is_nan(D_seq))}")
    
    if tf.reduce_any(tf.math.is_nan(I_seq)):
        # Find first NaN
        is_nan = tf.math.is_nan(I_seq)
        nan_indices = tf.where(is_nan)
        first_nan_idx = int(nan_indices[0, 0])
        print(f"   First NaN in I_seq at index: {first_nan_idx}")
    
except Exception as e:
    print(f"❌ Function failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)