from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import tensorflow as tf
import deepxde as dde


# --------------------------------------------------------------------------------------
# Patient presets (from your notebooks). kb is treated as a first-class parameter.
# --------------------------------------------------------------------------------------

_PATIENT_PRESETS: Dict[int, Dict[str, float]] = {
    2:  {"M": 72, "ksi": 197, "kl": 1.94, "Tu": 122, "ku_Vi": 59e-3, "Tr": 183, "kr_Vb": 2.40e-3, "kb": 128/72, "tend": 48*60},
    3:  {"M": 94, "ksi": 274, "kl": 1.72, "Tu": 88,  "ku_Vi": 62e-3, "Tr": 49,  "kr_Vb": 2.00e-3, "kb": 128/94, "tend": 48*60},
    4:  {"M": 74, "ksi": 191, "kl": 1.94, "Tu": 126, "ku_Vi": 61e-3, "Tr": 188, "kr_Vb": 2.47e-3, "kb": 128/74, "tend": 48*60},
    5:  {"M": 91, "ksi": 282, "kl": 1.67, "Tu": 85,  "ku_Vi": 64e-3, "Tr": 48,  "kr_Vb": 2.06e-3, "kb": 128/91, "tend": 48*60},
    6:  {"M": 70, "ksi": 203, "kl": 1.94, "Tu": 118, "ku_Vi": 57e-3, "Tr": 178, "kr_Vb": 2.33e-3, "kb": 128/70, "tend": 48*60},
    7:  {"M": 97, "ksi": 267, "kl": 1.77, "Tu": 91,  "ku_Vi": 60e-3, "Tr": 50,  "kr_Vb": 1.94e-3, "kb": 128/97, "tend": 48*60},
    8:  {"M": 73, "ksi": 200, "kl": 1.92, "Tu": 125, "ku_Vi": 60e-3, "Tr": 182, "kr_Vb": 2.38e-3, "kb": 128/73, "tend": 48*60},
    9:  {"M": 92, "ksi": 272, "kl": 1.71, "Tu": 87,  "ku_Vi": 61e-3, "Tr": 49,  "kr_Vb": 2.03e-3, "kb": 128/92, "tend": 48*60},
    10: {"M": 74, "ksi": 191, "kl": 1.94, "Tu": 126, "ku_Vi": 61e-3, "Tr": 188, "kr_Vb": 2.47e-3, "kb": 128/74, "tend": 48*60},
    11: {"M": 91, "ksi": 282, "kl": 1.67, "Tu": 85,  "ku_Vi": 64e-3, "Tr": 48,  "kr_Vb": 2.06e-3, "kb": 128/91, "tend": 48*60},
}


@dataclass
class MagdelaineParams:
    """All physiological parameters used in the Magdelaine ODEs."""
    M: float
    ksi: float
    kl: float
    kb: float
    Tu: float
    ku_Vi: float
    Tr: float
    kr_Vb: float
    tend_min: int = 48 * 60  # default 48h


@dataclass
class InverseParams:
    """Trainable log-space parameters for inverse runs (extend as needed)."""
    log_ksi: Optional[tf.Variable] = None


def make_params_from_preset(pat: int, *, override_kb: Optional[float] = None) -> MagdelaineParams:
    """Factory: build a MagdelaineParams from the embedded presets."""
    if pat not in _PATIENT_PRESETS:
        raise ValueError(f"Unknown patient preset: {pat}")
    p = _PATIENT_PRESETS[pat]
    kb = override_kb if override_kb is not None else float(p["kb"])
    return MagdelaineParams(
        M=float(p["M"]), ksi=float(p["ksi"]), kl=float(p["kl"]), kb=kb,
        Tu=float(p["Tu"]), ku_Vi=float(p["ku_Vi"]), Tr=float(p["Tr"]), kr_Vb=float(p["kr_Vb"]),
        tend_min=int(p["tend"])
    )


def make_inverse_params(enable: bool, ksi_init: Optional[float] = None) -> InverseParams:
    """Create trainable inverse parameters (currently only ksi)."""
    if not enable:
        return InverseParams(None)
    init = float(ksi_init) if ksi_init is not None else 235.0
    return InverseParams(log_ksi=tf.Variable(tf.math.log(init), dtype=tf.float32, name="log_ksi"))


# --------------------------------------------------------------------------------------
# Continuous residuals (DeepXDE autodiff) for PINN / Modified-MLP PINN
# Your normalizers are preserved EXACTLY: /100, /0.1, /5.0
# --------------------------------------------------------------------------------------

def residuals_dde(
    y: tf.Tensor,                 # [B,3] normalized -> [G,I,D]
    x: tf.Tensor,                 # [B,1] time in [0,1]
    params: MagdelaineParams,
    lookup,                       # object with .u and .r tensors; see models/input_lookup.py
    scales,                       # object with m_t, m_g, m_i, m_d; see utils/scaling.py
    *,
    inverse: Optional[InverseParams] = None,
    include_prior: bool = False
) -> list[tf.Tensor]:
    """
    Return [eq1, eq2, eq3] (optionally + prior) for DeepXDE-based models.
    """
    # Real time (minutes) and input lookup indexing
    t_real = tf.cast(x[:, 0:1] * float(scales.m_t), tf.float32)             # minutes
    idx = tf.cast(tf.round(t_real), tf.int32)
    idx = tf.clip_by_value(idx, 0, tf.shape(lookup.u)[0] - 1)
    ut = tf.gather(lookup.u, idx)  # [B,1] U/min
    rt = tf.gather(lookup.r, idx)  # [B,1] g/min

    # Denormalize outputs
    G = y[:, 0:1] * scales.m_g
    I = y[:, 1:2] * scales.m_i
    D = y[:, 2:3] * scales.m_d

    # Time derivatives (DeepXDE autodiff)
    dG_dt   = dde.grad.jacobian(y, x, i=0, j=0) * scales.m_g / scales.m_t
    dI_dt   = dde.grad.jacobian(y, x, i=1, j=0) * scales.m_i / scales.m_t
    dD_dt   = dde.grad.jacobian(y, x, i=2, j=0) * scales.m_d / scales.m_t
    d2I_dt2 = dde.grad.hessian(y, x, component=1, i=0, j=0) * scales.m_i / (scales.m_t ** 2)
    d2D_dt2 = dde.grad.hessian(y, x, component=2, i=0, j=0) * scales.m_d / (scales.m_t ** 2)

    # ksi (fixed or trainable inverse)
    ksi = tf.exp(inverse.log_ksi) if (inverse and inverse.log_ksi is not None) else params.ksi

    # Residuals (preserve your exact scalings)
    eq1 = (dG_dt - (-ksi * I + params.kl - params.kb + D)) / 100.0
    eq2 = (d2I_dt2 + (2.0 / params.Tu) * dI_dt + (1.0 / (params.Tu ** 2)) * I
           - ut * (params.ku_Vi / (params.Tu ** 2))) / 0.1
    eq3 = (d2D_dt2 + (2.0 / params.Tr) * dD_dt + (1.0 / (params.Tr ** 2)) * D
           - rt * (params.kr_Vb / (params.Tr ** 2))) / 5.0

    if include_prior and (inverse and inverse.log_ksi is not None):
        prior_mean, prior_std, prior_w = 220.0, 60.0, 1e-5
        prior = prior_w * tf.square((ksi - prior_mean) / prior_std)
        prior_res = tf.ones_like(eq1) * prior
        return [eq1, eq2, eq3, prior_res]

    return [eq1, eq2, eq3]


# --------------------------------------------------------------------------------------
# Discrete (forward-Euler) residual for BI-RNN pathway (G-only)
# --------------------------------------------------------------------------------------

def simulate_latents_euler(
    U_seq: np.ndarray | tf.Tensor,
    R_seq: np.ndarray | tf.Tensor,
    p: MagdelaineParams,
    dt: float = 1.0,
    I0: float | None = None,
    dI0: float = 0.0,
    D0: float = 0.0,
    dD0: float = 0.0,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Deterministically integrate Magdelaine latent states I(t), D(t) from inputs u(t), r(t)
    using forward Euler on the canonical 2nd-order ODEs rewritten as 1st-order systems.

    dI/dt = vI
    dvI/dt = -(2/Tu) vI - (1/Tu^2) I + u * (ku_Vi/Tu^2)

    dD/dt = vD
    dvD/dt = -(2/Tr) vD - (1/Tr^2) D + r * (kr_Vb/Tr^2)

    Returns:
        (I_seq, D_seq) as tf.float32 tensors shaped like U_seq/R_seq.
    """
    U = tf.convert_to_tensor(U_seq, dtype=tf.float32)
    R = tf.convert_to_tensor(R_seq, dtype=tf.float32)
    dt = tf.cast(dt, tf.float32)

    # Handle [T] vs [B,T]
    batch_mode = (len(U.shape) == 2)
    if not batch_mode:
        U = U[tf.newaxis, ...]  # [1, T]
        R = R[tf.newaxis, ...]  # [1, T]

    B = tf.shape(U)[0]
    T = tf.shape(U)[1]

    # Initial conditions: I0 defaults to physiological steady-state Ieq
    I0_val = p.kl - p.kb
    I0_val = I0_val / p.ksi if I0 is None else float(I0)
    I0_tf = tf.fill([B, 1], tf.cast(I0_val, tf.float32))
    vI0_tf = tf.fill([B, 1], tf.cast(dI0, tf.float32))
    D0_tf = tf.fill([B, 1], tf.cast(D0, tf.float32))
    vD0_tf = tf.fill([B, 1], tf.cast(dD0, tf.float32))

    I_list = [I0_tf]
    D_list = [D0_tf]
    vI = vI0_tf
    I = I0_tf
    vD = vD0_tf
    D = D0_tf

    cI = tf.cast(p.ku_Vi / (p.Tu ** 2), tf.float32)
    cD = tf.cast(p.kr_Vb / (p.Tr ** 2), tf.float32)
    aI = tf.cast(2.0 / p.Tu, tf.float32)
    bI = tf.cast(1.0 / (p.Tu ** 2), tf.float32)
    aD = tf.cast(2.0 / p.Tr, tf.float32)
    bD = tf.cast(1.0 / (p.Tr ** 2), tf.float32)

    # Unroll forward Euler
    for t in tf.range(T - 1):
        ut = U[:, t:t+1]
        rt = R[:, t:t+1]

        # I system
        dI_dt = vI
        dvI_dt = -aI * vI - bI * I + cI * ut
        I_next = I + dt * dI_dt
        vI_next = vI + dt * dvI_dt

        # D system
        dD_dt = vD
        dvD_dt = -aD * vD - bD * D + cD * rt
        D_next = D + dt * dD_dt
        vD_next = vD + dt * dvD_dt

        I, vI = I_next, vI_next
        D, vD = D_next, vD_next
        I_list.append(I)
        D_list.append(D)

    I_seq = tf.concat(I_list, axis=1)  # [B, T]
    D_seq = tf.concat(D_list, axis=1)  # [B, T]

    if not batch_mode:
        I_seq = tf.squeeze(I_seq, axis=0)  # [T]
        D_seq = tf.squeeze(D_seq, axis=0)  # [T]

    return I_seq, D_seq


def residuals_euler_seq(
    G_seq: np.ndarray | tf.Tensor,     # [T] or [B,T]
    U_seq: np.ndarray | tf.Tensor,     # [T] or [B,T]  (U/min)
    R_seq: np.ndarray | tf.Tensor,     # [T] or [B,T]  (g/min)
    p: MagdelaineParams,
    dt: float = 1.0,
    *,
    I_seq: Optional[np.ndarray | tf.Tensor] = None,
    D_seq: Optional[np.ndarray | tf.Tensor] = None,
    use_latent_sim: bool = True,
) -> dict[str, tf.Tensor]:
    """
    Compute BI-RNN biological residual (forward-Euler) consistent with your notebooks.
    By default, if I/D are not provided, we **simulate** them from u/r and params.

    Returns:
        {'LB': tensor} with shape [T-1] or [B,T-1]
    """
    G = tf.convert_to_tensor(G_seq, dtype=tf.float32)
    U = tf.convert_to_tensor(U_seq, dtype=tf.float32)
    R = tf.convert_to_tensor(R_seq, dtype=tf.float32)
    dt = tf.cast(dt, tf.float32)

    # Align batch/time dims
    batch_mode = (len(G.shape) == 2)
    if not batch_mode:
        G = G[tf.newaxis, ...]  # [1,T]
        U = U[tf.newaxis, ...]
        R = R[tf.newaxis, ...]
    # If I/D not provided, simulate them deterministically
    if (I_seq is None or D_seq is None) and use_latent_sim:
        I_sim, D_sim = simulate_latents_euler(U, R, p, dt=dt)
    else:
        I_sim = tf.convert_to_tensor(I_seq, dtype=tf.float32)
        D_sim = tf.convert_to_tensor(D_seq, dtype=tf.float32)
        if len(I_sim.shape) == 1: I_sim = I_sim[tf.newaxis, ...]
        if len(D_sim.shape) == 1: D_sim = D_sim[tf.newaxis, ...]

    # Finite-difference dG/dt
    dG = (G[..., 1:] - G[..., :-1]) / dt      # [B, T-1]
    # RHS of Magdelaine glucose ODE using I(t), D(t) at step t
    rhs = (-p.ksi * I_sim[..., :-1] + p.kl - p.kb + D_sim[..., :-1])
    eq1 = dG - rhs                             # [B, T-1]

    LB = eq1 if batch_mode else tf.squeeze(eq1, axis=0)
    return {"LB": LB}
