from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Reuse your physics for params (no duplication)
from src.physics.magdelaine import (  # optional for CLI convenience
    MagdelaineParams,
    make_params_from_preset,
)

# --------------------------------------------------------------------------------------
# Patient-specific presets (from your simulator notebook)
# --------------------------------------------------------------------------------------

# Glucose ICs (G0) per patient (mg/dL); I0 is steady state by default (Ieq = (kl-kb)/ksi)
_G0_BY_PAT: Dict[int, float] = {
    2: 220,
    3: 125,
    4: 150,
    5: 140,
    6: 180,
    7: 160,
    8: 170,
    9: 155,
    10: 145,
    11: 130,
}

# Carb impulses by patient: list of (minute, grams) – keep as GRAMS (no mg conversion)
_R_EVENTS_G: Dict[int, List[Tuple[int, float]]] = {
    2: [
        (24 * 60, 128),
        (25 * 60 + 30, 15),
        (37 * 60, 150),
        (41 * 60, 100),
        (42 * 60 + 30, 7.5),
        (44 * 60 + 30, 15),
    ],
    3: [
        (3 * 60, 15),
        (6 * 60, 15),
        (7 * 60, 20),
        (12 * 60 + 30, 15),
        (14 * 60, 15),
        (21 * 60, 15),
        (23 * 60 + 30, 20),
        (24 * 60, 129),
        (25 * 60 + 30, 15),
        (37 * 60, 149),
        (41 * 60, 97),
        (42 * 60 + 30, 8),
        (44 * 60 + 30, 15),
        (48 * 60, 129),
    ],
    4: [
        (24 * 60, 127),
        (25 * 60 + 30, 15),
        (37 * 60, 150),
        (41 * 60, 100),
        (42 * 60 + 30, 8),
        (44 * 60 + 30, 15),
        (48 * 60, 125),
    ],
    5: [
        (6 * 60, 15),
        (3 * 60, 15),
        (7 * 60, 20),
        (12 * 60 + 30, 15),
        (14 * 60, 15),
        (21 * 60, 15),
        (23 * 60 + 30, 20),
        (24 * 60, 128),
        (25 * 60 + 30, 15),
        (37 * 60, 150),
        (41 * 60, 100),
        (42 * 60 + 30, 8),
        (44 * 60 + 30, 15),
        (48 * 60, 129),
    ],
    6: [
        (24 * 60, 128),
        (25 * 60 + 30, 15),
        (37 * 60, 150),
        (41 * 60, 97),
        (42 * 60 + 30, 8),
        (44 * 60 + 30, 15),
        (48 * 60, 129),
    ],
    7: [
        (6 * 60, 15),
        (3 * 60, 15),
        (7 * 60, 19),
        (12 * 60 + 30, 15),
        (14 * 60, 15),
        (21 * 60, 15),
        (23 * 60 + 30, 20),
        (24 * 60, 129),
        (25 * 60 + 30, 15),
        (37 * 60, 150),
        (41 * 60, 97),
        (42 * 60 + 30, 8),
        (44 * 60 + 30, 15),
        (48 * 60, 129),
    ],
    8: [
        (24 * 60, 128),
        (25 * 60 + 30, 15),
        (37 * 60, 150),
        (41 * 60, 100),
        (42 * 60 + 30, 10),
        (44 * 60 + 30, 15),
        (48 * 60, 125),
    ],
    9: [
        (6 * 60, 15),
        (3 * 60, 15),
        (7 * 60, 19),
        (12 * 60 + 30, 15),
        (14 * 60, 15),
        (21 * 60, 15),
        (23 * 60 + 30, 20),
        (24 * 60, 128),
        (25 * 60 + 30, 15),
        (37 * 60, 150),
        (41 * 60, 100),
        (42 * 60 + 30, 8),
        (44 * 60 + 30, 15),
        (48 * 60, 129),
    ],
    10: [
        (24 * 60, 128),
        (25 * 60 + 30, 15),
        (37 * 60, 150),
        (41 * 60, 100),
        (42 * 60 + 30, 12),
        (44 * 60 + 30, 15),
        (48 * 60, 129),
    ],
    11: [
        (6 * 60, 15),
        (3 * 60, 15),
        (7 * 60, 19),
        (12 * 60 + 30, 15),
        (14 * 60, 15),
        (21 * 60, 15),
        (23 * 60 + 30, 20),
        (24 * 60, 129),
        (25 * 60 + 30, 15),
        (37 * 60, 150),
        (41 * 60, 97),
        (42 * 60 + 30, 8),
        (44 * 60 + 30, 15),
        (48 * 60, 129),
    ],
}

# Insulin presets:
# - baseline U/hr per patient group (converted to U/min),
# - optional patient-specific time segments with alternative basal U/hr,
# - bolus impulses (U) at specific minutes.
_BASELINE_U_PER_HR_BY_PAT: Dict[int, float] = {
    # From your notebook: 3,7,9,11 use 2 U/hr baseline; others default to 1 U/hr
    2: 1.0,
    3: 2.0,
    4: 1.0,
    5: 2.0,
    6: 1.0,
    7: 2.0,
    8: 1.0,
    9: 2.0,
    10: 1.0,
    11: 2.0,
}

# Per-patient basal SEGMENTS (U/hr) as (start_min, end_min, U/hr) – only for 3,7,9,11 in your notebook
_BASAL_SEGMENTS_U_PER_HR: Dict[int, List[Tuple[int, int, float]]] = {
    3: [
        (4 * 60, 8 * 60, 1.50),
        (22 * 60, 28 * 60, 1.60),
        (28 * 60, 34 * 60, 1.20),
        (34 * 60, 38 * 60, 1.60),
        (38 * 60, 46 * 60, 1.40),
        (46 * 60, 50 * 60, 1.00),
    ],
    7: [
        (4 * 60, 8 * 60, 1.55),
        (22 * 60, 28 * 60, 1.65),
        (28 * 60, 34 * 60, 1.24),
        (34 * 60, 38 * 60, 1.65),
        (38 * 60, 46 * 60, 1.45),
        (46 * 60, 50 * 60, 1.03),
    ],
    9: [
        (4 * 60, 8 * 60, 1.46),
        (22 * 60, 28 * 60, 1.55),
        (28 * 60, 34 * 60, 1.16),
        (34 * 60, 38 * 60, 1.55),
        (38 * 60, 46 * 60, 1.36),
        (46 * 60, 50 * 60, 0.97),
    ],
    11: [
        (4 * 60, 8 * 60, 1.55),
        (22 * 60, 28 * 60, 1.65),
        (28 * 60, 34 * 60, 1.24),
        (34 * 60, 38 * 60, 1.65),
        (38 * 60, 46 * 60, 1.45),
        (46 * 60, 50 * 60, 1.03),
    ],
}

# Bolus impulses (U) per patient – from your notebook
_BOLUS_EVENTS_U: Dict[int, List[Tuple[int, float]]] = {
    2: [
        (7 * 60 + 30, 0.5),
        (12 * 60 + 30, 2),
        (17 * 60, 2),
        (24 * 60, 22),
        (37 * 60, 18),
        (37 * 60 + 30, 17),
        (42 * 60 + 30, 16),
        (48 * 60, 19),
    ],
    3: [(24 * 60, 19), (36 * 60 + 30, 10), (37 * 60, 10), (41 * 60, 10)],
    4: [
        (7 * 60 + 30, 0.49),
        (12 * 60 + 30, 2),
        (17 * 60, 2),
        (24 * 60, 21),
        (37 * 60, 17),
        (37 * 60 + 30, 17),
        (42 * 60 + 30, 16),
        (48 * 60, 18),
    ],
    5: [(24 * 60, 18), (36 * 60 + 30, 10), (37 * 60, 10), (41 * 60, 10)],
    6: [
        (7 * 60 + 30, 0.52),
        (12 * 60 + 30, 2),
        (17 * 60, 2),
        (24 * 60, 17),
        (37 * 60, 15),
        (37 * 60 + 30, 13),
        (42 * 60 + 30, 12),
        (48 * 60, 5),
    ],
    7: [(24 * 60, 20), (36 * 60 + 30, 11), (37 * 60, 11), (41 * 60, 10)],
    8: [
        (7 * 60 + 30, 0.49),
        (12 * 60 + 30, 2),
        (17 * 60, 2),
        (24 * 60, 18),
        (37 * 60, 17),
        (37 * 60 + 30, 17),
        (42 * 60 + 30, 10),
        (48 * 60, 10),
    ],
    9: [(24 * 60, 18), (36 * 60 + 30, 10), (37 * 60, 10), (41 * 60, 10)],
    10: [
        (7 * 60 + 30, 0.52),
        (12 * 60 + 30, 2),
        (17 * 60, 2),
        (24 * 60, 23),
        (37 * 60, 18),
        (37 * 60 + 30, 18),
        (42 * 60 + 30, 17),
        (48 * 60, 20),
    ],
    11: [(24 * 60, 20), (36 * 60 + 30, 11), (37 * 60, 11), (41 * 60, 10)],
}


# --------------------------------------------------------------------------------------
# Config and helpers
# --------------------------------------------------------------------------------------
@dataclass
class SimConfig:
    minutes: int = 48 * 60
    dt: float = 0.1  # ← FIXED: Use 0.1 like notebook for numerical stability
    patient: int = 2
    # ICs (defaults mimic your notebook): I0 steady state, G0 from table, D0=0, derivatives 0
    G0: Optional[float] = None
    I0: Optional[float] = None
    D0: float = 0.0
    dI0: float = 0.0
    dD0: float = 0.0
    label: str = "Pat2"


def _time_grid(T: int, dt: float) -> np.ndarray:
    """Generate time grid from 0 to T with spacing dt."""
    n = int(round(T / dt)) + 1
    return np.arange(0, n, dtype=np.float32) * dt


def _build_inputs_for_patient(
    pat: int, minutes: int, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (t, u, r) matching your notebook behavior.

    CRITICAL FIXES:
    1. Time grid starts at 0
    2. Carbs are converted to mg (multiply by 1000) to match notebook
    3. Uses rounded time for event placement (handles dt=0.1 case)
    """
    t = _time_grid(minutes, dt)
    N = t.size

    # Create rounded integer time array for event indexing
    # This ensures events at minute 1440 land on index corresponding to t≈1440
    t_rounded = np.round(t).astype(int)

    # Baseline U/hr → U/min
    base_u_hr = _BASELINE_U_PER_HR_BY_PAT.get(pat, 1.0)
    u = np.full(N, base_u_hr / 60.0, dtype=np.float32)

    # Patient-specific basal segments (if any)
    for start_min, end_min, u_hr in _BASAL_SEGMENTS_U_PER_HR.get(pat, []):
        # Find indices where t_rounded is in [start_min, end_min)
        mask = (t_rounded >= start_min) & (t_rounded < end_min)
        u[mask] = u_hr / 60.0

    # Bolus impulses (U) - use exact minute matching
    for event_min, units in _BOLUS_EVENTS_U.get(pat, []):
        indices = np.where(t_rounded == event_min)[0]
        if len(indices) > 0:
            u[indices[0]] += float(units)

    # Carb impulses (grams) → CONVERT TO MG (multiply by 1000, matching notebook line 223)
    r = np.zeros(N, dtype=np.float32)
    for event_min, grams in _R_EVENTS_G.get(pat, []):
        indices = np.where(t_rounded == event_min)[0]
        if len(indices) > 0:
            r[indices[0]] += float(grams) * 1000.0  # ← FIXED: Convert g to mg

    return t, u, r


def _simulate_latents_euler_numpy(
    u, r, p: MagdelaineParams, dt, I0=None, dI0=0.0, D0=0.0, dD0=0.0
):
    """
    Pure NumPy forward-Euler for the Magdelaine I/D subsystems.
    Units:
      - u: U/min
      - r: mg/min (FIXED: now expects mg, not grams)
    Returns:
      I, dI, D, dD  (all np.ndarray, len = len(u))
    """
    T = len(u)
    I = np.zeros(T, dtype=np.float32)
    dI = np.zeros(T, dtype=np.float32)
    D = np.zeros(T, dtype=np.float32)
    dD = np.zeros(T, dtype=np.float32)

    # Initial conditions
    if I0 is None:
        I[0] = float((p.kl - p.kb) / p.ksi)  # steady-state if not provided
    else:
        I[0] = float(I0)
    dI[0] = float(dI0)
    D[0] = float(D0)
    dD[0] = float(dD0)

    Tu = float(p.Tu)
    Tr = float(p.Tr)
    ku_Vi = float(p.ku_Vi)
    kr_Vb = float(p.kr_Vb)

    inv_Tu = 1.0 / Tu
    inv_Tr = 1.0 / Tr
    inv_Tu2 = inv_Tu * inv_Tu
    inv_Tr2 = inv_Tr * inv_Tr

    # coefficients
    aI = -2.0 * inv_Tu
    bI = -inv_Tu2
    cI = ku_Vi * inv_Tu2

    aD = -2.0 * inv_Tr
    bD = -inv_Tr2
    cD = kr_Vb * inv_Tr2

    # semi-implicit Euler (d* uses updated slope)
    for k in range(T - 1):
        # I system: d2I = aI*dI + bI*I + cI*u
        d2I = aI * dI[k] + bI * I[k] + cI * float(u[k])
        dI[k + 1] = dI[k] + dt * d2I
        I[k + 1] = I[k] + dt * dI[k + 1]

        # D system: d2D = aD*dD + bD*D + cD*r
        d2D = aD * dD[k] + bD * D[k] + cD * float(r[k])
        dD[k + 1] = dD[k] + dt * d2D
        D[k + 1] = D[k] + dt * dD[k + 1]

    return I, dI, D, dD


def _integrate_glucose_from_latents(
    t: np.ndarray, I: np.ndarray, D: np.ndarray, p: MagdelaineParams, G0: float
) -> tuple[np.ndarray, np.ndarray]:
    """Forward Euler for glucose: dG/dt = -ksi * I + kl - kb + D."""
    dt = float(np.diff(t).mean()) if t.size > 1 else 0.1
    N = t.size
    G = np.zeros(N, dtype=np.float32)
    dG = np.zeros(N, dtype=np.float32)
    G[0] = float(G0)

    ksi = float(p.ksi)
    kl = float(p.kl)
    kb = float(p.kb)

    for k in range(N - 1):
        dGdt = -ksi * I[k] + kl - kb + D[k]
        G[k + 1] = G[k] + dt * dGdt
        dG[k + 1] = dGdt

    return G, dG


def simulate(params: MagdelaineParams, cfg: SimConfig) -> Dict[str, np.ndarray]:
    """Full synthetic trace for the selected patient using your paper presets."""
    t, u, r = _build_inputs_for_patient(cfg.patient, cfg.minutes, cfg.dt)

    # Latents via NumPy-only Euler (keeps simulator/test independent of TF)
    I, dI, D, dD = _simulate_latents_euler_numpy(
        u=u, r=r, p=params, dt=cfg.dt, I0=cfg.I0, dI0=cfg.dI0, D0=cfg.D0, dD0=cfg.dD0
    )

    # G0: table default if not provided
    G0 = cfg.G0 if cfg.G0 is not None else _G0_BY_PAT[cfg.patient]
    G, dG = _integrate_glucose_from_latents(t, I, D, params, G0)

    return {"t": t, "u": u, "r": r, "G": G, "dG": dG, "I": I, "dI": dI, "D": D}


def write_csv(path: str | Path, patient_label: str, sim: Dict[str, np.ndarray]) -> None:
    """
    Write simulation results to CSV with schema including u and r.

    IMPORTANT: This downsamples from dt=0.1 to 1-minute resolution to match notebook output.

    Columns: time, patient, glucose, glucose_derivative, insulin, insulin_derivative,
             carbohydrates, u, r
    """
    # Downsample to 1-minute resolution (matching notebook's gathert logic)
    # In notebook: gathert = int(1 / dt) = 10, so we save every 10th sample
    dt = float(np.diff(sim["t"]).mean()) if len(sim["t"]) > 1 else 0.1
    gather_interval = int(round(1.0 / dt))  # e.g., 10 for dt=0.1

    indices = np.arange(0, len(sim["t"]), gather_interval)

    df = pd.DataFrame(
        {
            "time": sim["t"][indices].astype(int),  # Convert to int minutes
            "patient": patient_label,
            "glucose": sim["G"][indices],
            "glucose_derivative": sim["dG"][indices],
            "insulin": sim["I"][indices],
            "insulin_derivative": sim["dI"][indices],
            "carbohydrates": sim["D"][indices],  # D state
            "u": sim["u"][indices],  # Insulin inputs (U/min)
            "r": sim["r"][indices] / 1000.0,  # Convert back to grams for CSV
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✓ Saved simulation to {path}")


# CLI interface for quick testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic patient data with Magdelaine model"
    )
    parser.add_argument("--patient", type=int, default=3, help="Patient number (2-11)")
    parser.add_argument(
        "--minutes", type=int, default=2880, help="Simulation duration in minutes"
    )
    parser.add_argument(
        "--dt", type=float, default=0.1, help="Time step in minutes (default 0.1)"
    )
    parser.add_argument(
        "--out_csv", type=str, default="runs/tmp/quick_demo.csv", help="Output CSV path"
    )
    args = parser.parse_args()

    from src.physics.magdelaine import make_params_from_preset

    params = make_params_from_preset(args.patient)
    cfg = SimConfig(
        minutes=args.minutes,
        dt=args.dt,
        patient=args.patient,
        label=f"Pat{args.patient}",
    )
    sim = simulate(params, cfg)
    write_csv(args.out_csv, cfg.label, sim)

# Usage:
# export PYTHONPATH=.
# python -m src.datasets.simulator --patient 3 --minutes 2880 --out_csv data/synthetic/Pat3.csv
