from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np
import pandas as pd
import json
from pathlib import Path


@dataclass
class TrainingWindow:
    """
    Unified container for training data (synthetic or real).

    This replaces the old mix of DataFrame/dict/PreparedWindow returns,
    providing a single consistent interface for all training code.
    """

    # === Always present (both synthetic and real) ===
    t_min: np.ndarray  # [0, 1, 2, ..., T-1] in minutes
    time_norm: np.ndarray  # [0, 1] normalized time
    glucose: np.ndarray  # mg/dL, shape (T,)
    u: np.ndarray  # insulin inputs U/min, shape (T,)
    r: np.ndarray  # carb inputs g/min, shape (T,)

    # === Optional: only for synthetic data ===
    glucose_deriv: Optional[np.ndarray] = None  # dG/dt
    insulin: Optional[np.ndarray] = None  # I(t) ground truth
    insulin_deriv: Optional[np.ndarray] = None  # dI/dt ground truth
    digestion: Optional[np.ndarray] = None  # D(t) ground truth
    # Note: no dD - not saved from simulator

    # === Metadata ===
    patient_id: str = "Unknown"
    data_source: Literal["synthetic", "real"] = "synthetic"

    # === Scaling factors ===
    m_t: float = 1.0  # time scaling (max time in minutes)
    m_g: float = 1.0  # glucose scaling (max glucose)
    m_i: float = 1.0  # insulin scaling (fixed at 1.0 in your notebooks)
    m_d: float = 1.0  # digestion scaling (max carb digestion rate)

    @property
    def has_latent_states(self) -> bool:
        """True if ground truth I/D are available (synthetic only)."""
        return self.insulin is not None and self.digestion is not None

    @property
    def scales_dict(self) -> dict:
        """Return scaling factors as dict (for backward compatibility with old code)."""
        return {"m_t": self.m_t, "m_g": self.m_g, "m_i": self.m_i, "m_d": self.m_d}

    def __repr__(self) -> str:
        latent_status = "with latents" if self.has_latent_states else "glucose only"
        return (
            f"TrainingWindow(patient={self.patient_id}, source={self.data_source}, "
            f"length={len(self.t_min)}, {latent_status})"
        )


def load_synthetic_window(
    patient: int,
    root: str | Path = "data/synthetic",
    t_start: int = 0,
    t_end: int = 2880,
) -> TrainingWindow:
    """
    Load synthetic data from Pat{patient}.csv with full ground truth.

    This function loads simulator-generated data with complete ground truth for
    glucose, insulin, and carbohydrate digestion states. The CSV must have been
    generated with the updated simulator that includes u and r columns.

    Args:
        patient: Patient number (2-11)
        root: Directory containing Pat{N}.csv files
        t_start: Start time in minutes (default 0)
        t_end: End time in minutes (default 2880 = 48 hours)

    Returns:
        TrainingWindow with insulin/digestion ground truth populated

    Raises:
        FileNotFoundError: If Pat{patient}.csv doesn't exist
        ValueError: If time range is invalid or no data in range

    Example:
        >>> window = load_synthetic_window(patient=3)
        >>> print(window.glucose.shape)  # (2881,)
        >>> print(window.has_latent_states)  # True
        >>> # Use in training:
        >>> model.fit(window.time_norm, window.glucose, ...)
    """
    root = Path(root)
    csv_path = root / f"Pat{patient}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Synthetic data not found: {csv_path}\n"
            f"Generate it with: python -m src.datasets.simulator --patient {patient}"
        )

    # Load CSV
    df = pd.read_csv(csv_path)

    # Expected columns (from updated simulator):
    # ['time', 'patient', 'glucose', 'glucose_derivative', 'insulin',
    #  'insulin_derivative', 'carbohydrates', 'u', 'r']

    required_cols = [
        "time",
        "glucose",
        "glucose_derivative",
        "insulin",
        "insulin_derivative",
        "carbohydrates",
        "u",
        "r",
    ]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}\n"
            f"Your CSV may be from the old simulator. Please regenerate with:\n"
            f"  python -m src.datasets.simulator --patient {patient} --out_csv {csv_path}"
        )

    # Slice time window
    mask = (df["time"] >= t_start) & (df["time"] <= t_end)
    df = df[mask].copy()

    if df.empty:
        raise ValueError(f"No data in time range [{t_start}, {t_end}]")

    # Extract arrays (matching your notebook conventions)
    t = df["time"].values.astype(np.float32)
    G = df["glucose"].values.astype(np.float32)
    dG = df["glucose_derivative"].values.astype(np.float32)
    I = df["insulin"].values.astype(np.float32)
    dI = df["insulin_derivative"].values.astype(np.float32)
    D = df["carbohydrates"].values.astype(np.float32)
    u = df["u"].values.astype(np.float32)
    r = df["r"].values.astype(np.float32)

    # Scaling factors (EXACTLY matching your notebook logic)
    m_t = float(t.max()) if t.max() > 0 else 1.0
    m_g = float(G.max()) if G.max() > 0 else 1.0
    m_i = float(I.max()) if I.max() > 0 else 1.0
    m_d = float(D.max()) if D.max() > 0 else 1.0

    # Normalize time to [0, 1] (matching your notebook: time_norm = t / m_t)
    t_min = t - t[0]  # Start from 0
    time_norm = t_min / m_t if m_t > 0 else t_min

    return TrainingWindow(
        t_min=t_min,
        time_norm=time_norm,
        glucose=G,
        u=u,
        r=r,
        glucose_deriv=dG,
        insulin=I,
        insulin_deriv=dI,
        digestion=D,
        patient_id=f"Pat{patient}",
        data_source="synthetic",
        m_t=m_t,
        m_g=m_g,
        m_i=m_i,
        m_d=m_d,
    )


def load_real_patient_csv(
    patient: int,
    root: str | Path = "data/processed",
    t_start: Optional[int] = None,
    t_end: Optional[int] = None,
) -> TrainingWindow:
    """
    Load real patient data from RealPat{patient}.csv.

    This function loads preprocessed real patient data from CSV files in the
    processed directory. Unlike synthetic data, real data typically only has
    glucose measurements and inputs, without ground truth for latent states.

    The CSV format matches synthetic patients with columns for insulin,
    insulin_derivative, and carbohydrates, but these are empty (blank strings)
    for real patients since we don't have ground truth.

    Args:
        patient: Patient number (1-15)
        root: Directory containing RealPat{N}.csv files
        t_start: Optional start time in minutes (for windowing)
        t_end: Optional end time in minutes (for windowing)

    Returns:
        TrainingWindow with glucose/u/r populated, latent states = None

    Raises:
        FileNotFoundError: If RealPat{patient}.csv doesn't exist
        ValueError: If CSV is missing required columns or time range invalid

    Example:
        >>> window = load_real_patient_csv(patient=3)
        >>> print(window.glucose.shape)  # (25000,) or similar
        >>> print(window.has_latent_states)  # False (no ground truth I, D)
        >>> # Use in training:
        >>> model.fit(window.time_norm, window.glucose, ...)

    Notes:
        - Real data may have NaN values (sensor dropouts)
        - Columns for insulin/carbohydrates exist but are empty
        - u and r inputs are present (derived from clinical records)
        - Sampling rate may be irregular (CGM artifacts)
    """
    root = Path(root)
    csv_path = root / f"RealPat{patient}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Real patient data not found: {csv_path}\n"
            f"Available files in {root}:\n"
            + "\n".join(f"  - {f.name}" for f in sorted(root.glob("RealPat*.csv")))
        )

    # Load CSV
    df = pd.read_csv(csv_path)

    # Expected columns (matching synthetic format):
    # ['time', 'patient', 'glucose', 'glucose_derivative', 'insulin',
    #  'insulin_derivative', 'carbohydrates', 'u', 'r']
    # Note: insulin, insulin_derivative, carbohydrates are EMPTY for real patients

    # Required columns (at minimum)
    required_cols = ["time", "glucose"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            f"Real patient CSVs must have at least 'time' and 'glucose'"
        )

    # Optional: window the data by time
    if t_start is not None or t_end is not None:
        t_start = t_start or df["time"].min()
        t_end = t_end or df["time"].max()
        mask = (df["time"] >= t_start) & (df["time"] <= t_end)
        df = df[mask].copy()

        if df.empty:
            raise ValueError(f"No data in time range [{t_start}, {t_end}]")

    # Extract required arrays
    t = df["time"].values.astype(np.float32)
    G = df["glucose"].values.astype(np.float32)

    # Helper function to extract column, handling empty strings (common in real data)
    def extract_column(df, col_name):
        """Extract column as float array, returning None if column is empty/all NaN."""
        if col_name not in df.columns:
            return None

        # Try to convert to float, empty strings become NaN
        arr = pd.to_numeric(df[col_name], errors="coerce").values.astype(np.float32)

        # If all values are NaN, treat as empty column
        if np.all(np.isnan(arr)):
            return None

        return arr

    # Extract optional arrays with proper empty-string handling
    u = extract_column(df, "u")
    r = extract_column(df, "r")

    # If u or r is None/empty, use zeros (no inputs)
    u = u if u is not None else np.zeros_like(t)
    r = r if r is not None else np.zeros_like(t)

    # Real data typically doesn't have ground truth for latent states
    # Columns may exist but be empty (empty strings in CSV)
    I = extract_column(df, "insulin")  # Will be None if empty
    D = extract_column(df, "carbohydrates")  # Will be None if empty
    dG = extract_column(df, "glucose_derivative")  # May or may not be present

    # Scaling factors
    m_t = float(t.max()) if t.max() > 0 else 1.0
    m_g = float(np.nanmax(G)) if np.nanmax(G) > 0 else 1.0  # Use nanmax for real data
    m_i = 1.0  # Standard scaling
    m_d = float(np.nanmax(D)) if D is not None and np.nanmax(D) > 0 else 1.0

    # Normalize time to [0, 1]
    t_min = t - t[0]  # Start from 0
    time_norm = t_min / m_t if m_t > 0 else t_min

    return TrainingWindow(
        t_min=t_min,
        time_norm=time_norm,
        glucose=G,
        u=u,
        r=r,
        # Real data has no ground truth for these (columns exist but are empty)
        glucose_deriv=dG,  # May be present from preprocessing
        insulin=I,  # None (empty column)
        insulin_deriv=None,
        digestion=D,  # None (empty column)
        patient_id=f"RealPat{patient}",
        data_source="real",
        m_t=m_t,
        m_g=m_g,
        m_i=m_i,
        m_d=m_d,
    )


def load_real_window(window_json: str | Path) -> TrainingWindow:
    """
    Load real patient window from JSON (created by prepare_data.py).

    This function loads real patient data that has been preprocessed and windowed
    by the prepare_data.py script. Real data only contains glucose measurements
    and inputs (u, r), with no ground truth for latent states I and D.

    Args:
        window_json: Path to window JSON file

    Returns:
        TrainingWindow with only glucose/u/r (no latent states)

    Raises:
        FileNotFoundError: If JSON file doesn't exist
        KeyError: If JSON is missing required fields

    Example:
        >>> window = load_real_window("data/processed/windows/pat123/win_xxx.json")
        >>> print(window.has_latent_states)  # False
        >>> print(np.isnan(window.glucose).sum())  # May have NaNs in real data
    """
    window_json = Path(window_json)

    if not window_json.exists():
        raise FileNotFoundError(f"Window JSON not found: {window_json}")

    with open(window_json, "r") as f:
        data = json.load(f)

    # JSON structure (from your prepare_data.py output):
    # {
    #   "patient_label": "RealPat_abc123",
    #   "time_index": ["2024-01-01 00:00:00", ...],
    #   "t_min": [0, 1, 2, ...],
    #   "time_norm": [0.0, 0.000347, ...],
    #   "glucose": [...],
    #   "u": [...],  # U/min
    #   "r": [...],  # g/min
    #   "scales": {"m_t": 2880, "m_g": 250, ...}
    # }

    required_keys = [
        "t_min",
        "time_norm",
        "glucose",
        "u",
        "r",
        "scales",
        "patient_label",
    ]
    missing = set(required_keys) - set(data.keys())
    if missing:
        raise KeyError(f"JSON missing required keys: {missing}")

    # Extract arrays
    t_min = np.array(data["t_min"], dtype=np.float32)
    time_norm = np.array(data["time_norm"], dtype=np.float32)
    glucose = np.array(data["glucose"], dtype=np.float32)
    u = np.array(data["u"], dtype=np.float32)
    r = np.array(data["r"], dtype=np.float32)

    scales = data["scales"]

    return TrainingWindow(
        t_min=t_min,
        time_norm=time_norm,
        glucose=glucose,
        u=u,
        r=r,
        # No latent states for real data
        glucose_deriv=None,
        insulin=None,
        insulin_deriv=None,
        digestion=None,
        patient_id=data["patient_label"],
        data_source="real",
        m_t=scales["m_t"],
        m_g=scales["m_g"],
        m_i=scales["m_i"],
        m_d=scales["m_d"],
    )


# Backward compatibility functions (if needed during migration)
def load_synthetic_csv(pat: int, root: str | Path = "data/synthetic") -> pd.DataFrame:
    """
    DEPRECATED: Legacy function for backward compatibility.
    Use load_synthetic_window() instead.

    This function maintains the old interface that returned a DataFrame.
    New code should use load_synthetic_window() which returns TrainingWindow.
    """
    import warnings

    warnings.warn(
        "load_synthetic_csv() is deprecated. Use load_synthetic_window() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    window = load_synthetic_window(pat, root)

    # Reconstruct old DataFrame format
    df = pd.DataFrame(
        {
            "t": window.t_min,
            "G": window.glucose,
            "dG": window.glucose_deriv,
            "I": window.insulin,
            "dI": window.insulin_deriv,
            "D": window.digestion,
        }
    )

    return df
