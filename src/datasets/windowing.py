# src/datasets/windowing.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

from .preprocessing import (
    upsample_glucose_to_min,
    project_events_to_grid,
)

@dataclass
class _CoveragePolicy:
    max_gap_min: int = 10      # max gap tolerated as "continuous"
    min_coverage: float = 0.9 # fraction of minutes that must have CGM

def _standardize_cgm(df_cgm: pd.DataFrame) -> pd.DataFrame:
    df = df_cgm.rename(columns={c: c.lower() for c in df_cgm.columns})
    # expected columns: timestamp, glucose (strings okay; we'll parse)
    if "timestamp" not in df or "glucose" not in df:
        raise ValueError("cgm.csv must have columns: timestamp, glucose")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df

def _standardize_events(df_events: pd.DataFrame) -> pd.DataFrame:
    df = df_events.rename(columns={c: c.lower() for c in df_events.columns})
    # expected columns: timestamp, kind, value
    if not {"timestamp","kind","value"}.issubset(df.columns):
        raise ValueError("events.csv must have columns: timestamp, kind, value")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    # keep only the 3 kinds we use
    df = df[df["kind"].isin(["basal","bolus","carbs"])].reset_index(drop=True)
    return df

def _contiguous_blocks_1min(idx: pd.DatetimeIndex, policy:_CoveragePolicy) -> List[slice]:
    # expects 1-min regular index; we'll still be tolerant to small gaps.
    if len(idx) == 0:
        return []
    blocks = []
    start = 0
    for i in range(1, len(idx)):
        gap = (idx[i] - idx[i-1]).total_seconds() / 60.0
        if gap > policy.max_gap_min:
            blocks.append(slice(start, i))
            start = i
    blocks.append(slice(start, len(idx)))
    return blocks

def _check_coverage(minute_index: pd.DatetimeIndex, covered_mask: np.ndarray, policy:_CoveragePolicy) -> bool:
    # covered_mask is True where we have CGM values (not NaN) on the minute grid
    return covered_mask.mean() >= policy.min_coverage

def _assemble_window_dict(
    time_index: List[str],
    glucose: np.ndarray,
    u: np.ndarray,
    r: np.ndarray,
    patient_label: str,
    dt_minutes: float,
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Assemble window in JSON-ready format for prepare_data.py.
    This matches the format expected by load_real_window().
    """
    T = len(time_index)
    t_min = np.arange(T, dtype=float) * dt_minutes
    m_t = float(t_min.max()) if T > 0 else 1.0
    
    # Scaling factors
    m_g = float(np.nanmax(glucose)) if np.nanmax(glucose) > 0 else 1.0
    m_i = 1.0  # For real data, we don't have I so we use 1.0
    m_d = float(np.nanmax(r)) if np.nanmax(r) > 0 else 1.0
    
    time_norm = t_min / m_t if m_t > 0 else t_min
    
    return {
        "patient_label": patient_label,
        "time_index": time_index,
        "t_min": t_min.tolist(),
        "time_norm": time_norm.tolist(),
        "glucose": glucose.tolist(),
        "u": u.tolist(),
        "r": r.tolist(),
        "dt_minutes": dt_minutes,
        "scales": {
            "m_t": m_t,
            "m_g": m_g,
            "m_i": m_i,
            "m_d": m_d,
        },
        "meta": meta,
    }

def select_window_manual(
    df_cgm: pd.DataFrame,
    df_events: pd.DataFrame,
    start_ts: str,
    end_ts: str,
    patient_label: str,
    dt_minutes: float = 1.0,
    max_gap_min: int = 10,
    min_coverage: float = 0.95,
) -> Dict[str, Any]:
    """Manual slice [start_ts, end_ts], then build 1-min grid CGM and u/r profiles."""
    pol = _CoveragePolicy(max_gap_min=max_gap_min, min_coverage=min_coverage)

    cgm = _standardize_cgm(df_cgm)
    ev  = _standardize_events(df_events)

    # 1) Upsample CGM to 1-min grid over the requested span
    t0 = pd.to_datetime(start_ts)
    t1 = pd.to_datetime(end_ts)
    if t1 <= t0:
        raise ValueError("end_ts must be after start_ts")

    cgm_range = cgm[(cgm["timestamp"] >= t0) & (cgm["timestamp"] <= t1)].copy()
    if cgm_range.empty:
        raise ValueError("No CGM in requested interval.")

    cgm_1min = upsample_glucose_to_min(cgm_range[["timestamp","glucose"]])
    minute_index = pd.date_range(cgm_1min["timestamp"].min(), cgm_1min["timestamp"].max(), freq="1min")
    # ensure full closed interval at 1 min
    cgm_1min = cgm_1min.set_index("timestamp").reindex(minute_index).rename_axis("timestamp").reset_index()
    cgm_1min = cgm_1min.rename(columns={"index":"timestamp"})
    covered = cgm_1min["glucose"].notna().to_numpy()

    coverage = covered.mean()
    if not _check_coverage(minute_index, covered, pol):
        print(
            f"[DEBUG] Coverage for {patient_label} "
            f"{start_ts} → {end_ts}: {coverage:.4f} "
            f"(threshold={pol.min_coverage})"
        )
        raise ValueError("Coverage below threshold in manual window.")

    # 2) Build per-minute u(t), r(t) on the same minute_index using canonical logic
    ev_range = ev[(ev["timestamp"] >= t0) & (ev["timestamp"] <= t1)].copy()

    # project_events_to_grid builds a per-minute index on [start, end]
    ur_full = project_events_to_grid(
        df_events=ev_range,
        start=minute_index[0],
        end=minute_index[-1],
    )

    # Align exactly to the CGM 1-min grid; fill any missing minutes with 0
    ur = ur_full.reindex(minute_index).fillna(0.0)
    # ur now has index = minute_index and columns: 'u' (U/min), 'r' (g/min)

    # 3) Package result using JSON-ready format
    win = _assemble_window_dict(
        time_index=[ts.strftime("%Y-%m-%d %H:%M:%S") for ts in minute_index],
        glucose=cgm_1min["glucose"].to_numpy(dtype=np.float32),
        u=ur["u"].to_numpy(dtype=np.float32),
        r=ur["r"].to_numpy(dtype=np.float32),
        patient_label=patient_label,
        dt_minutes=dt_minutes,
        meta={"mode": "manual", "start_ts": str(t0), "end_ts": str(t1)}
    )
    return win

def select_windows_auto(
    df_cgm: pd.DataFrame,
    df_events: pd.DataFrame,
    hours: int = 48,
    max_gap_min: int = 10,
    min_coverage: float = 0.98,
    stride_min: Optional[int] = None,
    patient_label: str = "RealPat",
    dt_minutes: float = 1.0,
) -> List[Dict[str, Any]]:
    """Find contiguous CGM blocks, then carve fixed-length windows and build aligned u/r."""
    pol = _CoveragePolicy(max_gap_min=max_gap_min, min_coverage=min_coverage)
    cgm = _standardize_cgm(df_cgm)
    ev  = _standardize_events(df_events)

    # 1) Get 1-min CGM over the full patient span
    cgm_1min = upsample_glucose_to_min(cgm[["timestamp","glucose"]])
    cgm_1min = cgm_1min.set_index("timestamp").sort_index()

    # regularize to 1-min integer grid across the full span
    full_index = pd.date_range(cgm_1min.index.min(), cgm_1min.index.max(), freq="1min")
    cgm_1min = cgm_1min.reindex(full_index)
    covered = cgm_1min["glucose"].notna().to_numpy()

    # 2) contiguous blocks by gap policy
    slices = _contiguous_blocks_1min(full_index, pol)
    win_len = int(hours * 60)
    step = int(stride_min) if stride_min else win_len  # non-overlap default

    out: List[Dict[str, Any]] = []
    for sl in slices:
        seg_idx = full_index[sl]
        seg_cov = covered[sl]
        if seg_idx.size < win_len:
            continue
        # slide windows within this segment
        for start_i in range(sl.start, sl.stop - win_len + 1, step):
            end_i = start_i + win_len
            rng = slice(start_i, end_i)
            minute_index = full_index[rng]
            mask = covered[rng]
            if mask.mean() < pol.min_coverage:
                continue

            cgm_seg = cgm_1min.iloc[rng]
            ev_seg  = ev[(ev["timestamp"] >= minute_index[0]) & (ev["timestamp"] <= minute_index[-1])].copy()

            # Build u(t), r(t) over this window using the canonical event logic
            ur_full = project_events_to_grid(
                df_events=ev_seg,
                start=minute_index[0],
                end=minute_index[-1],
            )

            # Align exactly to this window's minute grid
            ur = ur_full.reindex(minute_index).fillna(0.0)

            win = _assemble_window_dict(
                time_index=[ts.strftime("%Y-%m-%d %H:%M:%S") for ts in minute_index],
                glucose=cgm_seg["glucose"].to_numpy(dtype=np.float32),
                u=ur["u"].to_numpy(dtype=np.float32),
                r=ur["r"].to_numpy(dtype=np.float32),
                patient_label=patient_label,
                dt_minutes=dt_minutes,
                meta={"mode": "auto", "hours": hours}
            )
            out.append(win)

    return out