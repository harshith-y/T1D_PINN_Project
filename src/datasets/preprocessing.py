from __future__ import annotations

"""
Data preprocessing utilities for CGM and event data.

This module contains functions extracted from the original loader.py:
- upsample_glucose_to_min: Interpolate CGM to 1-minute grid
- project_events_to_grid: Convert events to per-minute u(t) and r(t) profiles

These are used by windowing.py and extractor.py but kept separate from
the main loader to avoid circular dependencies.
"""

import numpy as np
import pandas as pd


def upsample_glucose_to_min(
    df_cgm: pd.DataFrame,
    time_col: str = "timestamp",
    value_col: str | None = None,
    assume_local: bool = True,
) -> pd.DataFrame:
    """
    Resample CGM (typically every 5 min) to a 1-min grid via time interpolation.
    
    This function takes sparse CGM measurements (usually every 5 minutes) and
    interpolates them to a regular 1-minute grid. Duplicate timestamps are
    averaged, and missing values are interpolated linearly.
    
    Args:
        df_cgm: DataFrame with timestamp and glucose columns
        time_col: Name of timestamp column (default "timestamp")
        value_col: Name of glucose column (default auto-detected)
        assume_local: Whether to assume local timezone (default True)
        
    Returns:
        DataFrame with columns [time_col, 'glucose'] at 1-minute resolution
        
    Example:
        >>> df = pd.DataFrame({
        ...     'timestamp': ['2024-01-01 00:00:00', '2024-01-01 00:05:00'],
        ...     'glucose': [120, 125]
        ... })
        >>> df_1min = upsample_glucose_to_min(df)
        >>> len(df_1min)  # Should be 6 (0, 1, 2, 3, 4, 5 minutes)
        6
    """
    df = df_cgm.copy()

    # --- Choose glucose column if not explicitly provided ---
    if value_col is None:
        candidates = ["glucose_mgdl", "glucose", "value"]
        for c in candidates:
            if c in df.columns:
                value_col = c
                break
        if value_col is None:
            raise KeyError(
                f"Could not infer glucose column. "
                f"Available columns: {list(df.columns)}; "
                f"tried candidates {candidates}."
            )
    elif value_col not in df.columns:
        raise KeyError(f"Column not found: {value_col}")

    # Parse timestamps
    df[time_col] = pd.to_datetime(df[time_col], utc=False, infer_datetime_format=True)

    # Drop rows without glucose
    df = df.dropna(subset=[value_col])

    # Average duplicate timestamps
    df = (
        df.groupby(time_col, as_index=False)[value_col]
        .mean()
        .sort_values(time_col)
    )

    # Set index to timestamp for resampling
    df = df.set_index(time_col)

    # Per-minute index
    minute_idx = pd.date_range(
        start=df.index.min().floor("min"),
        end=df.index.max().ceil("min"),
        freq="1min",
        tz=df.index.tz,
    )

    # Interpolate to 1-minute grid
    G = (
        df[value_col]
        .astype(float)
        .reindex(minute_idx)
        .interpolate(method="time", limit_direction="both")
        .rename("glucose")  # Standardized column name
    )

    out = G.to_frame()
    out = out.rename_axis(time_col).reset_index()
    return out


def project_events_to_grid(
    df_events: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    time_col: str = "timestamp",
    type_col: str = "type",
    value_col: str = "value",
    units_col: str = "units",
    duration_col: str = "duration_minutes",
) -> pd.DataFrame:
    """
    Project long-form events (basal U/hr, bolus U, carbs g) onto a 1-minute grid.
    
    This function takes event data in long format and creates per-minute time series
    for insulin delivery (u) and carbohydrate intake (r). It handles:
    - Basal rates (U/hr → U/min, step function)
    - Temporary basal adjustments
    - Pump suspend/resume events
    - Bolus doses (U, impulse)
    - Carbohydrate intake (g, impulse)
    
    Args:
        df_events: Long-form events DataFrame
        start: Start timestamp for output grid
        end: End timestamp for output grid
        time_col: Name of timestamp column
        type_col: Name of event type column ('basal', 'bolus', 'carbs', etc.)
        value_col: Name of value column
        units_col: Name of units column (optional)
        duration_col: Name of duration column (optional, for basal)
        
    Returns:
        DataFrame indexed per minute with columns:
            - u: total insulin input per minute (U/min) [basal + bolus]
            - r: carb input per minute (g/min)
            
    Example:
        >>> events = pd.DataFrame({
        ...     'timestamp': ['2024-01-01 00:00:00', '2024-01-01 00:05:00'],
        ...     'type': ['basal', 'bolus'],
        ...     'value': [1.0, 5.0]  # 1 U/hr basal, 5 U bolus
        ... })
        >>> grid = project_events_to_grid(
        ...     events,
        ...     start=pd.Timestamp('2024-01-01 00:00:00'),
        ...     end=pd.Timestamp('2024-01-01 00:10:00')
        ... )
        >>> grid['u'].iloc[0]  # First minute: 1/60 U/min from basal
        0.0166...
        >>> grid['u'].iloc[5]  # Fifth minute: 1/60 + 5 (bolus added)
        5.0166...
    """
    ev = df_events.copy()

    # ---- Normalize column names between different sources ----
    # type_col: in KCL extractor we used 'kind' instead of 'type'
    if type_col not in ev.columns:
        if "kind" in ev.columns:
            type_col = "kind"
        else:
            raise KeyError(
                f"Expected an event type column '{type_col}' or 'kind'; "
                f"got columns: {list(ev.columns)}"
            )

    # value_col: KCL extractor uses 'value'
    if value_col not in ev.columns:
        if "value" in ev.columns:
            value_col = "value"
        else:
            raise KeyError(
                f"Expected an event value column '{value_col}' or 'value'; "
                f"got columns: {list(ev.columns)}"
            )

    # units_col and duration_col may not exist in KCL events; make them safe
    if units_col not in ev.columns:
        ev[units_col] = np.nan
    if duration_col not in ev.columns:
        ev[duration_col] = np.nan

    # Time column to datetime
    if time_col not in ev.columns:
        raise KeyError(
            f"Expected time column '{time_col}' in events; "
            f"got columns: {list(ev.columns)}"
        )
    ev[time_col] = pd.to_datetime(ev[time_col], utc=False, errors="coerce")
    ev = ev.dropna(subset=[time_col])

    # Create per-minute index
    minute_idx = pd.date_range(start.floor("min"), end.ceil("min"), freq="1min", tz=start.tz)
    u = pd.Series(0.0, index=minute_idx, name="u")  # U/min (rate + bolus)
    r = pd.Series(0.0, index=minute_idx, name="r")  # g/min

    # 2a) Basal profiles: step function in U/hr → U/min
    basal = ev[ev[type_col].str.lower().eq("basal")]
    if not basal.empty:
        # Persist each basal rate until the next basal event
        basal = basal[[time_col, value_col, units_col]].rename(columns={value_col: "rate", units_col: "units"})
        # Sanity: if units differ, you can insert conversions here
        basal["rate_u_per_min"] = basal["rate"].astype(float) / 60.0
        basal = basal.reset_index(drop=True)

        for i, row in basal.iterrows():
            t0 = row[time_col]
            t1 = basal.loc[i + 1, time_col] if i + 1 < len(basal) else end
            t0 = t0.floor("min")
            t1 = t1.floor("min")
            u.loc[t0:t1] = row["rate_u_per_min"]

    # 2b) Temporary basal segments (optional)
    if "temp_basal" in ev[type_col].str.lower().unique():
        tb = ev[ev[type_col].str.lower().eq("temp_basal")].copy()
        tb = tb[[time_col, value_col, units_col, duration_col]].rename(columns={value_col: "rate", units_col: "units"})
        tb["rate_u_per_min"] = tb["rate"].astype(float) / 60.0
        for _, row in tb.iterrows():
            t0 = row[time_col].floor("min")
            dur = int(row.get(duration_col, 0))  # minutes
            t1 = t0 + pd.Timedelta(minutes=max(dur, 0))
            u.loc[t0:t1] = row["rate_u_per_min"]  # overrides for the temp window

    # 2c) Pump suspend/resume (optional)
    if "suspend" in ev[type_col].str.lower().unique():
        sus = ev[ev[type_col].str.lower().eq("suspend")][time_col].dt.floor("min")
        for t in sus:
            u.loc[t:] = 0.0  # zero out until a resume (simple model)
    if "resume" in ev[type_col].str.lower().unique():
        # On resume, basal events after this time will already override appropriately
        pass

    # 2d) Bolus: instantaneous dose U → add to that minute bucket (additive with basal)
    bolus = ev[ev[type_col].str.lower().eq("bolus")]
    if not bolus.empty:
        for _, row in bolus.iterrows():
            t = row[time_col].floor("min")
            dose_u = float(row[value_col])
            if t in u.index:
                u.loc[t] += dose_u

    # 2e) Carbs: instantaneous grams → add to that minute bucket
    carbs = ev[ev[type_col].str.lower().eq("carbs")]
    if not carbs.empty:
        for _, row in carbs.iterrows():
            t = row[time_col].floor("min")
            grams = float(row[value_col])
            if t in r.index:
                r.loc[t] += grams

    return pd.concat([u, r], axis=1).fillna(0.0)