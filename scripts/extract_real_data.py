"""
Extract and process real patient data from raw health data file.

This script processes the raw health data export CSV into 
wide-format time series that match the synthetic data schema.

Input file (expected in data/raw/):
    - kcl_hdp_aug_25_export.csv (or specified via --input_file)

Output files (saved to data/real/):
    - patient_{ID}_wide.csv for each patient

Usage:
    python scripts/extract_real_data.py
    python scripts/extract_real_data.py --input_file data/raw/my_export.csv --output_dir data/real
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np


def extract_patient_data(
    user_id: str,
    data_df: pd.DataFrame,
    output_dir: Path,
    verbose: bool = True
) -> bool:
    """
    Extract and process data for a single patient.
    
    Args:
        user_id: Patient identifier
        data_df: DataFrame with all health data points
        output_dir: Directory to save processed CSV
        verbose: Print progress messages
    
    Returns:
        True if successful, False if no data for patient
    """
    if verbose:
        print(f"\nProcessing patient: {user_id}")
    
    # Filter data for this patient
    user_data = data_df[data_df['user_id'] == user_id].copy()
    if user_data.empty:
        if verbose:
            print("  ⚠️  No data for this patient")
        return False
    
    # Compute time in minutes from patient's first measurement
    patient_start = user_data['start_date'].min()
    user_data['time_min'] = (
        (user_data['start_date'] - patient_start).dt.total_seconds() / 60
    ).round().astype(int)
    
    # === Process basal insulin ===
    basal_data = user_data[user_data['type'] == 'basal'].copy()
    basal_data['duration_min'] = (
        (basal_data['end_date'] - basal_data['start_date']).dt.total_seconds() / 60
    )
    basal_data = basal_data[basal_data['duration_min'] >= 1]
    
    # Expand basal across its duration (U/hr → U/min)
    basal_expanded = []
    for _, row in basal_data.iterrows():
        t_start = int(((row['start_date'] - patient_start).total_seconds()) // 60)
        t_end = int(((row['end_date'] - patient_start).total_seconds()) // 60)
        rate_per_min = row['value'] / 60.0  # U/hr → U/min
        for t in range(t_start, t_end):
            basal_expanded.append({'time_min': t, 'ut_basal': rate_per_min})
    
    basal_df = pd.DataFrame(basal_expanded) if basal_expanded else pd.DataFrame(columns=['time_min', 'ut_basal'])
    
    # === Process bolus insulin ===
    bolus_data = user_data[user_data['type'] == 'bolus'].copy()
    if not bolus_data.empty:
        bolus_df = bolus_data[['time_min', 'value']].copy()
        bolus_df.rename(columns={'value': 'ut_bolus'}, inplace=True)
    else:
        bolus_df = pd.DataFrame(columns=['time_min', 'ut_bolus'])
    
    # === Process carbohydrates ===
    cho_data = user_data[user_data['type'] == 'carbs'].copy()
    if not cho_data.empty:
        cho_df = cho_data[['time_min', 'value']].copy()
        cho_df.rename(columns={'value': 'rt'}, inplace=True)
    else:
        cho_df = pd.DataFrame(columns=['time_min', 'rt'])
    
    # === Process glucose ===
    glucose_data = user_data[user_data['type'] == 'glucose'].copy()
    if not glucose_data.empty:
        glucose_data.sort_values(by='start_date', inplace=True)
        glucose_df = glucose_data.groupby('time_min').tail(1)[['time_min', 'value']].copy()
        glucose_df.rename(columns={'value': 'glucose'}, inplace=True)
    else:
        glucose_df = pd.DataFrame(columns=['time_min', 'glucose'])
    
    # === Merge into wide-format time series ===
    min_time = user_data['time_min'].min()
    max_time = user_data['time_min'].max()
    combined = pd.DataFrame({'time_min': np.arange(min_time, max_time + 1)})
    
    # Merge each data type
    if not basal_df.empty:
        combined = combined.merge(
            basal_df.groupby('time_min').sum().reset_index(), 
            on='time_min', 
            how='left'
        )
    else:
        combined['ut_basal'] = 0.0
    
    if not bolus_df.empty:
        combined = combined.merge(
            bolus_df.groupby('time_min').sum().reset_index(), 
            on='time_min', 
            how='left'
        )
    else:
        combined['ut_bolus'] = 0.0
    
    if not cho_df.empty:
        combined = combined.merge(
            cho_df.groupby('time_min').sum().reset_index(), 
            on='time_min', 
            how='left'
        )
    else:
        combined['rt'] = 0.0
    
    if not glucose_df.empty:
        combined = combined.merge(glucose_df, on='time_min', how='left')
    else:
        combined['glucose'] = np.nan
    
    # Fill missing impulses with 0
    combined['ut_basal'] = combined['ut_basal'].fillna(0)
    combined['ut_bolus'] = combined['ut_bolus'].fillna(0)
    combined['ut'] = combined['ut_basal'] + combined['ut_bolus']
    combined['rt'] = combined['rt'].fillna(0)
    
    # Trim to last glucose measurement
    if combined['glucose'].notna().any():
        max_glucose_time = combined[combined['glucose'].notna()]['time_min'].max()
        combined = combined[combined['time_min'] <= max_glucose_time]
    
    # Reorder columns to match expected format
    combined = combined[['time_min', 'ut_basal', 'ut_bolus', 'ut', 'rt', 'glucose']]
    
    # Save to file
    # Shorten user_id to first 8 chars for filename
    short_id = user_id[:8] if len(user_id) > 8 else user_id
    filename = output_dir / f"patient_{short_id}_wide.csv"
    combined.to_csv(filename, index=False)
    
    if verbose:
        n_glucose = combined['glucose'].notna().sum()
        duration_hours = combined['time_min'].max() / 60
        print(f"  ✓ Saved: {filename}")
        print(f"    Duration: {duration_hours:.1f} hours, Glucose readings: {n_glucose}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract real patient data from raw health data file"
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default='data/raw/kcl_hdp_aug_25_export.csv',
        help='Path to health data CSV file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/real',
        help='Directory to save processed patient CSV files'
    )
    parser.add_argument(
        '--patients',
        type=str,
        nargs='*',
        default=None,
        help='Specific patient IDs to process (default: all)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress'
    )
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check input file exists
    if not input_file.exists():
        raise FileNotFoundError(f"Health data file not found: {input_file}")
    
    print("=" * 80)
    print("EXTRACTING REAL PATIENT DATA")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    print("\nLoading health data file...")
    data_df = pd.read_csv(input_file)
    
    # Convert datetime columns
    data_df['start_date'] = pd.to_datetime(data_df['start_date'])
    data_df['end_date'] = pd.to_datetime(data_df['end_date'])
    
    # Get unique patients from data
    all_patients = data_df['user_id'].unique()
    
    print(f"  Patients found: {len(all_patients)}")
    print(f"  Data points: {len(data_df)}")
    print(f"  Data types: {sorted(data_df['type'].unique())}")
    
    # Determine which patients to process
    if args.patients:
        patients_to_process = [p for p in args.patients if p in all_patients]
        if len(patients_to_process) < len(args.patients):
            print(f"  ⚠️  Warning: {len(args.patients) - len(patients_to_process)} patient IDs not found")
    else:
        patients_to_process = all_patients
    
    print(f"\nProcessing {len(patients_to_process)} patients...")
    
    # Process each patient
    success_count = 0
    for i, user_id in enumerate(patients_to_process, 1):
        print(f"\n[{i}/{len(patients_to_process)}] {user_id[:8]}...")
        success = extract_patient_data(user_id, data_df, output_dir, verbose=args.verbose)
        if success:
            success_count += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Successfully processed: {success_count}/{len(patients_to_process)} patients")
    if success_count > 0:
        print(f"✅ Output files saved to: {output_dir}")
        print("\nNext steps:")
        print("  1. Review the extracted data")
        print("  2. Run: python scripts/prepare_data.py")
        print("  3. Update your loader to handle real patient data")
    else:
        print("❌ No patients processed successfully")


if __name__ == "__main__":
    main()