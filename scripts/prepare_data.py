"""
Prepare real patient data for training by converting to the same schema as synthetic data.

This script:
1. Loads wide-format patient CSVs from extract_real_data.py
2. Converts time_min to time (starting at 0)
3. Adds patient label
4. Converts to final schema matching synthetic data:
   [time, patient, glucose, glucose_derivative, insulin, insulin_derivative, carbohydrates, u, r]

Input: data/real/patient_*_wide.csv (from extract_real_data.py)
Output: data/real_prepared/RealPat{N}.csv (matching synthetic data format)

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --input_dir data/real --output_dir data/processed
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


def compute_derivative(y: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Compute numerical derivative using forward differences.

    Args:
        y: Array of values
        dt: Time step (default 1 minute)

    Returns:
        Derivative array (same length, last value repeated)
    """
    dy = np.diff(y) / dt
    # Append last value to maintain same length
    dy = np.append(dy, dy[-1] if len(dy) > 0 else 0)
    return dy


def interpolate_glucose(glucose: pd.Series, method: str = "linear") -> pd.Series:
    """
    Interpolate missing glucose values.

    Args:
        glucose: Series with glucose values (may have NaNs)
        method: Interpolation method ('linear', 'cubic', 'nearest')

    Returns:
        Interpolated glucose series
    """
    # Forward fill for leading NaNs
    glucose = glucose.ffill()  # Updated: replaced fillna(method='ffill')
    # Backward fill for trailing NaNs
    glucose = glucose.bfill()  # Updated: replaced fillna(method='bfill')
    # Interpolate remaining gaps
    glucose = glucose.interpolate(method=method, limit_direction="both")
    return glucose


def prepare_patient_data(
    input_file: Path,
    output_file: Path,
    patient_label: str,
    interpolate: bool = True,
    verbose: bool = True,
) -> bool:
    """
    Convert wide-format patient data to final training schema.

    Args:
        input_file: Path to patient_*_wide.csv
        output_file: Path to save prepared CSV
        patient_label: Patient identifier (e.g., "RealPat1")
        interpolate: Whether to interpolate missing glucose values
        verbose: Print progress

    Returns:
        True if successful, False otherwise
    """
    if verbose:
        print(f"  Processing: {input_file.name} → {patient_label}")

    try:
        # Load wide-format data
        df = pd.read_csv(input_file)

        # Check required columns
        required = ["time_min", "ut", "rt", "glucose"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            print(f"    ❌ Missing columns: {missing}")
            return False

        # Reset time to start at 0
        df["time"] = df["time_min"] - df["time_min"].min()

        # Handle glucose interpolation
        if interpolate:
            n_missing_before = df["glucose"].isna().sum()
            df["glucose"] = interpolate_glucose(df["glucose"])
            n_missing_after = df["glucose"].isna().sum()
            if verbose and n_missing_before > 0:
                print(
                    f"    Interpolated {n_missing_before} missing glucose values → {n_missing_after} remaining"
                )

        # Compute glucose derivative
        df["glucose_derivative"] = compute_derivative(df["glucose"].values)

        # For real data, we don't have true I(t) and D(t) states
        # We'll set them to NaN - the model will need to handle this
        # OR we could simulate them using the Magdelaine model (future enhancement)
        df["insulin"] = np.nan
        df["insulin_derivative"] = np.nan
        df["carbohydrates"] = np.nan

        # Rename columns to match synthetic data schema
        df["patient"] = patient_label
        df["u"] = df["ut"]  # Total insulin input (U/min)
        df["r"] = df["rt"]  # Carb input (g/min) - matches synthetic data

        # Select and order columns to match synthetic schema
        output_df = df[
            [
                "time",
                "patient",
                "glucose",
                "glucose_derivative",
                "insulin",  # NaN for real data (or simulated)
                "insulin_derivative",  # NaN for real data
                "carbohydrates",  # NaN for real data (or simulated)
                "u",
                "r",
            ]
        ]

        # Save to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_file, index=False)

        if verbose:
            duration_hours = output_df["time"].max() / 60
            n_glucose = output_df["glucose"].notna().sum()
            print(
                f"    ✓ Duration: {duration_hours:.1f}h, Glucose readings: {n_glucose}"
            )

        return True

    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Prepare real patient data for training"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/real",
        help="Directory with patient_*_wide.csv files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save prepared CSV files",
    )
    parser.add_argument(
        "--no_interpolate",
        action="store_true",
        help="Skip glucose interpolation (keep NaNs)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed progress"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print("=" * 80)
    print("PREPARING REAL PATIENT DATA FOR TRAINING")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Interpolate glucose: {not args.no_interpolate}")

    # Find all wide-format files
    wide_files = sorted(input_dir.glob("patient_*_wide.csv"))

    if not wide_files:
        print(f"\n❌ No patient files found in {input_dir}")
        print("   Expected files matching pattern: patient_*_wide.csv")
        print("   Run extract_real_data.py first!")
        return

    print(f"\nFound {len(wide_files)} patient files")

    # Process each patient
    print("\nProcessing patients...")
    success_count = 0

    for i, input_file in enumerate(wide_files, 1):
        # Generate patient label: RealPat1, RealPat2, etc.
        patient_label = f"RealPat{i}"
        output_file = output_dir / f"{patient_label}.csv"

        print(f"\n[{i}/{len(wide_files)}] {patient_label}")

        success = prepare_patient_data(
            input_file=input_file,
            output_file=output_file,
            patient_label=patient_label,
            interpolate=not args.no_interpolate,
            verbose=args.verbose,
        )

        if success:
            success_count += 1

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Successfully prepared: {success_count}/{len(wide_files)} patients")

    if success_count > 0:
        print(f"✅ Output files saved to: {output_dir}")
        print("\nData schema:")
        print("  - time: minutes from start")
        print("  - patient: patient ID (RealPat1, RealPat2, ...)")
        print("  - glucose: blood glucose (mg/dL)")
        print("  - glucose_derivative: dG/dt")
        print("  - insulin: NaN (not available for real data)")
        print("  - insulin_derivative: NaN")
        print("  - carbohydrates: NaN (digestion state not available)")
        print("  - u: insulin input (U/min)")
        print("  - r: carbohydrate input (g/min)")
        print("\nNext steps:")
        print("  1. Update src/datasets/loader.py to handle 'real_prepared' source")
        print("  2. Test loading: python tests/test_data_loading.py")
        print("  3. Start training with real data")
    else:
        print("❌ No patients prepared successfully")


if __name__ == "__main__":
    main()

# python << 'EOF'
# import pandas as pd
# import matplotlib.pyplot as plt

# # Compare one synthetic vs one real
# syn = pd.read_csv('data/synthetic/Pat3.csv')
# real = pd.read_csv('data/processed/RealPat1.csv')

# fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

# # Synthetic
# axes[0].plot(syn['time'] / 60, syn['glucose'], 'b-', linewidth=0.5, alpha=0.8)
# axes[0].set_ylabel('Glucose (mg/dL)', fontsize=12)
# axes[0].set_title('Synthetic Patient (Pat3) - Simulated, Complete', fontsize=14)
# axes[0].grid(True, alpha=0.3)

# # Real
# axes[1].plot(real['time'] / 60, real['glucose'], 'r-', linewidth=0.5, alpha=0.8)
# axes[1].set_ylabel('Glucose (mg/dL)', fontsize=12)
# axes[1].set_xlabel('Time (hours)', fontsize=12)
# axes[1].set_title('Real Patient (RealPat1) - Measured, Interpolated', fontsize=14)
# axes[1].grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig('data_comparison.png', dpi=150)
# print("✅ Saved: data_comparison.png")
# plt.show()
# EOF
