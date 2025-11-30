#!/usr/bin/env python3
"""
Regenerate all synthetic patient data (Pat2.csv through Pat11.csv)
with the updated simulator that includes u and r columns.

This script must be run once to update your synthetic data files
before using the refactored loader.

Usage:
    python scripts/regenerate_synthetic_data.py

Or with custom output directory:
    python scripts/regenerate_synthetic_data.py --out_dir data/synthetic
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.datasets.simulator import simulate, write_csv, SimConfig
from src.physics.magdelaine import make_params_from_preset


def regenerate_all_patients(out_dir: Path, patients: list[int] = None):
    """
    Regenerate synthetic data for all patients.

    Args:
        out_dir: Output directory for CSV files
        patients: List of patient numbers (default: 2-11)
    """
    if patients is None:
        patients = list(range(2, 12))  # Patients 2-11

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("REGENERATING SYNTHETIC PATIENT DATA")
    print("=" * 80)
    print(f"Output directory: {out_dir}")
    print(f"Patients: {patients}")
    print()

    success_count = 0
    failed = []

    for i, pat in enumerate(patients, 1):
        print(f"[{i}/{len(patients)}] Generating Patient {pat}...", end=" ")

        try:
            # Get patient-specific parameters
            params = make_params_from_preset(pat)

            # Configure simulation (48 hours at 1-minute resolution)
            cfg = SimConfig(
                minutes=48 * 60,  # 2880 minutes
                dt=1.0,  # 1-minute time step
                patient=pat,
                label=f"Pat{pat}",
            )

            # Run simulation
            sim = simulate(params, cfg)

            # Write to CSV
            out_path = out_dir / f"Pat{pat}.csv"
            write_csv(out_path, f"Pat{pat}", sim)

            print(f"✓ Saved to {out_path}")
            success_count += 1

        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed.append((pat, str(e)))

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Successfully generated: {success_count}/{len(patients)} patients")

    if failed:
        print(f"\nFailed patients:")
        for pat, error in failed:
            print(f"  - Patient {pat}: {error}")
        return 1
    else:
        print("\n✅ All patients generated successfully!")
        print("\nNext steps:")
        print("  1. Run tests: python tests/test_data_loading.py")
        print("  2. Update your training code to use new loader")
        print("  3. Verify one training run produces same results as notebook")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate synthetic patient data with updated simulator"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/synthetic",
        help="Output directory for CSV files (default: data/synthetic)",
    )
    parser.add_argument(
        "--patients",
        type=int,
        nargs="+",
        default=None,
        help="Specific patients to generate (default: all 2-11)",
    )

    args = parser.parse_args()

    return regenerate_all_patients(out_dir=Path(args.out_dir), patients=args.patients)


if __name__ == "__main__":
    sys.exit(main())
