#!/usr/bin/env python3
"""
Validation Script for Optimisation Results

Validates the best configurations found during hyperparameter optimisation:
- Phase 2: Run best configs on validation patients (Pat2-6)
- Phase 3: Test generalisation on held-out patients (Pat7-11)

Usage:
    # Validate best configs on validation set
    python scripts/validate_optimisation.py \
        --study-db results/optimisation/inverse_opt.db \
        --validation-patients 2 3 4 5 6 \
        --top-k 3

    # Test generalisation on held-out patients
    python scripts/validate_optimisation.py \
        --study-db results/optimisation/inverse_opt.db \
        --test-patients 7 8 9 10 11 \
        --top-k 2
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import optuna
import pandas as pd


def load_study(db_path: str, study_name: str = None) -> optuna.Study:
    """Load Optuna study from SQLite database."""
    storage = f"sqlite:///{db_path}"

    if study_name is None:
        study_summaries = optuna.study.get_all_study_summaries(storage)
        if not study_summaries:
            raise ValueError(f"No studies found in {db_path}")
        study_name = study_summaries[0].study_name

    study = optuna.load_study(study_name=study_name, storage=storage)
    return study


def get_top_configs(study: optuna.Study, top_k: int = 3) -> List[Dict]:
    """
    Get top K configurations per model type.

    Args:
        study: Optuna study
        top_k: Number of top configs per model

    Returns:
        List of config dictionaries
    """
    # Group trials by model type
    model_trials = {}
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        model_type = trial.params.get("model_type", "unknown")
        if model_type not in model_trials:
            model_trials[model_type] = []
        model_trials[model_type].append(trial)

    # Get top K per model
    top_configs = []
    for model_type, trials in model_trials.items():
        sorted_trials = sorted(trials, key=lambda t: t.value)[:top_k]
        for trial in sorted_trials:
            top_configs.append(
                {
                    "model_type": model_type,
                    "trial_number": trial.number,
                    "search_value": trial.value,
                    "params": trial.params.copy(),
                }
            )

    return top_configs


def run_validation(
    config: Dict,
    patient: int,
    mode: str,
    data_root: str = "data/synthetic",
    verbose: bool = True,
) -> Tuple[float, Dict]:
    """
    Run validation on a single patient.

    Args:
        config: Configuration dictionary from top configs
        patient: Patient number
        mode: 'forward' or 'inverse'
        data_root: Data directory
        verbose: Print progress

    Returns:
        (metric_value, results_dict)
    """
    model_type = config["model_type"]

    # Lazy imports
    from src.datasets.loader import load_synthetic_window
    from src.physics.magdelaine import make_params_from_preset
    from src.training.config import Config
    from src.training.search_space import params_to_config_dict

    try:
        # Convert to config
        params = config["params"].copy()
        params["model_type"] = model_type
        config_dict = params_to_config_dict(params, mode=mode)
        cfg = Config.from_dict(config_dict)

        if mode == "inverse":
            cfg.inverse_params = ["ksi"]

        # Load data
        data = load_synthetic_window(patient=patient, root=data_root)

        # Get true ksi
        true_ksi = None
        if mode == "inverse":
            true_params = make_params_from_preset(patient)
            true_ksi = getattr(true_params, "ksi")

        if verbose:
            print(f"    Building {model_type.upper()}...")

        # Build model
        if model_type == "birnn":
            from src.models.birnn import BIRNN

            model = BIRNN(cfg)
        elif model_type == "pinn":
            from src.models.pinn_feedforward import FeedforwardPINN

            model = FeedforwardPINN(cfg)
        elif model_type == "modified_mlp":
            from src.models.modified_mlp import ModifiedMLPPINN

            model = ModifiedMLPPINN(cfg)
        else:
            raise ValueError(f"Unknown model: {model_type}")

        model.build(data)
        model.compile()

        if verbose:
            print(f"    Training...")

        # Train
        if mode == "forward":
            model.train(display_every=5000)
            metrics = model.evaluate() if hasattr(model, "evaluate") else {}
            metric_value = metrics.get("rmse_total", 1.0)
            results = {"rmse_total": metric_value, "patient": patient}
        else:
            from src.training.inverse_trainer import InverseTrainer

            trainer = InverseTrainer(
                model=model,
                config=cfg,
                data=data,
                true_param_value=true_ksi,
            )
            history = trainer.train()

            metric_value = history["param_errors_percent"].get("ksi", 100.0)
            results = {
                "ksi_error_percent": metric_value,
                "ksi_estimated": history["final_params"].get("ksi"),
                "ksi_true": true_ksi,
                "patient": patient,
            }

        if verbose:
            metric_name = "RMSE" if mode == "forward" else "ksi error"
            print(f"    {metric_name}: {metric_value:.4f}")

        return metric_value, results

    except Exception as e:
        if verbose:
            print(f"    Error: {e}")
        return 100.0 if mode == "inverse" else 1.0, {"error": str(e)}


def validate_configs(
    configs: List[Dict],
    patients: List[int],
    mode: str,
    output_dir: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Validate configurations on multiple patients.

    Args:
        configs: List of config dictionaries
        patients: List of patient numbers
        mode: 'forward' or 'inverse'
        output_dir: Output directory
        verbose: Print progress

    Returns:
        DataFrame with validation results
    """
    results = []

    for i, config in enumerate(configs):
        model_type = config["model_type"]
        print(f"\n{'='*60}")
        print(
            f"Config {i+1}/{len(configs)}: {model_type.upper()} (trial {config['trial_number']})"
        )
        print(f"Search value: {config['search_value']:.4f}")
        print(f"{'='*60}")

        config_results = []

        for patient in patients:
            print(f"\n  Patient {patient}:")
            value, res = run_validation(
                config=config,
                patient=patient,
                mode=mode,
                verbose=verbose,
            )

            config_results.append(value)

            results.append(
                {
                    "model_type": model_type,
                    "trial_number": config["trial_number"],
                    "search_value": config["search_value"],
                    "patient": patient,
                    "validation_value": value,
                    **{k: v for k, v in res.items() if k != "patient"},
                }
            )

        # Summary for this config
        mean_val = np.mean(config_results)
        std_val = np.std(config_results)
        print(f"\n  Summary: mean={mean_val:.4f}, std={std_val:.4f}")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save results
    csv_path = output_dir / "validation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    return df


def generate_validation_report(df: pd.DataFrame, output_dir: Path, mode: str):
    """Generate validation summary report."""
    # Summary by model
    summary = (
        df.groupby("model_type")["validation_value"]
        .agg(["mean", "std", "min", "max", "count"])
        .round(4)
    )

    summary.columns = ["Mean", "Std", "Best", "Worst", "N"]

    # Save
    report_path = output_dir / "validation_summary.md"
    with open(report_path, "w") as f:
        metric_name = "RMSE" if mode == "forward" else "ksi Error (%)"
        f.write(f"# Validation Summary - {mode.capitalize()} Mode\n\n")
        f.write(f"## Results by Model ({metric_name})\n\n")
        f.write(summary.to_markdown())
        f.write("\n\n")

        # Best config per model
        f.write("## Best Configuration per Model\n\n")
        for model_type in df["model_type"].unique():
            model_df = df[df["model_type"] == model_type]
            best_idx = model_df["validation_value"].idxmin()
            best = model_df.loc[best_idx]
            f.write(f"### {model_type.upper()}\n")
            f.write(f"- Trial: {best['trial_number']}\n")
            f.write(f"- Validation {metric_name}: {best['validation_value']:.4f}\n")
            f.write(f"- Patient: {best['patient']}\n\n")

    print(f"Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate optimisation results on multiple patients"
    )
    parser.add_argument(
        "--study-db",
        type=str,
        required=True,
        help="Path to Optuna SQLite database",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Study name (default: first study in DB)",
    )
    parser.add_argument(
        "--validation-patients",
        type=int,
        nargs="+",
        default=None,
        help="Patient numbers for validation (e.g., 2 3 4 5 6)",
    )
    parser.add_argument(
        "--test-patients",
        type=int,
        nargs="+",
        default=None,
        help="Patient numbers for generalisation test (e.g., 7 8 9 10 11)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top configs per model to validate",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="inverse",
        choices=["forward", "inverse"],
        help="Optimisation mode",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory",
    )

    args = parser.parse_args()

    # Determine patients
    if args.validation_patients:
        patients = args.validation_patients
        phase = "validation"
    elif args.test_patients:
        patients = args.test_patients
        phase = "test"
    else:
        print("Error: Specify either --validation-patients or --test-patients")
        sys.exit(1)

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.study_db).parent / f"{phase}_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"VALIDATION - {phase.upper()} PHASE")
    print("=" * 70)
    print(f"Study: {args.study_db}")
    print(f"Patients: {patients}")
    print(f"Top K: {args.top_k}")
    print(f"Mode: {args.mode}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Load study
    print("\nLoading study...")
    study = load_study(args.study_db, args.study_name)
    print(f"  Loaded: {study.study_name}")

    # Get top configs
    print(f"\nGetting top {args.top_k} configs per model...")
    top_configs = get_top_configs(study, args.top_k)
    print(f"  Found {len(top_configs)} configurations")

    # Run validation
    print("\nRunning validation...")
    df = validate_configs(
        configs=top_configs,
        patients=patients,
        mode=args.mode,
        output_dir=output_dir,
    )

    # Generate report
    print("\nGenerating report...")
    generate_validation_report(df, output_dir, args.mode)

    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE")
    print(f"Results: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
