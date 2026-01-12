#!/usr/bin/env python3
"""
Unified Hyperparameter Optimisation Script

This script provides a single entry point for:
- Forward optimisation (minimise glucose RMSE)
- Inverse optimisation (minimise ksi estimation error)

Supports all three model architectures: BI-RNN, PINN, Modified-MLP

Usage:
    # Forward optimisation (all models)
    python scripts/optimise.py --mode forward --n-trials 300 --search-patient 5

    # Inverse optimisation (all models)
    python scripts/optimise.py --mode inverse --n-trials 300 --search-patient 5

    # Single model optimisation
    python scripts/optimise.py --mode inverse --model birnn --n-trials 100 --search-patient 5

    # Resume interrupted study
    python scripts/optimise.py --mode inverse --n-trials 300 --study-name my_study --resume

    # Quick test (3 trials)
    python scripts/optimise.py --mode inverse --n-trials 3 --search-patient 5 --no-upload
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Environment setup
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import optuna
import yaml

# Import after path setup
from src.training.search_space import SearchSpace, params_to_config_dict


def setup_tensorflow_mode(model_type: str):
    """
    Configure TensorFlow execution mode based on model type.

    DeepXDE models (PINN, Modified-MLP) require TF 1.x graph mode.
    BI-RNN uses TF 2.x eager execution.

    This must be called BEFORE importing the model classes.
    """
    import tensorflow as tf

    if model_type in ["pinn", "modified_mlp"]:
        # DeepXDE requires TF 1.x compatibility
        tf.compat.v1.disable_v2_behavior()
        tf.compat.v1.disable_eager_execution()
        print(f"  TensorFlow: Graph mode (TF 1.x compat) for {model_type}")
    else:
        # BI-RNN uses eager execution
        if not tf.executing_eagerly():
            print("  Warning: Cannot switch to eager mode after disabling it")
        else:
            print(f"  TensorFlow: Eager mode for {model_type}")


def run_trial(
    params: dict,
    mode: str,
    patient: int,
    data_root: str = "data/synthetic",
    verbose: bool = False,
) -> tuple:
    """
    Run a single optimisation trial.

    Args:
        params: Hyperparameters from Optuna
        mode: 'forward' or 'inverse'
        patient: Patient number
        data_root: Data directory
        verbose: Print progress

    Returns:
        (objective_value, results_dict)
    """
    model_type = params["model_type"]

    # Lazy imports to avoid TF mode issues
    from src.datasets.loader import load_synthetic_window
    from src.physics.magdelaine import make_params_from_preset
    from src.training.config import Config

    try:
        # Convert params to config
        config_dict = params_to_config_dict(params, mode=mode)
        config = Config.from_dict(config_dict)

        if mode == "inverse":
            config.inverse_params = ["ksi"]

        # Load data
        data = load_synthetic_window(patient=patient, root=data_root)

        # Get true ksi for inverse mode
        true_ksi = None
        if mode == "inverse":
            true_params = make_params_from_preset(patient)
            true_ksi = getattr(true_params, "ksi")

        if verbose:
            print(f"  Building {model_type.upper()} model...")

        # Build model (import inside function for TF mode control)
        if model_type == "birnn":
            from src.models.birnn import BIRNN

            model = BIRNN(config)
        elif model_type == "pinn":
            from src.models.pinn_feedforward import FeedforwardPINN

            model = FeedforwardPINN(config)
        elif model_type == "modified_mlp":
            from src.models.modified_mlp import ModifiedMLPPINN

            model = ModifiedMLPPINN(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.build(data)
        model.compile()

        if verbose:
            print(f"  Training {model_type.upper()}...")

        # Train based on mode
        if mode == "forward":
            model.train(display_every=max(500, params.get("epochs", 10000) // 20))
            metrics = model.evaluate() if hasattr(model, "evaluate") else {}
            objective_value = metrics.get("rmse_total", metrics.get("rmse", 1.0))

            results = {
                "rmse_total": objective_value,
                "model_type": model_type,
                "patient": patient,
            }

        else:  # inverse
            from src.training.inverse_trainer import InverseTrainer

            trainer = InverseTrainer(
                model=model,
                config=config,
                data=data,
                true_param_value=true_ksi,
            )
            history = trainer.train()

            objective_value = history["param_errors_percent"].get("ksi", 100.0)

            results = {
                "ksi_error_percent": objective_value,
                "ksi_estimated": history["final_params"].get("ksi"),
                "ksi_true": true_ksi,
                "model_type": model_type,
                "patient": patient,
            }

        if verbose:
            metric_name = "RMSE" if mode == "forward" else "ksi error"
            print(f"  {metric_name}: {objective_value:.4f}")

        return objective_value, results

    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
            import traceback

            traceback.print_exc()
        return 100.0 if mode == "inverse" else 1.0, {"error": str(e)}


class UnifiedObjective:
    """
    Unified objective function for both forward and inverse optimisation.
    """

    def __init__(
        self,
        mode: str,
        patient: int,
        model_type: str = None,
        data_root: str = "data/synthetic",
        verbose: bool = True,
        quick_test: bool = False,
    ):
        self.mode = mode
        self.patient = patient
        self.fixed_model_type = model_type
        self.data_root = data_root
        self.verbose = verbose
        self.quick_test = quick_test
        self.best_value = float("inf")
        self.best_params = None
        self.trial_count = 0

    def __call__(self, trial: optuna.Trial) -> float:
        self.trial_count += 1

        # Sample model type
        if self.fixed_model_type:
            model_type = self.fixed_model_type
            # Store as user attribute so it's available in trial.params-like lookups
            trial.set_user_attr("model_type_fixed", model_type)
        else:
            model_type = trial.suggest_categorical(
                "model_type", ["birnn", "pinn", "modified_mlp"]
            )

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Trial {trial.number} ({self.trial_count}): {model_type.upper()}")
            print(f"{'='*70}")

        # Sample hyperparameters
        search_space = SearchSpace(
            model_type=model_type, mode=self.mode, quick_test=self.quick_test
        )
        params = search_space.sample(trial)

        # Show key params
        if self.verbose:
            if self.mode == "inverse":
                print(
                    f"  stage1: {params.get('stage1_epochs')} epochs, lr={params.get('stage1_lr'):.2e}"
                )
                print(
                    f"  stage2: {params.get('stage2_epochs')} epochs, lr={params.get('stage2_lr'):.2e}"
                )
                print(
                    f"  stage3: {params.get('stage3_epochs')} epochs, lr={params.get('stage3_lr'):.2e}"
                )
            else:
                print(
                    f"  epochs: {params.get('epochs')}, lr={params.get('learning_rate'):.2e}"
                )

        # Run trial
        value, results = run_trial(
            params=params,
            mode=self.mode,
            patient=self.patient,
            data_root=self.data_root,
            verbose=self.verbose,
        )

        # Store results (convert numpy types to Python types for JSON serialization)
        for key, val in results.items():
            if val is not None and not isinstance(val, dict):
                # Convert numpy types to Python native types
                if hasattr(val, "item"):
                    val = val.item()
                elif hasattr(val, "__float__"):
                    val = float(val)
                trial.set_user_attr(key, val)

        # Track best
        if value < self.best_value:
            self.best_value = value
            self.best_params = params.copy()
            if self.verbose:
                print(f"  NEW BEST: {value:.4f}")

        return value


def save_best_config(study: optuna.Study, mode: str, output_dir: Path):
    """Save the best configuration as YAML."""
    best_trial = study.best_trial

    # Get model type from best trial
    model_type = best_trial.params.get("model_type", "birnn")

    # Reconstruct full params
    params = {"model_type": model_type}
    params.update(best_trial.params)

    # Convert to config dict
    config_dict = params_to_config_dict(params, mode=mode)

    # Save YAML
    config_path = output_dir / f"best_{model_type}_{mode}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    print(f"  Saved: {config_path}")


def export_results(study: optuna.Study, output_dir: Path):
    """Export study results to CSV and JSON."""
    # CSV export
    df = study.trials_dataframe()
    csv_path = output_dir / "all_trials.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path}")

    # JSON export (best trials per model)
    best_by_model = {}
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        # Check both params and user_attrs for model_type
        model_type = trial.params.get(
            "model_type", trial.user_attrs.get("model_type_fixed", "unknown")
        )
        if (
            model_type not in best_by_model
            or trial.value < best_by_model[model_type]["value"]
        ):
            best_by_model[model_type] = {
                "trial": trial.number,
                "value": trial.value,
                "params": trial.params,
            }

    json_path = output_dir / "best_by_model.json"
    with open(json_path, "w") as f:
        json.dump(best_by_model, f, indent=2)
    print(f"  JSON: {json_path}")


def upload_to_s3(
    local_dir: str, bucket: str, s3_prefix: str, region: str = "eu-west-2"
) -> bool:
    """Upload results to S3."""
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError

        s3_client = boto3.client("s3", region_name=region)
        local_path = Path(local_dir)

        uploaded = 0
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                relative = file_path.relative_to(local_path)
                s3_key = f"{s3_prefix}/{relative}"
                s3_client.upload_file(str(file_path), bucket, s3_key)
                uploaded += 1

        print(f"  Uploaded {uploaded} files to s3://{bucket}/{s3_prefix}/")
        return True

    except (NoCredentialsError, ClientError) as e:
        print(f"  S3 upload failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Unified hyperparameter optimisation for T1D-PINN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Forward optimisation (all models, 300 trials)
  python scripts/optimise.py --mode forward --n-trials 300 --search-patient 5

  # Inverse optimisation (single model)
  python scripts/optimise.py --mode inverse --model birnn --n-trials 100 --search-patient 5

  # Resume interrupted study
  python scripts/optimise.py --mode inverse --study-name my_study --resume

  # Quick test
  python scripts/optimise.py --mode inverse --n-trials 3 --search-patient 5 --no-upload
        """,
    )

    # Required arguments
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["forward", "inverse"],
        help="Optimisation mode: forward (RMSE) or inverse (ksi error)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        required=True,
        help="Number of optimisation trials",
    )
    parser.add_argument(
        "--search-patient",
        type=int,
        required=True,
        help="Patient number for hyperparameter search (2-11 for synthetic)",
    )

    # Optional arguments
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["birnn", "pinn", "modified_mlp"],
        help="Optimise single model type (default: all models)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Study name (default: auto-generated)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/optimisation/)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing study",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Disable S3 upload",
    )
    parser.add_argument(
        "--upload-s3",
        action="store_true",
        help="Upload results to S3 after completion",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=os.getenv("T1D_S3_BUCKET", "t1d-pinn-results-900630261719"),
        help="S3 bucket name",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Use minimal epochs for quick local testing (100 epochs per stage)",
    )

    args = parser.parse_args()

    # Validate patient number
    if not 2 <= args.search_patient <= 11:
        print("Error: search-patient must be between 2 and 11 (synthetic data)")
        sys.exit(1)

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"results/optimisation/{args.mode}_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup study name
    study_name = args.study_name or f"t1d_{args.mode}_{timestamp}"

    # Storage path
    storage_path = output_dir / f"{study_name}.db"
    storage = f"sqlite:///{storage_path}"

    verbose = not args.quiet

    print("=" * 70)
    print("T1D-PINN HYPERPARAMETER OPTIMISATION")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Patient: Pat{args.search_patient}")
    print(f"Trials: {args.n_trials}")
    print(f"Model: {args.model or 'all'}")
    print(f"Study: {study_name}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Create study with TPE sampler
    sampler = optuna.samplers.TPESampler(n_startup_trials=10, seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2000)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=args.resume,
    )

    # Create objective
    objective = UnifiedObjective(
        mode=args.mode,
        patient=args.search_patient,
        model_type=args.model,
        verbose=verbose,
        quick_test=args.quick_test,
    )

    # Run optimisation
    print(f"\nStarting optimisation ({args.n_trials} trials)...")
    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nOptimisation interrupted. Saving results...")

    # Results summary
    print("\n" + "=" * 70)
    print("OPTIMISATION COMPLETE")
    print("=" * 70)

    if len(study.trials) > 0:
        best_model = study.best_trial.params.get(
            "model_type", study.best_trial.user_attrs.get("model_type_fixed", "unknown")
        )
        print(f"\nBest trial:")
        print(f"  Trial: {study.best_trial.number}")
        print(f"  Value: {study.best_trial.value:.4f}")
        print(f"  Model: {best_model}")

        # Save results
        print(f"\nSaving results...")
        save_best_config(study, args.mode, output_dir)
        export_results(study, output_dir)

        # S3 upload
        if args.upload_s3 and not args.no_upload:
            print(f"\nUploading to S3...")
            s3_prefix = f"optimisation/{args.mode}/{timestamp}"
            upload_to_s3(str(output_dir), args.s3_bucket, s3_prefix)

    print(f"\nResults saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
