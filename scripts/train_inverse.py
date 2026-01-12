#!/usr/bin/env python3
"""
Enhanced Inverse Training Script v2

Uses InverseTrainer for flexible multi-stage parameter estimation.

Usage:
    # Basic inverse training (requires config with stages)
    python scripts/train_inverse_v2.py --config configs/birnn_inverse.yaml --patient 3

    # Custom patient and parameter
    python scripts/train_inverse_v2.py --config configs/pinn_inverse.yaml --patient 5 --param ksi
"""

import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# CRITICAL: Disable eager for DeepXDE models
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

import argparse
from datetime import datetime

import boto3
from botocore.exceptions import NoCredentialsError, ClientError

from src.datasets.loader import load_real_patient_csv, load_synthetic_window
from src.physics.magdelaine import make_params_from_preset
from src.training.checkpoint import CheckpointManager
from src.training.config import Config
from src.training.inverse_trainer import InverseTrainer
from src.training.predictions import PredictionManager


def upload_to_s3(local_dir: str, bucket: str, s3_prefix: str, region: str = "eu-west-2") -> bool:
    """
    Upload results directory to S3.

    Args:
        local_dir: Local directory to upload
        bucket: S3 bucket name
        s3_prefix: Prefix (folder path) in S3
        region: AWS region

    Returns:
        True if successful, False otherwise
    """
    try:
        s3_client = boto3.client("s3", region_name=region)
        local_path = Path(local_dir)

        uploaded_count = 0
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path)
                s3_key = f"{s3_prefix}/{relative_path}"
                s3_client.upload_file(str(file_path), bucket, s3_key)
                uploaded_count += 1

        print(f"✅ Uploaded {uploaded_count} files to s3://{bucket}/{s3_prefix}/")
        return True

    except NoCredentialsError:
        print("❌ S3 upload failed: AWS credentials not found")
        return False
    except ClientError as e:
        print(f"❌ S3 upload failed: {e}")
        return False


def load_model(model_name: str, config):
    """Load the appropriate model class."""
    if model_name == "birnn":
        from src.models.birnn import BIRNN

        return BIRNN(config)
    elif model_name == "pinn":
        from src.models.pinn_feedforward import FeedforwardPINN

        return FeedforwardPINN(config)
    elif model_name == "modified_mlp":
        from src.models.modified_mlp import ModifiedMLPPINN

        return ModifiedMLPPINN(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    """Main inverse training function."""
    parser = argparse.ArgumentParser(description="Enhanced inverse training v2")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to inverse config YAML with stages defined",
    )
    parser.add_argument(
        "--patient",
        type=int,
        required=True,
        help="Patient number (2-11 for synthetic, 1-15 for real)",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="synthetic",
        choices=["synthetic", "real"],
        help="Data type: synthetic (with ground truth) or real (estimates only)",
    )
    parser.add_argument(
        "--inverse-params",
        type=str,
        nargs="+",
        default=["ksi"],
        help="Parameters to estimate (default: ksi). "
        "Options: ksi kl ku_Vi kb Tu Tr kr_Vb M. "
        "Examples: --inverse-params ksi  OR  --inverse-params ksi kl ku_Vi",
    )
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument(
        "--upload-s3",
        action="store_true",
        help="Upload results to S3 after training",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=os.getenv("T1D_S3_BUCKET", "t1d-pinn-results-900630261719"),
        help="S3 bucket for results (default: from T1D_S3_BUCKET env or t1d-pinn-results-900630261719)",
    )
    parser.add_argument(
        "--s3-region",
        type=str,
        default=os.getenv("AWS_REGION", "eu-west-2"),
        help="AWS region (default: eu-west-2)",
    )

    args = parser.parse_args()

    # Validate patient number based on data type
    if args.data_type == "synthetic" and not (2 <= args.patient <= 11):
        print("❌ Error: Synthetic patients must be in range 2-11")
        sys.exit(1)
    elif args.data_type == "real" and not (1 <= args.patient <= 15):
        print("❌ Error: Real patients must be in range 1-15")
        sys.exit(1)

    print("=" * 80)
    print("INVERSE TRAINING v2 (Enhanced)")
    print("=" * 80)

    # ========================================================================
    # STEP 1: Load Configuration
    # ========================================================================
    print("\n[1/7] Loading configuration...")
    config = Config.from_yaml(args.config)

    # Validate config
    if config.mode != "inverse":
        print("⚠️  Warning: Config mode is not 'inverse', setting to 'inverse'")
        config.mode = "inverse"

    if not config.training.stages:
        print("❌ Error: Config must have training.stages defined for inverse training")
        print("   See configs/birnn_inverse.yaml for example")
        sys.exit(1)

    # Set inverse parameters from command line (overrides config)
    # Default is ['ksi'] if not specified
    config.inverse_params = args.inverse_params

    model_display = {"birnn": "BI-RNN", "pinn": "PINN", "modified_mlp": "Modified-MLP"}[
        config.model_name
    ]

    print(f"✅ Config loaded")
    print(f"   Model: {model_display}")
    print(f"   Parameters: {config.inverse_params}")
    print(f"   Stages: {len(config.training.stages)}")

    # Set output directory
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        patient_label = (
            f"Pat{args.patient}"
            if args.data_type == "synthetic"
            else f"RealPat{args.patient}"
        )
        args.save_dir = f"results/{config.model_name}_inverse/{patient_label}_{config.inverse_param}_{timestamp}"

    config.output.save_dir = args.save_dir
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print(f"   Output: {args.save_dir}")

    # ========================================================================
    # STEP 2: Load Data
    # ========================================================================
    print(f"\n[2/7] Loading {args.data_type} data...")

    if args.data_type == "synthetic":
        data = load_synthetic_window(patient=args.patient, root="data/synthetic")

        if not data.has_latent_states:
            print(
                "❌ Error: Inverse training requires synthetic data with ground truth"
            )
            sys.exit(1)

        # Get true parameter value (for error calculation)
        true_params = make_params_from_preset(args.patient)
        true_param_value = getattr(true_params, config.inverse_param)
        print(f"   True {config.inverse_param}: {true_param_value:.2f}")

    else:  # real patient
        data = load_real_patient_csv(
            patient=args.patient,
            root="data/processed",
            t_start=0,
            t_end=2880,  # 48 hours
        )

        # No ground truth for real patients
        true_param_value = None
        print(f"   ⚠️  Real patient: No ground truth available")
        print(f"   Will estimate {config.inverse_param} but cannot compute error")

    print(f"✅ Data loaded: {data}")

    # ========================================================================
    # STEP 3: Build Model
    # ========================================================================
    print(f"\n[3/7] Building {model_display} model...")
    model = load_model(config.model_name, config)
    model.build(data)
    print("✅ Model built")

    # ========================================================================
    # STEP 4: Initialize Managers
    # ========================================================================
    print("\n[4/7] Initializing managers...")
    checkpoint_mgr = CheckpointManager(args.save_dir, config)
    pred_mgr = PredictionManager(args.save_dir)
    print("✅ Managers initialized")

    # ========================================================================
    # STEP 5: Compile Model
    # ========================================================================
    print("\n[5/7] Compiling model...")
    model.compile()
    print("✅ Model compiled")

    # ========================================================================
    # STEP 6: Train with InverseTrainer (NEW!)
    # ========================================================================
    print(f"\n[6/7] Starting inverse training...")
    print("=" * 80)

    trainer = InverseTrainer(
        model=model, config=config, data=data, true_param_value=true_param_value
    )

    history = trainer.train()  # Runs all stages!

    print("=" * 80)
    print("✅ Inverse training completed")

    # ========================================================================
    # STEP 7: Save Everything
    # ========================================================================
    print("\n[7/7] Saving results...")

    # Evaluate
    metrics = model.evaluate() if hasattr(model, "evaluate") else {}

    # Add inverse parameter results (handle both single and multiple params)
    inverse_params_list = (
        config.inverse_params
        if isinstance(config.inverse_params, list)
        else [config.inverse_params]
    )

    for param in inverse_params_list:
        if param in history["final_params"]:
            metrics[f"{param}_estimated"] = history["final_params"][param]
            metrics[f"{param}_true"] = (
                true_param_value if param == inverse_params_list[0] else None
            )
            if param in history["param_errors_percent"]:
                metrics[f"{param}_error_percent"] = history["param_errors_percent"][
                    param
                ]

    # Save parameter evolution for each estimated parameter
    if history["param_history"]["epochs"]:
        import numpy as np

        for param in inverse_params_list:
            param_values_key = f"{param}_values"
            param_losses_key = f"{param}_losses"

            if param_values_key in history["param_history"]:
                pred_mgr.save_parameter_evolution(
                    epochs=np.array(history["param_history"]["epochs"]),
                    param_values=np.array(history["param_history"][param_values_key]),
                    param_name=param,
                    true_value=(
                        true_param_value if param == inverse_params_list[0] else None
                    ),
                    losses=np.array(history["param_history"][param_losses_key]),
                    stages=np.array(history["param_history"]["stages"]),
                )
        print("✅ Parameter evolution saved")

    # ========================================================================
    # Generate and Save Predictions
    # ========================================================================
    print("\n📊 Generating predictions...")

    # Use the prediction extractor utility (handles all model types)
    from src.training.prediction_extractor import extract_predictions

    predictions = extract_predictions(
        model=model, data=data, model_type=config.model_name
    )

    # Extract arrays
    time = predictions["time"]
    glucose_pred = predictions["glucose_pred"]
    glucose_true = predictions["glucose_true"]
    insulin_pred = predictions.get("insulin_pred")
    insulin_true = predictions.get("insulin_true")
    digestion_pred = predictions.get("digestion_pred")
    digestion_true = predictions.get("digestion_true")
    split_idx = predictions.get("split_idx", int(0.8 * len(time)))

    # Prepare metadata with parameter estimates
    metadata = {
        "model_name": config.model_name,
        "patient": (
            f"Pat{config.data.patient}"
            if hasattr(config.data, "patient")
            else "unknown"
        ),
        "mode": "inverse",
        "inverse_params": inverse_params_list,
        **metrics,
    }

    # Build predictions dict
    predictions_dict = {
        "time": time,
        "glucose_pred": glucose_pred,
        "glucose_true": glucose_true,
        "split_idx": split_idx,
        "metadata": metadata,
    }

    # Add latent states if available
    if insulin_pred is not None:
        predictions_dict.update(
            {
                "insulin_pred": insulin_pred,
                "insulin_true": insulin_true,
                "digestion_pred": digestion_pred,
                "digestion_true": digestion_true,
            }
        )

    # Save predictions
    pred_mgr.save(**predictions_dict)
    print("✅ Predictions saved")

    # Print evaluation metrics
    print(f"\n{'='*80}")
    print("EVALUATION METRICS")
    print(f"{'='*80}")
    if "rmse_interpolation" in metrics:
        print(f"rmse_interpolation: {metrics['rmse_interpolation']:.4f}")
    if "rmse_forecast" in metrics:
        print(f"rmse_forecast: {metrics['rmse_forecast']:.4f}")
    if "rmse_total" in metrics:
        print(f"rmse_total: {metrics['rmse_total']:.4f}")

    # Save final checkpoint
    # Get error for first parameter (for backward compatibility with is_best check)
    first_param = inverse_params_list[0]
    param_error = history["param_errors_percent"].get(first_param, 100)

    checkpoint_mgr.save(
        model=model,
        optimizer=getattr(model, "optimizer", None),
        epoch=sum(s["epochs"] for s in config.training.stages),
        metrics=metrics,
        is_final=True,
        is_best=(param_error < 10),
    )
    print("✅ Checkpoint saved")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ INVERSE TRAINING COMPLETE!")
    print("=" * 80)

    # Display results for each estimated parameter
    print(f"\n📊 Parameter Estimation:")
    for param in inverse_params_list:
        if param in history["final_params"]:
            print(f"   Parameter: {param}")
            print(f"   Estimated: {history['final_params'][param]:.2f}")
            if param == inverse_params_list[0] and true_param_value is not None:
                print(f"   True value: {true_param_value:.2f}")
                if param in history["param_errors_percent"]:
                    error = history["param_errors_percent"][param]
                    print(f"   Error: {error:.2f}%")

                    if error < 5:
                        print(f"   🎉 Excellent! Error < 5%")
                    elif error < 10:
                        print(f"   ✅ Good! Error < 10%")
                    elif error < 20:
                        print(f"   ⚠️  Acceptable. Error < 20%")
                    else:
                        print(f"   ❌ Poor. Error > 20%")

    print(f"\n📁 Output:")
    print(f"   {args.save_dir}")

    # ========================================================================
    # S3 Upload (if enabled)
    # ========================================================================
    if args.upload_s3:
        print("\n[S3] Uploading results to S3...")
        # Create S3 prefix from save_dir (e.g., results/pinn_inverse/Pat5_ksi_... -> pinn_inverse/Pat5_ksi_...)
        s3_prefix = str(Path(args.save_dir)).replace("results/", "")
        upload_to_s3(
            local_dir=args.save_dir,
            bucket=args.s3_bucket,
            s3_prefix=s3_prefix,
            region=args.s3_region,
        )
        print(f"   S3 URL: s3://{args.s3_bucket}/{s3_prefix}/")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Inverse training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
