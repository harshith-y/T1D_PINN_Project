#!/usr/bin/env python3
"""
Enhanced Forward Training Script v2

Based on train_with_visual.py with added:
- CheckpointManager for enhanced checkpointing
- PredictionManager for fast prediction save/load
- Resume capability
- Cleaner code using src/ utilities

Usage:
    # Basic training
    python scripts/train_forward_v2.py --model birnn --patient 3 --epochs 2000

    # Resume from checkpoint
    python scripts/train_forward_v2.py --model birnn --patient 3 --resume results/birnn_forward/Pat3_*/checkpoints/best

    # Real patient data
    python scripts/train_forward_v2.py --model pinn --patient 5 --data-type real --epochs 1000
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

from src.datasets.loader import load_real_patient_csv, load_synthetic_window
from src.training.checkpoint import CheckpointManager
from src.training.config import load_config
from src.training.prediction_extractor import extract_predictions
from src.training.predictions import PredictionManager
from src.visualisation.plotter import ExperimentPlotter


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
    """Main training function."""
    parser = argparse.ArgumentParser(description="Enhanced forward training v2")
    parser.add_argument(
        "--model", type=str, required=True, choices=["birnn", "pinn", "modified_mlp"]
    )
    parser.add_argument("--patient", type=int, required=True)
    parser.add_argument(
        "--data-type", type=str, default="synthetic", choices=["synthetic", "real"]
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint directory"
    )

    args = parser.parse_args()

    model_display = {"birnn": "BI-RNN", "pinn": "PINN", "modified_mlp": "Modified-MLP"}[
        args.model
    ]

    print("=" * 80)
    print(f"{model_display} FORWARD TRAINING v2 (Enhanced)")
    print("=" * 80)
    print(f"Model: {model_display}")
    print(f"Patient: {args.patient}")
    print(f"Data type: {args.data_type}")
    print(f"Epochs: {args.epochs}")
    if args.resume:
        print(f"Resume from: {args.resume}")
    print("=" * 80)

    # ========================================================================
    # STEP 1: Load Configuration
    # ========================================================================
    print("\n[1/8] Loading configuration...")
    config = load_config(model_name=args.model, mode="forward")
    config.training.epochs = args.epochs

    # Set output directory
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        patient_label = (
            f"Pat{args.patient}"
            if args.data_type == "synthetic"
            else f"RealPat{args.patient}"
        )
        args.save_dir = f"results/{args.model}_forward/{patient_label}_{timestamp}"

    config.output.save_dir = args.save_dir
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print(f"✅ Config loaded")
    print(f"   Output: {args.save_dir}")

    # ========================================================================
    # STEP 2: Load Data
    # ========================================================================
    print("\n[2/8] Loading data...")

    if args.data_type == "synthetic":
        data = load_synthetic_window(patient=args.patient, root="data/synthetic")
    else:
        data = load_real_patient_csv(
            patient=args.patient,
            root="data/processed",
            t_start=0,
            t_end=2880,  # 48 hours
        )

    print(f"✅ Data loaded: {data}")

    # ========================================================================
    # STEP 3: Build Model
    # ========================================================================
    print(f"\n[3/8] Building {model_display} model...")
    model = load_model(args.model, config)
    model.build(data)
    print("✅ Model built")

    # ========================================================================
    # STEP 4: Initialize Managers (NEW!)
    # ========================================================================
    print("\n[4/8] Initializing checkpoint and prediction managers...")
    checkpoint_mgr = CheckpointManager(args.save_dir, config)
    pred_mgr = PredictionManager(args.save_dir)
    print("✅ Managers initialized")

    # ========================================================================
    # STEP 5: Resume if Requested (NEW!)
    # ========================================================================
    start_epoch = 0
    if args.resume:
        print(f"\n[5/8] Resuming from checkpoint...")
        state = checkpoint_mgr.load(
            model,
            optimizer=getattr(model, "optimizer", None),
            checkpoint_path=Path(args.resume),
        )
        start_epoch = state["epoch"]
        print(f"✅ Resumed from epoch {start_epoch}")
    else:
        print(f"\n[5/8] Starting fresh training")

    # ========================================================================
    # STEP 6: Compile and Train
    # ========================================================================
    print("\n[6/8] Compiling model...")
    model.compile()
    print("✅ Model compiled")

    print(f"\n[7/8] Training for {args.epochs} epochs...")
    print("=" * 80)
    model.train(display_every=max(100, args.epochs // 20))
    print("=" * 80)
    print("✅ Training completed")

    # ========================================================================
    # STEP 8: Evaluate, Extract Predictions, Save Everything
    # ========================================================================
    print("\n[8/8] Evaluating and saving...")

    # Evaluate
    metrics = model.evaluate() if hasattr(model, "evaluate") else {}

    # Extract predictions (uses utility from src/)
    predictions = extract_predictions(model, data, args.model)

    # Add metadata
    predictions["metadata"] = {
        "model_name": args.model,
        "patient": args.patient,
        "data_type": args.data_type,
        "epochs": args.epochs,
        **metrics,
    }

    # Save predictions (NEW!)
    pred_mgr.save(**predictions)
    print("✅ Predictions saved")

    # Save checkpoint (NEW!)
    checkpoint_mgr.save(
        model=model,
        optimizer=getattr(model, "optimizer", None),
        epoch=args.epochs,
        metrics=metrics,
        is_final=True,
        is_best=True,
    )
    print("✅ Checkpoint saved")

    # Generate plots (using your existing plotter)
    plotter = ExperimentPlotter(
        save_dir=Path(args.save_dir) / "plots", model_name=f"{model_display} Forward"
    )

    plot_paths = plotter.plot_all(
        predictions={
            "time": predictions["time"],
            "glucose": predictions["glucose_pred"],
            "insulin": predictions.get("insulin_pred"),
            "digestion": predictions.get("digestion_pred"),
        },
        ground_truth={
            "glucose": predictions["glucose_true"],
            "insulin": predictions.get("insulin_true"),
            "digestion": predictions.get("digestion_true"),
        },
        metrics=metrics,
        split_idx=predictions.get("split_idx"),
    )

    print("✅ Plots generated")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Results:")
    print(f"   Model: {model_display}")
    print(f"   Patient: {args.patient}")
    print(f"   Epochs: {args.epochs}")
    if "test_loss" in metrics:
        print(f"   Test loss: {metrics['test_loss']:.4f}")

    print(f"\n📁 Output:")
    print(f"   {args.save_dir}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
