#!/usr/bin/env python3
"""
Rigorous Training Study: Compare 8 approaches for PINN inverse parameter estimation.

This script runs a comprehensive comparison of training approaches for the PINN
model (using DeepXDE) on the inverse problem (ksi estimation + glucose forecasting).

Usage:
    python scripts/run_training_study_pinn.py [--pilot] [--approach APPROACH] [--patient PAT]

    --pilot: Run pilot test (2 approaches x 2 patients x 1 seed)
    --approach: Run only specific approach (e.g., "fixed_curriculum")
    --patient: Run only specific patient (e.g., 2)
    --auto-shutdown: Shutdown EC2 instance after completion
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# CRITICAL: Disable eager execution BEFORE importing TensorFlow
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.datasets.loader import load_synthetic_window
from src.models.pinn_feedforward import FeedforwardPINN
from src.physics.magdelaine import make_params_from_preset
from src.training.config import Config
from src.training.inverse_trainer import InverseTrainer

# =============================================================================
# APPROACH CONFIGURATIONS FOR PINN
# =============================================================================

# Base architecture for PINN (matching pinn_inverse.yaml)
BASE_ARCHITECTURE = {
    "n_layers": 3,
    "n_neurons": 30,
    "activation": "tanh",
    "use_fourier": True,
    "fourier_features": [0, 1, 2, 3, 4, 5],
}

# Loss weights for PINN: [glucose_obs, ode_glucose, ode_insulin, ode_digestion]
# Standard values from pinn_inverse.yaml
PINN_LOSS_WEIGHTS_STANDARD = [5.97, 4.70, 2.71, 3.32]
PINN_LOSS_WEIGHTS_LOW_PHYSICS = [5.97, 0.5, 0.5, 0.5]
PINN_LOSS_WEIGHTS_MEDIUM_PHYSICS = [5.97, 2.0, 1.5, 1.5]
PINN_LOSS_WEIGHTS_HIGH_PHYSICS = [5.97, 8.0, 5.0, 5.0]
PINN_LOSS_WEIGHTS_GLUCOSE_ONLY = [5.97, 0.0, 0.0, 0.0]

# Learning rates
LR_HIGH = 1.0e-3
LR_MED = 5.0e-4
LR_LOW = 2.0e-4


def get_approach_configs() -> Dict[str, List[Dict[str, Any]]]:
    """Return stage configurations for all 8 approaches adapted for PINN."""

    approaches = {}

    # 1. BROKEN_BASELINE (Current implementation - for reference)
    # Stage 1 trains params with NO physics - ksi gets no signal
    approaches["broken_baseline"] = [
        {
            "name": "stage1_ksi_only_no_physics",
            "epochs": 5000,
            "learning_rate": LR_HIGH,
            "loss_weights": PINN_LOSS_WEIGHTS_GLUCOSE_ONLY,  # BUG: No physics!
            "train_inverse_params": True,
            "train_nn_weights": False,
            "optimizer": "adam",
            "display_every": 1000,
        },
        {
            "name": "stage2_nn_only",
            "epochs": 1000,
            "learning_rate": LR_MED,
            "loss_weights": PINN_LOSS_WEIGHTS_STANDARD,
            "train_inverse_params": False,
            "train_nn_weights": True,
            "optimizer": "adam",
            "display_every": 500,
        },
        {
            "name": "stage3_joint",
            "epochs": 1000,
            "learning_rate": LR_LOW,
            "loss_weights": PINN_LOSS_WEIGHTS_STANDARD,
            "train_inverse_params": True,
            "train_nn_weights": True,
            "optimizer": "adam",
            "display_every": 500,
        },
    ]

    # 2. FIXED_CURRICULUM (NN first, then ksi with physics)
    approaches["fixed_curriculum"] = [
        {
            "name": "stage1_nn_only_glucose",
            "epochs": 3000,
            "learning_rate": LR_HIGH,
            "loss_weights": PINN_LOSS_WEIGHTS_GLUCOSE_ONLY,  # Glucose only
            "train_inverse_params": False,
            "train_nn_weights": True,
            "optimizer": "adam",
            "display_every": 500,
        },
        {
            "name": "stage2_ksi_only_with_physics",
            "epochs": 3000,
            "learning_rate": LR_HIGH,
            "loss_weights": PINN_LOSS_WEIGHTS_STANDARD,  # Physics enabled
            "train_inverse_params": True,
            "train_nn_weights": False,
            "optimizer": "adam",
            "display_every": 500,
        },
        {
            "name": "stage3_joint",
            "epochs": 1000,
            "learning_rate": LR_LOW,
            "loss_weights": PINN_LOSS_WEIGHTS_STANDARD,
            "train_inverse_params": True,
            "train_nn_weights": True,
            "optimizer": "adam",
            "display_every": 500,
        },
    ]

    # 3. JOINT_FROM_START (Simple baseline - train everything together)
    approaches["joint_from_start"] = [
        {
            "name": "stage1_joint_all",
            "epochs": 7000,
            "learning_rate": LR_MED,
            "loss_weights": PINN_LOSS_WEIGHTS_STANDARD,
            "train_inverse_params": True,
            "train_nn_weights": True,
            "optimizer": "adam",
            "display_every": 500,
        },
    ]

    # 4. TWO_STAGE_NN_FIRST (Simplified curriculum)
    approaches["two_stage_nn_first"] = [
        {
            "name": "stage1_nn_pretrain",
            "epochs": 4000,
            "learning_rate": LR_HIGH,
            "loss_weights": PINN_LOSS_WEIGHTS_GLUCOSE_ONLY,  # Glucose only
            "train_inverse_params": False,
            "train_nn_weights": True,
            "optimizer": "adam",
            "display_every": 500,
        },
        {
            "name": "stage2_joint",
            "epochs": 3000,
            "learning_rate": LR_LOW,
            "loss_weights": PINN_LOSS_WEIGHTS_STANDARD,
            "train_inverse_params": True,
            "train_nn_weights": True,
            "optimizer": "adam",
            "display_every": 500,
        },
    ]

    # 5. GRADUAL_PHYSICS (Physics annealing)
    approaches["gradual_physics"] = [
        {
            "name": "stage1_low_physics",
            "epochs": 2000,
            "learning_rate": LR_HIGH,
            "loss_weights": PINN_LOSS_WEIGHTS_LOW_PHYSICS,  # Very low physics
            "train_inverse_params": True,
            "train_nn_weights": True,
            "optimizer": "adam",
            "display_every": 500,
        },
        {
            "name": "stage2_medium_physics",
            "epochs": 2000,
            "learning_rate": LR_MED,
            "loss_weights": PINN_LOSS_WEIGHTS_MEDIUM_PHYSICS,  # Medium physics
            "train_inverse_params": True,
            "train_nn_weights": True,
            "optimizer": "adam",
            "display_every": 500,
        },
        {
            "name": "stage3_standard_physics",
            "epochs": 2000,
            "learning_rate": LR_LOW,
            "loss_weights": PINN_LOSS_WEIGHTS_STANDARD,  # Standard physics
            "train_inverse_params": True,
            "train_nn_weights": True,
            "optimizer": "adam",
            "display_every": 500,
        },
        {
            "name": "stage4_high_physics",
            "epochs": 1000,
            "learning_rate": LR_LOW,
            "loss_weights": PINN_LOSS_WEIGHTS_HIGH_PHYSICS,  # High physics
            "train_inverse_params": True,
            "train_nn_weights": True,
            "optimizer": "adam",
            "display_every": 500,
        },
    ]

    # 6. ALTERNATING_OPT (Coordinate descent)
    approaches["alternating_opt"] = [
        {
            "name": "stage1_nn",
            "epochs": 1000,
            "learning_rate": LR_HIGH,
            "loss_weights": PINN_LOSS_WEIGHTS_STANDARD,
            "train_inverse_params": False,
            "train_nn_weights": True,
            "optimizer": "adam",
            "display_every": 500,
        },
        {
            "name": "stage2_ksi",
            "epochs": 1000,
            "learning_rate": LR_HIGH,
            "loss_weights": PINN_LOSS_WEIGHTS_STANDARD,
            "train_inverse_params": True,
            "train_nn_weights": False,
            "optimizer": "adam",
            "display_every": 500,
        },
        {
            "name": "stage3_nn",
            "epochs": 1000,
            "learning_rate": LR_MED,
            "loss_weights": PINN_LOSS_WEIGHTS_STANDARD,
            "train_inverse_params": False,
            "train_nn_weights": True,
            "optimizer": "adam",
            "display_every": 500,
        },
        {
            "name": "stage4_ksi",
            "epochs": 1000,
            "learning_rate": LR_MED,
            "loss_weights": PINN_LOSS_WEIGHTS_STANDARD,
            "train_inverse_params": True,
            "train_nn_weights": False,
            "optimizer": "adam",
            "display_every": 500,
        },
        {
            "name": "stage5_nn",
            "epochs": 1000,
            "learning_rate": LR_LOW,
            "loss_weights": PINN_LOSS_WEIGHTS_STANDARD,
            "train_inverse_params": False,
            "train_nn_weights": True,
            "optimizer": "adam",
            "display_every": 500,
        },
        {
            "name": "stage6_ksi",
            "epochs": 1000,
            "learning_rate": LR_LOW,
            "loss_weights": PINN_LOSS_WEIGHTS_STANDARD,
            "train_inverse_params": True,
            "train_nn_weights": False,
            "optimizer": "adam",
            "display_every": 500,
        },
        {
            "name": "stage7_joint",
            "epochs": 1000,
            "learning_rate": LR_LOW,
            "loss_weights": PINN_LOSS_WEIGHTS_STANDARD,
            "train_inverse_params": True,
            "train_nn_weights": True,
            "optimizer": "adam",
            "display_every": 500,
        },
    ]

    # 7. PHYSICS_HEAVY_START (Anchor ksi early with strong physics)
    approaches["physics_heavy_start"] = [
        {
            "name": "stage1_heavy_physics",
            "epochs": 3000,
            "learning_rate": LR_HIGH,
            "loss_weights": [2.0, 12.0, 8.0, 8.0],  # Very high physics
            "train_inverse_params": True,
            "train_nn_weights": True,
            "optimizer": "adam",
            "display_every": 500,
        },
        {
            "name": "stage2_standard",
            "epochs": 3000,
            "learning_rate": LR_MED,
            "loss_weights": PINN_LOSS_WEIGHTS_STANDARD,  # Standard
            "train_inverse_params": True,
            "train_nn_weights": True,
            "optimizer": "adam",
            "display_every": 500,
        },
        {
            "name": "stage3_nn_finetune",
            "epochs": 1000,
            "learning_rate": LR_LOW,
            "loss_weights": PINN_LOSS_WEIGHTS_GLUCOSE_ONLY,  # No physics, pure data fit
            "train_inverse_params": False,
            "train_nn_weights": True,
            "optimizer": "adam",
            "display_every": 500,
        },
    ]

    # 8. WARM_START_KSI (Initialize at population mean)
    approaches["warm_start_ksi"] = [
        {
            "name": "stage1_nn_with_prior",
            "epochs": 2000,
            "learning_rate": LR_HIGH,
            "loss_weights": PINN_LOSS_WEIGHTS_GLUCOSE_ONLY,  # No physics initially
            "train_inverse_params": False,
            "train_nn_weights": True,
            "optimizer": "adam",
            "display_every": 500,
        },
        {
            "name": "stage2_gentle_physics",
            "epochs": 2000,
            "learning_rate": LR_MED,
            "loss_weights": PINN_LOSS_WEIGHTS_MEDIUM_PHYSICS,  # Gentle physics
            "train_inverse_params": True,
            "train_nn_weights": True,
            "optimizer": "adam",
            "display_every": 500,
        },
        {
            "name": "stage3_standard",
            "epochs": 2000,
            "learning_rate": LR_LOW,
            "loss_weights": PINN_LOSS_WEIGHTS_STANDARD,
            "train_inverse_params": True,
            "train_nn_weights": True,
            "optimizer": "adam",
            "display_every": 500,
        },
        {
            "name": "stage4_high_physics",
            "epochs": 1000,
            "learning_rate": LR_LOW,
            "loss_weights": PINN_LOSS_WEIGHTS_HIGH_PHYSICS,
            "train_inverse_params": True,
            "train_nn_weights": True,
            "optimizer": "adam",
            "display_every": 500,
        },
    ]

    return approaches


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

VALIDATION_PATIENTS = [2, 3, 4, 5, 6]
TEST_PATIENTS = [7, 8, 9, 10, 11]
SEEDS = [42, 123, 456]

# Population mean for ksi (for warm_start_ksi approach)
KSI_POPULATION_MEAN = 235.0


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    # Identifiers
    approach: str
    patient: int
    seed: int

    # ksi parameter estimation
    ksi_true: float
    ksi_init: float  # Initial value (random or warm-start)
    ksi_estimated: float  # Final estimated value
    ksi_error_percent: float

    # Glucose prediction metrics
    rmse_interpolation: float  # Train set RMSE (mg/dL)
    rmse_forecast: float  # Test set RMSE (mg/dL) - CRITICAL
    rmse_total: float  # Full sequence RMSE

    # Training metadata
    training_time_seconds: float
    total_epochs: int
    num_stages: int

    # Trajectories for plotting
    ksi_trajectory: List[float]  # ksi value tracked during training
    epoch_trajectory: List[int]  # Corresponding epoch numbers
    loss_trajectory: List[float]  # Loss tracked during training
    stage_trajectory: List[str]  # Stage name for each point

    # Final state
    final_loss: float

    # Stage-by-stage breakdown
    stage_details: List[Dict[str, Any]]  # Detailed per-stage info


def run_single_experiment(
    approach_name: str,
    patient: int,
    seed: int,
    stages: List[Dict[str, Any]],
    output_dir: Path,
) -> ExperimentResult:
    """Run a single training experiment for PINN."""

    # Set seed for reproducibility
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    print(f"\n{'='*70}")
    print(f"Running: {approach_name} | Patient {patient} | Seed {seed}")
    print(f"{'='*70}")

    start_time = time.time()

    # Load data
    data = load_synthetic_window(patient=patient, root="data/synthetic")

    # Get true ksi
    true_params = make_params_from_preset(patient)
    true_ksi = float(getattr(true_params, "ksi"))

    # Build config for PINN
    config_dict = {
        "model_name": "pinn",
        "mode": "inverse",
        "device": "gpu",
        "seed": seed,
        "architecture": BASE_ARCHITECTURE.copy(),
        "training": {
            "stages": stages,
            "use_lbfgs_refinement": False,
            "optimizer": "adam",
            "learning_rate": LR_MED,  # Default, overridden by stages
            "epochs": 7000,  # Default, overridden by stages
        },
        "data": {
            "patient": f"Pat{patient}",
            "source": "synthetic",
            "mask_interval": 10,
        },
        "loss_weights": {
            "glucose": PINN_LOSS_WEIGHTS_STANDARD[0],
            "ode_glucose": PINN_LOSS_WEIGHTS_STANDARD[1],
            "ode_insulin": PINN_LOSS_WEIGHTS_STANDARD[2],
            "ode_digestion": PINN_LOSS_WEIGHTS_STANDARD[3],
        },
        "inverse_param": "ksi",
        "inverse_init_range": [150, 300],
        "output": {
            "save_dir": str(output_dir / f"{approach_name}_pat{patient}_seed{seed}"),
        },
    }

    # Special handling for warm_start_ksi - initialize at population mean
    if approach_name == "warm_start_ksi":
        config_dict["inverse_init_range"] = [
            KSI_POPULATION_MEAN - 1,
            KSI_POPULATION_MEAN + 1,
        ]

    cfg = Config.from_dict(config_dict)
    cfg.inverse_params = ["ksi"]

    # Build model
    model = FeedforwardPINN(cfg)
    model.build(data)
    model.compile()

    # Get initial ksi value (for PINN, we need to run a session to get the value)
    # Since we're in graph mode, we'll get this from the inverse_params_obj
    try:
        # Try to get the initialized value before training
        ksi_init = 225.0  # Default placeholder - will be updated after first training epoch
    except Exception:
        ksi_init = 225.0

    # Train using InverseTrainer
    trainer = InverseTrainer(
        model=model,
        config=cfg,
        data=data,
        true_param_value=true_ksi,
    )
    history = trainer.train()

    training_time = time.time() - start_time

    # Get results
    ksi_estimated = history["final_params"].get("ksi", 0.0)
    ksi_error_percent = history["param_errors_percent"].get("ksi", 100.0)

    # Get trajectories from trainer
    ksi_trajectory = history["param_history"].get("ksi_values", [])
    epoch_trajectory = history["param_history"].get("epochs", [])
    loss_trajectory = history["param_history"].get("ksi_losses", [])
    stage_trajectory = history["param_history"].get("stages", [])

    # Update ksi_init from trajectory if available
    if ksi_trajectory:
        ksi_init = ksi_trajectory[0]

    # Evaluate glucose prediction
    try:
        metrics = model.evaluate()
        rmse_interpolation = metrics.get("rmse_interpolation", 0.0)
        rmse_forecast = metrics.get("rmse_forecast", 0.0)
        rmse_total = metrics.get("rmse_total", 0.0)
    except Exception as e:
        print(f"Warning: Could not evaluate model: {e}")
        rmse_interpolation = rmse_forecast = rmse_total = -1.0

    # Calculate total epochs and stages
    total_epochs = sum(s["epochs"] for s in stages)
    num_stages = len(stages)

    # Get final loss
    final_loss = loss_trajectory[-1] if loss_trajectory else 0.0

    # Build stage details for comprehensive reporting
    stage_details = []
    cumulative_epoch = 0
    for i, stage in enumerate(stages):
        stage_info = {
            "stage_num": i + 1,
            "stage_name": stage["name"],
            "epochs": stage["epochs"],
            "learning_rate": stage["learning_rate"],
            "loss_weights": stage["loss_weights"],
            "train_ksi": stage["train_inverse_params"],
            "train_nn": stage["train_nn_weights"],
            "start_epoch": cumulative_epoch,
            "end_epoch": cumulative_epoch + stage["epochs"],
        }

        # Find ksi values at start and end of this stage
        stage_start_idx = None
        stage_end_idx = None
        for idx, ep in enumerate(epoch_trajectory):
            if ep >= cumulative_epoch and stage_start_idx is None:
                stage_start_idx = idx
            if ep >= cumulative_epoch + stage["epochs"] - 1:
                stage_end_idx = idx
                break
        if stage_end_idx is None and epoch_trajectory:
            stage_end_idx = len(epoch_trajectory) - 1

        if stage_start_idx is not None and ksi_trajectory:
            stage_info["ksi_start"] = ksi_trajectory[stage_start_idx]
        if stage_end_idx is not None and ksi_trajectory:
            stage_info["ksi_end"] = ksi_trajectory[
                min(stage_end_idx, len(ksi_trajectory) - 1)
            ]
        if stage_start_idx is not None and loss_trajectory:
            stage_info["loss_start"] = loss_trajectory[stage_start_idx]
        if stage_end_idx is not None and loss_trajectory:
            stage_info["loss_end"] = loss_trajectory[
                min(stage_end_idx, len(loss_trajectory) - 1)
            ]

        stage_details.append(stage_info)
        cumulative_epoch += stage["epochs"]

    result = ExperimentResult(
        approach=approach_name,
        patient=patient,
        seed=seed,
        ksi_true=true_ksi,
        ksi_init=ksi_init,
        ksi_estimated=ksi_estimated,
        ksi_error_percent=ksi_error_percent,
        rmse_interpolation=rmse_interpolation,
        rmse_forecast=rmse_forecast,
        rmse_total=rmse_total,
        training_time_seconds=training_time,
        total_epochs=total_epochs,
        num_stages=num_stages,
        ksi_trajectory=ksi_trajectory,
        epoch_trajectory=epoch_trajectory,
        loss_trajectory=loss_trajectory,
        stage_trajectory=stage_trajectory,
        final_loss=final_loss,
        stage_details=stage_details,
    )

    # Save individual result as JSON
    result_file = output_dir / f"{approach_name}_pat{patient}_seed{seed}.json"
    with open(result_file, "w") as f:
        result_dict = asdict(result)
        # Convert numpy types to Python types for JSON serialization
        json.dump(
            result_dict,
            f,
            indent=2,
            default=lambda x: float(x) if hasattr(x, "item") else x,
        )

    # Save trajectory as CSV for easy plotting
    if epoch_trajectory and ksi_trajectory:
        trajectory_file = (
            output_dir / f"{approach_name}_pat{patient}_seed{seed}_trajectory.csv"
        )
        trajectory_df = pd.DataFrame(
            {
                "epoch": epoch_trajectory,
                "ksi": ksi_trajectory,
                "loss": loss_trajectory if loss_trajectory else [0] * len(epoch_trajectory),
                "stage": stage_trajectory if stage_trajectory else ["unknown"] * len(epoch_trajectory),
            }
        )
        trajectory_df["ksi_true"] = true_ksi
        trajectory_df["ksi_error_percent"] = (
            abs(trajectory_df["ksi"] - true_ksi) / true_ksi * 100
        )
        trajectory_df.to_csv(trajectory_file, index=False)

    print(
        f"\nResults: ksi_error={ksi_error_percent:.2f}%, "
        f"RMSE_forecast={rmse_forecast:.2f} mg/dL, "
        f"time={training_time:.1f}s"
    )

    # Reset TensorFlow graph for next experiment
    tf.compat.v1.reset_default_graph()

    return result


def run_study(
    approaches: Optional[List[str]] = None,
    patients: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
    pilot: bool = False,
) -> pd.DataFrame:
    """Run the full training study."""

    # Get all approach configurations
    all_approaches = get_approach_configs()

    # Filter approaches if specified
    if approaches:
        approach_configs = {k: v for k, v in all_approaches.items() if k in approaches}
    else:
        approach_configs = all_approaches

    # Use default patients/seeds if not specified
    if patients is None:
        patients = VALIDATION_PATIENTS + TEST_PATIENTS
    if seeds is None:
        seeds = SEEDS

    # For pilot mode, reduce scope
    if pilot:
        approach_configs = {k: v for k, v in list(approach_configs.items())[:2]}
        patients = patients[:2]
        seeds = seeds[:1]
        print("PILOT MODE: Running reduced experiment")
        print(f"  Approaches: {list(approach_configs.keys())}")
        print(f"  Patients: {patients}")
        print(f"  Seeds: {seeds}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/training_study_pinn_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save study configuration
    study_config = {
        "model": "pinn",
        "approaches": list(approach_configs.keys()),
        "patients": patients,
        "seeds": seeds,
        "pilot": pilot,
        "timestamp": timestamp,
    }
    with open(output_dir / "study_config.json", "w") as f:
        json.dump(study_config, f, indent=2)

    # Run experiments
    results = []
    total_runs = len(approach_configs) * len(patients) * len(seeds)
    current_run = 0

    for approach_name, stages in approach_configs.items():
        for patient in patients:
            for seed in seeds:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] ", end="")

                try:
                    result = run_single_experiment(
                        approach_name=approach_name,
                        patient=patient,
                        seed=seed,
                        stages=stages,
                        output_dir=output_dir,
                    )
                    results.append(asdict(result))
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    # Record failed experiment
                    results.append(
                        {
                            "approach": approach_name,
                            "patient": patient,
                            "seed": seed,
                            "error": str(e),
                        }
                    )

    # Create summary DataFrame
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "all_results.csv", index=False)

    # Compute summary statistics
    summary = compute_summary_statistics(df, output_dir)

    # Generate comprehensive report
    generate_comprehensive_report(output_dir)

    print(f"\n{'='*70}")
    print(f"PINN STUDY COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}")

    return df, output_dir


def compute_summary_statistics(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Compute and save summary statistics for each approach."""

    # Filter out failed experiments
    df_valid = df[df["ksi_error_percent"].notna()].copy()

    if len(df_valid) == 0:
        print("Warning: No valid results to summarize")
        return pd.DataFrame()

    # Group by approach
    summary_rows = []

    for approach in df_valid["approach"].unique():
        approach_df = df_valid[df_valid["approach"] == approach]

        summary_rows.append(
            {
                "approach": approach,
                "n_runs": len(approach_df),
                "ksi_error_mean": approach_df["ksi_error_percent"].mean(),
                "ksi_error_std": approach_df["ksi_error_percent"].std(),
                "ksi_error_median": approach_df["ksi_error_percent"].median(),
                "ksi_error_min": approach_df["ksi_error_percent"].min(),
                "ksi_error_max": approach_df["ksi_error_percent"].max(),
                "rmse_forecast_mean": approach_df["rmse_forecast"].mean(),
                "rmse_forecast_std": approach_df["rmse_forecast"].std(),
                "rmse_interp_mean": approach_df["rmse_interpolation"].mean(),
                "rmse_interp_std": approach_df["rmse_interpolation"].std(),
                "time_mean_sec": approach_df["training_time_seconds"].mean(),
                "n_excellent": (approach_df["ksi_error_percent"] < 5).sum(),
                "n_good": (approach_df["ksi_error_percent"] < 10).sum(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("ksi_error_mean")

    # Save summary
    summary_df.to_csv(output_dir / "summary_statistics.csv", index=False)

    # Print summary table
    print("\n" + "=" * 90)
    print("SUMMARY STATISTICS")
    print("=" * 90)
    print(
        f"{'Approach':<25} {'ksi Error %':<20} {'RMSE Forecast':<20} {'Time (s)':<10}"
    )
    print(f"{'':<25} {'mean ± std':<20} {'mean ± std':<20}")
    print("-" * 90)

    for _, row in summary_df.iterrows():
        print(
            f"{row['approach']:<25} "
            f"{row['ksi_error_mean']:.2f} ± {row['ksi_error_std']:.2f}  "
            f"{row['rmse_forecast_mean']:.2f} ± {row['rmse_forecast_std']:.2f}  "
            f"{row['time_mean_sec']:.0f}"
        )

    print("=" * 90)

    return summary_df


def generate_comprehensive_report(output_dir: Path) -> None:
    """Generate a comprehensive markdown report from study results."""

    # Load all results
    all_results = pd.read_csv(output_dir / "all_results.csv")

    summary_path = output_dir / "summary_statistics.csv"
    if not summary_path.exists():
        print("Warning: No summary statistics to include in report")
        return

    summary = pd.read_csv(summary_path)

    # Load study config
    with open(output_dir / "study_config.json") as f:
        study_config = json.load(f)

    report_lines = []
    report_lines.append("# PINN Training Approach Comparison Study - Comprehensive Report\n")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"Results directory: `{output_dir}`\n\n")

    # Study Configuration
    report_lines.append("## Study Configuration\n")
    report_lines.append(f"- **Model**: PINN (DeepXDE)\n")
    report_lines.append(f"- **Approaches tested**: {len(study_config['approaches'])}\n")
    report_lines.append(f"- **Patients**: {study_config['patients']}\n")
    report_lines.append(f"- **Seeds**: {study_config['seeds']}\n")
    report_lines.append(f"- **Total runs**: {len(all_results)}\n")
    report_lines.append(f"- **Pilot mode**: {study_config['pilot']}\n\n")

    # Summary Table
    report_lines.append("## Summary Results\n\n")
    report_lines.append(
        "| Approach | ksi Error (%) | RMSE Forecast (mg/dL) | Time (s) | Excellent (<5%) | Good (<10%) |\n"
    )
    report_lines.append(
        "|----------|---------------|----------------------|----------|-----------------|-------------|\n"
    )

    for _, row in summary.iterrows():
        report_lines.append(
            f"| {row['approach']} | "
            f"{row['ksi_error_mean']:.2f} ± {row['ksi_error_std']:.2f} | "
            f"{row['rmse_forecast_mean']:.2f} ± {row['rmse_forecast_std']:.2f} | "
            f"{row['time_mean_sec']:.0f} | "
            f"{int(row['n_excellent'])}/{int(row['n_runs'])} | "
            f"{int(row['n_good'])}/{int(row['n_runs'])} |\n"
        )

    report_lines.append("\n")

    # Key Findings
    report_lines.append("## Key Findings\n\n")

    if len(summary) > 0:
        best_approach = summary.iloc[0]["approach"]
        worst_approach = summary.iloc[-1]["approach"]

        report_lines.append(
            f"1. **Best approach**: `{best_approach}` with mean ksi error of "
            f"{summary.iloc[0]['ksi_error_mean']:.2f}%\n"
        )
        report_lines.append(
            f"2. **Worst approach**: `{worst_approach}` with mean ksi error of "
            f"{summary.iloc[-1]['ksi_error_mean']:.2f}%\n"
        )

    report_lines.append("\n")

    # Files generated
    report_lines.append("## Files Generated\n\n")
    report_lines.append("- `all_results.csv` - All experiment results\n")
    report_lines.append("- `summary_statistics.csv` - Per-approach summary\n")
    report_lines.append("- `{approach}_pat{N}_seed{S}.json` - Full results per run\n")
    report_lines.append(
        "- `{approach}_pat{N}_seed{S}_trajectory.csv` - ksi evolution data\n"
    )
    report_lines.append("- `REPORT.md` - This report\n")

    # Write report
    report_path = output_dir / "REPORT.md"
    with open(report_path, "w") as f:
        f.writelines(report_lines)

    print(f"\nComprehensive report saved to: {report_path}")


def upload_results_to_s3(output_dir: Path, bucket: str = "t1d-pinn-results-900630261719") -> bool:
    """Upload results to S3 before shutdown."""
    import subprocess

    print("\n" + "=" * 70)
    print("UPLOADING RESULTS TO S3")
    print("=" * 70)

    try:
        # Sync results directory to S3
        s3_path = f"s3://{bucket}/training_study_pinn/{output_dir.name}/"
        cmd = ["aws", "s3", "sync", str(output_dir), s3_path]
        print(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Successfully uploaded to {s3_path}")
            return True
        else:
            print(f"S3 upload failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Warning: Could not upload to S3: {e}")
        return False


def shutdown_instance():
    """Shutdown the EC2 instance after experiment completion."""
    import subprocess

    print("\n" + "=" * 70)
    print("SHUTTING DOWN EC2 INSTANCE")
    print("=" * 70)

    try:
        # Use shutdown command
        result = subprocess.run(
            ["sudo", "shutdown", "-h", "now"],
            capture_output=True,
            text=True,
        )
        print("Shutdown command issued successfully")
    except Exception as e:
        print(f"Warning: Could not shutdown instance: {e}")
        print("Please manually stop the instance to avoid charges!")


def main():
    parser = argparse.ArgumentParser(
        description="Run PINN training approach comparison study"
    )
    parser.add_argument(
        "--pilot", action="store_true", help="Run pilot test (reduced scope)"
    )
    parser.add_argument("--approach", type=str, help="Run only specific approach")
    parser.add_argument("--patient", type=int, help="Run only specific patient")
    parser.add_argument(
        "--validation-only", action="store_true", help="Run only on validation patients"
    )
    parser.add_argument(
        "--test-only", action="store_true", help="Run only on test patients"
    )
    parser.add_argument(
        "--auto-shutdown",
        action="store_true",
        help="Shutdown EC2 instance after completion",
    )

    args = parser.parse_args()

    # Determine approaches
    approaches = None
    if args.approach:
        approaches = [args.approach]

    # Determine patients
    patients = None
    if args.patient:
        patients = [args.patient]
    elif args.validation_only:
        patients = VALIDATION_PATIENTS
    elif args.test_only:
        patients = TEST_PATIENTS

    # Run study
    df, output_dir = run_study(
        approaches=approaches,
        patients=patients,
        pilot=args.pilot,
    )

    # Auto-shutdown if requested
    if args.auto_shutdown:
        # Upload results to S3 first
        upload_results_to_s3(output_dir)
        # Then shutdown
        shutdown_instance()


if __name__ == "__main__":
    main()
