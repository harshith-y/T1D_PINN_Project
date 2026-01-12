#!/usr/bin/env python3
"""
Visualisation Script for Optimisation Results

Generates thesis-quality figures from Optuna optimisation studies:
- Optimisation history plots
- Parameter importance analysis
- Heatmaps for model x patient performance
- Loss weight sensitivity analysis
- Parallel coordinate plots
- Box plots for error distributions

Usage:
    python scripts/visualize_optimisation.py --study-db results/optimisation/study.db

    python scripts/visualize_optimisation.py \
        --study-db results/optimisation/inverse_opt.db \
        --output-dir results/optimisation/visualisations/
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns

# Style settings for thesis figures
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


def load_study(db_path: str, study_name: str = None) -> optuna.Study:
    """Load Optuna study from SQLite database."""
    storage = f"sqlite:///{db_path}"

    if study_name is None:
        # Get first study in database
        study_summaries = optuna.study.get_all_study_summaries(storage)
        if not study_summaries:
            raise ValueError(f"No studies found in {db_path}")
        study_name = study_summaries[0].study_name

    study = optuna.load_study(study_name=study_name, storage=storage)
    return study


def plot_optimisation_history(study: optuna.Study, output_dir: Path, mode: str):
    """Plot optimisation history showing best value over trials."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get trial values
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    values = [t.value for t in trials]
    trial_nums = [t.number for t in trials]

    # Calculate running best
    running_best = np.minimum.accumulate(values)

    # Plot
    ax.scatter(trial_nums, values, alpha=0.5, label="Trial value", s=30)
    ax.plot(trial_nums, running_best, "r-", linewidth=2, label="Best so far")

    # Labels
    metric = "RMSE" if mode == "forward" else "ksi Error (%)"
    ax.set_xlabel("Trial Number")
    ax.set_ylabel(metric)
    ax.set_title(f"Optimisation History - {mode.capitalize()} Mode")
    ax.legend()

    # Save
    output_path = output_dir / "optimisation_history.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_param_importance(study: optuna.Study, output_dir: Path, mode: str):
    """Plot parameter importance using Optuna's built-in analysis."""
    try:
        # Get importance scores
        importance = optuna.importance.get_param_importances(study)

        if not importance:
            print("  Warning: No parameter importance data available")
            return

        # Sort by importance
        params = list(importance.keys())
        scores = list(importance.values())

        # Create plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(params) * 0.4)))

        # Horizontal bar chart
        y_pos = np.arange(len(params))
        ax.barh(y_pos, scores, color="steelblue", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params)
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Parameter Importance - {mode.capitalize()} Mode")
        ax.invert_yaxis()  # Most important at top

        # Save
        output_path = output_dir / f"param_importance_{mode}.png"
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved: {output_path}")

    except Exception as e:
        print(f"  Warning: Could not generate parameter importance: {e}")


def plot_model_comparison(study: optuna.Study, output_dir: Path, mode: str):
    """Box plot comparing performance across model types."""
    # Get trial data
    df = study.trials_dataframe()

    if "params_model_type" not in df.columns:
        print("  Warning: No model_type parameter found in trials")
        return

    # Filter completed trials
    df = df[df["state"] == "COMPLETE"]

    if len(df) == 0:
        print("  Warning: No completed trials found")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Box plot
    model_order = ["birnn", "pinn", "modified_mlp"]
    available_models = [m for m in model_order if m in df["params_model_type"].values]

    if len(available_models) == 0:
        print("  Warning: No recognised model types found")
        return

    # Create box plot
    sns.boxplot(
        data=df,
        x="params_model_type",
        y="value",
        order=available_models,
        palette="Set2",
        ax=ax,
    )

    # Add individual points
    sns.stripplot(
        data=df,
        x="params_model_type",
        y="value",
        order=available_models,
        color="black",
        alpha=0.3,
        ax=ax,
    )

    # Labels
    metric = "RMSE" if mode == "forward" else "ksi Error (%)"
    ax.set_xlabel("Model Architecture")
    ax.set_ylabel(metric)
    ax.set_title(f"Model Comparison - {mode.capitalize()} Mode")

    # Better labels
    ax.set_xticklabels(["BI-RNN", "PINN", "Mod-MLP"][:len(available_models)])

    # Save
    output_path = output_dir / f"model_comparison_{mode}.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_parallel_coordinates(study: optuna.Study, output_dir: Path, mode: str):
    """Parallel coordinate plot for hyperparameter exploration."""
    try:
        fig = optuna.visualization.matplotlib.plot_parallel_coordinate(
            study,
            params=[
                "model_type",
                "stage1_lr" if mode == "inverse" else "learning_rate",
                "stage1_epochs" if mode == "inverse" else "epochs",
            ],
        )
        fig.set_size_inches(14, 8)

        output_path = output_dir / f"parallel_coordinates_{mode}.png"
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved: {output_path}")

    except Exception as e:
        print(f"  Warning: Could not generate parallel coordinates: {e}")


def plot_learning_rate_analysis(study: optuna.Study, output_dir: Path, mode: str):
    """Scatter plot of learning rate vs performance."""
    df = study.trials_dataframe()
    df = df[df["state"] == "COMPLETE"]

    if len(df) == 0:
        return

    # Determine LR column
    if mode == "inverse":
        lr_cols = ["params_stage1_lr", "params_stage2_lr", "params_stage3_lr"]
        available_lr = [c for c in lr_cols if c in df.columns]
        if not available_lr:
            return
        lr_col = available_lr[0]
    else:
        lr_col = "params_learning_rate"
        if lr_col not in df.columns:
            return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by model type if available
    if "params_model_type" in df.columns:
        for model_type in df["params_model_type"].unique():
            mask = df["params_model_type"] == model_type
            ax.scatter(
                df.loc[mask, lr_col],
                df.loc[mask, "value"],
                label=model_type.upper(),
                alpha=0.6,
                s=50,
            )
        ax.legend()
    else:
        ax.scatter(df[lr_col], df["value"], alpha=0.6, s=50)

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    metric = "RMSE" if mode == "forward" else "ksi Error (%)"
    ax.set_ylabel(metric)
    ax.set_title(f"Learning Rate Analysis - {mode.capitalize()} Mode")

    output_path = output_dir / f"lr_analysis_{mode}.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_loss_weight_heatmap(study: optuna.Study, output_dir: Path, mode: str):
    """Heatmap of loss weight combinations vs performance."""
    df = study.trials_dataframe()
    df = df[df["state"] == "COMPLETE"]

    if len(df) < 10:
        print("  Warning: Not enough trials for heatmap (need >= 10)")
        return

    # Check for loss weight columns
    if mode == "forward":
        w1_col = "params_lambda_g"
        w2_col = "params_lambda_B"
    else:
        w1_col = "params_lambda_g"
        w2_col = "params_lambda_B"

    if w1_col not in df.columns or w2_col not in df.columns:
        print(f"  Warning: Loss weight columns not found ({w1_col}, {w2_col})")
        return

    # Bin the weights
    df["w1_bin"] = pd.cut(df[w1_col], bins=5, labels=[f"{i}" for i in range(5)])
    df["w2_bin"] = pd.cut(df[w2_col], bins=5, labels=[f"{i}" for i in range(5)])

    # Create pivot table
    pivot = df.pivot_table(
        values="value",
        index="w1_bin",
        columns="w2_bin",
        aggfunc="mean",
    )

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",  # Red = bad, Green = good
        ax=ax,
    )
    ax.set_xlabel("lambda_B bins")
    ax.set_ylabel("lambda_g bins")
    metric = "RMSE" if mode == "forward" else "ksi Error (%)"
    ax.set_title(f"Loss Weight Sensitivity - {metric}")

    output_path = output_dir / f"loss_weight_heatmap_{mode}.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_epoch_analysis(study: optuna.Study, output_dir: Path, mode: str):
    """Analyse effect of total training epochs."""
    df = study.trials_dataframe()
    df = df[df["state"] == "COMPLETE"]

    if len(df) == 0:
        return

    # Calculate total epochs
    if mode == "inverse":
        epoch_cols = ["params_stage1_epochs", "params_stage2_epochs", "params_stage3_epochs"]
        available = [c for c in epoch_cols if c in df.columns]
        if available:
            df["total_epochs"] = df[available].sum(axis=1)
        else:
            return
    else:
        if "params_epochs" not in df.columns:
            return
        df["total_epochs"] = df["params_epochs"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot
    if "params_model_type" in df.columns:
        for model_type in df["params_model_type"].unique():
            mask = df["params_model_type"] == model_type
            ax.scatter(
                df.loc[mask, "total_epochs"],
                df.loc[mask, "value"],
                label=model_type.upper(),
                alpha=0.6,
                s=50,
            )
        ax.legend()
    else:
        ax.scatter(df["total_epochs"], df["value"], alpha=0.6, s=50)

    ax.set_xlabel("Total Training Epochs")
    metric = "RMSE" if mode == "forward" else "ksi Error (%)"
    ax.set_ylabel(metric)
    ax.set_title(f"Epoch Count Analysis - {mode.capitalize()} Mode")

    output_path = output_dir / f"epoch_analysis_{mode}.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def generate_summary_table(study: optuna.Study, output_dir: Path, mode: str):
    """Generate summary statistics table."""
    df = study.trials_dataframe()
    df = df[df["state"] == "COMPLETE"]

    if len(df) == 0:
        return

    # Summary by model type
    if "params_model_type" in df.columns:
        summary = df.groupby("params_model_type")["value"].agg([
            "count",
            "mean",
            "std",
            "min",
            ("best_trial", lambda x: df.loc[x.idxmin(), "number"]),
        ]).round(4)

        summary.columns = ["Trials", "Mean", "Std", "Best", "Best Trial"]
        summary.index = summary.index.map(lambda x: x.upper())

        # Save as CSV
        output_path = output_dir / f"summary_table_{mode}.csv"
        summary.to_csv(output_path)
        print(f"  Saved: {output_path}")

        # Also save as markdown for thesis
        md_path = output_dir / f"summary_table_{mode}.md"
        with open(md_path, "w") as f:
            f.write(f"# {mode.capitalize()} Optimisation Summary\n\n")
            f.write(summary.to_markdown())
        print(f"  Saved: {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualisations from optimisation results"
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
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as study DB)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="inverse",
        choices=["forward", "inverse"],
        help="Optimisation mode (for metric labels)",
    )

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.study_db).parent / "visualisations"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("OPTIMIZATION VISUALIZATION")
    print("=" * 70)
    print(f"Study DB: {args.study_db}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Load study
    print("\nLoading study...")
    study = load_study(args.study_db, args.study_name)
    print(f"  Study: {study.study_name}")
    print(f"  Trials: {len(study.trials)}")
    print(f"  Complete: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

    # Generate visualisations
    print("\nGenerating visualisations...")

    plot_optimisation_history(study, output_dir, args.mode)
    plot_param_importance(study, output_dir, args.mode)
    plot_model_comparison(study, output_dir, args.mode)
    plot_parallel_coordinates(study, output_dir, args.mode)
    plot_learning_rate_analysis(study, output_dir, args.mode)
    plot_loss_weight_heatmap(study, output_dir, args.mode)
    plot_epoch_analysis(study, output_dir, args.mode)
    generate_summary_table(study, output_dir, args.mode)

    print(f"\nAll visualisations saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
