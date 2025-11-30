#!/usr/bin/env python3
"""
Enhanced Visualization Script v2

Fast visualization using saved predictions (no model reload needed).

Usage:
    # Visualize saved results
    python scripts/visualize_v2.py --results-dir results/birnn_forward/Pat3_20241127_143022

    # Custom output directory
    python scripts/visualize_v2.py --results-dir results/birnn_forward/Pat3_* --output-dir custom_plots
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import argparse

from src.training.predictions import PredictionManager
from src.visualisation.plotter import ExperimentPlotter


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(
        description="Fast visualization from saved predictions"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Results directory containing predictions.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: results_dir/plots)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"❌ Error: Results directory not found: {results_dir}")
        sys.exit(1)

    print("=" * 80)
    print("VISUALIZING RESULTS v2 (Enhanced)")
    print("=" * 80)
    print(f"Results: {results_dir}")
    print("=" * 80)

    # ========================================================================
    # STEP 1: Load Predictions (FAST!)
    # ========================================================================
    print("\n[1/3] Loading predictions...")

    pred_mgr = PredictionManager(results_dir)

    if not pred_mgr.exists():
        print("❌ Error: No predictions found in results directory")
        print("   Run training first with prediction saving enabled")
        sys.exit(1)

    predictions = pred_mgr.load()
    metadata = predictions.get("metadata", {})

    print("✅ Predictions loaded")
    print(f"   Time points: {len(predictions['time'])}")
    print(f"   Has latent states: {predictions.get('insulin_pred') is not None}")

    # ========================================================================
    # STEP 2: Setup Plotter
    # ========================================================================
    print("\n[2/3] Setting up plotter...")

    # Determine output directory
    if args.output_dir is None:
        output_dir = results_dir / "plots"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get model name for styling
    model_name = metadata.get("model_name", "Model")
    model_display = {
        "birnn": "BI-RNN",
        "pinn": "PINN",
        "modified_mlp": "Modified-MLP",
    }.get(model_name, model_name)

    # Create plotter (uses YOUR existing plotter!)
    plotter = ExperimentPlotter(
        save_dir=output_dir, model_name=model_display, dpi=600  # Your thesis quality
    )

    print("✅ Plotter initialized")

    # ========================================================================
    # STEP 3: Generate All Plots
    # ========================================================================
    print("\n[3/3] Generating plots...")

    # Prepare data for plotter
    predictions_dict = {
        "time": predictions["time"],
        "glucose": predictions["glucose_pred"],
        "insulin": predictions.get("insulin_pred"),
        "digestion": predictions.get("digestion_pred"),
    }

    ground_truth_dict = {
        "glucose": predictions["glucose_true"],
        "insulin": predictions.get("insulin_true"),
        "digestion": predictions.get("digestion_true"),
    }

    # Call YOUR plot_all method!
    plot_paths = plotter.plot_all(
        predictions=predictions_dict,
        ground_truth=ground_truth_dict,
        metrics=metadata,
        split_idx=predictions.get("split_idx"),
    )

    print("✅ Plots generated:")
    for name, path in plot_paths.items():
        print(f"   📈 {name}: {path.name}")

    # Also plot parameter evolution if available (inverse mode)
    mode = metadata.get("mode", "forward")
    if mode == "inverse":
        print("\n   📊 INVERSE MODE: Generating parameter evolution plots...")

        # Get list of all estimated parameters
        inverse_params = metadata.get("inverse_params", ["ksi"])
        if not isinstance(inverse_params, list):
            inverse_params = [inverse_params]

        param_plots_generated = 0
        for param_name in inverse_params:
            param_data = pred_mgr.load_parameter_evolution(param_name)

            if param_data is not None:
                # Call existing plotter method with correct signature
                plotter.plot_parameter_evolution(
                    param_history=param_data["param_values"].tolist(),
                    true_value=param_data.get("true_value"),
                    param_name=param_name,
                    save_name=f"parameter_evolution_{param_name}.png",
                )
                print(f"   ✅ parameter_evolution_{param_name}.png")
                param_plots_generated += 1

        if param_plots_generated == 0:
            print("   ⚠️  No parameter evolution data found")
        else:
            print(
                f"   ✅ Generated {param_plots_generated} parameter evolution plot(s)"
            )

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ VISUALIZATION COMPLETE!")
    print("=" * 80)

    print(f"\n📁 Plots saved to:")
    print(f"   {output_dir}")

    print(f"\n💡 View plots:")
    print(f"   open {output_dir}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Visualization failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
