#!/usr/bin/env python3
"""Quick integration test for Modified-MLP PINN."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.config import load_config
from src.datasets.loader import load_synthetic_window
from src.models.modified_mlp import ModifiedMLPPINN
import numpy as np


def test_modified_mlp_integration():
    """Test that Modified-MLP integrates with Phase 1 data layer."""

    print("=" * 80)
    print("MODIFIED-MLP PINN INTEGRATION TEST")
    print("=" * 80)

    # Step 1: Load configuration
    print("\n[1/5] Loading configuration...")
    config = load_config(
        model_name="modified_mlp",
        mode="forward",
    )
    config.training.epochs = 100  # Short training for test
    config.training.use_lbfgs_refinement = False  # Skip L-BFGS for speed
    config.data.patient = "Pat3"
    print("✅ Configuration loaded")

    # Step 2: Load data
    print("\n[2/5] Loading synthetic data...")
    try:
        data_window = load_synthetic_window(patient=3, root="data/synthetic")
        print(f"✅ Data loaded: {data_window}")
        print(f"   Glucose shape: {data_window.glucose.shape}")
        print(f"   Has latent states: {data_window.has_latent_states}")
        print(
            f"   Scaling factors: m_t={data_window.m_t:.1f}, m_g={data_window.m_g:.1f}, "
            f"m_i={data_window.m_i:.5f}, m_d={data_window.m_d:.2f}"
        )
    except FileNotFoundError as e:
        print(f"❌ Data not found: {e}")
        print("\n   You need to generate synthetic data first:")
        print("   Run: python scripts/generate_synthetic_data.py")
        return False

    # Step 3: Build model
    print("\n[3/5] Building Modified-MLP model...")
    try:
        model = ModifiedMLPPINN(config)
        model.build(data_window)
        print("✅ Model built successfully")
    except Exception as e:
        print(f"❌ Model build failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Step 4: Compile model
    print("\n[4/5] Compiling model...")
    try:
        model.compile()
        print("✅ Model compiled")
    except Exception as e:
        print(f"❌ Model compilation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Step 5: Short training run
    print("\n[5/5] Running short training (100 epochs)...")
    try:
        model.train(display_every=50)
        print("✅ Training completed")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Step 6: Evaluation
    print("\n[6/6] Evaluating model...")
    try:
        metrics = model.evaluate()
        print("✅ Evaluation completed")

        # Check if metrics are reasonable
        for key, value in metrics.items():
            if np.isnan(value) or value > 1000:
                print(f"⚠️  Warning: {key} = {value} seems unusual")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Success!
    print("\n" + "=" * 80)
    print("✅ INTEGRATION TEST PASSED!")
    print("=" * 80)
    print("\nThe Modified-MLP PINN successfully:")
    print("  ✓ Loaded configuration")
    print("  ✓ Loaded Phase 1 data")
    print("  ✓ Built the custom gated architecture")
    print("  ✓ Compiled with DeepXDE")
    print("  ✓ Ran training loop")
    print("  ✓ Evaluated on test data")
    print("\n✨ Modified-MLP is ready to train!")

    return True


if __name__ == "__main__":
    success = test_modified_mlp_integration()
    sys.exit(0 if success else 1)
