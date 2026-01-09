# CLAUDE.md - T1D PINN Project

## Project Overview
Physics-Informed Neural Networks for Type 1 Diabetes glucose prediction and physiological parameter estimation. Three model architectures using the Magdelaine glucose-insulin-carbohydrate model. Goal: Build CV-worthy production code for industry roles.

## Quick Commands
```bash
# Activate environment
source T1D_PINN_venv/bin/activate

# Testing
pytest tests/ -v                          # All tests
pytest tests/unit/ -v                     # Unit only
pytest tests/unit/test_models.py -v       # Specific file

# Linting (run before commits)
black src/ scripts/ tests/
isort src/ scripts/ tests/
flake8 src/ scripts/ tests/

# Training examples
python scripts/train_forward.py --config configs/birnn.yaml --patient Pat2
python scripts/train_inverse.py --config configs/birnn.yaml --patient Pat2
```

## Current Priority: Fix CI/CD Pipeline
The GitHub Actions workflow (`.github/workflows/ci-test.yml`) has `continue-on-error: true` on most steps, masking actual test failures. 

**Task:** 
1. Run `pytest tests/ -v` to see real failures
2. Fix the underlying issues
3. Remove `continue-on-error: true` from workflow
4. Push and verify legitimate green CI

## Project Structure
```
T1D_PINN_Project/
├── src/
│   ├── models/
│   │   ├── birnn.py              # BI-RNN (TF2.x, eager execution)
│   │   ├── pinn_feedforward.py   # DeepXDE PINN (TF1.x compat)
│   │   └── modified_mlp.py       # Modified MLP (TF1.x compat)
│   ├── datasets/
│   │   ├── loader.py             # Unified data loading
│   │   ├── windowing.py          # Training windows
│   │   ├── preprocessing.py      # Data preprocessing
│   │   └── simulator.py          # Synthetic data generation
│   ├── physics/
│   │   └── magdelaine.py         # Physiological model
│   ├── training/                 # Training modules
│   └── visualization/            # Plotting modules
├── scripts/                      # Entry point scripts
├── tests/
│   ├── unit/
│   └── integration/
├── configs/                      # YAML model configs
├── data/
│   ├── synthetic/                # Pat2-Pat11 (safe to reference)
│   └── processed/                # CONFIDENTIAL patient data
└── .github/workflows/ci-test.yml # CI pipeline (needs fixing)
```

## Critical Technical Notes

### TensorFlow Version Conflicts
```python
# DeepXDE models (PINN, Modified-MLP) - MUST have at top of file:
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# BI-RNN - uses TF2.x eager execution
# These modes CANNOT be mixed in the same Python session
```

### Key Bug Fixes Already Applied
- **Timestep scaling:** Use `dt = 1.0` (physical time), not `dt = 1.0 / m_t` (normalized)
- **Normalization:** BI-RNN targets must be normalized to [0,1] before training
- **Memory:** Real patient data auto-limited to 48 hours to prevent crashes

### Best Results Achieved
- BI-RNN inverse training: **3.82% parameter estimation error**
- 3-stage training works: inverse params → NN weights → joint optimization

## Data Rules
- `data/synthetic/` (Pat2-Pat11): Safe to reference, has full ground truth
- `data/processed/` (RealPat1-15): **CONFIDENTIAL** - never commit, never expose
- Real patient data has glucose only (no latent states I(t), D(t))

## Model Architectures

| Model | Framework | TF Mode | Best For |
|-------|-----------|---------|----------|
| BI-RNN | TensorFlow 2.x | Eager | Best results, inverse training |
| PINN | DeepXDE | Graph (TF1.x) | Physics constraints |
| Modified-MLP | DeepXDE | Graph (TF1.x) | U-V encoding experiments |

## Config Files
- `configs/birnn.yaml` - BI-RNN settings
- `configs/pinn.yaml` - Feedforward PINN settings  
- `configs/modified_mlp.yaml` - Modified MLP settings

## When Debugging
1. Check TF execution mode matches model type
2. Verify data is properly normalized
3. Check timestep units in physics calculations
4. For DeepXDE: read parameters from log files, not TF sessions
