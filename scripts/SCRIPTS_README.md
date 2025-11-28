# 📚 Scripts Reference Guide

Complete reference for all enhanced training, visualization, and evaluation scripts.

---

## 📋 Table of Contents

1. [train_forward_v2.py](#1-train_forward_v2py) - Forward training with checkpointing
2. [train_inverse_v2.py](#2-train_inverse_v2py) - Multi-stage inverse parameter estimation
3. [visualize_v2.py](#3-visualize_v2py) - Fast visualization from saved predictions
4. [evaluate_v2.py](#4-evaluate_v2py) - Comprehensive evaluation and metrics

---

## 1. train_forward_v2.py

**Purpose:** Train models for glucose prediction (forward problem) with enhanced checkpointing and prediction saving.

### Arguments:

| Argument | Required | Type | Choices | Default | Description |
|----------|----------|------|---------|---------|-------------|
| `--model` | ✅ Yes | str | `birnn`, `pinn`, `modified_mlp` | - | Model architecture to train |
| `--patient` | ✅ Yes | int | - | - | Patient number (2-11 for synthetic, 1-15 for real) |
| `--data-type` | ❌ No | str | `synthetic`, `real` | `synthetic` | Data source type |
| `--epochs` | ❌ No | int | - | `2000` | Number of training epochs |
| `--save-dir` | ❌ No | str | - | Auto-generated | Custom output directory |
| `--resume` | ❌ No | str | - | `None` | Path to checkpoint directory to resume from |

### Usage Examples:

#### Basic Training (Synthetic Patient):
```bash
# Train BI-RNN on synthetic patient 3 for 2000 epochs
python scripts/train_forward_v2.py --model birnn --patient 3 --epochs 2000
```

#### Real Patient Data:
```bash
# Train PINN on real patient 5
python scripts/train_forward_v2.py --model pinn --patient 5 --data-type real --epochs 1000
```

#### Quick Test (Short Training):
```bash
# Quick 500-epoch test
python scripts/train_forward_v2.py --model birnn --patient 3 --epochs 500
```

#### Resume Interrupted Training:
```bash
# Resume from interrupted checkpoint
python scripts/train_forward_v2.py \
    --model birnn \
    --patient 3 \
    --resume results/birnn_forward/Pat3_20241127_143022/checkpoints/interrupted
```

#### Custom Output Directory:
```bash
# Save to custom location
python scripts/train_forward_v2.py \
    --model birnn \
    --patient 3 \
    --epochs 2000 \
    --save-dir my_experiments/test1
```

### Output Structure:

After training, results are saved to:
```
results/birnn_forward/Pat3_20241127_143022/
├── checkpoints/
│   ├── best/                    # Best model (lowest loss)
│   │   ├── model_weights.h5
│   │   ├── optimizer_state.npy
│   │   ├── training_state.json
│   │   └── metadata.json
│   └── final/                   # Final epoch checkpoint
├── predictions.npz              # Compressed predictions (fast reload!)
├── predictions_metadata.json
└── plots/                       # Generated during training
    ├── glucose_prediction.png
    ├── latent_variables.png     # If synthetic data
    └── ...
```

### Notes:

- **Synthetic patients (2-11):** Have complete ground truth (glucose, insulin, digestion)
- **Real patients (1-15):** Only glucose measurements available
- **Checkpoints** include optimizer state for smooth resume
- **Predictions** saved in compressed .npz format for instant reload

---

## 2. train_inverse_v2.py

**Purpose:** Multi-stage inverse training for parameter estimation (e.g., estimating ksi, kl, ku_Vi).

### Arguments:

| Argument | Required | Type | Choices | Default | Description |
|----------|----------|------|---------|---------|-------------|
| `--config` | ✅ Yes | str | - | - | Path to YAML config with training stages |
| `--patient` | ✅ Yes | int | - | - | Patient number (2-11 for synthetic, 1-15 for real) |
| `--data-type` | ❌ No | str | `synthetic`, `real` | `synthetic` | Data source type |
| `--param` | ❌ No | str | `ksi`, `kl`, `ku_Vi` | From config | Parameter to estimate (overrides config) |
| `--save-dir` | ❌ No | str | - | Auto-generated | Custom output directory |

### Usage Examples:

#### Basic Inverse Training (Synthetic):
```bash
# Estimate ksi for synthetic patient 3 (has ground truth for error calculation)
python scripts/train_inverse_v2.py \
    --config configs/birnn_inverse.yaml \
    --patient 3
```

#### Real Patient (No Ground Truth):
```bash
# Estimate parameters for real patient 5 (no error calculation, but gets estimates)
python scripts/train_inverse_v2.py \
    --config configs/birnn_inverse.yaml \
    --patient 5 \
    --data-type real
```

#### Override Parameter:
```bash
# Estimate kl instead of ksi (overrides config)
python scripts/train_inverse_v2.py \
    --config configs/birnn_inverse.yaml \
    --patient 3 \
    --param kl
```

#### Batch Inverse Training:
```bash
# Run on multiple patients
for patient in 3 5 7; do
    python scripts/train_inverse_v2.py \
        --config configs/birnn_inverse.yaml \
        --patient $patient
done
```

### Config File Format:

The config YAML must define training stages. Example `configs/birnn_inverse.yaml`:

```yaml
model_name: birnn
mode: inverse
inverse_param: ksi

training:
  stages:
    - name: "stage1_params_only"
      epochs: 1000
      learning_rate: 0.01
      train_inverse_params: true
      train_nn_weights: false
      loss_weights: [8.0, 4.82, 0.53]
    
    - name: "stage2_nn_only"
      epochs: 1000
      learning_rate: 0.001
      train_inverse_params: false
      train_nn_weights: true
      loss_weights: [8.0, 4.82, 0.53]
    
    - name: "stage3_joint"
      epochs: 1000
      learning_rate: 0.0001
      train_inverse_params: true
      train_nn_weights: true
      loss_weights: [8.0, 4.82, 0.53]
```

**Total training:** 1000 + 1000 + 1000 = 3000 epochs across 3 stages

### Output Structure:

```
results/birnn_inverse/Pat3_ksi_20241127_143022/
├── checkpoints/
│   ├── best/
│   ├── final/
│   └── stage1_complete/         # Checkpoint after each stage
├── predictions.npz
├── parameter_evolution_ksi.npz  # Parameter values over training
└── plots/
    ├── glucose_prediction.png
    ├── parameter_evolution_ksi.png
    └── ...
```

### Synthetic vs Real Patients:

| Patient Type | Can Estimate? | Ground Truth? | Output |
|--------------|---------------|---------------|--------|
| **Synthetic (2-11)** | ✅ Yes | ✅ Yes | Estimated value + Error % |
| **Real (1-15)** | ✅ Yes | ❌ No | Estimated value only |

**For real patients:** You get physiologically-informed parameter estimates, but can't compute error percentage. Still valuable for:
- Clinical personalization
- Cross-patient comparison
- Model improvement
- Research/thesis appendix

### Notes:

- **No `--epochs` flag** - epochs are defined in config file per stage
- **Flexible stages** - Can use 2, 3, 4, or more stages
- **Stage checkpoints** - Saves after each stage completes
- **Parameter tracking** - Full evolution history saved

---

## 3. visualize_v2.py

**Purpose:** Fast visualization from saved predictions (no model reload required).

### Arguments:

| Argument | Required | Type | Default | Description |
|----------|----------|------|---------|-------------|
| `--results-dir` | ✅ Yes | str | - | Path to results directory containing predictions.npz |
| `--output-dir` | ❌ No | str | `results_dir/plots` | Custom output directory for plots |

### Usage Examples:

#### Basic Visualization:
```bash
# Visualize specific experiment
python scripts/visualize_v2.py \
    --results-dir results/birnn_forward/Pat3_20241127_143022
```

#### Using Wildcards (Latest Result):
```bash
# Use wildcard to grab latest patient 3 result
python scripts/visualize_v2.py \
    --results-dir results/birnn_forward/Pat3_*
```

#### Custom Output Location:
```bash
# Save plots to custom directory
python scripts/visualize_v2.py \
    --results-dir results/birnn_forward/Pat3_* \
    --output-dir my_plots/experiment1
```

#### Visualize Inverse Training Results:
```bash
# Includes parameter evolution plots
python scripts/visualize_v2.py \
    --results-dir results/birnn_inverse/Pat3_ksi_*
```

### Generated Plots:

**For Forward Training:**
- `glucose_prediction.png` - Predicted vs true glucose
- `latent_variables.png` - Insulin and digestion (if synthetic)
- `loss_curves.png` - Training/test loss over epochs
- `residuals.png` - Prediction residuals

**For Inverse Training (additional):**
- `parameter_evolution_ksi.png` - Parameter convergence
- `parameter_error.png` - Error over training (if synthetic)

### Notes:

- ⚡ **Very fast** - Loads from predictions.npz (no model reload)
- 📊 Uses your existing `ExperimentPlotter` class
- 🎨 Publication quality (600 DPI)
- 🔄 Can re-run visualization without retraining

---

## 4. evaluate_v2.py

**Purpose:** Comprehensive evaluation with metrics computation and CSV export.

### Arguments:

| Argument | Required | Type | Default | Description |
|----------|----------|------|---------|-------------|
| `--results-dir` | ⚠️ * | str | - | Single results directory to evaluate |
| `--batch` | ⚠️ * | str | - | Batch pattern (e.g., `"results/birnn_forward/Pat*"`) |
| `--k-step` | ❌ No | int | `None` | Optional k-step ahead evaluation |
| `--output` | ❌ No | str | Auto-generated | Output CSV filename |

⚠️ *Must provide either `--results-dir` OR `--batch` (not both)

### Usage Examples:

#### Single Experiment Evaluation:
```bash
# Evaluate one experiment
python scripts/evaluate_v2.py \
    --results-dir results/birnn_forward/Pat3_20241127_143022 \
    --output metrics.csv
```

#### Using Wildcards:
```bash
# Evaluate latest patient 3 result
python scripts/evaluate_v2.py \
    --results-dir results/birnn_forward/Pat3_* \
    --output pat3_metrics.csv
```

#### Batch Evaluation:
```bash
# Evaluate all patient results
python scripts/evaluate_v2.py \
    --batch "results/birnn_forward/Pat*" \
    --output all_patients.csv
```

#### Compare All Models:
```bash
# Evaluate BI-RNN, PINN, and Modified-MLP
python scripts/evaluate_v2.py \
    --batch "results/*/Pat3_*" \
    --output model_comparison.csv
```

#### Inverse Training Evaluation:
```bash
# Evaluate parameter estimation results
python scripts/evaluate_v2.py \
    --batch "results/birnn_inverse/Pat*" \
    --output inverse_results.csv
```

#### With k-Step Evaluation:
```bash
# Add 15-step ahead prediction evaluation
python scripts/evaluate_v2.py \
    --results-dir results/birnn_forward/Pat3_* \
    --k-step 15 \
    --output metrics_with_kstep.csv
```

### Output CSV Format:

**For Forward Training:**
```csv
results_dir,model,patient,rmse,mae,mape,rmse_train,rmse_test,insulin_rmse,digestion_rmse
results/birnn_forward/Pat3_*,birnn,3,25.43,18.21,12.5,22.1,28.7,0.0034,0.0012
```

**For Inverse Training (additional columns):**
```csv
...,ksi_estimated,ksi_true,ksi_error_percent
...,271.50,274.00,0.91
```

### Computed Metrics:

**Glucose Metrics:**
- `rmse` - Root mean squared error (mg/dL)
- `mae` - Mean absolute error (mg/dL)
- `mape` - Mean absolute percentage error (%)
- `rmse_train` - Training set RMSE
- `rmse_test` - Test set RMSE

**Latent State Metrics (if available):**
- `insulin_rmse` - Insulin RMSE (U/dL)
- `insulin_mae` - Insulin MAE
- `digestion_rmse` - Digestion RMSE (mg/dL/min)
- `digestion_mae` - Digestion MAE

**Parameter Estimation Metrics (inverse mode):**
- `{param}_estimated` - Final estimated value
- `{param}_true` - True value (synthetic only)
- `{param}_error_percent` - Estimation error (synthetic only)

### Notes:

- 📊 Works with both forward and inverse training results
- 📈 Automatically detects inverse mode and includes parameter metrics
- 🔢 Batch mode creates one CSV with all results
- ⚡ Fast - loads from predictions.npz (no model reload)
- 📝 Import CSV to pandas/Excel for further analysis

---

## 🔄 Typical Workflows

### Workflow 1: Single Experiment
```bash
# 1. Train
python scripts/train_forward_v2.py --model birnn --patient 3 --epochs 2000

# 2. Already visualized during training, but can regenerate:
python scripts/visualize_v2.py --results-dir results/birnn_forward/Pat3_*

# 3. Get metrics
python scripts/evaluate_v2.py --results-dir results/birnn_forward/Pat3_* --output metrics.csv
```

### Workflow 2: Model Comparison Study
```bash
# Train all models
python scripts/train_forward_v2.py --model birnn --patient 3 --epochs 2000
python scripts/train_forward_v2.py --model pinn --patient 3 --epochs 2000
python scripts/train_forward_v2.py --model modified_mlp --patient 3 --epochs 2000

# Compare
python scripts/evaluate_v2.py --batch "results/*/Pat3_*" --output comparison.csv
```

### Workflow 3: Inverse Parameter Estimation
```bash
# Estimate parameters
python scripts/train_inverse_v2.py --config configs/birnn_inverse.yaml --patient 3

# Visualize parameter evolution
python scripts/visualize_v2.py --results-dir results/birnn_inverse/Pat3_ksi_*

# Get results
python scripts/evaluate_v2.py --results-dir results/birnn_inverse/Pat3_ksi_* --output inverse_metrics.csv
```

### Workflow 4: Systematic Patient Study
```bash
# Train on all synthetic patients
for patient in {2..11}; do
    python scripts/train_forward_v2.py --model birnn --patient $patient --epochs 2000
done

# Batch evaluate
python scripts/evaluate_v2.py --batch "results/birnn_forward/Pat*" --output all_patients.csv
```

### Workflow 5: Real Patient Analysis (Thesis Appendix)
```bash
# Train inverse on real patients
for patient in {1..15}; do
    python scripts/train_inverse_v2.py \
        --config configs/birnn_inverse.yaml \
        --patient $patient \
        --data-type real
done

# Evaluate (gets estimated parameters, no error since no ground truth)
python scripts/evaluate_v2.py \
    --batch "results/birnn_inverse/RealPat*" \
    --output real_patient_estimates.csv
```

---

## 🔍 Quick Reference

### Training Commands:
```bash
# Forward (synthetic)
python scripts/train_forward_v2.py --model birnn --patient 3 --epochs 2000

# Forward (real)
python scripts/train_forward_v2.py --model birnn --patient 5 --data-type real --epochs 1000

# Inverse (synthetic - with error calculation)
python scripts/train_inverse_v2.py --config configs/birnn_inverse.yaml --patient 3

# Inverse (real - estimates only, for appendix)
python scripts/train_inverse_v2.py --config configs/birnn_inverse.yaml --patient 5 --data-type real
```

### Analysis Commands:
```bash
# Visualize
python scripts/visualize_v2.py --results-dir results/birnn_forward/Pat3_*

# Evaluate single
python scripts/evaluate_v2.py --results-dir results/birnn_forward/Pat3_* --output metrics.csv

# Evaluate batch
python scripts/evaluate_v2.py --batch "results/birnn_forward/Pat*" --output all_metrics.csv
```

---

## 📊 Data Summary

### Synthetic Patients (2-11):
- Location: `data/synthetic/`
- Ground truth: ✅ All (glucose, insulin, digestion, parameters)
- Use for: Forward training, inverse training with error calculation

### Real Patients (1-15):
- Location: `data/processed/`
- Ground truth: ⚠️ Glucose only
- Use for: Forward training, inverse training (estimates only)

---

## ⚠️ Important Notes

1. **Inverse training on real patients:** 
   - Won't compute error percentage (no ground truth)
   - Still provides estimated parameter values
   - Valuable for thesis appendix, clinical validation

2. **TensorFlow environment:**
   - All scripts handle TF1.x/2.x compatibility automatically
   - No conflicts between BI-RNN and DeepXDE models

3. **Resume capability:**
   - Only available for forward training
   - Inverse training runs all stages sequentially

4. **Visualization speed:**
   - Uses saved predictions.npz (instant reload)
   - Can regenerate plots without retraining

5. **Batch evaluation:**
   - Creates single CSV with all results
   - Easy import to pandas/Excel for analysis

---

## 🆘 Troubleshooting

### "ModuleNotFoundError: No module named 'src'"
```bash
# Make sure you're in project root
cd /path/to/your/project
python scripts/train_forward_v2.py ...
```

### "Predictions file not found"
```bash
# Train first, then visualize/evaluate
python scripts/train_forward_v2.py --model birnn --patient 3 --epochs 500
python scripts/visualize_v2.py --results-dir results/birnn_forward/Pat3_*
```

### "Config must have training.stages defined"
```bash
# Create inverse config file first (see config format above)
# Or ask for help creating one!
```

### Scripts won't execute
```bash
# Make executable
chmod +x scripts/*.py
```

---

## 📚 Additional Resources

- **MASTER_DOWNLOAD_GUIDE.md** - Installation and setup
- **TEST_RESULTS.md** - Validation and testing
- **FINAL_SUMMARY.md** - Complete overview

---

**Ready to start validation!** 🚀
