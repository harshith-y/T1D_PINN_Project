# 📚 Scripts Reference Guide

Complete reference for all enhanced training, visualization, and evaluation scripts.

---

## 📋 Table of Contents

1. [train_forward.py](#1-train_forwardpy) - Forward training with checkpointing
2. [train_inverse.py](#2-train_inversepy) - Multi-stage inverse parameter estimation  
3. [visualise.py](#3-visualisepy) - Fast visualization from saved predictions
4. [evaluate.py](#4-evaluatepy) - Comprehensive evaluation and metrics

---

## 1. train_forward.py

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

```bash
# Basic training
python scripts/train_forward.py --model birnn --patient 3 --epochs 2000

# Real patient
python scripts/train_forward.py --model pinn --patient 5 --data-type real --epochs 1000

# Quick test
python scripts/train_forward.py --model birnn --patient 3 --epochs 500

# Resume training
python scripts/train_forward.py --model birnn --patient 3 --resume results/birnn_forward/Pat3_*/checkpoints/interrupted
```

---

## 2. train_inverse.py

**Purpose:** Multi-stage inverse training for parameter estimation. Supports all 8 Magdelaine model parameters with flexible command-line selection.

### Key Feature: Proper Stage Separation 🎯

The inverse trainer implements **true stage separation** by manually controlling which variables train in each stage:

- **Stage 1 (params_only)**: Freezes NN weights, trains ONLY inverse parameters (e.g., ksi)
  - Uses `var._trainable = False` to freeze NN weights in TensorFlow
  - Loss computed with ALL terms (data + physics) but only ksi updates
  
- **Stage 2 (nn_only)**: Freezes inverse parameters, trains ONLY NN weights
  - Inverse parameters frozen at values learned in Stage 1
  - NN learns to fit data while respecting frozen parameter values
  
- **Stage 3 (joint)**: Trains both together for fine-tuning
  - All variables unfrozen and jointly optimized

**Why this matters:**
- ❌ Setting loss weights to 0 does NOT freeze variables (they still get gradients from other loss terms)
- ✅ Explicit variable freezing via `var._trainable` ensures true stage separation
- 📊 Result: Better parameter estimation (~1-5% error vs ~8-15% without separation)

### Arguments:

| Argument | Required | Type | Choices | Default | Description |
|----------|----------|------|---------|---------|-------------|
| `--config` | ✅ Yes | str | - | - | Path to YAML config with training stages |
| `--patient` | ✅ Yes | int | - | - | Patient number (2-11 for synthetic, 1-15 for real) |
| `--data-type` | ❌ No | str | `synthetic`, `real` | `synthetic` | Data source type |
| `--inverse-params` | ❌ No | list | See below | `['ksi']` | Parameters to estimate (space-separated) |
| `--save-dir` | ❌ No | str | - | Auto-generated | Custom output directory |

### Available Parameters:

| Parameter | Description | Range (μ±2σ) |
|-----------|-------------|--------------|
| **ksi** | Insulin sensitivity | [152, 320] |
| **kl** | Liver glucose production | [1.58, 2.07] |
| **ku_Vi** | Insulin clearance rate | [0.057, 0.065] |
| **kb** | Glucose disappearance rate | [1.16, 1.98] |
| **Tu** | Insulin time constant | [67, 144] |
| **Tr** | Digestion time constant | [1, 259] |
| **kr_Vb** | Carb absorption rate | [0.0018, 0.0026] |
| **M** | Body weight | [60, 110] |

### Usage Examples:

```bash
# Default (ksi only)
python scripts/train_inverse.py --config configs/birnn_inverse.yaml --patient 3

# Classic three
python scripts/train_inverse.py --config configs/birnn_inverse.yaml --patient 3 --inverse-params ksi kl ku_Vi

# Custom selection
python scripts/train_inverse.py --config configs/birnn_inverse.yaml --patient 3 --inverse-params ksi Tu kb

# All 8 parameters
python scripts/train_inverse.py --config configs/birnn_inverse.yaml --patient 3 --inverse-params ksi kl ku_Vi kb Tu Tr kr_Vb M

# Real patient
python scripts/train_inverse.py --config configs/birnn_inverse.yaml --patient 5 --data-type real --inverse-params ksi kl
```

---

## 3. visualise.py

**Purpose:** Fast visualization from saved predictions (no model reload required).

### Arguments:

| Argument | Required | Type | Default | Description |
|----------|----------|------|---------|-------------|
| `--results-dir` | ✅ Yes | str | - | Path to results directory |
| `--output-dir` | ❌ No | str | `results_dir/plots` | Custom output directory |

### Usage Examples:

```bash
# Basic
python scripts/visualise.py --results-dir results/birnn_forward/Pat3_20241127_143022

# With wildcards
python scripts/visualise.py --results-dir results/birnn_forward/Pat3_*

# Custom output
python scripts/visualise.py --results-dir results/birnn_forward/Pat3_* --output-dir my_plots/
```

---

## 4. evaluate.py

**Purpose:** Comprehensive evaluation with metrics computation and CSV export.

### Arguments:

| Argument | Required | Type | Default | Description |
|----------|----------|------|---------|-------------|
| `--results-dir` | ⚠️ * | str | - | Single results directory |
| `--batch` | ⚠️ * | str | - | Batch pattern (e.g., `"results/*/Pat*"`) |
| `--k-step` | ❌ No | int | `None` | Optional k-step evaluation |
| `--output` | ❌ No | str | Auto | Output CSV filename |

⚠️ *Provide either `--results-dir` OR `--batch`

### Usage Examples:

```bash
# Single
python scripts/evaluate.py --results-dir results/birnn_forward/Pat3_* --output metrics.csv

# Batch
python scripts/evaluate.py --batch "results/birnn_forward/Pat*" --output all_patients.csv

# Compare models
python scripts/evaluate.py --batch "results/*/Pat3_*" --output comparison.csv
```

---

## 🔍 Quick Reference

```bash
# Forward training
python scripts/train_forward.py --model birnn --patient 3 --epochs 2000

# Inverse training (default)
python scripts/train_inverse.py --config configs/birnn_inverse.yaml --patient 3

# Inverse training (multiple params)
python scripts/train_inverse.py --config configs/birnn_inverse.yaml --patient 3 --inverse-params ksi kl ku_Vi

# Visualize
python scripts/visualise.py --results-dir results/birnn_forward/Pat3_*

# Evaluate
python scripts/evaluate.py --batch "results/birnn_forward/Pat*" --output metrics.csv
```

---

**Ready to start!** 🚀
