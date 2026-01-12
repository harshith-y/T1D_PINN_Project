# CLAUDE.md - T1D PINN Project

## Project Overview
Physics-Informed Neural Networks for Type 1 Diabetes glucose prediction and physiological parameter estimation. Three model architectures using the Magdelaine glucose-insulin-carbohydrate model. Goal: Build CV-worthy production code + generate publication results.

## Current Priority: AWS Deployment & Training

### AWS Status - READY TO DEPLOY
- **GPU Quota:** ✅ Approved (4 vCPUs for G instances in eu-west-2)
- **Credits:** ✅ $120 available (covers ~240 hours of g4dn.xlarge)
- **Region:** eu-west-2 (London)
- **Instance:** g4dn.xlarge (1x NVIDIA T4 GPU)

### Where We Left Off
1. ✅ Local optimization pipeline tested and working
2. ✅ Terraform files created/configured
3. ⏳ **NEXT:** Create EC2 instance and run full optimization pipeline
4. ⏳ Set up MLflow tracking for experiments

### What Needs to Happen
1. Deploy EC2 g4dn.xlarge instance via Terraform
2. SSH into instance, clone repo, set up environment
3. Run full training pipeline (3 architectures × 10 patients × 5 runs)
4. Track experiments with MLflow
5. Download results for publication

## Quick Commands
```bash
# Activate environment
source T1D_PINN_venv/bin/activate

# Testing
pytest tests/ -v

# Linting
black src/ scripts/ tests/
isort src/ scripts/ tests/

# Training
python scripts/train_forward.py --config configs/birnn.yaml --patient Pat2
python scripts/train_inverse.py --config configs/birnn.yaml --patient Pat2

# Terraform (from infrastructure/ or terraform/ directory)
terraform init
terraform plan
terraform apply
terraform destroy  # IMPORTANT: Run when done to stop charges
```

## Project Structure
```
T1D_PINN_Project/
├── src/
│   ├── models/
│   │   ├── birnn.py              # BI-RNN (TF2.x, eager execution)
│   │   ├── pinn_feedforward.py   # DeepXDE PINN (TF1.x compat)
│   │   └── modified_mlp.py       # Modified MLP (TF1.x compat)
│   ├── datasets/
│   ├── physics/
│   ├── training/
│   └── visualization/
├── scripts/                      # Entry point scripts
├── tests/
├── configs/                      # YAML model configs
├── data/
│   ├── synthetic/                # Pat2-Pat11 (safe for cloud)
│   └── processed/                # CONFIDENTIAL - DO NOT upload to cloud
├── terraform/ or infrastructure/ # AWS Terraform configs
└── .github/workflows/ci-test.yml
```

## Critical Technical Notes

### TensorFlow Version Conflicts
```python
# DeepXDE models (PINN, Modified-MLP):
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# BI-RNN uses TF2.x eager execution
# Cannot mix modes in same Python session
```

### Key Bug Fixes Applied
- **Timestep scaling:** Use `dt = 1.0` (physical time), not `dt = 1.0 / m_t`
- **Normalization:** BI-RNN targets must be normalized to [0,1]
- **Memory:** Real patient data auto-limited to 48 hours

### Best Results (Local)
- BI-RNN inverse training: **3.82% parameter estimation error**
- PINN inverse training: **6.66% mean relative error**
- 3-stage training: inverse params → NN weights → joint optimization

## Data Rules - IMPORTANT
- `data/synthetic/` (Pat2-Pat11): ✅ Safe for cloud training
- `data/processed/`: ❌ **CONFIDENTIAL** - NEVER upload to AWS/cloud

## Model Architectures

| Model | Framework | TF Mode | Best For |
|-------|-----------|---------|----------|
| BI-RNN | TensorFlow 2.x | Eager | Best accuracy, inverse training |
| PINN | DeepXDE | Graph (TF1.x) | Best parameter recovery |
| Modified-MLP | DeepXDE | Graph (TF1.x) | Real-world CGM flexibility |

## AWS Deployment Notes
- Use **only synthetic data** for cloud training (Pat2-Pat11)
- Instance: g4dn.xlarge (~$0.50/hr)
- Region: eu-west-2 (London)
- Remember: `terraform destroy` when done to stop charges
- MLflow for experiment tracking