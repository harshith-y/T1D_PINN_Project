# Physics-Informed Neural Networks for Type 1 Diabetes Glucose Prediction

## Overview
This project implements physics-informed neural networks (PINNs) for predicting blood glucose levels in Type 1 diabetes patients, incorporating the Magdelaine physiological model.

## Models
- **Feedforward PINN**: Basic physics-informed architecture
- **Modified-MLP PINN**: Enhanced architecture from research paper
- **BI-RNN**: Biologically-informed recurrent neural network

## Project Structure
```
T1D_PINN_Project/
├── data/                      # Data directory (not in repo)
│   ├── raw/                   # Raw health data exports
│   ├── real/                  # Extracted real patient data
│   ├── processed/             # Prepared training data
│   └── synthetic/             # Simulated patient data
├── src/
│   ├── datasets/              # Data loading and processing
│   │   ├── simulator.py       # Synthetic data generator
│   │   ├── loader.py          # Data loader
│   │   ├── preprocessing.py   # Data preprocessing
│   │   └── windowing.py       # Train/val/test splitting
│   ├── physics/               # Physics models
│   │   └── magdelaine.py      # Magdelaine physiological model
│   ├── models/                # Neural network architectures
│   │   ├── pinn.py            # PINN implementations
│   │   ├── birnn.py           # BI-RNN implementation
│   │   └── combined.py        # Combined model
│   └── training/              # Training infrastructure
│       ├── losses.py          # Loss functions
│       └── trainer.py         # Training loop
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── extract_real_data.py
│   ├── prepare_data.py
│   └── train_simple_pinn.py
├── tests/
│   └── test_data_loading.py
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites
- Python 3.8+
- TensorFlow 2.10+
- DeepXDE

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/T1D_PINN_Project.git
cd T1D_PINN_Project

# Create virtual environment
python -m venv T1D_PINN_venv
source T1D_PINN_venv/bin/activate  # On Windows: T1D_PINN_venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Generate Synthetic Data
```bash
python scripts/generate_synthetic_data.py
```

### Process Real Patient Data
```bash
# Extract from raw data
python scripts/extract_real_data.py --verbose

# Prepare for training
python scripts/prepare_data.py --verbose
```

### Train Models
```bash
# Train PINN on synthetic data
python scripts/train_simple_pinn.py --patient Pat3 --epochs 5000

# Train on real data
python scripts/train_simple_pinn.py --patient RealPat1 --source processed --epochs 5000
```

## Data
Data files are not included in this repository. To use:
1. Place raw data in `data/raw/`
2. Run extraction scripts to generate training data
3. Or generate synthetic data using the simulator

## Testing
```bash
python tests/test_data_loading.py
```

## Citation
If you use this code, please cite:
```
[Your paper citation here]
```

## License
[Your license here]

## Contact
[Your contact information]
