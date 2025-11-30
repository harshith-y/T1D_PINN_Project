# 🎯 T1D PINN MLOps Production Stack

## 📋 Overview

Complete production-grade MLOps infrastructure for Type 1 Diabetes glucose prediction and parameter estimation using Physics-Informed Neural Networks.

**Inspired by:** J0MT/AI_Drug_Discovery repository structure  
**Adapted for:** Medical AI with safety-critical glucose prediction  

---

## 🏗️ Project Structure

```
T1D_PINN_Project/
├── .github/
│   └── workflows/
│       ├── ci-test.yml              # Continuous integration testing
│       ├── train-models.yml         # Automated model training
│       └── deploy-mlflow.yml        # MLflow deployment
│
├── terraform/
│   ├── main.tf                      # AWS infrastructure definition
│   ├── variables.tf                 # Configuration variables
│   ├── outputs.tf                   # Infrastructure outputs
│   └── modules/
│       ├── ec2/                     # EC2 instance configuration
│       ├── s3/                      # S3 bucket for results
│       └── iam/                     # IAM roles and policies
│
├── docker/
│   ├── Dockerfile                   # Main training container
│   ├── docker-compose.yml           # Multi-service orchestration
│   ├── mlflow/
│   │   └── Dockerfile              # MLflow tracking server
│   └── airflow/
│       └── Dockerfile              # Airflow orchestration
│
├── airflow/
│   └── dags/
│       ├── comprehensive_training.py    # Batch experiment orchestration
│       ├── model_comparison.py          # Cross-model evaluation
│       └── real_patient_pipeline.py     # Real data processing
│
├── mlflow/
│   ├── mlflow_config.py            # MLflow integration utilities
│   └── model_registry.py           # Model versioning logic
│
├── dvc/
│   ├── .dvc/                       # DVC configuration
│   ├── .dvcignore                  # Files to ignore
│   └── data.dvc                    # Data version tracking
│
├── src/
│   ├── models/                     # Model architectures (existing)
│   ├── datasets/                   # Data loaders (existing)
│   ├── training/                   # Training logic (existing)
│   ├── physics/                    # Physics constraints (existing)
│   └── utils/                      # Utilities (existing)
│
├── scripts/
│   ├── train_forward.py            # Forward training (existing)
│   ├── train_inverse.py            # Inverse training (existing)
│   ├── visualise.py                # Visualization (existing)
│   ├── evaluate.py                 # Evaluation (existing)
│   └── batch_experiments.py        # NEW: Batch runner
│
├── tests/
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   └── conftest.py                 # Pytest configuration
│
├── configs/
│   ├── birnn_forward.yaml          # Existing configs
│   ├── pinn_inverse.yaml           # Existing configs
│   └── experiment_matrix.yaml      # NEW: Batch experiment config
│
├── data/                           # (Tracked by DVC)
│   ├── synthetic/                  # Synthetic patient data
│   └── processed/                  # Real patient data
│
├── results/                        # (Backed up to S3)
│   └── [experiment outputs]
│
├── docs/
│   ├── SETUP_GUIDE.md              # Infrastructure setup
│   ├── MLOPS_WORKFLOW.md           # Usage guide
│   └── API_REFERENCE.md            # Code documentation
│
├── .gitignore                      # Git ignore patterns
├── .dockerignore                   # Docker ignore patterns
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation
├── pytest.ini                      # Pytest configuration
├── .flake8                         # Linting configuration
├── .pre-commit-config.yaml         # Pre-commit hooks
└── README.md                       # Project documentation
```

---

## 🛠️ Technology Stack

### ML & Data
- **PyTorch**: Deep learning framework (BI-RNN, PINN, Modified-MLP)
- **DeepXDE**: Physics-Informed Neural Networks
- **MLflow**: Experiment tracking, model registry, artifact management
- **DVC**: Data versioning with S3 backend
- **Pandas, NumPy**: Data processing and feature engineering

### Infrastructure & DevOps
- **AWS EC2 + Terraform**: Infrastructure as Code deployment
- **Docker + Docker Compose**: Containerized services with network orchestration
- **GitHub Container Registry**: Private image storage and distribution
- **GitHub Actions**: CI/CD with automated model training triggers
- **AWS S3**: Results storage and DVC remote

### Orchestration & Workflow
- **Apache Airflow**: DAG-based experiment orchestration
- **tmux/screen**: Multi-experiment parallelization

### Development & Quality
- **pytest**: Comprehensive unit and integration testing
- **black, flake8**: Code formatting and linting
- **Type hints**: Static type checking for reliability
- **YAML configs**: Declarative hyperparameter management

---

## 🚀 Quick Start

### Prerequisites
- AWS Account (free tier eligible)
- Docker Desktop installed
- Git configured
- Python 3.9+

### 1. Clone and Setup
```bash
git clone https://github.com/YOUR_USERNAME/T1D_PINN_Project.git
cd T1D_PINN_Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Configure AWS Credentials
```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Enter region: eu-west-2 (or your preferred region)
```

### 3. Build Docker Images
```bash
docker-compose build
```

### 4. Deploy Infrastructure
```bash
cd terraform
terraform init
terraform plan
terraform apply
```

### 5. Start Services
```bash
docker-compose up -d
# MLflow UI: http://localhost:5000
# Airflow UI: http://localhost:8080
```

### 6. Run First Experiment
```bash
# Local test
python scripts/train_inverse.py --config configs/pinn_inverse.yaml --patient 3

# Via Docker
docker-compose run training python scripts/train_inverse.py --config configs/pinn_inverse.yaml --patient 3

# Via Airflow (batch)
# Navigate to http://localhost:8080 and trigger DAG
```

---

## 📊 Experiment Workflows

### Single Experiment
```bash
# Local execution
python scripts/train_inverse.py \
    --config configs/pinn_inverse.yaml \
    --patient 3 \
    --inverse-params ksi
```

### Batch Experiments (All Models × All Patients)
```bash
# Via Airflow DAG
airflow dags trigger comprehensive_training

# Or via script
python scripts/batch_experiments.py \
    --models birnn pinn modified_mlp \
    --patients 2 3 4 5 6 7 8 9 10 11 \
    --parallel 5
```

### View Results
```bash
# MLflow UI
open http://localhost:5000

# Or query programmatically
python -c "
import mlflow
client = mlflow.tracking.MlflowClient()
experiments = client.list_experiments()
for exp in experiments:
    print(f'{exp.name}: {exp.experiment_id}')
"
```

---

## 🏥 Medical AI Considerations

### Safety & Validation
- ✅ Comprehensive testing on synthetic data before real patients
- ✅ Parameter estimation error tracking (<5% target)
- ✅ Prediction RMSE monitoring (<5 mg/dL target)
- ✅ Physics constraint validation (ODE residuals)

### Reproducibility
- ✅ All experiments tracked in MLflow
- ✅ Data versioned with DVC
- ✅ Infrastructure versioned with Terraform
- ✅ Container images tagged and stored

### Compliance
- ✅ No PHI in version control
- ✅ Encrypted data storage (S3 + encryption)
- ✅ Audit trail via MLflow
- ✅ Reproducible pipelines for regulatory review

---

## 📈 Results Organization

### Experiment Naming Convention
```
{model}_{mode}_{patient}_ksi_{timestamp}/
```

Example:
```
pinn_inverse_Pat3_ksi_20251129_143022/
├── checkpoints/
├── predictions.npz
├── parameter_evolution_ksi.npz
└── metadata.json
```

### MLflow Organization
```
Experiments/
├── BIRNN_Forward/
├── BIRNN_Inverse/
├── PINN_Forward/
├── PINN_Inverse/
├── ModifiedMLP_Forward/
└── ModifiedMLP_Inverse/
```

---

## 🎯 Next Steps

1. **Week 1**: Set up infrastructure (Docker, Terraform, GitHub Actions)
2. **Week 2**: Integrate MLflow tracking into existing code
3. **Week 3**: Build Airflow DAGs for batch experiments
4. **Week 4**: Run comprehensive evaluation (all models × all patients)
5. **Week 5**: Collect results, generate visualizations, write paper

---

## 📚 Documentation

- **[SETUP_GUIDE.md](docs/SETUP_GUIDE.md)** - Detailed setup instructions
- **[MLOPS_WORKFLOW.md](docs/MLOPS_WORKFLOW.md)** - Day-to-day usage
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Common issues
- **[API_REFERENCE.md](docs/API_REFERENCE.md)** - Code documentation

---

## 🤝 Contributing

This is a research project for Type 1 Diabetes closed-loop control. Contributions welcome!

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **MLOps Structure**: Inspired by J0MT/AI_Drug_Discovery
- **Physics Model**: Based on Magdelaine et al. 2015
- **Research Group**: Imperial College London, Biomedical Engineering

---

## 📧 Contact

**Harsh [Your Last Name]**  
Imperial College London  
Email: [your-email]@imperial.ac.uk  
GitHub: @[your-github-username]

---

**Status**: 🚧 In Development | 📊 Paper in Progress | 🎯 Production-Ready MLOps
