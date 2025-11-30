# =============================================================================
# T1D PINN MLOps Complete Setup Script
# Run this to create entire project structure
# =============================================================================

## 🎯 STEP 1: CREATE PROJECT STRUCTURE
## Execute these commands in your terminal

# Navigate to your project root
cd /path/to/T1D_PINN_Project

# =============================================================================
# Create all directories
# =============================================================================
mkdir -p .github/workflows
mkdir -p terraform
mkdir -p docker
mkdir -p airflow/dags
mkdir -p airflow/logs
mkdir -p airflow/plugins
mkdir -p mlflow
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p scripts/setup
mkdir -p docs
mkdir -p .dvc

# Create __init__.py files for tests
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py

# =============================================================================
# Create configuration files
# =============================================================================

# .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
ENV/
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
results/
mlruns/
mlflow-artifacts/
data/synthetic/*.csv
data/processed/*.csv
*.npz
*.ckpt

# DVC
/data/synthetic
/data/processed
.dvc/tmp
.dvc/cache

# Terraform
terraform/.terraform/
terraform/*.tfstate
terraform/*.tfstate.backup
terraform/.terraform.lock.hcl

# Docker
*.log

# Secrets
*.pem
*.key
.env
secrets/
EOF

# .dockerignore
cat > .dockerignore << 'EOF'
__pycache__
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info
dist
build
.git
.github
.vscode
.idea
*.md
results/
mlruns/
.dvc/
terraform/
airflow/logs/
*.log
EOF

# .flake8
cat > .flake8 << 'EOF'
[flake8]
max-line-length = 127
exclude = 
    .git,
    __pycache__,
    build,
    dist,
    venv,
    .eggs,
    *.egg,
per-file-ignores =
    __init__.py:F401
ignore = 
    E203,
    W503,
    E501
EOF

# pytest.ini
cat > pytest.ini << 'EOF'
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
EOF

# =============================================================================
# STEP 2: DOWNLOAD FILES FROM CLAUDE
# =============================================================================

echo "
========================================
📥 DOWNLOAD THESE FILES:
========================================

From Claude's outputs, download and place in correct locations:

1. ROOT DIRECTORY:
   - Dockerfile                        → ./Dockerfile
   - docker-compose.yml                → ./docker-compose.yml

2. GITHUB WORKFLOWS (.github/workflows/):
   - ci-test.yml                       → ./.github/workflows/ci-test.yml
   - train-models.yml                  → ./.github/workflows/train-models.yml

3. TERRAFORM (terraform/):
   - terraform_main.tf                 → ./terraform/main.tf
   - terraform_variables.tf            → ./terraform/variables.tf
   - terraform_user_data.sh            → ./terraform/user_data.sh

4. DOCUMENTATION (docs/):
   - PROJECT_STRUCTURE.md              → ./docs/PROJECT_STRUCTURE.md
   - DEPLOYMENT_GUIDE.md               → ./docs/DEPLOYMENT_GUIDE.md

5. EXISTING CODE UPDATES:
   - inverse_trainer_WITH_CALLBACKS_v4_FINAL.py  → ./src/training/inverse_trainer.py
   - train_inverse_FIXED.py                      → ./scripts/train_inverse.py
   - SCRIPTS_README_UPDATED.md                   → ./docs/SCRIPTS_README.md
"

# =============================================================================
# STEP 3: CREATE TERRAFORM TFVARS
# =============================================================================

cat > terraform/terraform.tfvars << 'EOF'
# =============================================================================
# T1D PINN Terraform Variables
# EDIT THESE VALUES before running terraform apply
# =============================================================================

aws_region   = "eu-west-2"  # London
environment  = "dev"
owner_email  = "your-email@imperial.ac.uk"  # CHANGE THIS

# Instance configuration
instance_type     = "t3.medium"     # Change to "g4dn.xlarge" for GPU
root_volume_size  = 100             # GB
use_elastic_ip    = true

# SSH key (must create in AWS Console first!)
key_name = "t1d-pinn-key"  # CHANGE THIS to your key name

# Security (CHANGE THIS to your IP for better security)
allowed_ssh_cidrs = ["0.0.0.0/0"]  # WARNING: Open to world! Change to ["YOUR_IP/32"]
EOF

# =============================================================================
# STEP 4: CREATE AIRFLOW DAG
# =============================================================================

cat > airflow/dags/comprehensive_training.py << 'EOF'
"""
Comprehensive T1D PINN Training DAG
Trains all models on all patients
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'harsh',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    't1d_comprehensive_training',
    default_args=default_args,
    description='Train all models on all patients',
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['t1d', 'training', 'comprehensive'],
)

# Configuration
MODELS = ['birnn', 'pinn', 'modified_mlp']
PATIENTS = list(range(2, 12))  # Pat2-11

# Create tasks for each combination
tasks = []

for model in MODELS:
    for patient in PATIENTS:
        task_id = f'train_{model}_pat{patient}'
        
        task = DockerOperator(
            task_id=task_id,
            image='t1d-pinn:latest',
            command=f'python scripts/train_inverse.py --config configs/{model}_inverse.yaml --patient {patient}',
            docker_url='unix://var/run/docker.sock',
            network_mode='t1d-network',
            auto_remove=True,
            mounts=[
                {
                    'source': '/home/ubuntu/T1D_PINN_Project/data',
                    'target': '/data',
                    'type': 'bind',
                    'read_only': True,
                },
                {
                    'source': '/home/ubuntu/T1D_PINN_Project/results',
                    'target': '/results',
                    'type': 'bind',
                },
            ],
            environment={
                'MLFLOW_TRACKING_URI': 'http://mlflow:5000',
            },
            dag=dag,
        )
        
        tasks.append(task)

# Optional: Set dependencies (e.g., run models sequentially per patient)
# for i in range(1, len(tasks)):
#     tasks[i].set_upstream(tasks[i-1])
EOF

# =============================================================================
# STEP 5: CREATE MLFLOW INTEGRATION
# =============================================================================

cat > mlflow/mlflow_config.py << 'EOF'
"""
MLflow integration utilities for T1D PINN project
"""

import mlflow
import os
from pathlib import Path
from typing import Dict, Any, Optional

class MLflowTracker:
    """Wrapper for MLflow experiment tracking"""
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None
    ):
        """
        Initialize MLflow tracker
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking URI (default: http://localhost:5000)
            run_name: Optional run name
        """
        self.tracking_uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Set or create experiment
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self.run_name = run_name
        
    def start_run(self, run_name: Optional[str] = None):
        """Start MLflow run"""
        return mlflow.start_run(run_name=run_name or self.run_name)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact"""
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log directory of artifacts"""
        mlflow.log_artifacts(local_dir, artifact_path)
    
    def log_model(self, model, artifact_path: str = "model"):
        """Log model"""
        # TODO: Implement based on model type
        pass
    
    def end_run(self):
        """End MLflow run"""
        mlflow.end_run()

def create_tracker(model_name: str, mode: str, patient: int) -> MLflowTracker:
    """
    Factory function to create MLflow tracker
    
    Args:
        model_name: Model architecture (birnn, pinn, modified_mlp)
        mode: Training mode (forward, inverse)
        patient: Patient number
    
    Returns:
        MLflowTracker instance
    """
    experiment_name = f"{model_name.upper()}_{mode.capitalize()}"
    run_name = f"{model_name}_Pat{patient}_{mode}"
    
    return MLflowTracker(
        experiment_name=experiment_name,
        run_name=run_name
    )
EOF

# =============================================================================
# STEP 6: CREATE SAMPLE TESTS
# =============================================================================

cat > tests/unit/test_models.py << 'EOF'
"""
Unit tests for model architectures
"""

import pytest
import tensorflow as tf
import torch

def test_tensorflow_import():
    """Test TensorFlow import"""
    assert tf.__version__ is not None

def test_pytorch_import():
    """Test PyTorch import"""
    assert torch.__version__ is not None

# TODO: Add actual model tests
EOF

cat > tests/integration/test_training.py << 'EOF'
"""
Integration tests for training pipeline
"""

import pytest

# TODO: Add integration tests
EOF

cat > tests/conftest.py << 'EOF'
"""
Pytest configuration and fixtures
"""

import pytest

@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'model_name': 'pinn',
        'patient': 3,
        'mode': 'inverse',
    }
EOF

# =============================================================================
# STEP 7: CREATE DVC CONFIGURATION
# =============================================================================

cat > .dvc/config << 'EOF'
[core]
    remote = storage
    autostage = true
['remote "storage"']
    url = s3://t1d-pinn-data/dvc-storage
EOF

cat > .dvcignore << 'EOF'
# Add patterns of files dvc should ignore
*.pyc
__pycache__
.git
EOF

# =============================================================================
# STEP 8: CREATE SETUP SCRIPTS
# =============================================================================

cat > scripts/setup/install_dependencies.sh << 'EOF'
#!/bin/bash
# Install all dependencies on Ubuntu

set -e

echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3.9 \
    python3.9-venv \
    python3-pip \
    build-essential \
    git \
    curl \
    wget

echo "Creating virtual environment..."
python3.9 -m venv venv
source venv/bin/activate

echo "Installing Python packages..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .

echo "Done! Activate with: source venv/bin/activate"
EOF

chmod +x scripts/setup/install_dependencies.sh

cat > scripts/setup/setup_aws.sh << 'EOF'
#!/bin/bash
# Set up AWS credentials and DVC

set -e

echo "Setting up AWS..."

# Configure AWS CLI
aws configure

# Set up DVC remote
echo "Configuring DVC..."
cd "$(git rev-parse --show-toplevel)"
dvc remote add -d storage s3://t1d-pinn-data/dvc-storage
dvc remote modify storage region eu-west-2

echo "Done!"
EOF

chmod +x scripts/setup/setup_aws.sh

# =============================================================================
# STEP 9: CREATE README
# =============================================================================

cat > README.md << 'EOF'
# T1D PINN Production MLOps Pipeline

Production-grade MLOps infrastructure for Type 1 Diabetes glucose prediction using Physics-Informed Neural Networks.

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/T1D_PINN_Project.git
cd T1D_PINN_Project

# 2. Install dependencies
./scripts/setup/install_dependencies.sh

# 3. Build Docker images
docker-compose build

# 4. Run training
python scripts/train_inverse.py --config configs/pinn_inverse.yaml --patient 3
```

## 📚 Documentation

- [Project Structure](docs/PROJECT_STRUCTURE.md)
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)
- [Scripts Reference](docs/SCRIPTS_README.md)

## 🛠️ Technology Stack

- **ML**: PyTorch, TensorFlow, DeepXDE
- **MLOps**: MLflow, DVC, Airflow
- **Infrastructure**: Docker, AWS, Terraform
- **CI/CD**: GitHub Actions

## 📊 Models

- **BI-RNN**: Bidirectional RNN for glucose prediction
- **PINN**: Physics-Informed Neural Network
- **Modified-MLP**: Custom MLP with physics constraints

## 📄 License

MIT License

## 👤 Author

Harsh [Your Name] - Imperial College London
EOF

echo "
========================================
✅ PROJECT STRUCTURE CREATED!
========================================

Next steps:
1. Download files from Claude and place in correct locations
2. Edit terraform/terraform.tfvars with your values
3. Follow DEPLOYMENT_GUIDE.md for implementation

Run:  cat SETUP_COMPLETE.txt  for checklist
"

# Create completion checklist
cat > SETUP_COMPLETE.txt << 'EOF'
========================================
T1D PINN Setup Checklist
========================================

□ Directory structure created
□ Configuration files created (.gitignore, .dockerignore, etc.)
□ Downloaded all files from Claude
□ Placed files in correct directories
□ Edited terraform/terraform.tfvars
□ Created AWS SSH key pair
□ Committed to git

Next: docker-compose build
========================================
EOF
