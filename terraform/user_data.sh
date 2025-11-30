#!/bin/bash
# =============================================================================
# T1D PINN EC2 User Data Script
# Runs on instance first boot to set up environment
# =============================================================================

set -e  # Exit on error

# Log everything
exec > >(tee /var/log/user-data.log)
exec 2>&1

echo "========================================="
echo "Starting T1D PINN instance setup..."
echo "========================================="

# =============================================================================
# Update system
# =============================================================================
echo "Updating system packages..."
apt-get update -y
apt-get upgrade -y

# =============================================================================
# Install system dependencies
# =============================================================================
echo "Installing system dependencies..."
apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    tmux \
    htop \
    python3.9 \
    python3.9-venv \
    python3-pip \
    awscli \
    docker.io \
    docker-compose

# =============================================================================
# Configure Docker
# =============================================================================
echo "Configuring Docker..."
systemctl start docker
systemctl enable docker
usermod -aG docker ubuntu

# =============================================================================
# Install NVIDIA drivers (if GPU instance)
# =============================================================================
INSTANCE_TYPE=$(ec2-metadata --instance-type | cut -d " " -f 2)
if [[ $INSTANCE_TYPE == g4dn* ]] || [[ $INSTANCE_TYPE == p3* ]]; then
    echo "GPU instance detected, installing NVIDIA drivers..."
    
    # Install NVIDIA driver
    apt-get install -y ubuntu-drivers-common
    ubuntu-drivers autoinstall
    
    # Install NVIDIA Docker runtime
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
    apt-get update
    apt-get install -y nvidia-docker2
    systemctl restart docker
fi

# =============================================================================
# Set up Python environment
# =============================================================================
echo "Setting up Python environment..."
su - ubuntu << 'EOF'
cd /home/ubuntu

# Clone repository (replace with your repo URL)
# git clone https://github.com/YOUR_USERNAME/T1D_PINN_Project.git
# cd T1D_PINN_Project

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install pip and setuptools
pip install --upgrade pip setuptools wheel

# Install requirements (once repo is cloned)
# pip install -r requirements.txt
# pip install -e .

echo "Virtual environment created at /home/ubuntu/venv"
EOF

# =============================================================================
# Configure AWS CLI
# =============================================================================
echo "Configuring AWS CLI..."
su - ubuntu << 'EOF'
aws configure set region ${aws_region}
aws configure set output json
EOF

# =============================================================================
# Create directories
# =============================================================================
echo "Creating directory structure..."
mkdir -p /home/ubuntu/{data,results,mlruns,logs}
chown -R ubuntu:ubuntu /home/ubuntu

# =============================================================================
# Set up DVC remote (S3)
# =============================================================================
echo "Configuring DVC remote..."
cat > /home/ubuntu/.dvc_config << 'EOF'
# DVC configuration
# Run these commands after cloning the repository:
# cd /home/ubuntu/T1D_PINN_Project
# dvc remote add -d storage s3://${data_bucket}/dvc-storage
# dvc remote modify storage region ${aws_region}
EOF

# =============================================================================
# Create helpful aliases and environment
# =============================================================================
echo "Setting up user environment..."
cat >> /home/ubuntu/.bashrc << 'EOF'

# T1D PINN Environment
export T1D_PROJECT_DIR="/home/ubuntu/T1D_PINN_Project"
export MLFLOW_TRACKING_URI="http://localhost:5000"
export AWS_DEFAULT_REGION="${aws_region}"
export RESULTS_BUCKET="${results_bucket}"
export DATA_BUCKET="${data_bucket}"

# Aliases
alias activate='source /home/ubuntu/venv/bin/activate'
alias t1d='cd $T1D_PROJECT_DIR && activate'
alias results='aws s3 ls s3://$RESULTS_BUCKET/'
alias data='aws s3 ls s3://$DATA_BUCKET/'

# Auto-activate venv on login
if [ -f /home/ubuntu/venv/bin/activate ]; then
    source /home/ubuntu/venv/bin/activate
fi

echo "T1D PINN environment loaded!"
echo "Quick commands:"
echo "  t1d      - Go to project and activate venv"
echo "  results  - List results in S3"
echo "  data     - List data in S3"
EOF

# =============================================================================
# Install CloudWatch agent (for monitoring)
# =============================================================================
echo "Installing CloudWatch agent..."
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i -E ./amazon-cloudwatch-agent.deb
rm amazon-cloudwatch-agent.deb

# =============================================================================
# Final setup
# =============================================================================
echo "Finalizing setup..."
chown -R ubuntu:ubuntu /home/ubuntu

# Create welcome message
cat > /etc/motd << 'EOF'
========================================
T1D PINN Training Instance
========================================
Instance Type: $(ec2-metadata --instance-type | cut -d " " -f 2)
Region: ${aws_region}

Results Bucket: ${results_bucket}
Data Bucket: ${data_bucket}

Quick Start:
1. Clone repo: git clone https://github.com/YOUR_USERNAME/T1D_PINN_Project.git
2. Install: cd T1D_PINN_Project && pip install -r requirements.txt
3. Train: python scripts/train_inverse.py --config configs/pinn_inverse.yaml --patient 3

Commands:
- t1d      : Go to project directory
- results  : List S3 results
- data     : List S3 data

MLflow UI: http://$(ec2-metadata --public-ipv4):5000
Airflow UI: http://$(ec2-metadata --public-ipv4):8080

========================================
EOF

echo "========================================="
echo "Setup complete! Instance ready."
echo "========================================="

# Reboot if GPU drivers were installed
if [[ $INSTANCE_TYPE == g4dn* ]] || [[ $INSTANCE_TYPE == p3* ]]; then
    echo "Rebooting to load NVIDIA drivers..."
    shutdown -r +1
fi
