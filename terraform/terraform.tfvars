# =============================================================================
# T1D PINN Terraform Variables - GPU CONFIG
# EDIT THESE VALUES before running terraform apply
# =============================================================================

aws_region   = "eu-west-2"  # London
environment  = "dev"
owner_email  = "your-email@imperial.ac.uk"  # CHANGE THIS

# Instance configuration - GPU for training
# g4dn.xlarge: 4 vCPU, 16GB RAM, 1x T4 GPU (16GB) - ~$0.526/hour
instance_type     = "g4dn.xlarge"
root_volume_size  = 100             # GB (for models and data)
use_elastic_ip    = false           # IP changes on restart, but saves money

# SSH key (already created)
key_name = "t1d-pinn-key"

# Security (CHANGE THIS to your IP for better security)
allowed_ssh_cidrs = ["0.0.0.0/0"]  # WARNING: Open to world! Change to ["YOUR_IP/32"]
