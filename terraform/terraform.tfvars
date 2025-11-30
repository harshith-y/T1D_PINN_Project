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
