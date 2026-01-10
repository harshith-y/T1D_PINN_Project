# =============================================================================
# T1D PINN Terraform Variables - GPU CONFIG
# EDIT THESE VALUES before running terraform apply
# =============================================================================

aws_region   = "eu-west-2"  # London
environment  = "dev"
owner_email  = "your-email@imperial.ac.uk"  # CHANGE THIS

# Instance configuration - CPU for testing (upgrade to GPU later)
# t3.micro: 2 vCPU, 1GB RAM - ~$0.01/hour
# After GPU quota approved, change to: g4dn.xlarge
instance_type     = "t3.micro"
root_volume_size  = 30              # GB (enough for testing)
use_elastic_ip    = false           # IP changes on restart, but saves money

# SSH key (already created)
key_name = "t1d-pinn-key"

# Security (CHANGE THIS to your IP for better security)
allowed_ssh_cidrs = ["0.0.0.0/0"]  # WARNING: Open to world! Change to ["YOUR_IP/32"]
