# =============================================================================
# T1D PINN Terraform Variables
# Configure these values in terraform.tfvars
# =============================================================================

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "eu-west-2"  # London
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "owner_email" {
  description = "Email of the project owner"
  type        = string
  default     = "your-email@imperial.ac.uk"
}

variable "instance_type" {
  description = "EC2 instance type (t3.micro for budget, t3.medium for CPU, g4dn.xlarge for GPU)"
  type        = string
  default     = "t3.micro"

  validation {
    condition     = can(regex("^(t2\\.|t3\\.|g4dn\\.|p3\\.|p4d\\.)", var.instance_type))
    error_message = "Instance type must be t2.*, t3.*, g4dn.*, p3.*, or p4d.*"
  }
}

variable "root_volume_size" {
  description = "Size of root EBS volume in GB"
  type        = number
  default     = 30

  validation {
    condition     = var.root_volume_size >= 20 && var.root_volume_size <= 500
    error_message = "Root volume size must be between 20 and 500 GB"
  }
}

variable "key_name" {
  description = "Name of SSH key pair (must exist in AWS)"
  type        = string
  default     = "t1d-pinn-key"
}

variable "allowed_ssh_cidrs" {
  description = "CIDR blocks allowed to SSH (your IP)"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Change to your IP for security
}

variable "use_elastic_ip" {
  description = "Whether to use Elastic IP (recommended for stability)"
  type        = bool
  default     = true
}

# =============================================================================
# Instance Type Presets
# =============================================================================

# CPU instances
# t3.medium:  2 vCPU, 4 GB RAM,  $0.0416/hour - Good for development
# t3.large:   2 vCPU, 8 GB RAM,  $0.0832/hour - Better for small experiments
# t3.xlarge:  4 vCPU, 16 GB RAM, $0.1664/hour - Best CPU option

# GPU instances
# g4dn.xlarge:  4 vCPU, 16 GB RAM, 1x T4 GPU (16GB),  $0.526/hour - Recommended
# g4dn.2xlarge: 8 vCPU, 32 GB RAM, 1x T4 GPU (16GB),  $0.752/hour - More CPU
# p3.2xlarge:   8 vCPU, 61 GB RAM, 1x V100 GPU (16GB), $3.06/hour - Fastest

# Recommended:
# - Development/testing: t3.medium
# - Small experiments: t3.large
# - Production training: g4dn.xlarge
