# =============================================================================
# T1D PINN AWS Infrastructure
# Main Terraform configuration for EC2, S3, IAM, and networking
# =============================================================================

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  # Optional: Store state in S3 (uncomment after first run)
  # backend "s3" {
  #   bucket = "t1d-pinn-terraform-state"
  #   key    = "infrastructure/terraform.tfstate"
  #   region = "eu-west-2"
  # }
}

# =============================================================================
# Provider Configuration
# =============================================================================
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "T1D_PINN"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Owner       = var.owner_email
    }
  }
}

# =============================================================================
# Data Sources
# =============================================================================

# Get latest Ubuntu AMI
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Get current AWS account ID
data "aws_caller_identity" "current" {}

# =============================================================================
# VPC and Networking (using default VPC for simplicity)
# =============================================================================

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# =============================================================================
# Security Group for EC2
# =============================================================================

resource "aws_security_group" "training_instance" {
  name        = "t1d-pinn-training-sg"
  description = "Security group for T1D PINN training instance"
  vpc_id      = data.aws_vpc.default.id

  # SSH access
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }

  # MLflow UI
  ingress {
    description = "MLflow UI"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }

  # Airflow UI
  ingress {
    description = "Airflow UI"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }

  # Allow all outbound
  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "t1d-pinn-training-sg"
  }
}

# =============================================================================
# IAM Role for EC2
# =============================================================================

resource "aws_iam_role" "training_instance" {
  name = "t1d-pinn-training-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "t1d-pinn-training-role"
  }
}

# IAM policy for S3 access
resource "aws_iam_role_policy" "s3_access" {
  name = "t1d-pinn-s3-access"
  role = aws_iam_role.training_instance.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.results.arn,
          "${aws_s3_bucket.results.arn}/*",
          aws_s3_bucket.data.arn,
          "${aws_s3_bucket.data.arn}/*"
        ]
      }
    ]
  })
}

# Attach SSM policy for remote command execution
resource "aws_iam_role_policy_attachment" "ssm_policy" {
  role       = aws_iam_role.training_instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# Instance profile
resource "aws_iam_instance_profile" "training_instance" {
  name = "t1d-pinn-training-profile"
  role = aws_iam_role.training_instance.name
}

# =============================================================================
# S3 Buckets
# =============================================================================

# Results bucket
resource "aws_s3_bucket" "results" {
  bucket = "t1d-pinn-results-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name        = "T1D PINN Results"
    Description = "Storage for training results and model artifacts"
  }
}

# Enable versioning
resource "aws_s3_bucket_versioning" "results" {
  bucket = aws_s3_bucket.results.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

# Enable encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "results" {
  bucket = aws_s3_bucket.results.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Data bucket (for DVC)
resource "aws_s3_bucket" "data" {
  bucket = "t1d-pinn-data-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name        = "T1D PINN Data"
    Description = "DVC remote storage for datasets"
  }
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# =============================================================================
# SSH Key Pair (you'll need to create this manually or import existing)
# =============================================================================

# Uncomment if you want Terraform to create the key
# resource "aws_key_pair" "training" {
#   key_name   = "t1d-pinn-training-key"
#   public_key = file("~/.ssh/id_rsa.pub")
# }

# =============================================================================
# EC2 Instance
# =============================================================================

resource "aws_instance" "training" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name              = var.key_name  # You must create this in AWS console first
  iam_instance_profile  = aws_iam_instance_profile.training_instance.name
  vpc_security_group_ids = [aws_security_group.training_instance.id]
  
  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_size
    delete_on_termination = true
    encrypted             = true
  }

  user_data = templatefile("${path.module}/user_data.sh", {
    aws_region      = var.aws_region
    results_bucket  = aws_s3_bucket.results.id
    data_bucket     = aws_s3_bucket.data.id
  })

  tags = {
    Name = "T1D-PINN-Training"
  }

  lifecycle {
    ignore_changes = [ami]  # Don't recreate if AMI updates
  }
}

# Elastic IP (optional, for stable IP)
resource "aws_eip" "training" {
  count    = var.use_elastic_ip ? 1 : 0
  instance = aws_instance.training.id
  domain   = "vpc"

  tags = {
    Name = "T1D-PINN-Training-EIP"
  }
}

# =============================================================================
# CloudWatch Alarms (optional monitoring)
# =============================================================================

resource "aws_cloudwatch_metric_alarm" "high_cpu" {
  alarm_name          = "t1d-pinn-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors ec2 cpu utilization"
  alarm_actions       = []  # Add SNS topic ARN for notifications

  dimensions = {
    InstanceId = aws_instance.training.id
  }
}

# =============================================================================
# Outputs
# =============================================================================

output "instance_id" {
  description = "ID of the EC2 instance"
  value       = aws_instance.training.id
}

output "instance_public_ip" {
  description = "Public IP of the EC2 instance"
  value       = var.use_elastic_ip ? aws_eip.training[0].public_ip : aws_instance.training.public_ip
}

output "instance_public_dns" {
  description = "Public DNS of the EC2 instance"
  value       = aws_instance.training.public_dns
}

output "results_bucket_name" {
  description = "Name of the S3 results bucket"
  value       = aws_s3_bucket.results.id
}

output "data_bucket_name" {
  description = "Name of the S3 data bucket"
  value       = aws_s3_bucket.data.id
}

output "ssh_command" {
  description = "SSH command to connect to instance"
  value       = "ssh -i ~/.ssh/${var.key_name}.pem ubuntu@${var.use_elastic_ip ? aws_eip.training[0].public_ip : aws_instance.training.public_ip}"
}

output "mlflow_url" {
  description = "MLflow UI URL"
  value       = "http://${var.use_elastic_ip ? aws_eip.training[0].public_ip : aws_instance.training.public_ip}:5000"
}

output "airflow_url" {
  description = "Airflow UI URL"
  value       = "http://${var.use_elastic_ip ? aws_eip.training[0].public_ip : aws_instance.training.public_ip}:8080"
}
