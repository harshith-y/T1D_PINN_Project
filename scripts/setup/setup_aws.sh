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
