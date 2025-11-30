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
