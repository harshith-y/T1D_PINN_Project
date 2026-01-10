from setuptools import setup, find_packages

setup(
    name="t1d_pinn",
    version="1.0.0",
    description="Physics-Informed Neural Networks for Type 1 Diabetes",
    author="Harsh",
    author_email="your-email@imperial.ac.uk",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.26,<2.0",
        "pandas>=2.0,<2.3",
        "matplotlib>=3.8,<3.10",
        "scipy>=1.11,<1.14",
        "tensorflow==2.16.1",
        "DeepXDE==1.12.0",
        "scikit-learn>=1.3,<1.6",
        "optuna>=3.5,<4",
        "tqdm>=4.66,<5",
        "mlflow>=2.10,<3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "black",
            "flake8",
            "jupyterlab>=4.2,<5",
            "ipykernel>=6.29,<8",
            "seaborn>=0.13,<0.14",
        ]
    },
)
