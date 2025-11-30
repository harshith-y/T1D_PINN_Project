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
        "numpy>=2.0,<3",
        "pandas>=2.3,<2.4",
        "matplotlib>=3.9,<3.11",
        "scipy>=1.16,<1.17",
        "tensorflow==2.20.0",
        "DeepXDE==1.14.0",
        "scikit-learn>=1.4,<1.8",
        "optuna>=3.6,<5",
        "tqdm>=4.66,<5",
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
