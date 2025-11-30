# T1D PINN Production Training Container
# Multi-stage build for optimized image size

# =============================================================================
# Stage 1: Base Python environment with system dependencies
# =============================================================================
FROM python:3.13-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Stage 2: Python dependencies
# =============================================================================
FROM base as dependencies

WORKDIR /tmp

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# =============================================================================
# Stage 3: Final production image
# =============================================================================
FROM base as production

# Create non-root user for security
RUN useradd -m -u 1000 t1d && \
    mkdir -p /app /data /results && \
    chown -R t1d:t1d /app /data /results

WORKDIR /app

# Copy installed packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=t1d:t1d . .

# Install package in editable mode
RUN pip install -e .

# Switch to non-root user
USER t1d

# Set up volume mount points
VOLUME ["/data", "/results"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import tensorflow as tf; import torch; print('healthy')" || exit 1

# Default command
CMD ["python", "scripts/train_inverse.py", "--help"]

# =============================================================================
# Labels for metadata
# =============================================================================
LABEL maintainer="[Your Name] <[your-email]@imperial.ac.uk>"
LABEL description="Production container for T1D PINN training and inference"
LABEL version="1.0.0"
LABEL project="T1D_PINN_Production"
