# =============================================================================
# Train Service Dockerfile
# =============================================================================
# Handles model training and MLflow integration
# Multi-stage build to reduce image size
# =============================================================================

# -----------------------------------------------------------------------------
# Builder stage: Install dependencies with build tools
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

LABEL maintainer="MLOps Team"
LABEL description="Model training and registry service - Builder stage"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt pyproject.toml setup.py README.md ./
COPY src /app/src
RUN pip install --upgrade pip && \
    pip install --user --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Runtime stage: Copy only installed packages, no build tools
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="MLOps Team"
LABEL description="Model training and registry service"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/root/.local/bin:$PATH

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash mlops
WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application files
COPY pyproject.toml setup.py README.md ./
COPY src /app/src
COPY services/common /app/services/common
COPY services/train_service /app/services/train_service

# Clean Python bytecode and change ownership
RUN find /app -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true \
    && find /app -type f -name "*.pyc" -delete \
    && chown -R mlops:mlops /app

USER mlops

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the service
CMD ["uvicorn", "services.train_service.main:app", "--host", "0.0.0.0", "--port", "8002"]

