# =============================================================================
# Auth Service Dockerfile
# =============================================================================
# Handles authentication and user management
# =============================================================================

FROM python:3.11-slim AS base

LABEL maintainer="MLOps Team"
LABEL description="Authentication and user management service"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash mlops
WORKDIR /app

# Install Python dependencies
# Copy files needed for package installation first
COPY requirements.txt pyproject.toml setup.py README.md ./
COPY src /app/src
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy common module
COPY services/common /app/services/common

# Copy service code
COPY services/auth_service /app/services/auth_service

# Change ownership to non-root user
RUN chown -R mlops:mlops /app
USER mlops

# Expose port
EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# Run the service
CMD ["uvicorn", "services.auth_service.main:app", "--host", "0.0.0.0", "--port", "8004"]
