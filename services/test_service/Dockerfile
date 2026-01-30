# =============================================================================
# Test Service Dockerfile
# =============================================================================
# Runs API endpoint tests for all microservices
# =============================================================================

FROM python:3.11-slim AS base

LABEL maintainer="MLOps Team"
LABEL description="API endpoint test service for microservices"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
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

# Copy test files
COPY tests /app/tests

# Change ownership to non-root user
RUN chown -R mlops:mlops /app
USER mlops

# Default command runs all tests
# Can be overridden via docker-compose command or environment variables
CMD ["pytest", "tests/", "-v", "--tb=short"]
