# Streamlit Dashboard - Control Center UI
# Build from repo root: docker build -f services/dashboard/Dockerfile .
# Multi-stage build to reduce image size

# -----------------------------------------------------------------------------
# Builder stage: Install dependencies with build tools
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/dashboard/requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Runtime stage: Copy only installed packages
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH=/root/.local/bin:$PATH

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application files
COPY services/dashboard/.streamlit /app/.streamlit
COPY services/dashboard/app /app/app

# Clean Python bytecode
RUN find /app -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true \
    && find /app -type f -name "*.pyc" -delete

EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
