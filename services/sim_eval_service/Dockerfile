# Multi-stage build to reduce image size
# -----------------------------------------------------------------------------
# Builder stage: Install dependencies with build tools
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies (if needed for some packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Runtime stage: Copy only installed packages
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/root/.local/bin:$PATH

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY main.py .

# Clean Python bytecode
RUN find /app -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true \
    && find /app -type f -name "*.pyc" -delete

CMD ["python3", "main.py"]