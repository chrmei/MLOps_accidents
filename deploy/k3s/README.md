# k3s Deployment Guide

This directory contains Kubernetes manifests for deploying the MLOps Accidents microservices to k3s (lightweight Kubernetes).

## Overview

The deployment includes:
- **Multi-replica predict service** (2+ replicas) behind nginx load balancer
- **Shared model cache** via PVC for faster model loading
- **Model reload mechanism** via Job + rolling restart
- **Optional HPA** for automatic scaling based on CPU usage
- **All microservices**: auth, data, train, predict, geocode, weather, docs, nginx

## Prerequisites

1. **k3s installed** (single-node):
   ```bash
   curl -sfL https://get.k3s.io | sh
   ```

2. **kubectl configured**:
   ```bash
   # k3s default kubeconfig location
   export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
   # Or copy to ~/.kube/config
   mkdir -p ~/.kube
   sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
   sudo chown $USER ~/.kube/config
   ```

3. **Docker images built** (same images as Docker Compose)

4. **.env file** with required secrets (see below)

## Quick Start

### 1. Build and Import Images

```bash
# Build all images
make k3s-build-images

# Import into k3s (local strategy, no registry)
make k3s-import-images
```

### 2. Create Secrets

**⚠️ SECURITY**: Never commit real secrets to git! The `.env` file is gitignored.

**Important**: Create `.env` file with required secrets before deploying:

```bash
# Copy from .env.example if needed
cp .env.example .env

# Edit .env with your actual values:
# - JWT_SECRET_KEY (generate: openssl rand -hex 32)
# - MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD
# - POSTGRES_PASSWORD (use strong password)
# - ADMIN_PASSWORD (use strong password)

# Create secret in k3s (reads from .env, creates secret directly in Kubernetes)
make k3s-create-secrets
```

**Note**: Secrets are created directly in Kubernetes via `make k3s-create-secrets` which reads from `.env` (never stored in git). See `.env.example` for required secret variables.

### 3. Deploy

```bash
# Deploy all services
make k3s-deploy

# Check status
make k3s-status

# Get node IP for access
make k3s-get-node-ip
```

### 4. Access the API

Access the API at: `http://<node-ip>:30080`

Example:
```bash
curl http://localhost:30080/health
curl http://localhost:30080/api/v1/auth/health
```

## Architecture

```
┌─────────────────────────────────────────┐
│  Client                                   │
└──────────────┬────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Nginx (NodePort 30080)                │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴───────┬──────────────┐
       ▼               ▼              ▼
┌──────────┐   ┌──────────┐   ┌──────────┐
│  Auth    │   │  Data    │   │  Train   │
│  (8004)  │   │  (8001)  │   │  (8002)  │
└──────────┘   └──────────┘   └──────────┘
       │               │              │
       └───────┬───────┴──────────────┘
               │
       ┌───────┴───────┬──────────────┐
       ▼               ▼              ▼
┌──────────┐   ┌──────────┐   ┌──────────┐
│ Predict  │   │ Geocode  │   │ Weather  │
│ x N      │   │  (8005)  │   │  (8006)  │
│ (8003)   │   └──────────┘   └──────────┘
└────┬─────┘
     │
     ▼
┌──────────────────┐
│  model-cache PVC │
│  (shared cache)  │
└──────────────────┘
```

## Model Cache and Reload

The predict service uses a shared model cache PVC for faster startup:

1. **First deployment**: Predict pods load from MLflow (fallback)
2. **After training**: Run model-reload Job to populate cache
3. **Rolling restart**: Predict pods restart and load from cache

### Reload Model Workflow

```bash
# Step 1: Train a new model (via train service API or locally)
# Step 2: Promote model to Production in MLflow
# Step 3: Reload model into cache and restart predict pods
make k3s-reload-model
```

This will:
1. Delete old model-reload job
2. Create new job that loads Production model from MLflow
3. Save model to model-cache PVC
4. Rolling restart predict deployment (pods load from cache)

## Scaling

### Manual Scaling

```bash
# Scale predict service to 3 replicas
make k3s-scale-predict REPLICAS=3
```

### Automatic Scaling (HPA)

Enable HPA for automatic scaling based on CPU:

```bash
# Ensure metrics-server is installed
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Apply HPA
kubectl apply -f deploy/k3s/25-hpa-predict.yaml

# Check HPA status
kubectl get hpa -n mlops
```

HPA configuration:
- **Min replicas**: 2
- **Max replicas**: 5
- **Target CPU**: 70%
- **Scale up**: Aggressive (100% or +2 pods per 30s)
- **Scale down**: Conservative (50% per 60s, 5min stabilization)

## Monitoring and Debugging

### Check Status

```bash
# All resources
make k3s-status

# Specific service logs
make k3s-logs SERVICE=predict
make k3s-logs SERVICE=nginx

# Predict service logs (follow)
make k3s-logs-predict
```

### Common Issues

**Pods not starting:**
```bash
# Check pod events
kubectl describe pod <pod-name> -n mlops

# Check logs
kubectl logs <pod-name> -n mlops
```

**Model not loading:**
```bash
# Check if model-reload job succeeded
kubectl logs job/model-reload -n mlops

# Check cache PVC
kubectl exec -it deployment/predict -n mlops -- ls -la /app/model-cache
```

**Secrets not found:**
```bash
# Verify secrets exist
kubectl get secret mlops-secrets -n mlops

# Recreate secrets
make k3s-create-secrets
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `k3s-create-secrets` | Create/update Secret from .env file |
| `k3s-build-images` | Build all Docker images |
| `k3s-import-images` | Import images into k3s (local) |
| `k3s-deploy` | Deploy all services |
| `k3s-deploy-predict-only` | Deploy only predict + nginx |
| `k3s-destroy` | Delete all resources |
| `k3s-status` | Show pods, services, PVCs |
| `k3s-scale-predict REPLICAS=N` | Scale predict deployment |
| `k3s-reload-model` | Reload model + restart predict |
| `k3s-logs-predict` | Follow predict logs |
| `k3s-logs SERVICE=name` | Follow logs for service |
| `k3s-get-node-ip` | Show node IPs |

## File Structure

Manifests are applied in this order:

1. `00-namespace.yaml` - Create namespace
2. `01-configmap.yaml` - Nginx config + env vars
3. `03-pvc.yaml` - Persistent volumes (data, models, model-cache)
4. `05-postgres.yaml` - PostgreSQL database
5. `10-*.yaml` - Microservices (auth, data, train, predict, geocode, weather, docs)
6. `20-nginx.yaml` - Nginx reverse proxy (NodePort 30080)
7. `25-hpa-predict.yaml` - Optional HPA for predict
8. `30-model-reload-job.yaml` - Model reload job

**Security Note**: 
- Secrets are created directly in Kubernetes via `make k3s-create-secrets` (reads from `.env`)
- `.env` file is gitignored and never committed
- See `.env.example` for required secret variables

## Docker Compose vs k3s

| Aspect | Docker Compose | k3s |
|--------|----------------|-----|
| **Use case** | Local dev, CI/CD | Production deployment |
| **Scaling** | Manual (replicas in compose) | Manual + HPA |
| **Model cache** | N/A | Shared PVC |
| **Load balancing** | Nginx | Nginx + k3s Service |
| **Storage** | Bind mounts | PVCs |
| **Access** | localhost:80 | NodePort 30080 |

**Recommendation**: Use Docker Compose for local development and testing (`make docker-up`, `make docker-test`), and k3s for production-like deployments.

## Troubleshooting

### Images not found

```bash
# Rebuild and reimport
make k3s-build-images
make k3s-import-images

# Verify images in k3s
sudo k3s ctr images list | grep mlops-accidents
```

### PVC not bound

```bash
# Check PVC status
kubectl get pvc -n mlops

# Check storage class
kubectl get storageclass

# k3s uses local-path by default, should work out of the box
```

### Predict pods failing health checks

```bash
# Check if model cache exists
kubectl exec -it deployment/predict -n mlops -- ls -la /app/model-cache

# If cache is empty, run model-reload job
make k3s-reload-model
```

## Next Steps

- **Production**: Use a container registry instead of local import
- **Multi-node**: Configure k3s for multi-node cluster
- **Monitoring**: Add Prometheus/Grafana for metrics
- **Ingress**: Replace NodePort with Ingress controller
- **TLS**: Add cert-manager for HTTPS
