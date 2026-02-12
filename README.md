# MLOps Road Accident Prediction

[![CI Pipeline](https://github.com/chrmei/MLOps_accidents/actions/workflows/ci.yaml/badge.svg)](https://github.com/chrmei/MLOps_accidents/actions/workflows/ci.yaml)
[![Docker Build and Publish](https://github.com/chrmei/MLOps_accidents/actions/workflows/docker-publish.yml/badge.svg?branch=master)](https://github.com/chrmei/MLOps_accidents/actions/workflows/docker-publish.yml)
[![Lint Workflow](https://github.com/chrmei/MLOps_accidents/actions/workflows/lint.yaml/badge.svg?branch=master)](https://github.com/chrmei/MLOps_accidents/actions/workflows/lint.yaml)
[![Test Workflow](https://github.com/chrmei/MLOps_accidents/actions/workflows/test.yaml/badge.svg)](https://github.com/chrmei/MLOps_accidents/actions/workflows/test.yaml)

A **containerized** MLOps project for predicting road accidents using machine learning. This project implements a complete MLOps workflow from data ingestion to model serving, with a focus on **reproducibility, versioning, and containerization**.

## Table of Contents

- [Project Overview](#project-overview)
- [Quick Start (Docker Compose)](#quick-start-docker-compose)
- [Docker Compose Services](#docker-compose-services)
- [k3s Deployment (Production)](#k3s-deployment-production)
- [Current State](#current-state)
- [Documentation](#documentation)
- [License](#license)

## Project Overview

This project is an MLOps implementation for road accident prediction, designed as a learning and production-ready template. The system processes road accident data from multiple sources, trains machine learning models to predict accident severity, and serves predictions through a REST API.

### Core Principles

- **Containerization First**: All workflows run in Docker containers for environment parity
- **Reproducibility**: Ensure all experiments and workflows can be reproduced across different environments
- **Versioning**: Track data, models, and experiments using DVC and MLflow
- **Automation**: CI/CD pipelines for testing, linting, and deployment
- **Monitoring**: Track model performance and detect data drift in production

### What Has Been Done

- **ML Pipeline**: 5-step pipeline (Import -> Preprocess -> Feature Engineering -> Training -> Prediction) with multi-model support (XGBoost, Random Forest, Logistic Regression, LightGBM)
- **Microservices Architecture**: Auth, Data, Train, Predict services behind an Nginx API gateway
- **MLflow Integration**: Experiment tracking, model registry with versioning and staging (None -> Staging -> Production -> Archived)
- **Docker Compose**: Full local development and testing environment
- **k3s Deployment**: Production-ready Kubernetes manifests with horizontal scaling, shared model cache, and monitoring (Prometheus/Grafana)
- **CI/CD**: GitHub Actions for linting, testing, and Docker image publishing
- **API Testing**: Comprehensive test suite for all microservice endpoints

## Quick Start (Docker Compose)

> This project is designed to run in Docker containers. Docker ensures environment parity across development, training, and production.

**Prerequisites:** Docker & Docker Compose, Git, Make

1. **Clone the repository**

```bash
git clone https://github.com/chrmei/MLOps_accidents.git
cd MLOps_accidents
```

2. **Set up environment**

```bash
cp .env.example .env
# Edit .env with your credentials (Dagshub, MLflow, PostgreSQL, etc.)
```

3. **Build and start all services**

```bash
docker compose build
docker compose up -d
```

4. **Run the ML pipeline**

```bash
# Complete training pipeline
docker compose up train

# Or step-by-step in dev container
docker compose up dev
# Inside container:
make run-import      # Step 1: Download data
make run-preprocess  # Step 2: Clean & merge
make run-features    # Step 3: Feature engineering
make run-train       # Step 4: Train models
make run-predict     # Step 5: Make predictions
```

5. **Run API tests**

```bash
make docker-test
# Or: docker compose --profile test run --rm test
```

6. **Access the API** at `http://localhost` (via Nginx gateway)

For local development without Docker, see [doc/local_development.md](doc/local_development.md).

## Docker Compose Services

| Service | Purpose | Port |
|---------|---------|------|
| **nginx** | Reverse proxy / API Gateway | 80 |
| **auth** | Authentication (JWT) | 8004 |
| **data** | Data preprocessing & feature engineering | 8001 |
| **train** | Model training & MLflow integration | 8002 |
| **predict** | Model inference API | 8003 |
| **postgres** | Persistent storage (users, job logs) | 5432 |
| **test** | API endpoint tests (profile: test) | - |
| **dev** | Interactive development shell | - |

## k3s Deployment (Production)

The project includes Kubernetes manifests for deploying to k3s with horizontal scaling support.

**Prerequisites:** k3s installed, kubectl configured, `.env` file with secrets (copy from `.env.example`)

### Deploy

```bash
# 1. Build and import images
make k3s-build-images
make k3s-import-images

# 2. Create secrets from .env
make k3s-create-secrets

# 3. Deploy all services
make k3s-deploy

# 4. Check status
make k3s-status

# 5. Access API at http://<node-ip>:30080
make k3s-get-node-ip
```

### Key Features

- **Multi-replica predict service** (2+ replicas) with load balancing
- **Shared model cache** via PVC for faster startup
- **Model reload workflow**: `make k3s-reload-model`
- **Optional HPA** for automatic scaling (CPU-based)
- **Monitoring**: Prometheus + Grafana

### Scaling

```bash
# Manual scaling
make k3s-scale-predict REPLICAS=3

# Enable HPA (automatic scaling)
kubectl apply -f deploy/k3s/25-hpa-predict.yaml
```

### When to Use

- **Docker Compose**: Local development, CI/CD, testing
- **k3s**: Production deployments, horizontal scaling, shared storage

For detailed k3s deployment instructions, see [deploy/k3s/README.md](deploy/k3s/README.md).

## Current State

### Completed

- Project structure (cookiecutter-data-science), reproducible Python environment (pyproject.toml, UV)
- Multi-stage Dockerfile (dev, train, prod) and Docker Compose with all microservices
- Feature engineering pipeline with temporal features, cyclic encoding, categorical transformations
- Multi-model training framework (XGBoost, Random Forest, Logistic Regression, LightGBM + SMOTE)
- MLflow integration for experiment tracking and model registry
- Microservices (Auth, Data, Train, Predict) with API test suite
- k3s deployment with horizontal scaling and monitoring
- CI/CD pipelines (GitHub Actions)

### In Progress

- DVC + Dagshub integration for data versioning
- Data validation (Pandera or manual schema)
- Unit tests for model training and prediction

For detailed roadmap, see [doc/Roadmap.md](doc/Roadmap.md).

## Documentation

Detailed documentation is available in the `doc/` directory:

| Document | Description |
|----------|-------------|
| [Pipeline Overview](doc/pipeline_overview.md) | 5-step ML pipeline diagram and key files |
| [Workflow](doc/workflow.md) | Detailed step-by-step pipeline explanation |
| [MLflow Model Registry](doc/mlflow_model_registry.md) | Model versioning, staging, and deployment |
| [Multi-Model Training](doc/multi_model_training.md) | Training framework for multiple ML algorithms |
| [Feature Selection](doc/feature_selection.md) | Automatic feature selection and canonical input schema |
| [Technology Stack](doc/technology_stack.md) | Complete technology overview |
| [API Testing](doc/api_testing.md) | API endpoint test suite documentation |
| [Development Commands](doc/development_commands.md) | All Makefile commands reference |
| [Project Structure](doc/project_structure.md) | Directory layout and file descriptions |
| [Local Development](doc/local_development.md) | Setup without Docker |
| [Contributing](doc/contributing.md) | Contribution guidelines |
| [GO-LIVE Checklist](doc/go_live_checklist.md) | Production deployment checklist |
| [Team Structure](doc/team_structure.md) | Team roles and responsibilities |
| [Roadmap](doc/Roadmap.md) | Project phases and milestones |
| [Phase 1 Plan](doc/Plan_Phase_01.md) | Detailed Phase 1 execution plan |
| [k3s Deployment](deploy/k3s/README.md) | Detailed Kubernetes deployment guide |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/). Data sourced from French road accident databases.
