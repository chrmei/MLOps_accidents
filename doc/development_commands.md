# Development Commands

Run `make help` to see all commands. Key commands:

## Setup & Dependencies

- `make install-dev` - Install with dev dependencies
- `make setup-venv` - Create venv and install dependencies
- `make clean` - Remove build artifacts

## Code Quality

- `make format` - Format with black & isort
- `make lint` - Lint with flake8
- `make type-check` - Type checking with mypy

## Testing

- `make test` - Run pytest
- `make test-cov` - Run with coverage report
- `make test-api` - Run API endpoint tests (requires services running)
- `make test-api-cov` - Run API tests with coverage
- `make test-api-auth` - Run auth service tests only
- `make test-api-data` - Run data service tests only
- `make test-api-train` - Run train service tests only
- `make test-api-predict` - Run predict service tests only
- `make docker-test` - Run all API tests in Docker (auto-starts services)
- `make docker-test-build` - Build test service Docker image
- `make docker-test-run CMD="..."` - Run custom test command in Docker
- `make docker-test-cov` - Run API tests with coverage in Docker
- `make docker-test-service SERVICE=auth` - Run tests for specific service

## Data Pipeline

- `make run-import` - Import raw data
- `make run-preprocess` - Preprocess data
- `make run-features` - Build features
- `make run-train` - Train models
- `make run-predict` - Interactive predictions
- `make run-predict-file FILE=path` - Predictions from JSON

## Docker

- `make docker-build-dev` - Build development image
- `make docker-build-train` - Build training image
- `make docker-build-services` - Build all microservice images
- `make docker-up` - Start all microservices
- `make docker-down` - Stop all microservices
- `make docker-health` - Check health of all services
- `make docker-run-dev` - Run dev container (interactive)
- `make docker-run-train` - Run training pipeline
- `make docker-run-dev-exec CMD="..."` - Run one-off command
- `make docker-dvc-pull` - Pull data/models from DVC remote via Docker Compose

## DVC (Data Version Control)

- `make dvc-init` - Initialize DVC
- `make dvc-setup-remote` - Configure Dagshub remote (**requires manually edited .env**)
- `make dvc-status` / `dvc-push` / `dvc-pull` / `dvc-repro`

> **Important**: For `make dvc-setup-remote`, you must first manually edit `.env` with your Dagshub credentials (copy from `.env.example`).

## k3s Deployment

- `make k3s-build-images` - Build all Docker images for k3s
- `make k3s-import-images` - Import images into k3s
- `make k3s-create-secrets` - Create Kubernetes secrets from .env
- `make k3s-deploy` - Deploy all services to k3s
- `make k3s-status` - Check deployment status
- `make k3s-get-node-ip` - Get node IP for API access
- `make k3s-reload-model` - Reload model and restart predict pods
- `make k3s-scale-predict REPLICAS=N` - Scale predict service
