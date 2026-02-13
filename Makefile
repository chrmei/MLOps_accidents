.PHONY: help install install-dev sync lock update clean lint format type-check test test-cov test-api test-api-cov test-api-service test-api-service-cov test-api-auth test-api-data test-api-train test-api-predict run-import run-preprocess run-features run-train run-train-grid run-predict run-predict-file workflow-all workflow-data workflow-ml dvc-init dvc-setup-remote dvc-status dvc-push dvc-pull dvc-repro docker-up docker-down docker-build-services docker-build-services-no-cache docker-logs docker-status docker-health docker-restart docker-dev docker-build docker-build-dev docker-build-train docker-dvc-pull docker-clean docker-test docker-test-build docker-test-run docker-test-cov docker-test-service k3s-create-secrets k3s-build-images k3s-build-test-images k3s-import-images k3s-import-test-images k3s-deploy k3s-deploy-predict-only k3s-destroy k3s-shutdown k3s-status k3s-scale-predict k3s-restart k3s-reload-model k3s-logs-predict k3s-logs k3s-get-node-ip k3s-test k3s-test-run k3s-test-cov k3s-test-service k3s-test-logs k3s-test-clean

# Load .env file if it exists (cross-platform with GNU Make)
# Note: On Windows, use Git Bash, WSL, or another Unix-like environment
ifneq (,$(wildcard .env))
    include .env
    export
endif

# Default Python version (can be overridden)
PYTHON_VERSION ?= 3.11

# UV executable (use uv if available, otherwise suggest installation)
UV := $(shell command -v uv 2> /dev/null)

# Detect virtual environment and use its Python if available
# Check if .venv/bin/python exists, otherwise fall back to system python
PYTHON := $(shell if [ -f .venv/bin/python ]; then echo .venv/bin/python; else echo python; fi)

# Detect virtual environment tools (flake8, black, isort, mypy, pytest, dvc)
# Use venv versions if available, otherwise use system versions
FLAKE8 := $(shell if [ -f .venv/bin/flake8 ]; then echo .venv/bin/flake8; else echo flake8; fi)
BLACK := $(shell if [ -f .venv/bin/black ]; then echo .venv/bin/black; else echo black; fi)
ISORT := $(shell if [ -f .venv/bin/isort ]; then echo .venv/bin/isort; else echo isort; fi)
MYPY := $(shell if [ -f .venv/bin/mypy ]; then echo .venv/bin/mypy; else echo mypy; fi)
PYTEST := $(shell if [ -f .venv/bin/pytest ]; then echo .venv/bin/pytest; else echo pytest; fi)
DVC := $(shell if [ -f .venv/bin/dvc ]; then echo .venv/bin/dvc; else echo dvc; fi)

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

check-uv:
	@if [ -z "$(UV)" ]; then \
		echo "Error: uv is not installed. Please install it first:"; \
		echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		echo "  or visit: https://github.com/astral-sh/uv"; \
		exit 1; \
	fi

install: check-uv ## Install project dependencies using UV
	@echo "Installing dependencies with UV..."
	uv pip install -e .

install-dev: check-uv ## Install project with development dependencies
	@echo "Installing dependencies with UV (including dev dependencies)..."
	uv pip install -e ".[dev]"

sync: check-uv ## Sync dependencies from pyproject.toml (UV equivalent of pip-sync)
	@echo "Syncing dependencies with UV..."
	uv pip install -e ".[dev]"

lock: check-uv ## Generate lock file (if using UV's lock feature)
	@echo "Note: UV manages dependencies directly from pyproject.toml"
	@echo "No separate lock file needed with UV pip install"

update: check-uv ## Update all dependencies to latest versions
	@echo "Updating dependencies with UV..."
	uv pip install --upgrade -e ".[dev]"

clean: ## Remove build artifacts and cache files
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -r {} + 2>/dev/null || true

lint: ## Run linting with flake8
	@echo "Running flake8..."
	$(FLAKE8) src/ --max-line-length=88 --extend-ignore=E203,W503

format: ## Format code with black and isort (matches GitHub Actions scope: src/ only)
	@echo "Formatting code with black..."
	$(BLACK) src/
	@echo "Sorting imports with isort..."
	$(ISORT) src/

type-check: ## Run type checking with mypy
	@echo "Running mypy..."
	$(MYPY) src/

test: ## Run tests with pytest
	@echo "Running tests..."
	$(PYTEST)

test-cov: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	$(PYTEST) --cov=src --cov-report=term-missing --cov-report=html

# =============================================================================
# API Endpoint Tests
# =============================================================================

test-api: ## Run API endpoint tests (requires services to be running)
	@echo "Running API endpoint tests..."
	@if [ -z "$(BASE_URL)" ]; then \
		echo "Using default BASE_URL: http://localhost"; \
		BASE_URL=http://localhost $(PYTEST) tests/ -v --tb=short; \
	else \
		echo "Using BASE_URL: $(BASE_URL)"; \
		BASE_URL=$(BASE_URL) $(PYTEST) tests/ -v --tb=short; \
	fi

test-api-cov: ## Run API endpoint tests with coverage
	@echo "Running API endpoint tests with coverage..."
	@if [ -z "$(BASE_URL)" ]; then \
		echo "Using default BASE_URL: http://localhost"; \
		BASE_URL=http://localhost $(PYTEST) tests/ -v --tb=short --cov=services --cov-report=term-missing --cov-report=html; \
	else \
		echo "Using BASE_URL: $(BASE_URL)"; \
		BASE_URL=$(BASE_URL) $(PYTEST) tests/ -v --tb=short --cov=services --cov-report=term-missing --cov-report=html; \
	fi

test-api-auth: ## Run auth service API tests only
	@echo "Running auth service API tests..."
	@if [ -z "$(BASE_URL)" ]; then \
		BASE_URL=http://localhost $(PYTEST) tests/test_auth_service.py -v --tb=short; \
	else \
		BASE_URL=$(BASE_URL) $(PYTEST) tests/test_auth_service.py -v --tb=short; \
	fi

test-api-data: ## Run data service API tests only
	@echo "Running data service API tests..."
	@if [ -z "$(BASE_URL)" ]; then \
		BASE_URL=http://localhost $(PYTEST) tests/test_data_service.py -v --tb=short; \
	else \
		BASE_URL=$(BASE_URL) $(PYTEST) tests/test_data_service.py -v --tb=short; \
	fi

test-api-train: ## Run train service API tests only
	@echo "Running train service API tests..."
	@if [ -z "$(BASE_URL)" ]; then \
		BASE_URL=http://localhost $(PYTEST) tests/test_train_service.py -v --tb=short; \
	else \
		BASE_URL=$(BASE_URL) $(PYTEST) tests/test_train_service.py -v --tb=short; \
	fi

test-api-predict: ## Run predict service API tests only
	@echo "Running predict service API tests..."
	@if [ -z "$(BASE_URL)" ]; then \
		BASE_URL=http://localhost $(PYTEST) tests/test_predict_service.py -v --tb=short; \
	else \
		BASE_URL=$(BASE_URL) $(PYTEST) tests/test_predict_service.py -v --tb=short; \
	fi

test-api-service: ## Run API tests in Docker container (requires docker-compose)
	@echo "Running API tests in Docker container..."
	@echo "Note: This will start services if not already running..."
	docker compose --profile test run --rm test

test-api-service-cov: ## Run API tests in Docker container with coverage
	@echo "Running API tests in Docker container with coverage..."
	docker compose --profile test run --rm test pytest tests/ -v --tb=short --cov=services --cov-report=term-missing --cov-report=html

# Data pipeline commands
run-import: ## Import raw data from S3
	@echo "Importing raw data..."
	$(PYTHON) src/data/import_raw_data.py

run-preprocess: ## Create interim dataset from raw data
	@echo "Preprocessing data (creating interim dataset)..."
	NOCONFIRM=$(NOCONFIRM) $(PYTHON) src/data/make_dataset.py

run-features: ## Build features from interim dataset (XGBoost-optimized)
	@echo "Building features from interim dataset..."
	NOCONFIRM=$(NOCONFIRM) $(PYTHON) src/features/build_features.py

run-train: ## Train multiple models with default parameters (fast, no grid search)
	@echo "Training multiple models with default parameters..."
	$(PYTHON) src/models/train_multi_model.py

run-train-grid: ## Train multiple models with grid search for hyperparameter tuning (slow)
	@echo "Training multiple models with grid search (this may take a while)..."
	$(PYTHON) src/models/train_multi_model.py --grid-search

run-predict: ## Make predictions using default JSON file (src/models/test_features.json) - uses local model (development mode)
	@echo "Making predictions from default JSON file (local model)..."
	$(PYTHON) src/models/predict_model.py

run-predict-mlflow: ## Make predictions using MLflow Production model (best practice for production)
	@echo "Making predictions using MLflow Production model..."
	$(PYTHON) src/models/predict_model.py --use-mlflow-production

run-predict-best: ## Make predictions using best Production model across all model types (automatically selects best)
	@echo "Making predictions using best Production model (auto-selected)..."
	$(PYTHON) src/models/predict_model.py --use-best-model

run-predict-file: ## Make predictions from JSON file (usage: make run-predict-file FILE=src/models/test_features.json)
	@if [ -z "$(FILE)" ]; then \
		echo "Error: FILE variable is required. Usage: make run-predict-file FILE=path/to/file.json"; \
		exit 1; \
	fi
	@echo "Making predictions from $(FILE) (local model)..."
	$(PYTHON) src/models/predict_model.py $(FILE)

run-predict-file-mlflow: ## Make predictions from JSON file using MLflow Production (usage: make run-predict-file-mlflow FILE=src/models/test_features.json)
	@if [ -z "$(FILE)" ]; then \
		echo "Error: FILE variable is required. Usage: make run-predict-file-mlflow FILE=path/to/file.json"; \
		exit 1; \
	fi
	@echo "Making predictions from $(FILE) using MLflow Production model..."
	$(PYTHON) src/models/predict_model.py $(FILE) --use-mlflow-production

# Complete workflow commands
workflow-all: $(if $(NOCONFIRM),,run-import) run-preprocess run-features $(if $(filter 1,$(GRID_SEARCH)),run-train-grid,run-train) run-predict ## Run complete pipeline (use GRID_SEARCH=1 for grid search, NOCONFIRM=1 to skip raw data import and auto-confirm overwrites): import → preprocess → features → train → predict
	@echo "Complete workflow finished!"

workflow-data: run-import run-preprocess ## Run data pipeline: import → preprocess
	@echo "Data pipeline finished!"

workflow-ml: run-features run-train ## Run ML pipeline: features → train
	@echo "ML pipeline finished!"

# Setup commands
setup: check-uv ## Initial project setup (install dependencies)
	@echo "Setting up project..."
	@echo "Python version: $(PYTHON_VERSION)"
	@echo "Installing dependencies..."
	uv pip install -e ".[dev]"
	@echo "Setup complete!"

setup-venv: check-uv ## Create virtual environment and install dependencies
	@echo "Creating virtual environment with Python $(PYTHON_VERSION)..."
	python$(PYTHON_VERSION) -m venv .venv
	@echo "Activating virtual environment and installing dependencies..."
	.venv/bin/pip install --upgrade pip
	uv pip install -e ".[dev]"
	@echo "Virtual environment created! Activate it with: source .venv/bin/activate"

# DVC commands
dvc-init: ## Initialize DVC repository
	@echo "Initializing DVC..."
	$(DVC) init
	@echo "DVC initialized! Run 'make dvc-setup-remote' to configure Dagshub remote."

dvc-setup-remote: ## Configure DVC remote with Dagshub (requires .env file)
	@if [ -z "$(DAGSHUB_USERNAME)" ] || [ -z "$(DAGSHUB_TOKEN)" ] || [ -z "$(DAGSHUB_REPO)" ]; then \
		echo "Error: DAGSHUB_USERNAME, DAGSHUB_TOKEN, and DAGSHUB_REPO must be set in .env file"; \
		echo "Please ensure .env file exists and contains these variables."; \
		exit 1; \
	fi
	@echo "Setting up DVC remote with Dagshub..."
	@$(DVC) remote add origin s3://dvc 2>/dev/null || $(DVC) remote modify origin url s3://dvc
	@$(DVC) remote modify origin endpointurl https://dagshub.com/$(DAGSHUB_REPO).s3
	@$(DVC) remote modify origin --local access_key_id $(DAGSHUB_TOKEN)
	@$(DVC) remote modify origin --local secret_access_key $(DAGSHUB_TOKEN)
	@echo "DVC remote configured successfully!"
	@echo "Repository: $(DAGSHUB_REPO)"
	@echo "Endpoint: https://dagshub.com/$(DAGSHUB_REPO).s3"

dvc-status: ## Check DVC status
	@echo "Checking DVC status..."
	$(DVC) status

dvc-push: ## Push data to DVC remote
	@echo "Pushing data to DVC remote..."
	$(DVC) push

dvc-pull: ## Pull data from DVC remote
	@echo "Pulling data from DVC remote..."
	$(DVC) pull

dvc-repro: ## Reproduce DVC pipeline
	@echo "Pulling data from DVC remote (if available)..."
	@$(DVC) pull || echo "Warning: Some files not found in remote, will regenerate..."
	@echo "Reproducing DVC pipeline..."
	@$(DVC) repro

# =============================================================================
# Microservices Docker commands
# =============================================================================

docker-up: ## Start all microservices (nginx, auth, data, train, predict) - excludes test services
	@echo "Starting microservices..."
	docker compose up -d postgres node-exporter geocode predict weather prometheus data auth train grafana docs nginx streamlit
	@echo "Services started! API available at http://localhost"
	@echo "Note: Test services (sim-traffic, sim-eval) are excluded. Use 'make docker-test' to include them."

docker-up-all: ## Start all microservices including test service
	@echo "Starting all microservices including test..."
	docker compose up -d
	@echo "All services started!"

docker-down: ## Stop all microservices
	@echo "Stopping microservices..."
	docker compose down
	@echo "Services stopped."

docker-build-services: ## Build all microservice images
	@echo "Building microservice images..."
	docker compose build
	@echo "All microservice images built!"

docker-build-services-no-cache: ## Build all microservice images
	@echo "Building microservice images..."
	docker compose build --no-cache
	@echo "All microservice images built!"

docker-build-test: ## Build test service image
	@echo "Building test service image..."
	docker compose build test
	@echo "Test service image built!"

docker-logs: ## Show logs from all services (follow mode)
	docker compose logs -f

docker-logs-nginx: ## Show nginx logs
	docker compose logs -f nginx

docker-logs-data: ## Show data service logs
	docker compose logs -f data

docker-logs-train: ## Show train service logs
	docker compose logs -f train

docker-logs-predict: ## Show predict service logs
	docker compose logs -f predict

docker-status: ## Show status of all services
	docker compose ps

docker-health: ## Check health of all services via API
	@echo "Checking service health..."
	@echo "--- Nginx Gateway ---"
	@curl -sf http://localhost/health 2>/dev/null && echo "" || echo "  Not responding"
	@echo "--- Auth Service ---"
	@curl -sf http://localhost/api/v1/auth/health 2>/dev/null && echo "" || echo "  Not responding"
	@echo "--- Data Service ---"
	@curl -sf http://localhost/api/v1/data/health 2>/dev/null && echo "" || echo "  Not responding"
	@echo "--- Train Service ---"
	@curl -sf http://localhost/api/v1/train/health 2>/dev/null && echo "" || echo "  Not responding"
	@echo "--- Predict Service ---"
	@curl -sf http://localhost/api/v1/predict/health 2>/dev/null && echo "" || echo "  Not responding"

docker-restart: ## Restart all microservices
	@echo "Restarting microservices..."
	docker compose restart nginx auth data train predict

# Compose files: base + test overrides (relaxed auth rate limits and nginx for tests)
COMPOSE_TEST := -f docker-compose.yml -f docker-compose.test.yml

docker-test: ## Run API tests in Docker (starts services if needed, runs tests, stops test container) - includes sim-traffic and sim-eval
	@echo "Running API tests in Docker..."
	@echo "Building and starting services if not running..."
	@docker compose $(COMPOSE_TEST) build nginx auth data train predict sim-traffic sim-eval
	@docker compose $(COMPOSE_TEST) up -d nginx auth data train predict sim-traffic sim-eval || true
	@echo "Waiting for services to be healthy..."
	@sleep 5
	@echo "Running tests..."
	docker compose $(COMPOSE_TEST) --profile test run --rm test
	@echo "Tests completed!"

docker-test-build: ## Build test service Docker image
	@echo "Building test service..."
	docker compose $(COMPOSE_TEST) build test

docker-test-run: ## Run API tests in Docker with custom command (usage: make docker-test-run CMD="pytest tests/test_auth_service.py -v")
	@if [ -z "$(CMD)" ]; then \
		echo "Running all API tests..."; \
		docker compose $(COMPOSE_TEST) --profile test run --rm test; \
	else \
		echo "Running custom test command: $(CMD)"; \
		docker compose $(COMPOSE_TEST) --profile test run --rm test $(CMD); \
	fi

docker-test-cov: ## Run API tests in Docker with coverage report
	@echo "Running API tests with coverage..."
	docker compose $(COMPOSE_TEST) --profile test run --rm test pytest tests/ -v --tb=short --cov=services --cov-report=term-missing --cov-report=html

docker-test-service: ## Run tests for specific service (usage: make docker-test-service SERVICE=auth)
	@if [ -z "$(SERVICE)" ]; then \
		echo "Error: SERVICE variable required. Usage: make docker-test-service SERVICE=auth"; \
		echo "Available services: auth, data, train, predict"; \
		exit 1; \
	fi
	@echo "Running tests for $(SERVICE) service..."
	docker compose $(COMPOSE_TEST) --profile test run --rm test pytest tests/test_$(SERVICE)_service.py -v --tb=short

docker-dev: ## Start development shell container (interactive)
	@echo "Starting development container..."
	docker compose --profile dev run --rm dev

# =============================================================================
# Legacy Docker commands (standalone containers)
# =============================================================================

docker-build: ## Build legacy Docker images (dev, train)
	@echo "Building legacy Docker images..."
	docker build -t mlops-accidents:dev --target dev .
	docker build -t mlops-accidents:train --target train .
	@echo "Legacy Docker images built!"

docker-build-dev: ## Build development Docker image (standalone)
	@echo "Building development Docker image..."
	docker build -t mlops-accidents:dev --target dev .

docker-build-train: ## Build training Docker image (standalone)
	@echo "Building training Docker image..."
	docker build -t mlops-accidents:train --target train .

docker-dvc-pull: ## Pull data/models from DVC remote using Docker Compose
	@echo "Pulling data/models from DVC remote via Docker..."
	docker compose --profile dvc run --rm dvc-pull

docker-clean: ## Remove all Docker containers, images, and volumes
	@echo "Stopping and removing containers..."
	docker compose down --rmi local --volumes 2>/dev/null || true
	@echo "Removing legacy images..."
	docker rmi mlops-accidents:dev mlops-accidents:train 2>/dev/null || true
	@echo "Cleanup complete!"

# =============================================================================
# k3s Deployment Commands
# =============================================================================

K3S_DIR := deploy/k3s
K3S_NAMESPACE := mlops

k3s-check: ## Check if k3s is running and kubectl is configured
	@echo "Checking k3s status..."
	@kubectl cluster-info >/dev/null 2>&1 || \
		(echo "❌ Error: k3s is not running or kubectl is not configured." && \
		 echo "   Start k3s with: sudo systemctl start k3s" && \
		 echo "   Configure kubectl with: mkdir -p ~/.kube && sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config && sudo chown $$USER ~/.kube/config" && \
		 exit 1)
	@echo "✅ k3s is running and kubectl is configured"

k3s-create-secrets: k3s-check ## Create/update Secret from .env file (requires .env with secrets)
	@if [ ! -f .env ]; then \
		echo "Error: .env file not found. Please create .env file with required secrets."; \
		exit 1; \
	fi
	@echo "Ensuring namespace exists..."
	@kubectl apply -f $(K3S_DIR)/00-namespace.yaml
	@echo "Creating/updating Secret from .env..."
	@kubectl create secret generic mlops-secrets \
		--from-env-file=.env \
		-n $(K3S_NAMESPACE) \
		--dry-run=client -o yaml | kubectl apply -f -
	@echo "Secret created/updated successfully!"

k3s-build-images: ## Build all Docker images for k3s deployment
	@echo "Building Docker images for k3s..."
	docker compose build nginx auth data train predict geocode weather docs streamlit
	@echo "All images built successfully!"

k3s-build-test-images: ## Build test service Docker images for k3s (test, sim-traffic, sim-eval)
	@echo "Building test service images..."
	docker compose build test sim-traffic sim-eval
	@echo "Test service images built successfully!"

k3s-import-images: k3s-check ## Import built images into k3s (local strategy, no registry)
	@echo "Importing images into k3s..."
	@echo "Tagging images for k3s..."
	@docker tag mlops_accidents-nginx:latest mlops-accidents:nginx 2>/dev/null || true
	@docker tag mlops_accidents-auth:latest mlops-accidents:auth_service 2>/dev/null || true
	@docker tag mlops_accidents-data:latest mlops-accidents:data_service 2>/dev/null || true
	@docker tag mlops_accidents-train:latest mlops-accidents:train_service 2>/dev/null || true
	@docker tag mlops_accidents-predict:latest mlops-accidents:predict_service 2>/dev/null || true
	@docker tag mlops_accidents-geocode:latest mlops-accidents:geocode_service 2>/dev/null || true
	@docker tag mlops_accidents-weather:latest mlops-accidents:weather_service 2>/dev/null || true
	@docker tag mlops_accidents-docs:latest mlops-accidents:docs_service 2>/dev/null || true
	@docker tag mlops_accidents-streamlit:latest mlops-accidents:dashboard 2>/dev/null || true
	@echo "Saving and importing images..."
	@docker save mlops-accidents:nginx mlops-accidents:auth_service mlops-accidents:data_service \
		mlops-accidents:train_service mlops-accidents:predict_service \
		mlops-accidents:geocode_service mlops-accidents:weather_service \
		mlops-accidents:docs_service mlops-accidents:dashboard 2>/dev/null | sudo k3s ctr images import - || \
		(echo "Error: Failed to import images. Ensure images are built and k3s is running." && exit 1)
	@echo "Images imported successfully!"

k3s-import-test-images: k3s-check ## Import test service images into k3s (test, sim-traffic, sim-eval)
	@echo "Importing test service images into k3s..."
	@echo "Tagging test images for k3s..."
	@docker tag mlops_accidents-test:latest mlops-accidents:test_service 2>/dev/null || \
		(echo "Error: Test image not found. Build it first with 'make k3s-build-test-images'" && exit 1)
	@docker tag mlops_accidents-sim-traffic:latest mlops-accidents:sim_traffic_service 2>/dev/null || \
		(echo "Error: sim-traffic image not found. Build it first with 'make k3s-build-test-images'" && exit 1)
	@docker tag mlops_accidents-sim-eval:latest mlops-accidents:sim_eval_service 2>/dev/null || \
		(echo "Error: sim-eval image not found. Build it first with 'make k3s-build-test-images'" && exit 1)
	@echo "Saving and importing test images..."
	@docker save mlops-accidents:test_service mlops-accidents:sim_traffic_service mlops-accidents:sim_eval_service 2>/dev/null | sudo k3s ctr images import - || \
		(echo "Error: Failed to import test images. Ensure images are built and k3s is running." && exit 1)
	@echo "Test images imported successfully!"

k3s-deploy: k3s-check ## Deploy all services to k3s (namespace, configmap, PVCs, services)
	@echo "Deploying to k3s..."
	@kubectl apply -f $(K3S_DIR)/00-namespace.yaml
	@kubectl apply -f $(K3S_DIR)/01-configmap.yaml
	@echo "⚠️  SECURITY: Ensure secrets are created with 'make k3s-create-secrets' before deploying services!"
	@echo "   (This reads from .env file, which is gitignored and never committed)"
	@kubectl apply -f $(K3S_DIR)/03-pvc.yaml
	@kubectl apply -f $(K3S_DIR)/05-postgres.yaml
	@kubectl apply -f $(K3S_DIR)/07-prometheus-configmap.yaml
	@kubectl apply -f $(K3S_DIR)/07-prometheus.yaml
	@kubectl apply -f $(K3S_DIR)/08-grafana-configmap.yaml
	@kubectl apply -f $(K3S_DIR)/08-grafana.yaml
	@kubectl apply -f $(K3S_DIR)/09-node-exporter.yaml
	@kubectl apply -f $(K3S_DIR)/10-auth.yaml
	@kubectl apply -f $(K3S_DIR)/10-data.yaml
	@kubectl apply -f $(K3S_DIR)/10-train.yaml
	@kubectl apply -f $(K3S_DIR)/10-predict.yaml
	@kubectl apply -f $(K3S_DIR)/10-geocode.yaml
	@kubectl apply -f $(K3S_DIR)/10-weather.yaml
	@kubectl apply -f $(K3S_DIR)/10-docs.yaml
	@kubectl apply -f $(K3S_DIR)/10-dashboard.yaml
	@kubectl apply -f $(K3S_DIR)/20-nginx.yaml
	@echo "Deployment completed! Access API at http://<node-ip>:30080"
	@echo "Optional: Apply HPA with 'kubectl apply -f $(K3S_DIR)/25-hpa-predict.yaml'"

k3s-deploy-predict-only: ## Deploy only predict service and nginx (when other services exist)
	@echo "Deploying predict service and nginx..."
	@kubectl apply -f $(K3S_DIR)/10-predict.yaml
	@kubectl apply -f $(K3S_DIR)/20-nginx.yaml
	@echo "Predict service and nginx deployed!"

k3s-destroy: ## Delete all k3s resources (namespace deletion removes everything)
	@echo "Destroying k3s deployment..."
	@kubectl delete namespace $(K3S_NAMESPACE) || echo "Namespace already deleted or doesn't exist"
	@echo "k3s deployment destroyed!"

k3s-shutdown: ## Safely shut down workloads without losing data (keeps namespace, PVCs, secrets)
	@echo "Shutting down k3s workloads (data preserved)..."
	@kubectl delete deployment --all -n $(K3S_NAMESPACE) --ignore-not-found=true
	@kubectl delete job --all -n $(K3S_NAMESPACE) --ignore-not-found=true
	@kubectl delete cronjob --all -n $(K3S_NAMESPACE) --ignore-not-found=true
	@kubectl delete service --all -n $(K3S_NAMESPACE) --ignore-not-found=true
	@kubectl delete configmap --all -n $(K3S_NAMESPACE) --ignore-not-found=true
	@kubectl delete hpa --all -n $(K3S_NAMESPACE) --ignore-not-found=true
	@echo "✅ Workloads stopped. Namespace, PVCs (data), and secrets are unchanged."
	@echo "   Restart with: make k3s-deploy"

k3s-status: ## Show status of all pods, services, and PVCs
	@echo "=== Pods ==="
	@kubectl get pods -n $(K3S_NAMESPACE)
	@echo ""
	@echo "=== Services ==="
	@kubectl get svc -n $(K3S_NAMESPACE)
	@echo ""
	@echo "=== PVCs ==="
	@kubectl get pvc -n $(K3S_NAMESPACE)

k3s-scale-predict: ## Scale predict deployment (usage: make k3s-scale-predict REPLICAS=3)
	@if [ -z "$(REPLICAS)" ]; then \
		echo "Error: REPLICAS variable required. Usage: make k3s-scale-predict REPLICAS=3"; \
		exit 1; \
	fi
	@echo "Scaling predict deployment to $(REPLICAS) replicas..."
	@kubectl scale deployment predict -n $(K3S_NAMESPACE) --replicas=$(REPLICAS)
	@echo "Scaled successfully!"

k3s-restart: k3s-check ## Safely restart all services after rebuilding images (imports images + rolling restart)
	@echo "🔄 Safely restarting all k3s services..."
	@echo ""
	@echo "Step 1: Importing new images..."
	@$(MAKE) k3s-import-images
	@echo ""
	@echo "Step 2: Rolling restart of all deployments (zero-downtime)..."
	@echo "Restarting monitoring services..."
	@kubectl rollout restart deployment/prometheus -n $(K3S_NAMESPACE) || true
	@kubectl rollout restart deployment/grafana -n $(K3S_NAMESPACE) || true
	@echo "Restarting backend services..."
	@kubectl rollout restart deployment/auth -n $(K3S_NAMESPACE)
	@kubectl rollout restart deployment/data -n $(K3S_NAMESPACE)
	@kubectl rollout restart deployment/train -n $(K3S_NAMESPACE)
	@kubectl rollout restart deployment/predict -n $(K3S_NAMESPACE)
	@kubectl rollout restart deployment/geocode -n $(K3S_NAMESPACE)
	@kubectl rollout restart deployment/weather -n $(K3S_NAMESPACE)
	@kubectl rollout restart deployment/docs -n $(K3S_NAMESPACE)
	@kubectl rollout restart deployment/dashboard -n $(K3S_NAMESPACE)
	@echo "Restarting nginx (depends on other services)..."
	@kubectl rollout restart deployment/nginx -n $(K3S_NAMESPACE)
	@echo ""
	@echo "Step 3: Waiting for all rollouts to complete..."
	@echo "Waiting for prometheus..."
	@kubectl rollout status deployment/prometheus -n $(K3S_NAMESPACE) --timeout=300s || true
	@echo "Waiting for grafana..."
	@kubectl rollout status deployment/grafana -n $(K3S_NAMESPACE) --timeout=300s || true
	@echo "Waiting for auth..."
	@kubectl rollout status deployment/auth -n $(K3S_NAMESPACE) --timeout=300s || true
	@echo "Waiting for data..."
	@kubectl rollout status deployment/data -n $(K3S_NAMESPACE) --timeout=300s || true
	@echo "Waiting for train..."
	@kubectl rollout status deployment/train -n $(K3S_NAMESPACE) --timeout=300s || true
	@echo "Waiting for predict..."
	@kubectl rollout status deployment/predict -n $(K3S_NAMESPACE) --timeout=300s || true
	@echo "Waiting for geocode..."
	@kubectl rollout status deployment/geocode -n $(K3S_NAMESPACE) --timeout=300s || true
	@echo "Waiting for weather..."
	@kubectl rollout status deployment/weather -n $(K3S_NAMESPACE) --timeout=300s || true
	@echo "Waiting for docs..."
	@kubectl rollout status deployment/docs -n $(K3S_NAMESPACE) --timeout=300s || true
	@echo "Waiting for dashboard..."
	@kubectl rollout status deployment/dashboard -n $(K3S_NAMESPACE) --timeout=300s || true
	@echo "Waiting for nginx..."
	@kubectl rollout status deployment/nginx -n $(K3S_NAMESPACE) --timeout=300s || true
	@echo ""
	@echo "Step 4: Verifying deployment status..."
	@kubectl get pods -n $(K3S_NAMESPACE)
	@echo ""
	@echo "✅ Restart completed! All services should be running with new images."
	@echo "Check status with: make k3s-status"

k3s-reload-model: ## Run model-reload Job then rollout restart predict deployment
	@echo "Reloading model..."
	@echo "Step 1: Deleting old model-reload job if exists..."
	@kubectl delete job model-reload -n $(K3S_NAMESPACE) --ignore-not-found=true
	@echo "Step 2: Creating model-reload job..."
	@kubectl apply -f $(K3S_DIR)/30-model-reload-job.yaml
	@echo "Step 3: Waiting for job to complete..."
	@kubectl wait --for=condition=complete --timeout=300s job/model-reload -n $(K3S_NAMESPACE) || \
		(echo "Error: Job failed or timed out. Check logs with: kubectl logs job/model-reload -n $(K3S_NAMESPACE)" && exit 1)
	@echo "Step 4: Rolling restart predict deployment..."
	@kubectl rollout restart deployment/predict -n $(K3S_NAMESPACE)
	@echo "Step 5: Waiting for rollout to complete..."
	@kubectl rollout status deployment/predict -n $(K3S_NAMESPACE)
	@echo "Model reload completed successfully!"

k3s-logs-predict: ## Follow logs for predict pods
	@echo "Following predict service logs..."
	@kubectl logs -n $(K3S_NAMESPACE) -l app=predict -f

k3s-logs: ## Follow logs for all services (usage: make k3s-logs SERVICE=auth)
	@if [ -z "$(SERVICE)" ]; then \
		echo "Error: SERVICE variable required. Usage: make k3s-logs SERVICE=auth"; \
		echo "Available services: auth, data, train, predict, geocode, weather, docs, dashboard, nginx, postgres"; \
		exit 1; \
	fi
	@echo "Following logs for $(SERVICE)..."
	@kubectl logs -n $(K3S_NAMESPACE) -l app=$(SERVICE) -f || \
		kubectl logs -n $(K3S_NAMESPACE) deployment/$(SERVICE) -f

k3s-get-node-ip: ## Get node IP for accessing NodePort service
	@echo "Node IPs:"
	@kubectl get nodes -o wide
	@echo ""
	@echo "Access API at: http://<node-ip>:30080"

k3s-test: k3s-check ## Run API tests in k3s (builds/imports test images, creates job, waits for completion)
	@echo "Running API tests in k3s..."
	@echo ""
	@echo "Step 1: Building test service images..."
	@$(MAKE) k3s-build-test-images
	@echo ""
	@echo "Step 2: Importing test images into k3s..."
	@$(MAKE) k3s-import-test-images
	@echo ""
	@echo "Step 3: Checking if services are running..."
	@kubectl get pods -n $(K3S_NAMESPACE) -l app=nginx --no-headers 2>/dev/null | grep -q Running || \
		(echo "❌ Error: Services are not running. Deploy them first with 'make k3s-deploy'" && exit 1)
	@echo "✅ Services are running"
	@echo ""
	@echo "Step 4: Deleting old test job if exists..."
	@kubectl delete job api-test -n $(K3S_NAMESPACE) --ignore-not-found=true
	@echo ""
	@echo "Step 5: Creating test job..."
	@kubectl apply -f $(K3S_DIR)/35-test-job.yaml
	@echo ""
	@echo "Step 6: Waiting for test job to complete (max 10 minutes)..."
	@kubectl wait --for=condition=complete --timeout=600s job/api-test -n $(K3S_NAMESPACE) || \
		(echo "❌ Error: Test job failed or timed out. Check logs with: kubectl logs job/api-test -n $(K3S_NAMESPACE)" && exit 1)
	@echo ""
	@echo "Step 7: Showing test results..."
	@kubectl logs job/api-test -n $(K3S_NAMESPACE)
	@echo ""
	@echo "✅ Tests completed successfully!"

k3s-test-run: k3s-check ## Run API tests in k3s with custom command (usage: make k3s-test-run CMD="pytest tests/test_auth_service.py -v")
	@if [ -z "$(CMD)" ]; then \
		echo "Error: CMD variable required. Usage: make k3s-test-run CMD='pytest tests/test_auth_service.py -v'"; \
		exit 1; \
	fi
	@echo "Running custom test command in k3s: $(CMD)"
	@echo "Deleting old test pod if exists..."
	@kubectl delete pod api-test -n $(K3S_NAMESPACE) --ignore-not-found=true
	@echo "Creating test pod with custom command..."
	@kubectl run api-test --image=mlops-accidents:test_service --image-pull-policy=Never \
		-n $(K3S_NAMESPACE) --restart=Never \
		--env="BASE_URL=http://nginx:80" --env="PYTHONPATH=/app" --env="PYTHONUNBUFFERED=1" \
		-- sh -c "$(CMD)"
	@echo "Waiting for test pod to complete (max 10 minutes)..."
	@timeout=600; \
	while [ $$timeout -gt 0 ]; do \
		phase=$$(kubectl get pod api-test -n $(K3S_NAMESPACE) -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound"); \
		if [ "$$phase" = "Succeeded" ]; then \
			echo "✅ Test pod completed successfully"; \
			break; \
		elif [ "$$phase" = "Failed" ]; then \
			echo "❌ Test pod failed"; \
			kubectl logs pod/api-test -n $(K3S_NAMESPACE); \
			exit 1; \
		fi; \
		sleep 2; \
		timeout=$$((timeout - 2)); \
	done; \
	if [ $$timeout -le 0 ]; then \
		echo "❌ Error: Test pod timed out. Check logs with: kubectl logs pod/api-test -n $(K3S_NAMESPACE)"; \
		kubectl logs pod/api-test -n $(K3S_NAMESPACE); \
		exit 1; \
	fi
	@echo ""
	@echo "Showing test results..."
	@kubectl logs pod/api-test -n $(K3S_NAMESPACE)
	@echo ""
	@echo "✅ Custom test command completed!"

k3s-test-cov: k3s-check ## Run API tests in k3s with coverage report
	@echo "Running API tests with coverage in k3s..."
	@echo ""
	@echo "Step 1: Building test service images..."
	@$(MAKE) k3s-build-test-images
	@echo ""
	@echo "Step 2: Importing test images into k3s..."
	@$(MAKE) k3s-import-test-images
	@echo ""
	@echo "Step 3: Checking if services are running..."
	@kubectl get pods -n $(K3S_NAMESPACE) -l app=nginx --no-headers 2>/dev/null | grep -q Running || \
		(echo "❌ Error: Services are not running. Deploy them first with 'make k3s-deploy'" && exit 1)
	@echo "✅ Services are running"
	@echo ""
	@echo "Step 4: Deleting old test pod if exists..."
	@kubectl delete pod api-test -n $(K3S_NAMESPACE) --ignore-not-found=true
	@echo ""
	@echo "Step 5: Creating test pod with coverage..."
	@kubectl run api-test --image=mlops-accidents:test_service --image-pull-policy=Never \
		-n $(K3S_NAMESPACE) --restart=Never \
		--env="BASE_URL=http://nginx:80" --env="PYTHONPATH=/app" --env="PYTHONUNBUFFERED=1" \
		-- pytest tests/ -v --tb=short --cov=services --cov-report=term-missing --cov-report=html
	@echo "Waiting for test pod to complete (max 10 minutes)..."
	@timeout=600; \
	while [ $$timeout -gt 0 ]; do \
		phase=$$(kubectl get pod api-test -n $(K3S_NAMESPACE) -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound"); \
		if [ "$$phase" = "Succeeded" ]; then \
			echo "✅ Test pod completed successfully"; \
			break; \
		elif [ "$$phase" = "Failed" ]; then \
			echo "❌ Test pod failed"; \
			kubectl logs pod/api-test -n $(K3S_NAMESPACE); \
			exit 1; \
		fi; \
		sleep 2; \
		timeout=$$((timeout - 2)); \
	done; \
	if [ $$timeout -le 0 ]; then \
		echo "❌ Error: Test pod timed out. Check logs with: kubectl logs pod/api-test -n $(K3S_NAMESPACE)"; \
		kubectl logs pod/api-test -n $(K3S_NAMESPACE); \
		exit 1; \
	fi
	@echo ""
	@echo "Showing test results..."
	@kubectl logs pod/api-test -n $(K3S_NAMESPACE)
	@echo ""
	@echo "✅ Coverage tests completed!"

k3s-test-service: k3s-check ## Run tests for specific service (usage: make k3s-test-service SERVICE=auth)
	@if [ -z "$(SERVICE)" ]; then \
		echo "Error: SERVICE variable required. Usage: make k3s-test-service SERVICE=auth"; \
		echo "Available services: auth, data, train, predict"; \
		exit 1; \
	fi
	@echo "Running tests for $(SERVICE) service in k3s..."
	@echo "Deleting old test pod if exists..."
	@kubectl delete pod api-test -n $(K3S_NAMESPACE) --ignore-not-found=true
	@echo "Creating test pod for $(SERVICE) service..."
	@kubectl run api-test --image=mlops-accidents:test_service --image-pull-policy=Never \
		-n $(K3S_NAMESPACE) --restart=Never \
		--env="BASE_URL=http://nginx:80" --env="PYTHONPATH=/app" --env="PYTHONUNBUFFERED=1" \
		-- pytest tests/test_$(SERVICE)_service.py -v --tb=short
	@echo "Waiting for test pod to complete (max 10 minutes)..."
	@timeout=600; \
	while [ $$timeout -gt 0 ]; do \
		phase=$$(kubectl get pod api-test -n $(K3S_NAMESPACE) -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound"); \
		if [ "$$phase" = "Succeeded" ]; then \
			echo "✅ Test pod completed successfully"; \
			break; \
		elif [ "$$phase" = "Failed" ]; then \
			echo "❌ Test pod failed"; \
			kubectl logs pod/api-test -n $(K3S_NAMESPACE); \
			exit 1; \
		fi; \
		sleep 2; \
		timeout=$$((timeout - 2)); \
	done; \
	if [ $$timeout -le 0 ]; then \
		echo "❌ Error: Test pod timed out. Check logs with: kubectl logs pod/api-test -n $(K3S_NAMESPACE)"; \
		kubectl logs pod/api-test -n $(K3S_NAMESPACE); \
		exit 1; \
	fi
	@echo ""
	@echo "Showing test results..."
	@kubectl logs pod/api-test -n $(K3S_NAMESPACE)
	@echo ""
	@echo "✅ Tests for $(SERVICE) service completed!"

k3s-test-logs: k3s-check ## Show logs from test job/pod
	@echo "Showing test logs..."
	@kubectl logs job/api-test -n $(K3S_NAMESPACE) 2>/dev/null || \
		kubectl logs pod/api-test -n $(K3S_NAMESPACE) 2>/dev/null || \
		(echo "Error: Test job/pod not found. Run tests first with 'make k3s-test'" && exit 1)

k3s-test-clean: k3s-check ## Delete test job and pod
	@echo "Deleting test resources..."
	@kubectl delete job api-test -n $(K3S_NAMESPACE) --ignore-not-found=true
	@kubectl delete pod api-test -n $(K3S_NAMESPACE) --ignore-not-found=true
	@echo "Test resources deleted!"
