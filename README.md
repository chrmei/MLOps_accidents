# 🚦 MLOps Road Accident Prediction

A containerized MLOps project for predicting road accidents using machine learning. This project implements a complete MLOps workflow from data ingestion to model serving, with a focus on reproducibility, versioning, and best practices.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Current State](#current-state)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [Workflow](#workflow)
- [Team Structure](#team-structure)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project is an MLOps implementation for road accident prediction, designed as a learning and production-ready template. The system processes road accident data from multiple sources, trains machine learning models to predict accident severity, and serves predictions through a REST API.

### Objectives

- **Reproducibility**: Ensure all experiments and workflows can be reproduced across different environments
- **Versioning**: Track data, models, and experiments using DVC and MLflow
- **Containerization**: Isolate environments using Docker for consistent deployments
- **Automation**: Implement CI/CD pipelines for testing, linting, and deployment
- **Monitoring**: Track model performance and detect data drift in production

### Success Metrics

- **Model Performance**: F1 Score, Precision, and Recall for accident severity classification
- **Baseline Model**: RandomForest with default parameters as initial benchmark
- **Minimum Performance Threshold**: To be defined based on baseline results

## 📊 Current State

**Phase**: Phase 1 - Foundations (Deadline: December 19th)

### ✅ Completed

- Project structure based on cookiecutter-data-science template
- Reproducible Python environment setup with `pyproject.toml` and UV
- Development automation with Makefile
- Python version management (`.python-version`, `.tool-versions`)

- Initial data exploration notebook

### 🚧 In Progress

- Containerization with Docker
- Data import pipeline (`import_raw_data.py`)
- Data preprocessing pipeline (`make_dataset.py`)
- Baseline model training (`train_model.py`)
- Model prediction script (`predict_model.py`)
- DVC + Dagshub integration for data versioning
- MLflow integration for experiment tracking
- FastAPI service for model serving
- Unit testing suite
- CI/CD pipeline with GitHub Actions

### 📝 Planned

See [Roadmap](#roadmap) section for detailed phase breakdown.

## 📁 Project Structure

```
MLOps_accidents/
├── .github/workflows/         # CI/CD pipelines (to be implemented)
│   ├── lint.yaml
│   └── test.yaml
├── data/                      # Data directory (created at runtime)
│   ├── external/              # Data from third party sources
│   ├── interim/               # Intermediate data that has been transformed
│   ├── processed/             # The final, canonical data sets for modeling
│   └── raw/                   # The original, immutable data dump
├── doc/                       # Project documentation
│   ├── README_INITIAL.md      # Initial project documentation
│   ├── Plan_Phase_01.md       # Detailed Phase 1 execution plan
│   └── Roadmap.md             # Project roadmap and milestones
├── logs/                      # Logs from training and predicting
├── models/                    # Trained and serialized models
├── notebooks/                 # Jupyter notebooks
│   └── 1.0-ldj-initial-data-exploration.ipynb
├── references/                # Data dictionaries, manuals, and explanatory materials
├── reports/                   # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures/               # Generated graphics and figures
├── src/                       # Source code for use in this project
│   ├── __init__.py
│   ├── config/                # Configuration files (YAML configs)
│   ├── data/                  # Scripts to download or generate data
│   │   ├── __init__.py
│   │   ├── check_structure.py
│   │   ├── import_raw_data.py # Downloads data from S3
│   │   └── make_dataset.py    # Preprocesses raw data
│   ├── features/              # Scripts to turn raw data into features
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/                # Scripts to train models and make predictions
│   │   ├── __init__.py
│   │   ├── predict_model.py   # Model inference script
│   │   ├── train_model.py     # Model training script
│   │   └── test_features.json # Example features for testing
│   └── visualization/         # Scripts to create visualizations
│       ├── __init__.py
│       └── visualize.py
├── tests/                     # Pytest suite (to be implemented)
│   ├── test_data.py
│   ├── test_models.py
│   └── test_api.py
├── .dockerignore              # Docker ignore file (to be created)
├── .python-version            # Python version for pyenv
├── .tool-versions             # Python version for asdf
├── Dockerfile                 # Multi-stage Dockerfile (to be created)
├── docker-compose.yaml        # Local orchestration (to be created)
├── dvc.yaml                   # DVC pipeline definition (to be created)
├── LICENSE                    # MIT License
├── Makefile                   # Development commands and automation
├── pyproject.toml             # Python project configuration and dependencies
├── requirements.txt           # Python dependencies (legacy, use pyproject.toml)
├── setup.py                   # Package setup configuration (legacy)
└── README.md                  # This file
```

## 🛠️ Technology Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Data Versioning** | **DVC + Dagshub** | Versioning raw data and artifacts without AWS |
| **Experiment Tracking** | **MLflow (Dagshub)** | Tracking metrics, parameters, and models |
| **Pipeline Stages** | **Docker Containers** | Isolated environments for Data, Training, and API |
| **Model Serving** | **FastAPI** | REST API for real-time accident prediction |
| **CI/CD** | **GitHub Actions** | Automated testing, linting, and image building |
| **Machine Learning** | **scikit-learn** | Model training and evaluation |
| **Data Processing** | **pandas, numpy** | Data manipulation and preprocessing |
| **Testing** | **pytest** | Unit and integration testing |
| **Code Quality** | **black, flake8, isort** | Code formatting and linting |
| **Dependency Management** | **UV** | Fast Python package installer and resolver |
| **Build System** | **setuptools** | Package building and distribution |

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+** (3.11 recommended - see `.python-version`)
- **UV** - Fast Python package installer ([Install](https://github.com/astral-sh/uv): `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **DVC** - Installed automatically with dependencies
- **Docker & Docker Compose** - For containerized workflow
- **Git** & **Make** - Usually pre-installed
- **Dagshub account** - [Sign up](https://dagshub.com) for data versioning

### Quick Start

1. **Clone and setup Python** (optional)
   ```bash
   git clone https://github.com/chrmei/MLOps_accidents.git
   cd MLOps_accidents
   # Using pyenv: pyenv install 3.11.0 && pyenv local 3.11.0
   # Using asdf: asdf install python 3.11.0 && asdf reshim python
   ```

2. **Install dependencies**
   ```bash
   make install-dev  # Installs project + dev dependencies (pytest, black, isort, mypy, etc.)
   # Alternative: uv pip install -e ".[dev]"
   ```

3. **Verify installation**
   ```bash
   python --version && make help
   ```

5. **Set up DVC and Dagshub (Optional but recommended)**
   
   DVC is used for data versioning. To set it up:
   
   ```bash
   # Step 1: Create .env file from template
   cp .env.example .env
   
   # Step 2: MANUALLY EDIT .env file with your Dagshub credentials:
   #   - DAGSHUB_USERNAME: Your Dagshub username
   #   - DAGSHUB_TOKEN: Get from https://dagshub.com/user/settings/tokens
   #   - DAGSHUB_REPO: Your repository (e.g., chrmei/MLOps_accidents)
   # 
   # IMPORTANT: You must manually edit .env before running the next commands!
   
   # Step 3: Initialize DVC and configure remote using Makefile
   make dvc-init
   make dvc-setup-remote
   ```
   
   **Important Notes**:
   - The `.env` file must be **manually edited** with your credentials before running `make dvc-setup-remote`
   - The `.env` file is gitignored and will never be committed
   - Each team member should create their own `.env` file with their personal Dagshub credentials
   - See [DVC Commands](#dvc-data-version-control) section for all available DVC commands

### Run the Pipeline

```bash
# 1. Import raw data (downloads 4 CSV files from AWS S3)
make run-import

# 2. Preprocess data (creates train/test splits in data/preprocessed/)
make run-preprocess

# 3. Train baseline model (saves to src/models/trained_model.joblib)
make run-train

# 4. Make predictions
make run-predict                    # Interactive mode
make run-predict-file FILE=path     # From JSON file
```

## 🛠️ Development Commands

Run `make help` to see all commands. Key commands:

**Setup & Dependencies**
- `make install-dev` - Install with dev dependencies
- `make setup-venv` - Create venv and install dependencies
- `make clean` - Remove build artifacts

**Code Quality**
- `make format` - Format with black & isort
- `make lint` - Lint with flake8
- `make type-check` - Type checking with mypy

**Testing**
- `make test` - Run pytest
- `make test-cov` - Run with coverage report

**Data Pipeline**
- `make run-import` - Import raw data
- `make run-preprocess` - Preprocess data
- `make run-train` - Train model
- `make run-predict` - Interactive predictions
- `make run-predict-file FILE=path` - Predictions from JSON

**DVC (Data Version Control)**
- `make dvc-init` - Initialize DVC
- `make dvc-setup-remote` - Configure Dagshub remote (**requires manually edited .env**)
- `make dvc-status` / `dvc-push` / `dvc-pull` / `dvc-repro`

> **Important**: For `make dvc-setup-remote`, you must first manually edit `.env` with your Dagshub credentials (copy from `.env.example`).

## 🔄 Workflow

**Current (Phase 1)**: Data Ingestion → Preprocessing → Training → Inference  
**Target (Post Phase 1)**: DVC Pipeline → MLflow Tracking → FastAPI Serving → CI/CD

## 👥 Team Structure

The project is designed for a 3-person team with clear separation of concerns:

### **Engineer A: Data & Pipeline Infrastructure**
- **Focus**: Getting data from raw to "model-ready"
- **Deliverables**: DVC pipelines, data validation (Pandera or manual), Dagshub integration
- **Primary Files**: `src/data/`, `dvc.yaml`, `params.yaml`

### **Engineer B: ML Modeling & Tracking**
- **Focus**: ML model development and experiment logging
- **Deliverables**: Training scripts, MLflow tracking, model registry management
- **Primary Files**: `src/models/`, `src/config/model_config.yaml`

### **Engineer C: API, Docker & CI/CD**
- **Focus**: Containerization, API development, and automation
- **Deliverables**: FastAPI application, Dockerfiles, GitHub Actions pipelines
- **Primary Files**: `src/api/`, `Dockerfile`, `.github/workflows/`

## 🗺️ Roadmap

### Phase 1: Foundations (Deadline: December 19th, 2024)
- ✅ Define project objectives and key metrics
- 🚧 Set up reproducible development environment (containerization, Docker)
- 🚧 Collect and preprocess data (ML Pipeline)
- 🚧 Build and evaluate baseline ML model, implement unit tests
- 🚧 Implement basic inference API

### Phase 2: Microservices, Tracking & Versioning (Deadline: January 16th, 2026)
- Set up experiment tracking with MLflow
- Implement data & model versioning (MLflow, DVC)
- Decompose application into microservices and design orchestration

### Phase 3: Orchestration & Deployment (Deadline: January 29th, 2026)
- Finalize end-to-end orchestration
- Create CI Pipeline (GitHub Actions: linter and others)
- Optimize and secure the API
- Implement scalability with Docker/Kubernetes

### Phase 4: Monitoring & Maintenance (Deadline: February 6th, 2026)
- Set up performance monitoring using Prometheus/Grafana
- Implement drift detection with Evidently
- Develop automated model and component updates
- Finalize technical documentation

**Final Presentation (Defence)**: February 9th, 2026

For detailed execution plans, see:
- [Phase 1 Plan](doc/Plan_Phase_01.md)
- [Roadmap](doc/Roadmap.md)

## 📝 Development Guidelines

**Code**: Run scripts from project root, use conventional commits (`feat:`, `fix:`), follow PEP 8, add type hints  
**Branches**: `feature/<ticket-id>-description`, PRs require approval + passing CI  
**Testing**: >70% coverage for `src/data/`, `src/models/`, `src/api/`; run `make test` before PRs  
**Dependencies**: Managed in `pyproject.toml` via UV (`make install-dev`); Python version in `.python-version`

## 🤝 Contributing

1. Setup: `make install-dev`
2. Branch: `git checkout -b feature/your-feature-name`
3. Develop: Make changes following guidelines
4. Quality: `make format && make lint && make type-check`
5. Test: `make test` (coverage: `make test-cov`)
6. PR: Ensure all checks pass, submit with clear description

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/)
- Data sourced from French road accident databases

## 📚 Additional Documentation

- [Initial README](doc/README_INITIAL.md) - Original project documentation
- [Phase 1 Plan](doc/Plan_Phase_01.md) - Detailed Phase 1 execution plan
- [Roadmap](doc/Roadmap.md) - Project roadmap and milestones

---

**Note**: This project is in active development. The structure and workflows are being refined as we progress through Phase 1. For the most up-to-date information, refer to the documentation in the `doc/` directory.

