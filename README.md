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
- Data import pipeline (`import_raw_data.py`)
- Data preprocessing pipeline (`make_dataset.py`)
- Baseline model training (`train_model.py`)
- Model prediction script (`predict_model.py`)
- Initial data exploration notebook

### 🚧 In Progress

- Containerization with Docker
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

- **Python 3.8+** (Python 3.11 recommended - specified in `.python-version` and `.tool-versions`)
- **UV** - Fast Python package installer ([Installation guide](https://github.com/astral-sh/uv))
- **Docker and Docker Compose** (for containerized workflow)
- **Git**
- **Make** (for using Makefile commands - usually pre-installed on Linux/Mac)

### Installing UV

If you don't have UV installed, you can install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or visit the [UV installation guide](https://github.com/astral-sh/uv) for other installation methods.

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MLOps_accidents
   ```

2. **Set up Python version** (optional but recommended)
   
   If using **pyenv**:
   ```bash
   pyenv install 3.11.0  # Install Python 3.11 if not already installed
   pyenv local 3.11.0    # Use the version specified in .python-version
   ```
   
   If using **asdf**:
   ```bash
   asdf install python 3.11.0  # Install Python 3.11 if not already installed
   asdf reshim python           # Reshim after installation
   ```

3. **Install dependencies using UV and Make**
   ```bash
   make install-dev
   ```
   
   This will:
   - Check if UV is installed
   - Install the project in editable mode with all dependencies
   - Include development dependencies (pytest, black, isort, mypy, etc.)
   
   **Alternative**: Install directly with UV
   ```bash
   uv pip install -e ".[dev]"
   ```

4. **Verify installation**
   ```bash
   python --version  # Should show Python 3.8 or higher
   make help         # View all available Make commands
   ```

### Initial Setup

You can use Make commands for common tasks:

1. **Import raw data**
   ```bash
   make run-import
   # or
   python src/data/import_raw_data.py
   ```
   This will download 4 datasets from AWS S3:
   - `caracteristiques-2021.csv`
   - `lieux-2021.csv`
   - `usagers-2021.csv`
   - `vehicules-2021.csv`

2. **Preprocess data**
   ```bash
   make run-preprocess
   # or
   python src/data/make_dataset.py
   ```
   This processes the raw data and creates train/test splits in `data/preprocessed/`.

3. **Train the baseline model**
   ```bash
   make run-train
   # or
   python src/models/train_model.py
   ```
   This trains a RandomForest classifier and saves it to `src/models/trained_model.joblib`.

4. **Make predictions**
   
   Using a JSON file:
   ```bash
   make run-predict-file FILE=src/models/test_features.json
   # or
   python src/models/predict_model.py src/models/test_features.json
   ```
   
   Or interactively (you'll be prompted to enter features manually):
   ```bash
   make run-predict
   # or
   python src/models/predict_model.py
   ```

## 🛠️ Development Commands

The project includes a `Makefile` with common development tasks. Run `make help` to see all available commands:

### Dependency Management
- `make install` - Install project dependencies (production)
- `make install-dev` - Install project with development dependencies
- `make sync` - Sync dependencies from `pyproject.toml`
- `make update` - Update all dependencies to latest versions
- `make clean` - Remove build artifacts and cache files

### Code Quality
- `make lint` - Run linting with flake8
- `make format` - Format code with black and isort
- `make type-check` - Run type checking with mypy

### Testing
- `make test` - Run tests with pytest
- `make test-cov` - Run tests with coverage report

### Data Pipeline
- `make run-import` - Import raw data from S3
- `make run-preprocess` - Preprocess raw data
- `make run-train` - Train the baseline model
- `make run-predict` - Make predictions (interactive)
- `make run-predict-file FILE=path/to/file.json` - Make predictions from JSON file

### Setup
- `make setup` - Initial project setup (install dependencies)
- `make setup-venv` - Create virtual environment and install dependencies

## 🔄 Workflow

### Current Workflow (Phase 1)

1. **Data Ingestion**: Download raw data from S3 using `import_raw_data.py`
2. **Data Preprocessing**: Transform and clean data using `make_dataset.py`
3. **Model Training**: Train RandomForest model using `train_model.py`
4. **Model Inference**: Make predictions using `predict_model.py`

### Target Workflow (Post Phase 1)

1. **Data Pipeline** (DVC): Automated data ingestion → validation → preprocessing
2. **Model Training** (MLflow): Config-driven training with experiment tracking
3. **Model Serving** (FastAPI): REST API for real-time predictions
4. **CI/CD** (GitHub Actions): Automated testing and deployment

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

### Code Conventions

- All Python scripts must be run from the project root specifying the relative file path
- Use conventional commits (e.g., `feat:`, `fix:`, `docs:`)
- Follow PEP 8 style guidelines
- Use type hints where appropriate

### Branching Strategy

- Use feature branches: `feature/<ticket-id>-description`
- Pull requests require at least 1 approval and passing CI checks
- Main branch should always be in a deployable state

### Testing

- Unit tests should cover core functionality with >70% code coverage
- Tests should be written for `src/data/`, `src/models/`, and `src/api/` modules
- Run tests before submitting PRs: `make test` or `pytest`
- Generate coverage reports: `make test-cov`

### Dependency Management

- All dependencies are managed in `pyproject.toml`
- Use **UV** for installing dependencies: `make install-dev` or `uv pip install -e ".[dev]"`
- The legacy `requirements.txt` file is kept for reference but `pyproject.toml` is the source of truth
- Python version is specified in `.python-version` (for pyenv) and `.tool-versions` (for asdf)

## 🤝 Contributing

1. **Set up your development environment**
   ```bash
   make install-dev  # Install dependencies with development tools
   ```

2. **Create a feature branch from `main`**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes following the development guidelines**

4. **Format and lint your code**
   ```bash
   make format    # Format with black and isort
   make lint      # Check with flake8
   make type-check # Type checking with mypy (if applicable)
   ```

5. **Write or update tests as needed**
   ```bash
   make test      # Run tests
   make test-cov  # Run tests with coverage
   ```

6. **Ensure all tests pass and code is properly formatted**
   - All tests must pass: `make test`
   - Code must be formatted: `make format`
   - No linting errors: `make lint`

7. **Submit a pull request with a clear description**

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

