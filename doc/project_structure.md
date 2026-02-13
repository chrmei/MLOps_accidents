# Project Structure

```
MLOps_accidents/
├── .github/workflows/         # CI/CD pipelines
│   ├── ci.yaml                # CI pipeline (tests, linting)
│   ├── docker-publish.yml     # Docker image building and publishing
│   └── lint.yaml              # Linting pipeline
├── data/                      # Data directory (created at runtime)
│   ├── external/              # Data from third party sources
│   ├── interim/               # Intermediate data that has been transformed
│   ├── processed/             # The final, canonical data sets for modeling
│   └── raw/                   # The original, immutable data dump
├── deploy/                    # Deployment configurations
│   └── k3s/                   # Kubernetes (k3s) manifests
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
├── services/                  # Microservice implementations
│   ├── auth/                  # Authentication service
│   ├── data/                  # Data preprocessing service
│   ├── predict/               # Prediction service
│   └── train/                 # Training service
├── src/                       # Source code for use in this project
│   ├── __init__.py
│   ├── config/                # Configuration files (YAML configs)
│   │   └── model_config.yaml
│   ├── data/                  # Scripts to download or generate data
│   │   ├── __init__.py
│   │   ├── check_structure.py
│   │   ├── import_raw_data.py # Downloads data from S3
│   │   └── make_dataset.py    # Preprocesses raw data
│   ├── features/              # Scripts to turn raw data into features
│   │   ├── __init__.py
│   │   ├── build_features.py
│   │   └── preprocess.py      # Reusable preprocessing for inference
│   ├── models/                # Scripts to train models and make predictions
│   │   ├── __init__.py
│   │   ├── predict_model.py   # Model inference script
│   │   ├── train_multi_model.py # Multi-model training framework
│   │   └── test_features.json # Example features for testing
│   └── visualization/         # Scripts to create visualizations
│       ├── __init__.py
│       └── visualize.py
├── scripts/                   # Utility scripts
│   ├── manage_model_registry.py # MLflow model registry management
│   └── setup_dvc_remote.sh
├── tests/                     # Pytest suite for API endpoint testing
│   ├── __init__.py
│   ├── conftest.py           # Shared fixtures and configuration
│   ├── test_auth_service.py  # Auth service endpoint tests
│   ├── test_data_service.py  # Data service endpoint tests
│   ├── test_train_service.py # Train service endpoint tests
│   ├── test_predict_service.py # Predict service endpoint tests
│   ├── README.md             # Test suite documentation
│   ├── TEST_SERVICE.md        # Docker test service usage guide
│   └── run_tests.sh          # Convenience script for running tests
├── Dockerfile                 # Multi-stage Dockerfile
├── docker-compose.yml         # Docker Compose configuration
├── dvc.yaml                   # DVC pipeline definition
├── LICENSE                    # MIT License
├── Makefile                   # Development commands and automation
├── pyproject.toml             # Python project configuration and dependencies
├── requirements.txt           # Python dependencies (legacy, use pyproject.toml)
├── setup.py                   # Package setup configuration (legacy)
└── README.md                  # Project overview and quick start
```
