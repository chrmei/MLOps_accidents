# Technology Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Containerization** | **Docker** | Multi-stage containers for dev, train, and prod |
| **Orchestration** | **k3s (Kubernetes)** | Production deployment with horizontal scaling |
| **Data Versioning** | **DVC + Dagshub** | Versioning raw data and artifacts without AWS |
| **Experiment Tracking** | **MLflow (Dagshub)** | Tracking metrics, parameters, and models |
| **Model Serving** | **FastAPI** | REST API for real-time accident prediction |
| **CI/CD** | **GitHub Actions** | Automated testing, linting, and image building |
| **Machine Learning** | **scikit-learn, XGBoost, LightGBM** | Model training and evaluation |
| **Data Processing** | **pandas, numpy** | Data manipulation and preprocessing |
| **Testing** | **pytest** | Unit and integration testing |
| **Code Quality** | **black, flake8, isort** | Code formatting and linting |
| **Dependency Management** | **UV** | Fast Python package installer and resolver |
| **Build System** | **setuptools** | Package building and distribution |
| **Monitoring** | **Prometheus, Grafana** | Performance monitoring and dashboards |
| **API Gateway** | **Nginx** | Reverse proxy and request routing |
| **Database** | **PostgreSQL** | Persistent storage for users and job logs |
