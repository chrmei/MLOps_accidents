# Team Structure

The project is designed for a 3-person team with clear separation of concerns:

## Engineer A: Data & Pipeline Infrastructure

- **Focus**: Getting data from raw to "model-ready"
- **Deliverables**: DVC pipelines, data validation (Pandera or manual), Dagshub integration
- **Primary Files**: `src/data/`, `dvc.yaml`, `params.yaml`
- **Status**: 
  - DVC + Dagshub integration in progress
  - Data validation pending

## Engineer B: ML Modeling & Tracking

- **Focus**: ML model development and experiment logging
- **Deliverables**: Training scripts, MLflow tracking, model registry management
- **Primary Files**: `src/models/`, `src/features/`, `src/config/model_config.yaml`
- **Status**: 
  - Feature engineering pipeline complete
  - Model training and prediction scripts functional
  - Config-driven training (ML-1) - Complete
  - MLflow integration (ML-2) - Model registry with versioning and staging implemented
  - DVC metrics format (ML-3) - Partial (metrics saved, DVC format pending)
  - Unit tests (TEST-1) - Pending

## Engineer C: API, Docker & CI/CD

- **Focus**: Containerization, API development, and automation
- **Deliverables**: FastAPI application, Dockerfiles, GitHub Actions pipelines
- **Primary Files**: `src/api/`, `Dockerfile`, `.github/workflows/`
- **Status**:
  - Multi-stage Dockerfile implemented (dev, train, prod template)
  - Docker Compose configuration complete
  - Microservices Architecture implemented
  - k3s deployment manifests created
  - CI/CD pipeline implemented
