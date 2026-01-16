# Microservices Architecture Implementation Plan

## Architecture Overview

Transform the monolithic MLOps project into 3 microservices:

1. **Data Service**: Handles data preprocessing (`make_dataset.py`, `build_features.py`)
2. **Train Service**: Handles model training (`train_multi_model.py`, `base_trainer.py`)
3. **Predict Service**: Handles model inference (`predict_model.py`)

All services will be containerized with individual Dockerfiles, orchestrated via docker-compose, and fronted by Nginx as a reverse proxy with JWT authentication.

## Architecture Diagram

```flowchart LR
        Client["Client"] --> Nginx["Nginx"]

        subgraph Docker["Docker Network"]
            direction TB
            Nginx --> Data["Data :8001"]
            Nginx --> Train["Train :8002"]
            Nginx --> Predict["Predict :8003"]

            Volume["Shared Volume"]
            Data --- Volume
            Train --- Volume
            Predict --- Volume
        end

        subgraph ML["ML Infrastructure"]
            direction TB
            S3["S3 / DVC"]
            MLflow["MLflow"]
        end

        Data -->|DVC| S3
        Train -->|DVC| S3
        Predict -->|MLflow| MLflow
```

## Implementation Structure

### 1. Project Structure

```
services/
├── common/                    # Shared code across services
│   ├── __init__.py
│   ├── auth.py               # JWT authentication & RBAC
│   ├── models.py             # Common Pydantic models
│   ├── dependencies.py       # FastAPI dependencies
│   └── config.py             # Shared configuration
│
├── data_service/
│   ├── Dockerfile
│   ├── main.py               # FastAPI app
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py         # API endpoints
│   │   └── schemas.py        # Request/Response models
│   └── core/
│       ├── __init__.py
│       ├── preprocessing.py  # Wraps make_dataset.py
│       └── features.py       # Wraps build_features.py
│
├── train_service/
│   ├── Dockerfile
│   ├── main.py               # FastAPI app
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── schemas.py
│   └── core/
│       ├── __init__.py
│       └── trainer.py        # Wraps train_multi_model.py
|
├── predict_service/
│   ├── Dockerfile
│   ├── main.py               # FastAPI app
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── schemas.py
│   └── core/
│       ├── __init__.py
│       └── predictor.py      # Wraps predict_model.py
└── nginx/
    ├── nginx.conf             # Nginx configuration
    └── Dockerfile             # Optional: custom Nginx image
```

### 2. Common Security Module (`services/common/auth.py`)

- JWT token generation/validation using `python-jose[cryptography]`
- Password hashing with `passlib[bcrypt]`
- Role-based access control decorators/dependencies
- User management (admin/normal user)
- Token refresh mechanism

### 3. Individual Dockerfiles

Each service gets its own Dockerfile:

- Base image: `python:3.11-slim`
- Install dependencies from `pyproject.toml`
- Copy service-specific code + common module
- Expose service port (8001, 8002, 8003)
- Health check endpoints

### 4. Docker Compose Configuration

- 3 service containers (data, train, predict)
- Nginx container as reverse proxy
- Shared volumes for `data/` and `models/`
- Environment variables for configuration
- Network configuration for service communication
- MLflow service (optional, if using local MLflow)

### 5. Nginx Configuration

- Reverse proxy rules for each service
- JWT validation middleware (or forward to services for validation)
- SSL/TLS termination (optional)
- Rate limiting
- CORS configuration
- Health check endpoints

### 6. API Endpoints Design

**Data Service** (`/api/v1/data/`):

- `POST /preprocess` - Run make_dataset.py (admin only)
- `POST /build-features` - Run build_features.py (admin only)
- `GET /status` - Check preprocessing status (admin only)

**Train Service** (`/api/v1/train/`):

- `POST /train` - Train models (admin only)
- `POST /fetch_best_model` - fetches best model from MLflow (reference make docker-run-predict-best implements this function already)
- `GET /status/{job_id}` - Check training status (admin only)
- `GET /models` - List available models (admin only, MLFLow)
- `GET /metrics/{model_type}` - Get model metrics (admin only)


**Predict Service** (`/api/v1/predict/`):

- `POST /predict` - Make prediction (all authenticated users)
- `GET /models` - List available models (all authenticated users)
- `GET /health` - Health check (public)

### 7. Security Implementation

- JWT tokens with expiration
- Role-based access: `admin` vs `user`
- Password-based authentication endpoint
- Protected routes using FastAPI dependencies
- Token refresh endpoint

### 8. Shared Storage Strategy

- **Development**: Docker volumes for `data/` and `models/`
- [OPTIONAL] **Production**: S3/DVC remote for data, MLflow for models
- Services read/write to shared locations
- DVC integration maintained in data and train services

## Key Files to Create/Modify

1. **New Files**:

   - `services/common/auth.py` - JWT authentication
   - `services/data_service/main.py` - Data service FastAPI app
   - `services/train_service/main.py` - Train service FastAPI app
   - `services/predict_service/main.py` - Predict service FastAPI app
   - `services/nginx/nginx.conf` - Nginx configuration
   - `docker-compose.microservices.yml` - New compose file
   - Individual Dockerfiles for each service

2. **Modified Files**:

   - Existing Python modules will be imported/wrapped by services
   - `docker-compose.yml` - Keep for development, add new compose file

3. **Configuration**:

   - Environment variables for JWT secrets, service ports, MLflow URI
   - `.env.example` for configuration template

## Dependencies to Add

- `fastapi` - API framework
- `uvicorn[standard]` - ASGI server
- `python-jose[cryptography]` - JWT handling
- `passlib[bcrypt]` - Password hashing
- `python-multipart` - Form data handling
- `pydantic` - Data validation (likely already in FastAPI deps)

## Migration Strategy

1. Create common auth module
2. Create data service (test with existing code)
3. Create train service (test with existing code)
4. Create predict service (test with existing code)
5. Set up Nginx reverse proxy
6. Integrate JWT authentication
7. Test end-to-end workflow
8. Update documentation