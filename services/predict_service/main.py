"""
Predict service FastAPI application.

The best Production model from MLflow is loaded once at container startup
and reused for all prediction requests. The train service operates
independently and registers models to MLflow; this service only reads from it.

In k3s deployments, the service checks MODEL_CACHE_DIR first for a cached model
before falling back to MLflow. This enables faster startup and model updates
via the model-reload Job.
"""

import logging
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.common.config import settings
from services.common.models import HealthResponse

from .api.routes import router as predict_router

logger = logging.getLogger(__name__)

# Config path relative to project root (WORKDIR /app in container)
MODEL_CONFIG_PATH = "src/config/model_config.yaml"


def load_model_from_cache(cache_dir: str):
    """
    Load model from cache directory if available.
    
    Returns:
        tuple: (model, label_encoders, metadata, model_type, model_uri) or None if not found
    """
    cache_path = Path(cache_dir)
    model_file = cache_path / "model.pkl"
    encoders_file = cache_path / "label_encoders.pkl"
    metadata_file = cache_path / "metadata.pkl"
    model_type_file = cache_path / "model_type.txt"
    model_uri_file = cache_path / "model_uri.txt"
    
    if not model_file.exists() or not encoders_file.exists() or not metadata_file.exists():
        return None
    
    try:
        logger.info(f"Loading model from cache: {cache_dir}")
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        with open(encoders_file, "rb") as f:
            label_encoders = pickle.load(f)
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
        
        model_type = "unknown"
        if model_type_file.exists():
            with open(model_type_file, "r") as f:
                model_type = f.read().strip()
        
        model_uri = None
        if model_uri_file.exists():
            with open(model_uri_file, "r") as f:
                model_uri = f.read().strip()
        
        logger.info(f"Successfully loaded model from cache (model_type={model_type})")
        return model, label_encoders, metadata, model_type, model_uri
    except Exception as e:
        logger.warning(f"Failed to load model from cache: {e}", exc_info=True)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the best Production model at startup.
    
    Priority:
    1. MODEL_CACHE_DIR (if set and cache exists) - for k3s deployments
    2. MLflow Production model - fallback or when cache not available
    """
    from src.models.predict_model import load_best_production_model

    # Check for cached model first (k3s deployment)
    model_cache_dir = os.getenv("MODEL_CACHE_DIR")
    if model_cache_dir:
        cached_model = load_model_from_cache(model_cache_dir)
        if cached_model:
            model, label_encoders, metadata, model_type, model_uri = cached_model
            app.state.model_cache = {
                "model": model,
                "label_encoders": label_encoders,
                "metadata": metadata,
                "model_type": model_type,
                "model_uri": model_uri,
            }
            logger.info(
                "Loaded model from cache (model_type=%s, cache_dir=%s)",
                model_type,
                model_cache_dir,
            )
            yield
            return
    
    # Fallback to MLflow
    try:
        logger.info("Loading best Production model from MLflow...")
        result = load_best_production_model(
            config_path=MODEL_CONFIG_PATH,
        )
        # Handle both old (4-tuple) and new (5-tuple) return formats
        if len(result) == 5:
            model, label_encoders, metadata, model_type, model_uri = result
        else:
            model, label_encoders, metadata, model_type = result
            model_uri = None
    except Exception as e:
        logger.exception("Failed to load best Production model from MLflow at startup")
        raise RuntimeError(
            "Predict service requires a Production model in MLflow or cached model. "
            "Ensure MLFLOW_TRACKING_URI is set and at least one model is in Production, "
            "or run the model-reload Job to populate the cache."
        ) from e

    app.state.model_cache = {
        "model": model,
        "label_encoders": label_encoders,
        "metadata": metadata,
        "model_type": model_type,
        "model_uri": model_uri,
    }
    logger.info("Loaded best Production model from MLflow (model_type=%s)", model_type)
    yield
    # Shutdown: nothing to release


app = FastAPI(
    title="MLOps Predict Service",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def healthcheck() -> HealthResponse:
    """Service liveness probe."""
    return HealthResponse(service=settings.SERVICE_NAME)


app.include_router(predict_router)
