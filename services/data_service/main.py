"""
Data service FastAPI application.
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.common.config import settings
from services.common.models import HealthResponse

from .api.routes import router as data_router

app = FastAPI(title="MLOps Data Service", version="0.1.0")

logging.getLogger().setLevel(logging.INFO)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_data_dirs() -> None:
    """Create data directories if they do not exist."""
    for path in (
        settings.DATA_DIR,
        settings.RAW_DATA_DIR,
        settings.PREPROCESSED_DATA_DIR,
        settings.MODELS_DIR,
    ):
        if not path:
            continue
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            logging.exception("Failed to create data directory: %s", path)


@app.on_event("startup")
async def _startup() -> None:
    _ensure_data_dirs()


@app.get("/health", response_model=HealthResponse)
async def healthcheck() -> HealthResponse:
    """Service liveness probe."""
    return HealthResponse(service=settings.SERVICE_NAME)


app.include_router(data_router)
