"""
Geocoding service FastAPI application.

Provides address to latitude/longitude conversion using Nominatim
(designed to be replaceable with Google Geocoding API).
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.common.config import settings
from services.common.models import HealthResponse

from .api.routes import router as geocode_router

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MLOps Geocoding Service",
    version="0.1.0",
    description="Geocoding service for address to latitude/longitude conversion",
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


app.include_router(geocode_router)
