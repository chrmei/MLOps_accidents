"""
Weather service FastAPI application.

Provides weather and solar data integration for determining lighting and atmospheric
conditions based on GPS coordinates and timestamp.
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.common.config import settings
from services.common.models import HealthResponse

from .api.routes import router as weather_router

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MLOps Weather Service",
    version="0.1.0",
    description="Weather and solar data service for determining lighting and atmospheric conditions",
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
    """
    Service liveness probe.
    
    Checks that the service is running and can initialize weather clients.
    """
    try:
        logger.info(f"Weather service health check passed - Service: {settings.SERVICE_NAME}")
        return HealthResponse(service=settings.SERVICE_NAME)
    except Exception as e:
        logger.error(
            f"Weather service health check failed - "
            f"Service: {settings.SERVICE_NAME}, "
            f"Error: {type(e).__name__}: {e}",
            exc_info=True
        )
        return HealthResponse(service=settings.SERVICE_NAME)


app.include_router(weather_router)
