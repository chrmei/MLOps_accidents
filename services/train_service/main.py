"""
Train service FastAPI application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.common.config import settings
from services.common.models import HealthResponse

from .api.routes import router as train_router

app = FastAPI(title="MLOps Train Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def healthcheck() -> HealthResponse:
    return HealthResponse(service=settings.SERVICE_NAME)


app.include_router(train_router)
