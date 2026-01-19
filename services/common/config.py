"""
Shared configuration for MLOps microservices.

Configuration is loaded from environment variables with sensible defaults.
"""

from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ==========================================================================
    # JWT Configuration
    # ==========================================================================
    JWT_SECRET_KEY: str = "CHANGE_ME_IN_PRODUCTION_USE_STRONG_SECRET_KEY"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ==========================================================================
    # Service Configuration
    # ==========================================================================
    SERVICE_NAME: str = "mlops-service"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ==========================================================================
    # Service URLs (for inter-service communication)
    # ==========================================================================
    DATA_SERVICE_URL: str = "http://data:8001"
    TRAIN_SERVICE_URL: str = "http://train:8002"
    PREDICT_SERVICE_URL: str = "http://predict:8003"

    # ==========================================================================
    # MLflow Configuration (DagsHub Remote)
    # ==========================================================================
    MLFLOW_TRACKING_URI: str = ""  # Set via environment variable
    MLFLOW_TRACKING_USERNAME: str = ""  # DagsHub username
    MLFLOW_TRACKING_PASSWORD: str = ""  # DagsHub token
    MLFLOW_EXPERIMENT_NAME: str = "accident_severity"

    # ==========================================================================
    # Data Paths
    # ==========================================================================
    DATA_DIR: str = "/app/data"
    MODELS_DIR: str = "/app/models"
    RAW_DATA_DIR: str = "/app/data/raw"
    PREPROCESSED_DATA_DIR: str = "/app/data/preprocessed"

    # ==========================================================================
    # CORS Configuration
    # ==========================================================================
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    # ==========================================================================
    # Database Configuration (for user storage - SQLite by default)
    # ==========================================================================
    DATABASE_URL: str = "sqlite:///./users.db"

    # ==========================================================================
    # Initial Admin User
    # ==========================================================================
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "CHANGE_ME_ADMIN_PASSWORD"
    ADMIN_EMAIL: str = "admin@mlops.local"

    class Config:
        """Pydantic settings configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()


# Global settings instance
settings = get_settings()

