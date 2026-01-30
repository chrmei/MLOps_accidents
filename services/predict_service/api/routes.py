"""
FastAPI routes for the predict service.
"""

from fastapi import APIRouter, Depends

from services.common.dependencies import AuthenticatedUser

from ..core.predictor import make_batch_prediction, make_prediction
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)

router = APIRouter(prefix="/api/v1/predict", tags=["predict"])

# ============================================================================
# Prediction Routes
# ============================================================================


@router.post("/", response_model=PredictionResponse)
async def single_prediction(
    request: PredictionRequest,
    current_user: AuthenticatedUser,
):
    """Make a single prediction (authenticated users)."""
    result = make_prediction(
        features=request.features,
        model_type=request.model_type,
    )

    return PredictionResponse(
        prediction=result["prediction"],
        model_type=result["model_type"],
    )


@router.post("/batch", response_model=BatchPredictionResponse)
async def batch_prediction(
    request: BatchPredictionRequest,
    current_user: AuthenticatedUser,
):
    """Make batch predictions (authenticated users)."""
    result = make_batch_prediction(
        features_list=request.features_list,
        model_type=request.model_type,
    )

    return BatchPredictionResponse(
        predictions=result["predictions"],
        count=result["count"],
        model_type=result["model_type"],
    )


@router.get("/models")
async def list_available_models(current_user: AuthenticatedUser):
    """List available models for prediction."""
    return {
        "models": ["xgboost", "random_forest", "logistic_regression", "lightgbm"],
        "default": "best_model",
    }
