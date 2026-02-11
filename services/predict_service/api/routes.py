"""
FastAPI routes for the predict service.
"""

import asyncio
import time

from fastapi import APIRouter, Depends, Request

from services.common.dependencies import AuthenticatedUser
from services.common.models import ModelMetrics

from ..core.predictor import evaluate_test_set, make_batch_prediction, make_prediction
from ..core.prom_metrics import (
    api_request_duration_seconds,
    api_requests_total,
    evidently_col_drift_share,
    model_accuracy_score,
    model_f1_score,
    model_precision_score,
    model_recall_score,
)
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    EvaluationRequest,
    EvaluationResponse,
    FeatureEngineeringConfig,
    InputFeaturesResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

router = APIRouter(prefix="/api/v1/predict", tags=["predict"])


def get_model_cache(request: Request):
    """Dependency: cached model loaded at startup."""
    return request.app.state.model_cache


@router.post("/", response_model=PredictionResponse)
async def single_prediction(
    request: PredictionRequest,
    http_request: Request,
    current_user: AuthenticatedUser,
    model_cache=Depends(get_model_cache),
):
    """Make a single prediction using the loaded Production model."""
    start_time = time.time()
    result = await asyncio.to_thread(
        make_prediction,
        features=request.features,
        model_cache=model_cache,
    )
    end_time = time.time()
    duration = end_time - start_time
    api_request_duration_seconds.labels(
        endpoint=http_request.url.path, method="POST", status_code="200"
    ).observe(duration)
    api_requests_total.labels(
        endpoint=http_request.url.path, method="POST", status_code="200"
    ).inc()
    return PredictionResponse(
        prediction=result["prediction"],
        probability=result["probability"],
        model_type=result["model_type"],
    )


@router.post("/batch", response_model=BatchPredictionResponse)
async def batch_prediction(
    request: BatchPredictionRequest,
    http_request: Request,
    current_user: AuthenticatedUser,
    model_cache=Depends(get_model_cache),
):
    """Make batch predictions using the loaded Production model."""
    start_time = time.time()
    result = await asyncio.to_thread(
        make_batch_prediction,
        features_list=request.features_list,
        model_cache=model_cache,
    )
    end_time = time.time()
    duration = end_time - start_time
    api_request_duration_seconds.labels(
        endpoint=http_request.url.path, method="POST", status_code="200"
    ).observe(duration)
    api_requests_total.labels(
        endpoint=http_request.url.path, method="POST", status_code="200"
    ).inc()
    return BatchPredictionResponse(
        predictions=result["predictions"],
        probabilities=result["probabilities"],
        count=result["count"],
        model_type=result["model_type"],
    )


@router.get("/models")
async def list_loaded_model(
    http_request: Request,
    current_user: AuthenticatedUser,
    model_cache=Depends(get_model_cache),
):
    """Return the currently loaded Production model (loaded at container start)."""
    api_requests_total.labels(
        endpoint=http_request.url.path, method="GET", status_code="200"
    ).inc()
    return {
        "loaded_model_type": model_cache["model_type"],
        "source": "MLflow Production (best by f1_score)",
    }


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_model(
    request: EvaluationRequest,
    http_request: Request,
    current_user: AuthenticatedUser,
    model_cache=Depends(get_model_cache),
):
    """Evaluate the currently loaded Production model on provided evaluation data."""
    start_time = time.time()
    result = await asyncio.to_thread(
        evaluate_test_set,
        eval_data=request.eval_data,
        ref_data=request.ref_data,
        model_cache=model_cache,
    )
    end_time = time.time()
    duration = end_time - start_time
    api_request_duration_seconds.labels(
        endpoint=http_request.url.path, method="POST", status_code="200"
    ).observe(duration)
    api_requests_total.labels(
        endpoint=http_request.url.path, method="POST", status_code="200"
    ).inc()

    # Update Prometheus Model Metrics
    model_accuracy_score.set(result["metrics"]["accuracy"])
    model_precision_score.set(result["metrics"]["precision"])
    model_recall_score.set(result["metrics"]["recall"])
    model_f1_score.set(result["metrics"]["f1_score"])
    if result["col_drift_share"] is not None:
        evidently_col_drift_share.set(result["col_drift_share"])

    return EvaluationResponse(
        metrics=ModelMetrics(**result["metrics"]),
        col_drift_share=result["col_drift_share"],
        model_type=result["model_type"],
    )


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info(
    current_user: AuthenticatedUser,
    model_cache=Depends(get_model_cache),
):
    """Get detailed model information including feature engineering config."""
    metadata = model_cache.get("metadata", {})
    model_type = model_cache.get("model_type", "Unknown")

    # Try to get MLflow signature if model_uri is available
    mlflow_signature_available = False
    input_features = None

    # Check if model_uri is in cache (might be added during model loading)
    model_uri = model_cache.get("model_uri")
    if model_uri:
        try:
            from mlflow.models.model import get_model_info

            model_info = get_model_info(model_uri)
            if model_info.signature:
                mlflow_signature_available = True
                input_features = model_info.signature.inputs.input_names()
        except Exception:
            pass

    # Fallback to metadata if signature not available
    if input_features is None:
        input_features = metadata.get("input_features")
        if input_features is None:
            # Last resort: use canonical schema
            from src.features.schema import get_canonical_input_features

            input_features = get_canonical_input_features()

    # Build feature engineering config
    fe_config = FeatureEngineeringConfig(
        feature_engineering_version=metadata.get("feature_engineering_version"),
        uses_grouped_features=metadata.get("uses_grouped_features", False),
        grouped_feature_mappings=metadata.get("grouped_feature_mappings"),
        removed_features=metadata.get("removed_features"),
        apply_cyclic_encoding=metadata.get("apply_cyclic_encoding", True),
        apply_interactions=metadata.get("apply_interactions", True),
    )

    return ModelInfoResponse(
        model_type=model_type,
        model_version=None,  # Could be extracted from MLflow if needed
        input_features=input_features,
        feature_engineering_config=fe_config,
        mlflow_signature_available=mlflow_signature_available,
    )


@router.get("/input-features", response_model=InputFeaturesResponse)
async def get_input_features(
    current_user: AuthenticatedUser,
    model_cache=Depends(get_model_cache),
):
    """Get canonical input features expected by the model."""
    metadata = model_cache.get("metadata", {})

    # Try MLflow signature first (most reliable)
    model_uri = model_cache.get("model_uri")
    source = "inferred"
    input_features = None

    if model_uri:
        try:
            from mlflow.models.model import get_model_info

            model_info = get_model_info(model_uri)
            if model_info.signature:
                input_features = model_info.signature.inputs.input_names()
                source = "mlflow_signature"
        except Exception:
            pass

    # Fallback to metadata
    if input_features is None and metadata.get("input_features"):
        input_features = metadata["input_features"]
        source = "metadata"

    # Last resort: use canonical schema
    if input_features is None:
        from src.features.schema import get_canonical_input_features

        input_features = get_canonical_input_features()
        source = "canonical_schema"

    return InputFeaturesResponse(
        input_features=input_features,
        uses_grouped_features=metadata.get("uses_grouped_features", False),
        feature_engineering_version=metadata.get("feature_engineering_version"),
        source=source,
    )
