"""
FastAPI routes for the predict service.
"""

import asyncio
import time
from fastapi import APIRouter, Depends, Request

from services.common.dependencies import AuthenticatedUser
from services.common.models import ModelMetrics

from ..core.predictor import make_batch_prediction, make_prediction, evaluate_test_set
from ..core.prom_metrics import api_requests_total, api_request_duration_seconds, model_accuracy_score, model_precision_score, model_recall_score, model_f1_score, evidently_data_drift_detected_status
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
    EvaluationRequest,
    EvaluationResponse,
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
    api_request_duration_seconds.labels(endpoint=http_request.url.path, method="POST", status_code="200").observe(duration)
    api_requests_total.labels(endpoint=http_request.url.path, method="POST", status_code="200").inc()
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
    api_request_duration_seconds.labels(endpoint=http_request.url.path, method="POST", status_code="200").observe(duration)
    api_requests_total.labels(endpoint=http_request.url.path, method="POST", status_code="200").inc()
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
    api_requests_total.labels(endpoint=http_request.url.path, method="GET", status_code="200").inc()
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
    api_request_duration_seconds.labels(endpoint=http_request.url.path, method="POST", status_code="200").observe(duration)
    api_requests_total.labels(endpoint=http_request.url.path, method="POST", status_code="200").inc()

    # Update Prometheus Model Metrics
    model_accuracy_score.set(result["metrics"]["accuracy"])
    model_precision_score.set(result["metrics"]["precision"])
    model_recall_score.set(result["metrics"]["recall"])
    model_f1_score.set(result["metrics"]["f1_score"])
    if result["data_drift"] is not None:
        evidently_data_drift_detected_status.set(int(result["data_drift"]))

    return EvaluationResponse(
        metrics= ModelMetrics(**result["metrics"]),
        data_drift=result["data_drift"],
        model_type=result["model_type"],
    )