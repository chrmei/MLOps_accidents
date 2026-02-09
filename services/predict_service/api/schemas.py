"""
Pydantic schemas for the predict service API.

Predict service always uses the best Production model from MLflow,
loaded once at container startup. No per-request model selection.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from services.common.models import ModelMetrics


class PredictionRequest(BaseModel):
    """Request payload for single prediction."""

    features: Dict[str, Any]


class PredictionResponse(BaseModel):
    """Response payload for single prediction."""

    prediction: Any
    probability: float
    model_type: str


class BatchPredictionRequest(BaseModel):
    """Request payload for batch predictions."""

    features_list: List[Dict[str, Any]]


class BatchPredictionResponse(BaseModel):
    """Response payload for batch predictions."""

    predictions: List[Any]
    probabilities: List[float]
    count: int
    model_type: str

class EvaluationRequest(BaseModel):
    """Request payload for model evaluation."""

    eval_data: List[Dict[str, Any]]
    ref_data: Optional[List[Dict[str, Any]]]=None

class EvaluationResponse(BaseModel):
    """Response payload for model evaluation."""
    
    metrics: ModelMetrics
    data_drift: bool
    model_type: str