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
    ref_data: Optional[List[Dict[str, Any]]] = None


class EvaluationResponse(BaseModel):
    """Response payload for model evaluation."""

    metrics: ModelMetrics
    data_drift: Optional[bool]
    model_type: str


class FeatureEngineeringConfig(BaseModel):
    """Feature engineering configuration."""

    feature_engineering_version: Optional[str] = None
    uses_grouped_features: bool = False
    grouped_feature_mappings: Optional[Dict[str, str]] = None
    removed_features: Optional[List[str]] = None
    apply_cyclic_encoding: bool = True
    apply_interactions: bool = True


class ModelInfoResponse(BaseModel):
    """Response payload for model information."""

    model_type: str
    model_version: Optional[str] = None
    input_features: List[str]
    feature_engineering_config: FeatureEngineeringConfig
    mlflow_signature_available: bool = False


class InputFeaturesResponse(BaseModel):
    """Response payload for input features."""

    input_features: List[str]
    uses_grouped_features: bool
    feature_engineering_version: Optional[str] = None
    source: str  # "mlflow_signature", "metadata", or "inferred"
