"""
Pydantic schemas for the predict service API.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """Request payload for single prediction."""

    features: Dict[str, Any]
    model_type: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response payload for single prediction."""

    prediction: Any
    model_type: str


class BatchPredictionRequest(BaseModel):
    """Request payload for batch predictions."""

    features_list: List[Dict[str, Any]]
    model_type: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """Response payload for batch predictions."""

    predictions: List[Any]
    count: int
    model_type: str


class LoginRequest(BaseModel):
    """Request payload for login."""

    username: str
    password: str


class LoginResponse(BaseModel):
    """Response payload for login."""

    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int
