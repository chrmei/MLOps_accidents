"""
Prediction helpers for the predict service.

Uses the model loaded once at container startup (best Production from MLflow).
No per-request model loading.
"""

import logging
from typing import Any, Dict, List

from src.features.preprocess import align_features_with_model, preprocess_for_inference
from src.models.predict_model import get_expected_features

logger = logging.getLogger(__name__)

# Display names for model_type in responses
MODEL_TYPE_DISPLAY = {
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "random_forest": "RandomForest",
    "randomforest": "RandomForest",
    "logistic_regression": "LogisticRegression",
    "logisticregression": "LogisticRegression",
}


def _model_type_display(model_type: str) -> str:
    if not model_type:
        return "unknown"
    lower = model_type.lower()
    return MODEL_TYPE_DISPLAY.get(lower) or model_type.replace("_", " ").title().replace(" ", "")


def _preprocess_and_predict_one(
    features: Dict[str, Any],
    model: Any,
    label_encoders: Any,
    metadata: Any,
    model_type: str,
) -> Any:
    """Preprocess features and run model.predict. Returns raw prediction (e.g. array)."""
    if metadata and not isinstance(metadata, dict):
        metadata = None
    apply_cyclic_encoding = (
        metadata.get("apply_cyclic_encoding", True) if isinstance(metadata, dict) else True
    )
    apply_interactions = (
        metadata.get("apply_interactions", True) if isinstance(metadata, dict) else True
    )
    model_type_display = _model_type_display(model_type)
    df_features = preprocess_for_inference(
        features,
        label_encoders=label_encoders,
        apply_cyclic_encoding=apply_cyclic_encoding,
        apply_interactions=apply_interactions,
        model_type=model_type_display,
    )
    expected_features = get_expected_features(model, metadata)
    if expected_features is None:
        expected_features = list(df_features.columns)
    else:
        df_features = align_features_with_model(df_features, expected_features)
    prediction = model.predict(df_features)
    return prediction


def make_prediction(features: Dict[str, Any], model_cache: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a single prediction using the cached Production model.

    Args:
        features: Input features dict.
        model_cache: Must have keys: model, label_encoders, metadata, model_type.

    Returns:
        Dict with "prediction" and "model_type".
    """
    model = model_cache["model"]
    label_encoders = model_cache["label_encoders"]
    metadata = model_cache["metadata"]
    model_type = model_cache["model_type"]
    pred = _preprocess_and_predict_one(
        features, model, label_encoders, metadata, model_type
    )
    out = pred.tolist() if hasattr(pred, "tolist") else pred
    # Single sample: take first element if array
    if hasattr(out, "__getitem__") and len(out) == 1:
        out = out[0]
    return {
        "prediction": out,
        "model_type": _model_type_display(model_type),
    }


def make_batch_prediction(
    features_list: List[Dict[str, Any]],
    model_cache: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Make batch predictions using the cached Production model.

    Args:
        features_list: List of feature dicts.
        model_cache: Must have keys: model, label_encoders, metadata, model_type.

    Returns:
        Dict with "predictions", "count", "model_type".
    """
    model = model_cache["model"]
    label_encoders = model_cache["label_encoders"]
    metadata = model_cache["metadata"]
    model_type = model_cache["model_type"]
    predictions = []
    for features in features_list:
        pred = _preprocess_and_predict_one(
            features, model, label_encoders, metadata, model_type
        )
        out = pred.tolist() if hasattr(pred, "tolist") else pred
        if hasattr(out, "__getitem__") and len(out) == 1:
            out = out[0]
        predictions.append(out)
    return {
        "predictions": predictions,
        "count": len(predictions),
        "model_type": _model_type_display(model_type),
    }
