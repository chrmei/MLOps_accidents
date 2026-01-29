"""
Prediction helpers for the predict service.
"""

import logging
from typing import Dict, List, Optional

from src.models.predict_model import predict as predict_fn

logger = logging.getLogger(__name__)


def make_prediction(
    features: Dict,
    model_type: Optional[str] = None,
    use_mlflow_production: bool = True,
) -> Dict:
    """
    Make a prediction using the trained model.

    Args:
        features: Dictionary of input features.
        model_type: Model type to use (e.g., "XGBoost"). Defaults to best model.
        use_mlflow_production: Load from MLflow Production stage.

    Returns:
        Dictionary with prediction result and metadata.
    """
    prediction = predict_fn(
        features=features,
        model_type=model_type or "XGBoost",
        use_mlflow_production=use_mlflow_production,
        use_best_model=model_type is None,  # Auto-select best if no specific type
    )

    return {
        "prediction": prediction.tolist() if hasattr(prediction, "tolist") else prediction,
        "model_type": model_type or "best_model",
    }


def make_batch_prediction(
    features_list: List[Dict],
    model_type: Optional[str] = None,
    use_mlflow_production: bool = True,
) -> Dict:
    """
    Make batch predictions.

    Args:
        features_list: List of feature dictionaries.
        model_type: Model type to use.
        use_mlflow_production: Load from MLflow Production stage.

    Returns:
        Dictionary with predictions and metadata.
    """
    predictions = []
    for features in features_list:
        pred = predict_fn(
            features=features,
            model_type=model_type or "XGBoost",
            use_mlflow_production=use_mlflow_production,
            use_best_model=model_type is None,
        )
        predictions.append(pred.tolist() if hasattr(pred, "tolist") else pred)

    return {
        "predictions": predictions,
        "count": len(predictions),
        "model_type": model_type or "best_model",
    }
