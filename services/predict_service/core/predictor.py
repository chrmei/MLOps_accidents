"""
Prediction helpers for the predict service.

Uses the model loaded once at container startup (best Production from MLflow).
No per-request model loading.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.metrics import Accuracy, DriftedColumnsCount, F1Score, Precision, Recall

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
    return MODEL_TYPE_DISPLAY.get(lower) or model_type.replace(
        "_", " "
    ).title().replace(" ", "")


def _preprocess(
    features: Union[Dict[str, Any], List[Dict[str, Any]]],
    model: Any,
    label_encoders: Any,
    metadata: Any,
    model_type: str,
) -> pd.DataFrame:
    """Preprocess features and return the feature dataframe"""
    if metadata and not isinstance(metadata, dict):
        metadata = None
    apply_cyclic_encoding = (
        metadata.get("apply_cyclic_encoding", True)
        if isinstance(metadata, dict)
        else True
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
        metadata=metadata,
    )
    expected_features = get_expected_features(model, metadata)
    if expected_features is None:
        logger.warning(
            f"Could not extract expected features from model/metadata. "
            f"Using all {len(df_features.columns)} preprocessed features. "
            f"This may cause feature mismatch errors."
        )
        expected_features = list(df_features.columns)
    else:
        logger.info(
            f"Extracted {len(expected_features)} expected features from model/metadata. "
            f"Preprocessed features: {len(df_features.columns)}"
        )
        df_features = align_features_with_model(df_features, expected_features)
        logger.info(
            f"After alignment: {len(df_features.columns)} features match model expectations"
        )
    return df_features


def _preprocess_and_predict_one(
    features: Dict[str, Any],
    model: Any,
    label_encoders: Any,
    metadata: Any,
    model_type: str,
) -> Tuple[Any, Optional[float]]:
    """Preprocess features, run model.predict and predict_proba when available.
    Returns (raw prediction, probability of positive class or None)."""
    df_features = _preprocess(features, model, label_encoders, metadata, model_type)
    prediction = model.predict(df_features)
    proba: Optional[float] = None
    if hasattr(model, "predict_proba"):
        try:
            proba_arr = model.predict_proba(df_features)
            # Binary classification: probability of class 1 (positive)
            if proba_arr.shape[1] >= 2:
                p = proba_arr[:, 1]
                proba = float(p.flat[0]) if hasattr(p, "flat") else float(p[0])
        except Exception as e:  # noqa: BLE001
            logger.debug("predict_proba failed: %s", e)
    return prediction, proba


def make_prediction(
    features: Dict[str, Any], model_cache: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Make a single prediction using the cached Production model.

    Args:
        features: Input features dict.
        model_cache: Must have keys: model, label_encoders, metadata, model_type.

    Returns:
        Dict with "prediction", "probability" (optional), and "model_type".
    """
    model = model_cache["model"]
    label_encoders = model_cache["label_encoders"]
    metadata = model_cache["metadata"]
    model_type = model_cache["model_type"]
    pred, proba = _preprocess_and_predict_one(
        features, model, label_encoders, metadata, model_type
    )
    out = pred.tolist() if hasattr(pred, "tolist") else pred
    # Single sample: take first element if array
    if hasattr(out, "__getitem__") and len(out) == 1:
        out = out[0]
    # Always include probability: from predict_proba when available, else 0.0/1.0 from class
    probability = proba if proba is not None else float(out)
    return {
        "prediction": out,
        "probability": probability,
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
        Dict with "predictions", "probabilities" (optional), "count", "model_type".
    """
    model = model_cache["model"]
    label_encoders = model_cache["label_encoders"]
    metadata = model_cache["metadata"]
    model_type = model_cache["model_type"]
    predictions: List[Any] = []
    probabilities: List[float] = []
    for features in features_list:
        pred, proba = _preprocess_and_predict_one(
            features, model, label_encoders, metadata, model_type
        )
        out = pred.tolist() if hasattr(pred, "tolist") else pred
        if hasattr(out, "__getitem__") and len(out) == 1:
            out = out[0]
        predictions.append(out)
        if proba is not None:
            probabilities.append(proba)
    # Always include probabilities: from predict_proba when available, else 0.0/1.0 from class
    if len(probabilities) != len(predictions):
        probabilities = [float(p) for p in predictions]
    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "count": len(predictions),
        "model_type": _model_type_display(model_type),
    }


def evaluate_test_set(
    eval_data: List[Dict[str, Any]],
    model_cache: Dict[str, Any],
    ref_data: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Evaluate model on a test set using Evidently.
    Args:
        eval_data: List of feature dicts including true labels.
        ref_data: Optional list of feature dicts to use as reference for data drift detection.
        model_cache: Must have keys: model, label_encoders, metadata, model_type.
    Returns:
        Dict with evaluation "metrics" (incl. Accuracy, Precision, Recall, F1 Score), "col_drift_share", and "model_type".
    """

    # Define feature types
    CAT_FEATS = [
        "place",
        "place_group",
        "catu",
        "sexe",
        "secu1",
        "secu_group",
        "locp",
        "actp",
        "etatp",
        "catv",
        "catv_group",
        "obs",
        "obs_group",
        "obsm",
        "obsm_group",
        "motor",
        "motor_group",
        "catr",
        "v1",
        "circ",
        "vosp",
        "prof",
        "plan",
        "surf",
        "infra",
        "situ",
        "lum",
        "agg_",
        "int",
        "atm",
        "col",
        "hour",
        "month",
        "day_of_week",
        "is_weekend",
        "season",
        "victim_age_binned",
        "hour_weekend",
        "atm_lum",
    ]
    NUM_FEATS = [
        "larrout",
        "vma",
        "jour",
        "mois",
        "an",
        "hrmn",
        "dep",
        "com",
        "lat",
        "long",
        "nb_victim",
        "nb_vehicles",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "victim_age",
        "victims_per_vehicle",
    ]
    TARGET = "grav"
    PREDICTION = "grav_pred"

    metrics = [
        Accuracy(),
        Precision(),
        Recall(),
        F1Score(),
    ]

    model = model_cache["model"]
    label_encoders = model_cache["label_encoders"]
    metadata = model_cache["metadata"]
    model_type = model_cache["model_type"]

    eval_targets = [row.pop("grav", None) for row in eval_data]
    eval_features_df = _preprocess(
        eval_data, model, label_encoders, metadata, model_type
    )
    eval_predictions = model.predict(eval_features_df)
    eval_features_df[TARGET] = eval_targets
    eval_features_df[PREDICTION] = eval_predictions

    # Define Evidently Dataset
    schema = DataDefinition(
        classification=[
            BinaryClassification(
                target=TARGET,
                prediction_labels=PREDICTION,
            )
        ],
        numerical_columns=[
            feat for feat in NUM_FEATS if feat in eval_features_df.columns
        ],
        categorical_columns=[
            feat for feat in CAT_FEATS if feat in eval_features_df.columns
        ],
    )
    eval_dataset = Dataset.from_pandas(data=eval_features_df, data_definition=schema)

    ref_dataset = None
    if ref_data is not None:
        ref_targets = [row.pop("grav", None) for row in ref_data]
        ref_features_df = _preprocess(
            ref_data, model, label_encoders, metadata, model_type
        )
        ref_predictions = model.predict(ref_features_df)
        ref_features_df[TARGET] = ref_targets
        ref_features_df[PREDICTION] = ref_predictions
        ref_dataset = Dataset.from_pandas(data=ref_features_df, data_definition=schema)
        metrics.append(DriftedColumnsCount())

    # Generate Evidently Report
    report = Report(metrics=metrics)
    eval_rep = report.run(
        current_data=eval_dataset,
        reference_data=ref_dataset,
    )

    # Extract metrics and data drift status from the report
    eval_dict = eval_rep.dict()
    accuracy = eval_dict["metrics"][0]["value"]
    precision = float(eval_dict["metrics"][1]["value"])
    recall = float(eval_dict["metrics"][2]["value"])
    f1_score = float(eval_dict["metrics"][3]["value"])

    col_drift_count = None
    col_drift_share = None
    if len(eval_dict["metrics"]) == 5:
        col_drift_count = eval_dict["metrics"][4]["value"]["count"]
        col_drift_share = eval_dict["metrics"][4]["value"]["share"]

    return {
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        },
        "col_drift_share": col_drift_share,
        "model_type": _model_type_display(model_type),
    }
