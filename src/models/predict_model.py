# -*- coding: utf-8 -*-
"""
Model prediction script for inference.

This script loads a trained model and makes predictions on new data.
It uses the same preprocessing pipeline used during training to ensure consistency.
"""
import json
import logging
import os
import sys

import joblib
import pandas as pd

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.features.preprocess import align_features_with_model, preprocess_for_inference

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model_artifacts(model_path: str):
    """
    Load model and associated artifacts (encoders, metadata).
    
    Parameters
    ----------
    model_path : str
        Path to the trained model file
        
    Returns
    -------
    tuple
        (model, label_encoders, metadata) tuple
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Please run 'make run-train' first to train the model."
        )

    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")

    # Load label encoders
    encoders_path = model_path.replace("trained_model.joblib", "label_encoders.joblib")
    label_encoders = None
    if os.path.exists(encoders_path):
        label_encoders = joblib.load(encoders_path)
        logger.info(f"Loaded label encoders from {encoders_path}")
    else:
        logger.warning(f"Label encoders not found at {encoders_path}")

    # Load feature metadata
    metadata_path = model_path.replace("trained_model.joblib", "trained_model_metadata.joblib")
    metadata = None
    if os.path.exists(metadata_path):
        metadata = joblib.load(metadata_path)
        logger.info(f"Loaded feature metadata from {metadata_path}")
    else:
        logger.warning(f"Feature metadata not found at {metadata_path}")

    return model, label_encoders, metadata


def get_expected_features(model, metadata=None):
    """
    Extract expected feature names from model or metadata.
    
    Parameters
    ----------
    model : object
        Trained model (pipeline)
    metadata : dict, optional
        Feature metadata dictionary
        
    Returns
    -------
    list
        List of expected feature names
    """
    # Try metadata first
    if metadata and "feature_names" in metadata:
        return metadata["feature_names"]

    # Try to extract from model
    if hasattr(model, "steps") and len(model.steps) > 0:
        xgb_model = model.steps[-1][1]
        if hasattr(xgb_model, "feature_names_in_") and xgb_model.feature_names_in_ is not None:
            return list(xgb_model.feature_names_in_)
        elif hasattr(xgb_model, "get_booster"):
            booster = xgb_model.get_booster()
            if hasattr(booster, "feature_names"):
                return booster.feature_names

    return None


def predict(features: dict, model_path: str = "models/trained_model.joblib"):
    """
    Make prediction from input features.
    
    This function applies the same preprocessing pipeline used during training.
    
    Parameters
    ----------
    features : dict
        Input features dictionary
    model_path : str
        Path to the trained model file
        
    Returns
    -------
    prediction : array
        Model prediction
    """
    # Load model artifacts
    model, label_encoders, metadata = load_model_artifacts(model_path)

    # Get preprocessing parameters from metadata or use defaults
    apply_cyclic_encoding = (
        metadata.get("apply_cyclic_encoding", True) if metadata else True
    )
    apply_interactions = metadata.get("apply_interactions", True) if metadata else True

    # Preprocess features
    logger.info("Preprocessing input features...")
    df_features = preprocess_for_inference(
        features,
        label_encoders=label_encoders,
        apply_cyclic_encoding=apply_cyclic_encoding,
        apply_interactions=apply_interactions,
    )

    # Get expected features from model
    expected_features = get_expected_features(model, metadata)
    if expected_features is None:
        logger.warning("Could not determine expected features, using all preprocessed features")
        expected_features = list(df_features.columns)
    else:
        # Align features with model expectations
        df_features = align_features_with_model(df_features, expected_features)

    logger.info(f"Making prediction with {len(expected_features)} features...")

    # Make prediction
    prediction = model.predict(df_features)

    return prediction


if __name__ == "__main__":
    # Default JSON file path
    default_json_file = "src/models/test_features.json"

    if len(sys.argv) == 2:
        json_file = sys.argv[1]
    else:
        json_file = default_json_file
        print(f"Using default JSON file: {json_file}")
        print(
            "To use a different file, run: python src/models/predict_model.py <path/to/file.json>"
        )

    # Load features from JSON file
    try:
        with open(json_file, "r") as file:
            features = json.load(file)
        print(f"Loaded features from {json_file}")
    except FileNotFoundError:
        print(f"Error: JSON file not found: {json_file}")
        print(
            f"Please create {json_file} with the required features or specify a different file."
        )
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {json_file}: {e}")
        sys.exit(1)

    # Make prediction
    try:
        result = predict(features)
        print(f"\nPrediction: {result[0]}")
        print(f"Interpretation: {'Priority' if result[0] == 1 else 'Non-Priority'}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        sys.exit(1)
