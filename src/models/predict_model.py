# -*- coding: utf-8 -*-
"""
Model prediction script for inference.

This script loads a trained model and makes predictions on new data.
It uses the same preprocessing pipeline used during training to ensure consistency.

Supports loading models from:
- Local filesystem (default)
- MLflow Model Registry (by stage or version)
"""
import json
import logging
import os
import sys

import joblib
import pandas as pd
import yaml

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.features.preprocess import align_features_with_model, preprocess_for_inference

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model_from_registry(
    model_name: str, stage: str = None, version: int = None, config_path: str = None
):
    """
    Load model from MLflow Model Registry.
    
    Parameters
    ----------
    model_name : str
        Name of the registered model
    stage : str, optional
        Stage name (Staging, Production). Required if version is not provided.
    version : int, optional
        Model version number. Required if stage is not provided.
    config_path : str, optional
        Path to configuration file for MLflow setup
        
    Returns
    -------
    tuple
        (model, label_encoders, metadata) tuple
    """
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        raise ImportError(
            "MLflow is required to load models from registry. "
            "Install with: pip install mlflow"
        )

    # Setup MLflow tracking URI
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        mlflow_config = config.get("mlflow", {})
        tracking_uri = mlflow_config.get("tracking_uri") or os.getenv("MLFLOW_TRACKING_URI")
    else:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        dagshub_repo = os.getenv("DAGSHUB_REPO")
        if dagshub_repo:
            tracking_uri = f"https://dagshub.com/{dagshub_repo}.mlflow"
        else:
            raise ValueError(
                "MLflow tracking URI not configured. "
                "Set MLFLOW_TRACKING_URI or DAGSHUB_REPO environment variable."
            )

    mlflow.set_tracking_uri(tracking_uri)

    # Construct model URI
    if stage:
        model_uri = f"models:/{model_name}/{stage}"
        logger.info(f"Loading model from MLflow registry: {model_uri}")
    elif version:
        model_uri = f"models:/{model_name}/{version}"
        logger.info(f"Loading model version {version} from MLflow registry: {model_uri}")
    else:
        raise ValueError("Either 'stage' or 'version' must be provided")

    # Load model from registry
    model = mlflow.sklearn.load_model(model_uri)
    logger.info(f"Successfully loaded model from MLflow registry")

    # Note: When loading from MLflow registry, encoders and metadata should be
    # stored as artifacts in the MLflow run. For now, we'll try to load from
    # local filesystem as fallback, but ideally these should be stored in MLflow.
    # This is a limitation - ideally encoders and metadata should be logged as
    # MLflow artifacts during training.
    label_encoders = None
    metadata = None

    # Try to load from local filesystem (fallback)
    local_encoders_path = "models/label_encoders.joblib"
    local_metadata_path = "models/trained_model_metadata.joblib"

    if os.path.exists(local_encoders_path):
        label_encoders = joblib.load(local_encoders_path)
        logger.info(f"Loaded label encoders from local filesystem: {local_encoders_path}")
    else:
        logger.warning(
            "Label encoders not found. Consider storing encoders as MLflow artifacts."
        )

    if os.path.exists(local_metadata_path):
        metadata = joblib.load(local_metadata_path)
        logger.info(f"Loaded metadata from local filesystem: {local_metadata_path}")
    else:
        logger.warning(
            "Model metadata not found. Consider storing metadata as MLflow artifacts."
        )

    return model, label_encoders, metadata


def load_model_artifacts(model_path: str):
    """
    Load model and associated artifacts (encoders, metadata) from local filesystem.
    
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


def predict(
    features: dict,
    model_path: str = None,
    model_name: str = None,
    stage: str = None,
    version: int = None,
    config_path: str = "src/config/model_config.yaml",
):
    """
    Make prediction from input features.
    
    This function applies the same preprocessing pipeline used during training.
    
    Parameters
    ----------
    features : dict
        Input features dictionary
    model_path : str, optional
        Path to the trained model file (for local filesystem loading)
    model_name : str, optional
        Name of the registered model (for MLflow registry loading)
    stage : str, optional
        Stage name (Staging, Production) - required if loading from registry
    version : int, optional
        Model version number - alternative to stage for registry loading
    config_path : str
        Path to configuration file (for MLflow setup)
        
    Returns
    -------
    prediction : array
        Model prediction
    """
    # Determine loading method
    if model_name:
        # Load from MLflow registry
        model, label_encoders, metadata = load_model_from_registry(
            model_name, stage=stage, version=version, config_path=config_path
        )
    else:
        # Load from local filesystem (default)
        if model_path is None:
            model_path = "models/trained_model.joblib"
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Make predictions using trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "json_file",
        nargs="?",
        default="src/models/test_features.json",
        help="Path to JSON file with input features",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to local model file (default: models/trained_model.joblib)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Name of registered model in MLflow (for registry loading)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["Staging", "Production"],
        help="Stage name for registry loading (required if --model-name is used)",
    )
    parser.add_argument(
        "--version",
        type=int,
        help="Model version number for registry loading (alternative to --stage)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/model_config.yaml",
        help="Path to model configuration file",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.model_name and not args.stage and not args.version:
        parser.error("--stage or --version is required when using --model-name")

    # Load features from JSON file
    try:
        with open(args.json_file, "r") as file:
            features = json.load(file)
        print(f"Loaded features from {args.json_file}")
    except FileNotFoundError:
        print(f"Error: JSON file not found: {args.json_file}")
        print(
            f"Please create {args.json_file} with the required features or specify a different file."
        )
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {args.json_file}: {e}")
        sys.exit(1)

    # Make prediction
    try:
        result = predict(
            features,
            model_path=args.model_path,
            model_name=args.model_name,
            stage=args.stage,
            version=args.version,
            config_path=args.config,
        )
        print(f"\nPrediction: {result[0]}")
        print(f"Interpretation: {'Priority' if result[0] == 1 else 'Non-Priority'}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        sys.exit(1)
