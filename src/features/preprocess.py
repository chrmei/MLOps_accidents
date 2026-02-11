# -*- coding: utf-8 -*-
"""
Preprocessing utilities for model inference.

This module provides reusable preprocessing functions for inference,
ensuring consistency with the training pipeline.
"""
import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .build_features import build_features, prepare_input_for_feature_engineering

logger = logging.getLogger(__name__)


def preprocess_for_inference(
    features: Union[Dict[str, Any], List[Dict[str, Any]]],
    label_encoders: Optional[Dict] = None,
    apply_cyclic_encoding: Optional[bool] = None,
    apply_interactions: Optional[bool] = None,
    model_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Preprocess input features for model inference.

    This is the main preprocessing function that should be used for inference.
    It applies the same feature engineering pipeline used during training.

    Parameters
    ----------
    features : dict
        Input features dictionary (canonical input features)
    label_encoders : dict, optional
        Pre-fitted label encoders from training. Required for inference.
    apply_cyclic_encoding : bool, optional
        Whether to apply cyclic encoding (must match training).
        If None, inferred from metadata.
    apply_interactions : bool, optional
        Whether to create interaction features (must match training).
        If None, inferred from metadata.
    model_type : str, optional
        Model type (e.g., "XGBoost", "LightGBM", "RandomForest") for logging purposes.
    metadata : dict, optional
        Model metadata containing feature engineering configuration.
        If provided, will be used to determine preprocessing settings.

    Returns
    -------
    pd.DataFrame
        Preprocessed features ready for model prediction
    """
    # Extract feature engineering config from metadata if available
    if metadata and isinstance(metadata, dict):
        # Use metadata values if explicit parameters not provided
        if apply_cyclic_encoding is None:
            apply_cyclic_encoding = metadata.get("apply_cyclic_encoding", True)
        if apply_interactions is None:
            apply_interactions = metadata.get("apply_interactions", True)

        # Log feature engineering version if available
        fe_version = metadata.get("feature_engineering_version")
        if fe_version:
            logger.info(f"Using feature engineering version: {fe_version}")

        uses_grouped_features = metadata.get("uses_grouped_features", False)
        if uses_grouped_features:
            logger.info("Model uses grouped features - grouping will be applied")
    else:
        # Default values if no metadata
        if apply_cyclic_encoding is None:
            apply_cyclic_encoding = True
        if apply_interactions is None:
            apply_interactions = True

    # Prepare input for feature engineering
    df_interim = prepare_input_for_feature_engineering(features)

    # Apply feature engineering pipeline
    df_features, _ = build_features(
        df_interim,
        apply_cyclic_encoding=apply_cyclic_encoding,
        apply_interactions=apply_interactions,
        label_encoders=label_encoders,
        model_type=model_type,
    )

    return df_features


def align_features_with_model(
    df_features: pd.DataFrame, expected_features: list, fill_value: float = 0.0
) -> pd.DataFrame:
    """
    Align feature DataFrame with model's expected feature order and names.

    Parameters
    ----------
    df_features : pd.DataFrame
        Features DataFrame from preprocessing
    expected_features : list
        List of feature names expected by the model
    fill_value : float
        Value to use for missing features

    Returns
    -------
    pd.DataFrame
        Aligned features DataFrame
    """
    missing_features = set(expected_features) - set(df_features.columns)
    if missing_features:
        logger.warning(
            f"Missing features ({len(missing_features)}): {list(missing_features)[:10]}..."
        )
        for feat in missing_features:
            df_features[feat] = fill_value

    extra_features = set(df_features.columns) - set(expected_features)
    if extra_features:
        logger.warning(
            f"Extra features ({len(extra_features)}) will be dropped: {list(extra_features)[:10]}..."
        )

    # Reorder and select only expected features
    df_features = df_features.reindex(columns=expected_features, fill_value=fill_value)

    return df_features
