# -*- coding: utf-8 -*-
"""
Model training script with XGBoost and SMOTE.

This script:
1. Loads configuration from model_config.yaml
2. Loads feature-engineered data
3. Performs train/test split
4. Trains XGBoost model with SMOTE pipeline:
   - By default: Uses parameters from config file (fast training)
   - Optional: Runs GridSearchCV for hyperparameter tuning (slow)
5. Evaluates and saves metrics
6. Saves the trained model

Usage:
    # Fast training with default parameters from config
    python src/models/train_model.py

    # Grid search for hyperparameter tuning
    python src/models/train_model.py --grid-search

    # Override config with CLI arguments
    python src/models/train_model.py --test-size 0.2 --random-state 123
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import click
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

logger = logging.getLogger(__name__)


def load_config(config_path="src/config/model_config.yaml"):
    """
    Load model configuration from YAML file.

    Parameters
    ----------
    config_path : str
        Path to configuration YAML file

    Returns
    -------
    dict
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Please ensure model_config.yaml exists in src/config/"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def load_features(features_path="data/preprocessed/features.csv"):
    """
    Load feature-engineered dataset and separate features from target.

    Parameters
    ----------
    features_path : str
        Path to features CSV file

    Returns
    -------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    """
    logger.info(f"Loading features from {features_path}")
    df_features = pd.read_csv(features_path)

    # Separate target and features
    if "grav" not in df_features.columns:
        raise ValueError("Target column 'grav' not found in features file")

    X = df_features.drop(columns=["grav"])
    y = df_features["grav"]

    logger.info(f"Loaded {len(X)} samples with {len(X.columns)} features")
    return X, y


def get_default_xgb_params(config):
    """
    Get default XGBoost parameters from config file.

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    dict
        Dictionary of default XGBoost parameters
    """
    xgb_config = config.get("xgboost", {}).get("default_params", {})
    # Remove random_state and eval_metric as they're handled separately
    params = {k: v for k, v in xgb_config.items() if k not in ["random_state", "eval_metric"]}
    return params


def train_with_default_params(X_train, y_train, config):
    """
    Train XGBoost model with default parameters from config and SMOTE pipeline.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    config : dict
        Configuration dictionary

    Returns
    -------
    model : Pipeline
        Trained model pipeline
    """
    logger.info("Training XGBoost model with default parameters from config")

    # Get default parameters from config
    default_params = get_default_xgb_params(config)
    xgb_config = config.get("xgboost", {}).get("default_params", {})
    random_state = xgb_config.get("random_state", 42)
    eval_metric = xgb_config.get("eval_metric", "logloss")

    # Get SMOTE config
    smote_config = config.get("smote", {})
    smote_random_state = smote_config.get("random_state", 42)
    smote_enabled = smote_config.get("enabled", True)

    # Create pipeline with SMOTE + XGBoost
    pipeline_steps = []
    if smote_enabled:
        pipeline_steps.append(("smote", SMOTE(random_state=smote_random_state)))
        logger.info("SMOTE enabled for class imbalance handling")

    pipeline_steps.append(
        (
            "xgb",
            xgb.XGBClassifier(
                random_state=random_state,
                eval_metric=eval_metric,
                **default_params,
            ),
        )
    )

    pipeline = ImbPipeline(pipeline_steps)

    logger.info(f"Default parameters: {default_params}")
    logger.info("Fitting model...")

    pipeline.fit(X_train, y_train)

    logger.info("Model training completed!")

    return pipeline


def train_with_grid_search(X_train, y_train, config, cv=None, verbose=None):
    """
    Train XGBoost model with GridSearchCV and SMOTE pipeline.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    config : dict
        Configuration dictionary
    cv : int, optional
        Number of cross-validation folds (overrides config if provided)
    verbose : int, optional
        Verbosity level for GridSearchCV (overrides config if provided)

    Returns
    -------
    grid_search : GridSearchCV
        Fitted GridSearchCV object
    """
    logger.info("Setting up XGBoost + SMOTE pipeline for grid search")

    # Get grid search config
    grid_config = config.get("grid_search", {})
    cv = cv if cv is not None else grid_config.get("cv", 5)
    verbose = verbose if verbose is not None else grid_config.get("verbose", 1)
    scoring = grid_config.get("scoring", "f1")
    n_jobs = grid_config.get("n_jobs", -1)
    param_grid = grid_config.get("param_grid", {})

    # Get XGBoost and SMOTE config
    xgb_config = config.get("xgboost", {}).get("default_params", {})
    random_state = xgb_config.get("random_state", 42)
    eval_metric = xgb_config.get("eval_metric", "logloss")

    smote_config = config.get("smote", {})
    smote_random_state = smote_config.get("random_state", 42)
    smote_enabled = smote_config.get("enabled", True)

    # Create pipeline with SMOTE + XGBoost
    # SMOTE will be applied only to training portion of each CV fold
    pipeline_steps = []
    if smote_enabled:
        pipeline_steps.append(("smote", SMOTE(random_state=smote_random_state)))

    pipeline_steps.append(
        ("xgb", xgb.XGBClassifier(random_state=random_state, eval_metric=eval_metric))
    )

    pipeline = ImbPipeline(pipeline_steps)

    # Create GridSearchCV object with pipeline
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    # Fit grid search on training data
    # SMOTE will be applied automatically within each CV fold
    logger.info(f"Starting grid search with {cv}-fold CV...")
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Parameter grid: {list(param_grid.keys())}")
    if smote_enabled:
        logger.info("Note: SMOTE will be applied to training portion of each CV fold")

    grid_search.fit(X_train, y_train)

    logger.info("Grid search completed!")
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV {scoring} score: {grid_search.best_score_:.4f}")

    return grid_search


def evaluate_model(model, X_test, y_test, config=None):
    """
    Evaluate model performance on test set.

    Parameters
    ----------
    model : sklearn estimator
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    config : dict, optional
        Configuration dictionary (for future confusion matrix saving)

    Returns
    -------
    dict
        Dictionary containing all metrics
    """
    logger.info("Evaluating model on test set...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Classification report
    report = classification_report(
        y_test, y_pred, target_names=["Non-Priority", "Priority"], output_dict=True
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "classification_report": report,
    }

    # Add confusion matrix if requested in config
    if config and config.get("evaluation", {}).get("save_confusion_matrix", False):
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

    logger.info(f"Test Set Metrics:")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")

    return metrics


def setup_mlflow(config):
    """
    Setup MLflow tracking with Dagshub.

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    bool
        True if MLflow is enabled and configured, False otherwise
    """
    mlflow_config = config.get("mlflow", {})

    if not mlflow_config.get("enabled", False):
        logger.info("MLflow tracking is disabled in config")
        return False

    # Get tracking URI from config or environment
    tracking_uri = mlflow_config.get("tracking_uri") or os.getenv("MLFLOW_TRACKING_URI")

    # If not set, construct from DAGSHUB_REPO
    if not tracking_uri:
        dagshub_repo = os.getenv("DAGSHUB_REPO")
        if dagshub_repo:
            tracking_uri = f"https://dagshub.com/{dagshub_repo}.mlflow"
        else:
            logger.warning(
                "MLflow tracking URI not configured. "
                "Set MLFLOW_TRACKING_URI or DAGSHUB_REPO environment variable."
            )
            return False

    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri)

    # Set experiment name
    experiment_name = mlflow_config.get("experiment_name", "accident_prediction")
    mlflow.set_experiment(experiment_name)

    logger.info(f"MLflow tracking enabled: {tracking_uri}")
    logger.info(f"MLflow experiment: {experiment_name}")

    return True


def log_to_mlflow(
    config,
    model,
    metrics,
    params,
    best_cv_score,
    X_train,
    y_train,
    X_test,
    y_test,
    use_grid_search=False,
):
    """
    Log experiment to MLflow.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    model : sklearn estimator
        Trained model
    metrics : dict
        Evaluation metrics
    params : dict
        Model parameters
    best_cv_score : float, optional
        Best cross-validation score (if grid search was used)
    X_train, y_train : training data
    X_test, y_test : test data
    use_grid_search : bool
        Whether grid search was used
    """
    mlflow_config = config.get("mlflow", {})

    if not mlflow_config.get("enabled", False):
        return

    try:
        with mlflow.start_run():
            logger.info("Logging experiment to MLflow...")

            # Log parameters
            logger.info("Logging parameters to MLflow...")
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

            # Log data split parameters
            data_split = config.get("data_split", {})
            mlflow.log_param("test_size", data_split.get("test_size", 0.3))
            mlflow.log_param("random_state", data_split.get("random_state", 42))

            # Log model hyperparameters from config
            xgb_params = config.get("xgboost", {}).get("default_params", {})
            for param_name, param_value in xgb_params.items():
                if param_name not in ["random_state", "eval_metric"]:
                    mlflow.log_param(f"xgb_{param_name}", param_value)

            # Log feature engineering flags
            feature_config = config.get("feature_engineering", {})
            mlflow.log_param(
                "apply_cyclic_encoding", feature_config.get("apply_cyclic_encoding", True)
            )
            mlflow.log_param("apply_interactions", feature_config.get("apply_interactions", True))

            # Log SMOTE configuration
            smote_config = config.get("smote", {})
            mlflow.log_param("smote_enabled", smote_config.get("enabled", True))

            # Log grid search info
            mlflow.log_param("used_grid_search", use_grid_search)
            if use_grid_search and best_cv_score is not None:
                mlflow.log_metric("best_cv_f1_score", best_cv_score)

            # Log metrics
            logger.info("Logging metrics to MLflow...")
            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("precision", metrics["precision"])
            mlflow.log_metric("recall", metrics["recall"])
            mlflow.log_metric("f1_score", metrics["f1_score"])

            # Log dataset info
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_classes", len(y_train.unique()))

            # Log model
            if mlflow_config.get("log_model", True):
                logger.info("Logging model to MLflow...")
                mlflow.sklearn.log_model(
                    model, "model", registered_model_name="XGBoost_Accident_Prediction"
                )

            # Log artifacts (metrics files, confusion matrix, etc.)
            if mlflow_config.get("log_artifacts", True):
                logger.info("Logging artifacts to MLflow...")
                metrics_dir = config.get("paths", {}).get("metrics_dir", "data/metrics")
                if os.path.exists(metrics_dir):
                    mlflow.log_artifacts(metrics_dir, "metrics")

            logger.info("MLflow logging completed successfully!")

    except Exception as e:
        logger.warning(f"Failed to log to MLflow: {e}")
        logger.warning("Continuing without MLflow logging...")


def save_metrics(
    metrics,
    best_params,
    best_cv_score=None,
    y_test=None,
    y_pred=None,
    output_dir="data/metrics",
    use_grid_search=True,
):
    """
    Save training metrics to JSON file.

    Parameters
    ----------
    metrics : dict
        Evaluation metrics
    best_params : dict
        Best parameters from grid search
    best_cv_score : float
        Best cross-validation score
    y_test : array-like, optional
        True labels for classification report
    y_pred : array-like, optional
        Predicted labels for classification report
    output_dir : str
        Directory to save metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create metrics dictionary with all information
    metrics_dict = {
        "timestamp": datetime.now().isoformat(),
        "parameters": best_params,
        "test_metrics": metrics,
    }

    if use_grid_search and best_cv_score is not None:
        metrics_dict["best_cv_f1_score"] = float(best_cv_score)
        metrics_dict["used_grid_search"] = True
    else:
        metrics_dict["used_grid_search"] = False
        metrics_dict["note"] = "Trained with default parameters (no grid search)"

    # Save to JSON file
    metrics_file = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    logger.info(f"Metrics saved to {metrics_file}")

    # Also save a human-readable text report
    report_file = os.path.join(output_dir, "training_report.txt")
    with open(report_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Model Training Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {metrics_dict['timestamp']}\n\n")
        f.write("Model Parameters:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        if use_grid_search and best_cv_score is not None:
            f.write(f"\nBest CV F1 Score: {best_cv_score:.4f}\n")
        else:
            f.write(f"\nNote: Used default parameters (no grid search performed)\n")
        f.write("\n")
        f.write("Test Set Metrics:\n")
        f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {metrics['recall']:.4f}\n")
        f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n\n")

        if y_test is not None and y_pred is not None:
            f.write("Classification Report:\n")
            f.write(
                classification_report(
                    y_test,
                    y_pred,
                    target_names=["Non-Priority", "Priority"],
                )
            )
            f.write("\n")

        f.write("Detailed Classification Report (JSON):\n")
        f.write(json.dumps(metrics["classification_report"], indent=2))

    logger.info(f"Training report saved to {report_file}")


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    default="src/config/model_config.yaml",
    help="Path to model configuration YAML file",
)
@click.option(
    "--features-path",
    type=click.Path(exists=True),
    default=None,
    help="Path to features CSV file (overrides config)",
)
@click.option(
    "--model-output",
    type=click.Path(),
    default=None,
    help="Path to save trained model (overrides config)",
)
@click.option(
    "--metrics-dir",
    type=click.Path(),
    default=None,
    help="Directory to save metrics (overrides config)",
)
@click.option(
    "--grid-search/--no-grid-search",
    default=None,
    help="Enable grid search for hyperparameter tuning (overrides config)",
)
@click.option(
    "--cv",
    type=int,
    default=None,
    help="Number of CV folds for grid search (overrides config)",
)
@click.option(
    "--test-size",
    type=float,
    default=None,
    help="Proportion of data for test set (overrides config)",
)
@click.option(
    "--random-state",
    type=int,
    default=None,
    help="Random state for reproducibility (overrides config)",
)
@click.option(
    "--verbose",
    type=int,
    default=None,
    help="Verbosity level for GridSearchCV (0-3, overrides config)",
)
def main(
    config,
    features_path,
    model_output,
    metrics_dir,
    grid_search,
    cv,
    test_size,
    random_state,
    verbose,
):
    """
    Train XGBoost model with optional grid search.

    Configuration is loaded from model_config.yaml. CLI arguments override config values.
    By default, uses parameters from config file for fast training.
    Use --grid-search flag to enable hyperparameter tuning.
    """
    logger.info("=" * 60)
    logger.info("Starting Model Training")
    logger.info("=" * 60)

    # Load configuration
    config_dict = load_config(config)

    # Get paths from config (CLI args override config)
    paths_config = config_dict.get("paths", {})
    features_path = features_path or paths_config.get("features", "data/preprocessed/features.csv")
    model_output = model_output or paths_config.get("model_output", "models/trained_model.joblib")
    metrics_dir = metrics_dir or paths_config.get("metrics_dir", "data/metrics")

    # Get data split parameters (CLI args override config)
    data_split_config = config_dict.get("data_split", {})
    test_size = test_size if test_size is not None else data_split_config.get("test_size", 0.3)
    random_state = (
        random_state if random_state is not None else data_split_config.get("random_state", 42)
    )

    # Get grid search setting (CLI arg overrides config)
    grid_search_config = config_dict.get("grid_search", {})
    grid_search = (
        grid_search if grid_search is not None else grid_search_config.get("enabled", False)
    )

    # Get feature engineering config
    feature_config = config_dict.get("feature_engineering", {})
    apply_cyclic_encoding = feature_config.get("apply_cyclic_encoding", True)
    apply_interactions = feature_config.get("apply_interactions", True)

    logger.info(f"Configuration loaded from: {config}")
    logger.info(f"Grid search: {'Enabled' if grid_search else 'Disabled'}")

    # Setup MLflow
    mlflow_enabled = setup_mlflow(config_dict)

    # Load features
    X, y = load_features(features_path)

    # Train/test split
    logger.info(f"Splitting data (test_size={test_size}, random_state={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=data_split_config.get("shuffle", True)
    )
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    # Train model (with or without grid search)
    if grid_search:
        logger.info("Grid search enabled - this may take a while...")
        grid_search_result = train_with_grid_search(
            X_train, y_train, config_dict, cv=cv, verbose=verbose
        )
        model = grid_search_result.best_estimator_
        best_params = grid_search_result.best_params_
        best_cv_score = grid_search_result.best_score_
        logger.info("Using best model from grid search")
    else:
        logger.info("Using default parameters from config for fast training")
        model = train_with_default_params(X_train, y_train, config_dict)
        best_params = get_default_xgb_params(config_dict)
        # Add xgb__ prefix to match grid search format
        best_params = {f"xgb__{k}": v for k, v in best_params.items()}
        best_cv_score = None

    # Evaluate on test set
    metrics = evaluate_model(model, X_test, y_test, config_dict)

    # Get predictions for report
    y_pred = model.predict(X_test)

    # Save metrics
    save_metrics(
        metrics,
        best_params,
        best_cv_score,
        y_test=y_test,
        y_pred=y_pred,
        output_dir=metrics_dir,
        use_grid_search=grid_search,
    )

    # Save model
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    joblib.dump(model, model_output)
    logger.info(f"Model saved to {model_output}")

    # Save feature metadata for inference
    # Extract feature names from the model
    feature_names = list(X_train.columns)
    if hasattr(model, "steps") and len(model.steps) > 0:
        xgb_model = model.steps[-1][1]
        if hasattr(xgb_model, "feature_names_in_") and xgb_model.feature_names_in_ is not None:
            feature_names = list(xgb_model.feature_names_in_)
        elif hasattr(xgb_model, "get_booster"):
            booster = xgb_model.get_booster()
            if hasattr(booster, "feature_names"):
                feature_names = booster.feature_names

    metadata = {
        "feature_names": feature_names,
        "apply_cyclic_encoding": apply_cyclic_encoding,
        "apply_interactions": apply_interactions,
    }

    metadata_path = model_output.replace(".joblib", "_metadata.joblib")
    joblib.dump(metadata, metadata_path)
    logger.info(f"Feature metadata saved to {metadata_path}")

    # Log to MLflow
    if mlflow_enabled:
        log_to_mlflow(
            config_dict,
            model,
            metrics,
            best_params,
            best_cv_score,
            X_train,
            y_train,
            X_test,
            y_test,
            use_grid_search=grid_search,
        )

    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
