#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MLflow Model Registry Management Script

This script provides utilities for managing models in the MLflow Model Registry:
- List registered models and versions
- Transition models between stages (None, Staging, Production, Archived)
- Promote models to production
- Get model information by stage or version

Usage:
    # List all registered models
    python scripts/manage_model_registry.py list-models

    # List all versions of a model
    python scripts/manage_model_registry.py list-versions --model-name XGBoost_Accident_Prediction

    # Transition a model version to a stage
    python scripts/manage_model_registry.py transition --model-name XGBoost_Accident_Prediction --version 1 --stage Production

    # Promote latest version to Production
    python scripts/manage_model_registry.py promote --model-name XGBoost_Accident_Prediction --stage Production

    # Get model info by stage
    python scripts/manage_model_registry.py get-model --model-name XGBoost_Accident_Prediction --stage Production

    # Archive a model version
    python scripts/manage_model_registry.py archive --model-name XGBoost_Accident_Prediction --version 1
"""
import argparse
import logging
import os
import sys

import mlflow
import yaml
from mlflow.tracking import MlflowClient

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_mlflow_client(config_path="src/config/model_config.yaml"):
    """
    Setup MLflow client using configuration from config file.

    Parameters
    ----------
    config_path : str
        Path to configuration YAML file

    Returns
    -------
    MlflowClient
        Configured MLflow client
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Please ensure model_config.yaml exists in src/config/"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    mlflow_config = config.get("mlflow", {})

    if not mlflow_config.get("enabled", False):
        raise ValueError("MLflow is disabled in configuration")

    # Get tracking URI from config or environment
    tracking_uri = mlflow_config.get("tracking_uri") or os.getenv("MLFLOW_TRACKING_URI")

    # If not set, construct from DAGSHUB_REPO
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
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    return MlflowClient()


def list_models(client):
    """List all registered models."""
    logger.info("Fetching registered models...")
    try:
        models = client.search_registered_models()
        if not models:
            print("No registered models found.")
            return

        print("\n" + "=" * 80)
        print("Registered Models")
        print("=" * 80)
        for model in models:
            print(f"\nModel Name: {model.name}")
            print(f"  Latest Versions: {len(model.latest_versions)}")
            if model.latest_versions:
                for version in model.latest_versions:
                    print(
                        f"    Version {version.version}: {version.current_stage} "
                        f"(Created: {version.creation_timestamp})"
                    )
        print("\n" + "=" * 80)
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise


def list_versions(client, model_name):
    """List all versions of a registered model."""
    logger.info(f"Fetching versions for model '{model_name}'...")
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            print(f"No versions found for model '{model_name}'")
            return

        print("\n" + "=" * 80)
        print(f"Versions for Model: {model_name}")
        print("=" * 80)
        for version in sorted(versions, key=lambda v: v.version, reverse=True):
            print(f"\nVersion {version.version}")
            print(f"  Stage: {version.current_stage}")
            print(f"  Status: {version.status}")
            print(f"  Created: {version.creation_timestamp}")
            print(f"  Run ID: {version.run_id}")
            if version.description:
                print(f"  Description: {version.description}")
        print("\n" + "=" * 80)
    except Exception as e:
        logger.error(f"Failed to list versions: {e}")
        raise


def transition_model(client, model_name, version, stage):
    """
    Transition a model version to a specific stage.

    Parameters
    ----------
    client : MlflowClient
        MLflow client
    model_name : str
        Name of the registered model
    version : int
        Model version number
    stage : str
        Target stage (None, Staging, Production, Archived)
    """
    valid_stages = ["None", "Staging", "Production", "Archived"]
    if stage not in valid_stages:
        raise ValueError(f"Invalid stage '{stage}'. Must be one of: {valid_stages}")

    logger.info(f"Transitioning {model_name} version {version} to {stage}...")
    try:
        client.transition_model_version_stage(
            name=model_name, version=version, stage=stage
        )
        print(f"✓ Successfully transitioned {model_name} version {version} to {stage}")
    except Exception as e:
        logger.error(f"Failed to transition model: {e}")
        raise


def promote_model(client, model_name, stage="Production"):
    """
    Promote the latest version of a model to a specific stage.

    Parameters
    ----------
    client : MlflowClient
        MLflow client
    model_name : str
        Name of the registered model
    stage : str
        Target stage (default: Production)
    """
    logger.info(f"Promoting latest version of {model_name} to {stage}...")
    try:
        # Get latest version
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise ValueError(f"No versions found for model '{model_name}'")

        latest_version = max(versions, key=lambda v: int(v.version))
        transition_model(client, model_name, int(latest_version.version), stage)
    except Exception as e:
        logger.error(f"Failed to promote model: {e}")
        raise


def get_model_by_stage(client, model_name, stage):
    """
    Get model information by stage.

    Parameters
    ----------
    client : MlflowClient
        MLflow client
    model_name : str
        Name of the registered model
    stage : str
        Stage name (Staging, Production, etc.)
    """
    logger.info(f"Fetching {model_name} from {stage} stage...")
    try:
        model_version = client.get_latest_versions(model_name, stages=[stage])
        if not model_version:
            print(f"No model found in {stage} stage for '{model_name}'")
            return

        version = model_version[0]
        print("\n" + "=" * 80)
        print(f"Model: {model_name} (Stage: {stage})")
        print("=" * 80)
        print(f"Version: {version.version}")
        print(f"Stage: {version.current_stage}")
        print(f"Status: {version.status}")
        print(f"Created: {version.creation_timestamp}")
        print(f"Run ID: {version.run_id}")
        print(f"Model URI: models:/{model_name}/{stage}")
        if version.description:
            print(f"Description: {version.description}")
        print("=" * 80 + "\n")
    except Exception as e:
        logger.error(f"Failed to get model: {e}")
        raise


def archive_model(client, model_name, version):
    """
    Archive a model version.

    Parameters
    ----------
    client : MlflowClient
        MLflow client
    model_name : str
        Name of the registered model
    version : int
        Model version number
    """
    transition_model(client, model_name, version, "Archived")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage MLflow Model Registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="src/config/model_config.yaml",
        help="Path to model configuration YAML file",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List models command
    list_models_parser = subparsers.add_parser("list-models", help="List all registered models")

    # List versions command
    list_versions_parser = subparsers.add_parser(
        "list-versions", help="List all versions of a model"
    )
    list_versions_parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the registered model",
    )

    # Transition command
    transition_parser = subparsers.add_parser(
        "transition", help="Transition a model version to a stage"
    )
    transition_parser.add_argument(
        "--model-name", type=str, required=True, help="Name of the registered model"
    )
    transition_parser.add_argument(
        "--version", type=int, required=True, help="Model version number"
    )
    transition_parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["None", "Staging", "Production", "Archived"],
        help="Target stage",
    )

    # Promote command
    promote_parser = subparsers.add_parser(
        "promote", help="Promote latest version to a stage"
    )
    promote_parser.add_argument(
        "--model-name", type=str, required=True, help="Name of the registered model"
    )
    promote_parser.add_argument(
        "--stage",
        type=str,
        default="Production",
        choices=["Staging", "Production"],
        help="Target stage (default: Production)",
    )

    # Get model command
    get_model_parser = subparsers.add_parser(
        "get-model", help="Get model information by stage"
    )
    get_model_parser.add_argument(
        "--model-name", type=str, required=True, help="Name of the registered model"
    )
    get_model_parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["Staging", "Production"],
        help="Stage name",
    )

    # Archive command
    archive_parser = subparsers.add_parser("archive", help="Archive a model version")
    archive_parser.add_argument(
        "--model-name", type=str, required=True, help="Name of the registered model"
    )
    archive_parser.add_argument(
        "--version", type=int, required=True, help="Model version number"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        client = setup_mlflow_client(args.config)

        if args.command == "list-models":
            list_models(client)
        elif args.command == "list-versions":
            list_versions(client, args.model_name)
        elif args.command == "transition":
            transition_model(client, args.model_name, args.version, args.stage)
        elif args.command == "promote":
            promote_model(client, args.model_name, args.stage)
        elif args.command == "get-model":
            get_model_by_stage(client, args.model_name, args.stage)
        elif args.command == "archive":
            archive_model(client, args.model_name, args.version)

    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

