# -*- coding: utf-8 -*-
"""
Model-specific trainer implementations.

This module contains trainer classes for different model types:
- XGBoostTrainer
- RandomForestTrainer
- LogisticRegressionTrainer
- LightGBMTrainer
"""
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from src.models.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")


class XGBoostTrainer(BaseTrainer):
    """Trainer for XGBoost models."""
    
    model_type = "xgboost"

    def _get_default_params(self) -> Dict:
        """Get default XGBoost parameters from config."""
        xgb_config = self.config.get("xgboost", {}).get("default_params", {})
        # Remove random_state and eval_metric as they're handled separately
        params = {
            k: v
            for k, v in xgb_config.items()
            if k not in ["random_state", "eval_metric"]
        }
        return params

    def _build_model(
        self, X_train: pd.DataFrame, y_train: pd.Series, use_grid_search: bool = False
    ) -> Tuple[object, Dict, Optional[float]]:
        """Build and train XGBoost model."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")

        xgb_config = self.config.get("xgboost", {}).get("default_params", {})
        random_state = xgb_config.get("random_state", 42)
        eval_metric = xgb_config.get("eval_metric", "logloss")

        smote_config = self.config.get("smote", {})
        smote_random_state = smote_config.get("random_state", 42)
        smote_enabled = smote_config.get("enabled", True)

        # Create pipeline with SMOTE + XGBoost
        pipeline_steps = []
        if smote_enabled:
            pipeline_steps.append(("smote", SMOTE(random_state=smote_random_state)))
            logger.info("SMOTE enabled for class imbalance handling")

        if use_grid_search:
            # For grid search, create base model without parameters
            pipeline_steps.append(
                ("xgb", xgb.XGBClassifier(random_state=random_state, eval_metric=eval_metric))
            )
            pipeline = ImbPipeline(pipeline_steps)

            # Get grid search config
            grid_config = self.config.get("grid_search", {})
            param_grid = grid_config.get("param_grid", {})
            cv = grid_config.get("cv", 5)
            scoring = grid_config.get("scoring", "f1")
            n_jobs = grid_config.get("n_jobs", -1)
            verbose = grid_config.get("verbose", 1)

            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose
            )

            logger.info(f"Starting grid search with {cv}-fold CV...")
            grid_search.fit(X_train, y_train)

            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV {scoring} score: {grid_search.best_score_:.4f}")

            return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
        else:
            # Use default parameters
            default_params = self._get_default_params()
            pipeline_steps.append(
                (
                    "xgb",
                    xgb.XGBClassifier(
                        random_state=random_state, eval_metric=eval_metric, **default_params
                    ),
                )
            )
            pipeline = ImbPipeline(pipeline_steps)

            logger.info(f"Default parameters: {default_params}")
            logger.info("Fitting model...")
            pipeline.fit(X_train, y_train)

            # Format params for logging (add xgb__ prefix)
            best_params = {f"xgb__{k}": v for k, v in default_params.items()}
            return pipeline, best_params, None

    def _get_model_params(self, model) -> Dict:
        """Extract XGBoost model parameters."""
        # Extract XGBoost model from pipeline
        xgb_model = model
        if hasattr(model, "steps"):
            for step_name, step_model in model.steps:
                if "xgb" in step_name.lower() or isinstance(step_model, xgb.XGBClassifier):
                    xgb_model = step_model
                    break

        params = {}
        if hasattr(xgb_model, "get_params"):
            params = xgb_model.get_params()

        return params


class RandomForestTrainer(BaseTrainer):
    """Trainer for Random Forest models."""
    
    model_type = "random_forest"

    def _get_default_params(self) -> Dict:
        """Get default Random Forest parameters from config."""
        rf_config = self.config.get("random_forest", {}).get("default_params", {})
        # Remove random_state as it's handled separately
        params = {k: v for k, v in rf_config.items() if k != "random_state"}
        return params

    def _build_model(
        self, X_train: pd.DataFrame, y_train: pd.Series, use_grid_search: bool = False
    ) -> Tuple[object, Dict, Optional[float]]:
        """Build and train Random Forest model."""
        rf_config = self.config.get("random_forest", {}).get("default_params", {})
        random_state = rf_config.get("random_state", 42)

        smote_config = self.config.get("smote", {})
        smote_random_state = smote_config.get("random_state", 42)
        smote_enabled = smote_config.get("enabled", True)

        # Create pipeline with SMOTE + Random Forest
        pipeline_steps = []
        if smote_enabled:
            pipeline_steps.append(("smote", SMOTE(random_state=smote_random_state)))
            logger.info("SMOTE enabled for class imbalance handling")

        if use_grid_search:
            # For grid search
            pipeline_steps.append(
                ("rf", RandomForestClassifier(random_state=random_state))
            )
            pipeline = ImbPipeline(pipeline_steps)

            # Get grid search config
            grid_config = self.config.get("grid_search", {})
            param_grid = grid_config.get("param_grid_rf", {})
            cv = grid_config.get("cv", 5)
            scoring = grid_config.get("scoring", "f1")
            n_jobs = grid_config.get("n_jobs", -1)
            verbose = grid_config.get("verbose", 1)

            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose
            )

            logger.info(f"Starting grid search with {cv}-fold CV...")
            grid_search.fit(X_train, y_train)

            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV {scoring} score: {grid_search.best_score_:.4f}")

            return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
        else:
            # Use default parameters
            default_params = self._get_default_params()
            pipeline_steps.append(
                (
                    "rf",
                    RandomForestClassifier(
                        random_state=random_state, **default_params
                    ),
                )
            )
            pipeline = ImbPipeline(pipeline_steps)

            logger.info(f"Default parameters: {default_params}")
            logger.info("Fitting model...")
            pipeline.fit(X_train, y_train)

            # Format params for logging (add rf__ prefix)
            best_params = {f"rf__{k}": v for k, v in default_params.items()}
            return pipeline, best_params, None

    def _get_model_params(self, model) -> Dict:
        """Extract Random Forest model parameters."""
        rf_model = model
        if hasattr(model, "steps"):
            for step_name, step_model in model.steps:
                if "rf" in step_name.lower() or isinstance(step_model, RandomForestClassifier):
                    rf_model = step_model
                    break

        params = {}
        if hasattr(rf_model, "get_params"):
            params = rf_model.get_params()

        return params


class LogisticRegressionTrainer(BaseTrainer):
    """Trainer for Logistic Regression models."""
    
    model_type = "logistic_regression"

    def _get_default_params(self) -> Dict:
        """Get default Logistic Regression parameters from config."""
        lr_config = self.config.get("logistic_regression", {}).get("default_params", {})
        # Remove random_state as it's handled separately
        params = {k: v for k, v in lr_config.items() if k != "random_state"}
        return params

    def _build_model(
        self, X_train: pd.DataFrame, y_train: pd.Series, use_grid_search: bool = False
    ) -> Tuple[object, Dict, Optional[float]]:
        """Build and train Logistic Regression model."""
        lr_config = self.config.get("logistic_regression", {}).get("default_params", {})
        random_state = lr_config.get("random_state", 42)

        smote_config = self.config.get("smote", {})
        smote_random_state = smote_config.get("random_state", 42)
        smote_enabled = smote_config.get("enabled", True)

        # Create pipeline with SMOTE + StandardScaler + Logistic Regression
        pipeline_steps = []
        if smote_enabled:
            pipeline_steps.append(("smote", SMOTE(random_state=smote_random_state)))
            logger.info("SMOTE enabled for class imbalance handling")

        # Add scaling for logistic regression
        pipeline_steps.append(("scaler", StandardScaler()))

        if use_grid_search:
            # For grid search
            pipeline_steps.append(
                ("lr", LogisticRegression(random_state=random_state, max_iter=1000))
            )
            pipeline = ImbPipeline(pipeline_steps)

            # Get grid search config
            grid_config = self.config.get("grid_search", {})
            param_grid = grid_config.get("param_grid_lr", {})
            cv = grid_config.get("cv", 5)
            scoring = grid_config.get("scoring", "f1")
            n_jobs = grid_config.get("n_jobs", -1)
            verbose = grid_config.get("verbose", 1)

            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose
            )

            logger.info(f"Starting grid search with {cv}-fold CV...")
            grid_search.fit(X_train, y_train)

            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV {scoring} score: {grid_search.best_score_:.4f}")

            return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
        else:
            # Use default parameters
            default_params = self._get_default_params()
            pipeline_steps.append(
                (
                    "lr",
                    LogisticRegression(
                        random_state=random_state, max_iter=1000, **default_params
                    ),
                )
            )
            pipeline = ImbPipeline(pipeline_steps)

            logger.info(f"Default parameters: {default_params}")
            logger.info("Fitting model...")
            pipeline.fit(X_train, y_train)

            # Format params for logging (add lr__ prefix)
            best_params = {f"lr__{k}": v for k, v in default_params.items()}
            return pipeline, best_params, None

    def _get_model_params(self, model) -> Dict:
        """Extract Logistic Regression model parameters."""
        lr_model = model
        if hasattr(model, "steps"):
            for step_name, step_model in model.steps:
                if "lr" in step_name.lower() or isinstance(step_model, LogisticRegression):
                    lr_model = step_model
                    break

        params = {}
        if hasattr(lr_model, "get_params"):
            params = lr_model.get_params()

        return params


class LightGBMTrainer(BaseTrainer):
    """Trainer for LightGBM models."""
    
    model_type = "lightgbm"

    def _get_default_params(self) -> Dict:
        """Get default LightGBM parameters from config."""
        lgbm_config = self.config.get("lightgbm", {}).get("default_params", {})
        # Remove random_state as it's handled separately
        params = {k: v for k, v in lgbm_config.items() if k != "random_state"}
        return params

    def _build_model(
        self, X_train: pd.DataFrame, y_train: pd.Series, use_grid_search: bool = False
    ) -> Tuple[object, Dict, Optional[float]]:
        """Build and train LightGBM model."""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")

        lgbm_config = self.config.get("lightgbm", {}).get("default_params", {})
        random_state = lgbm_config.get("random_state", 42)

        smote_config = self.config.get("smote", {})
        smote_random_state = smote_config.get("random_state", 42)
        smote_enabled = smote_config.get("enabled", True)

        # Create pipeline with SMOTE + LightGBM
        pipeline_steps = []
        if smote_enabled:
            pipeline_steps.append(("smote", SMOTE(random_state=smote_random_state)))
            logger.info("SMOTE enabled for class imbalance handling")

        if use_grid_search:
            # For grid search
            pipeline_steps.append(
                ("lgbm", lgb.LGBMClassifier(random_state=random_state, verbose=-1))
            )
            pipeline = ImbPipeline(pipeline_steps)

            # Get grid search config
            grid_config = self.config.get("grid_search", {})
            param_grid = grid_config.get("param_grid_lgbm", {})
            cv = grid_config.get("cv", 5)
            scoring = grid_config.get("scoring", "f1")
            n_jobs = grid_config.get("n_jobs", -1)
            verbose = grid_config.get("verbose", 1)

            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose
            )

            logger.info(f"Starting grid search with {cv}-fold CV...")
            grid_search.fit(X_train, y_train)

            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV {scoring} score: {grid_search.best_score_:.4f}")

            return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
        else:
            # Use default parameters
            default_params = self._get_default_params()
            pipeline_steps.append(
                (
                    "lgbm",
                    lgb.LGBMClassifier(
                        random_state=random_state, verbose=-1, **default_params
                    ),
                )
            )
            pipeline = ImbPipeline(pipeline_steps)

            logger.info(f"Default parameters: {default_params}")
            logger.info("Fitting model...")
            pipeline.fit(X_train, y_train)

            # Format params for logging (add lgbm__ prefix)
            best_params = {f"lgbm__{k}": v for k, v in default_params.items()}
            return pipeline, best_params, None

    def _get_model_params(self, model) -> Dict:
        """Extract LightGBM model parameters."""
        lgbm_model = model
        if hasattr(model, "steps"):
            for step_name, step_model in model.steps:
                if "lgbm" in step_name.lower() or isinstance(step_model, lgb.LGBMClassifier):
                    lgbm_model = step_model
                    break

        params = {}
        if hasattr(lgbm_model, "get_params"):
            params = lgbm_model.get_params()

        return params

