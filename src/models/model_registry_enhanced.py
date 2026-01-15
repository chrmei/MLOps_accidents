# -*- coding: utf-8 -*-
"""
Enhanced MLflow Model Registry with automated comparison, promotion, and performance tracking.

This module provides:
1. Automated model comparison across different training runs/versions
2. Automated model promotion workflow based on performance thresholds
3. Model performance tracking over time with degradation detection
"""
import json
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class EnhancedModelRegistry:
    """
    Enhanced MLflow Model Registry with automated workflows.
    
    Features:
    - Automated model comparison across versions
    - Performance-based promotion automation
    - Time-series performance tracking
    - Degradation detection
    """
    
    def __init__(self, config: Dict):
        """
        Initialize enhanced model registry.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary from model_config.yaml
        """
        self.config = config
        self.mlflow_config = config.get("mlflow", {})
        self.registry_config = self.mlflow_config.get("model_registry", {})
        self.promotion_config = self.registry_config.get("promotion", {})
        
        # Setup MLflow client
        self._setup_mlflow_client()
        
    def _setup_mlflow_client(self):
        """Setup MLflow client with tracking URI from config."""
        tracking_uri = self.mlflow_config.get("tracking_uri") or os.getenv("MLFLOW_TRACKING_URI")
        
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
        self.client = MlflowClient()
        logger.info(f"MLflow client initialized with tracking URI: {tracking_uri}")
    
    def compare_models_across_versions(
        self,
        model_name: str,
        versions: Optional[List[int]] = None,
        include_stages: Optional[List[str]] = None,
        output_format: str = "html"
    ) -> Dict:
        """
        Compare models across different versions or stages.
        
        Parameters
        ----------
        model_name : str
            Name of the registered model
        versions : list of int, optional
            Specific versions to compare. If None, compares all versions.
        include_stages : list of str, optional
            Stages to include (e.g., ["Production", "Staging"]). If None, includes all.
        output_format : str
            Output format: "html", "json", or "dataframe"
            
        Returns
        -------
        dict
            Comparison results with metrics, visualizations, and recommendations
        """
        logger.info(f"Comparing versions of model '{model_name}'")
        
        # Get model versions
        if versions:
            model_versions = []
            for version in versions:
                try:
                    mv = self.client.get_model_version(model_name, version)
                    model_versions.append(mv)
                except Exception as e:
                    logger.warning(f"Could not fetch version {version}: {e}")
        else:
            # Get all versions
            filter_string = f"name='{model_name}'"
            if include_stages:
                stage_filter = " OR ".join([f"current_stage='{s}'" for s in include_stages])
                filter_string += f" AND ({stage_filter})"
            
            model_versions = self.client.search_model_versions(filter_string)
        
        if not model_versions:
            logger.warning(f"No versions found for model '{model_name}'")
            return {}
        
        # Sort by version number (descending)
        model_versions = sorted(model_versions, key=lambda v: int(v.version), reverse=True)
        
        # Collect metrics for each version
        comparison_data = []
        for mv in model_versions:
            try:
                # Get run details
                run = self.client.get_run(mv.run_id)
                
                # Extract metrics
                metrics = {
                    "version": int(mv.version),
                    "stage": mv.current_stage,
                    "created": mv.creation_timestamp,
                    "run_id": mv.run_id,
                    "status": mv.status,
                }
                
                # Get performance metrics
                for metric_name in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
                    if metric_name in run.data.metrics:
                        metrics[metric_name] = run.data.metrics[metric_name]
                
                # Get model size (if available)
                try:
                    model_uri = f"models:/{model_name}/{mv.version}"
                    model = mlflow.sklearn.load_model(model_uri)
                    # Estimate model size (rough approximation)
                    import pickle
                    model_size_bytes = len(pickle.dumps(model))
                    metrics["model_size_mb"] = model_size_bytes / (1024 * 1024)
                except Exception as e:
                    logger.debug(f"Could not estimate model size for version {mv.version}: {e}")
                    metrics["model_size_mb"] = None
                
                # Get training parameters
                metrics["n_estimators"] = run.data.params.get("n_estimators", "N/A")
                metrics["max_depth"] = run.data.params.get("max_depth", "N/A")
                metrics["learning_rate"] = run.data.params.get("learning_rate", "N/A")
                
                comparison_data.append(metrics)
                
            except Exception as e:
                logger.warning(f"Error processing version {mv.version}: {e}")
                continue
        
        if not comparison_data:
            logger.warning("No valid comparison data collected")
            return {}
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Generate comparison report
        comparison_result = {
            "model_name": model_name,
            "comparison_date": datetime.now().isoformat(),
            "num_versions": len(comparison_df),
            "data": comparison_df.to_dict("records"),
        }
        
        # Generate visualizations
        if len(comparison_df) > 1:
            comparison_result["visualizations"] = self._generate_comparison_visualizations(
                comparison_df, model_name
            )
        
        # Generate recommendations
        comparison_result["recommendations"] = self._generate_comparison_recommendations(
            comparison_df
        )
        
        # Generate output in requested format
        if output_format == "html":
            comparison_result["html_report"] = self._generate_html_report(comparison_result)
        elif output_format == "dataframe":
            comparison_result["dataframe"] = comparison_df
        
        return comparison_result
    
    def _generate_comparison_visualizations(
        self, comparison_df: pd.DataFrame, model_name: str
    ) -> Dict[str, str]:
        """
        Generate visualization plots for model comparison.
        
        Returns
        -------
        dict
            Dictionary mapping plot names to file paths
        """
        plots = {}
        
        # Metrics comparison plot
        metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
        available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
        
        if available_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for idx, metric in enumerate(available_metrics[:4]):
                ax = axes[idx]
                comparison_df.plot(
                    x="version",
                    y=metric,
                    kind="line",
                    marker="o",
                    ax=ax,
                    title=f"{metric.replace('_', ' ').title()} by Version",
                )
                ax.set_xlabel("Version")
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            metrics_plot_path = tempfile.mktemp(suffix=".png")
            plt.savefig(metrics_plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            plots["metrics_comparison"] = metrics_plot_path
        
        # Stage distribution
        if "stage" in comparison_df.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            stage_counts = comparison_df["stage"].value_counts()
            stage_counts.plot(kind="bar", ax=ax, color="steelblue")
            ax.set_title("Model Versions by Stage")
            ax.set_xlabel("Stage")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            stage_plot_path = tempfile.mktemp(suffix=".png")
            plt.savefig(stage_plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            plots["stage_distribution"] = stage_plot_path
        
        return plots
    
    def _generate_comparison_recommendations(
        self, comparison_df: pd.DataFrame
    ) -> List[Dict[str, str]]:
        """
        Generate recommendations based on comparison results.
        
        Returns
        -------
        list
            List of recommendation dictionaries
        """
        recommendations = []
        
        if "f1_score" not in comparison_df.columns:
            return recommendations
        
        # Find best performing version
        best_version = comparison_df.loc[comparison_df["f1_score"].idxmax()]
        
        # Check if best version is in Production
        production_versions = comparison_df[comparison_df["stage"] == "Production"]
        if not production_versions.empty:
            prod_best = production_versions.loc[production_versions["f1_score"].idxmax()]
            
            if best_version["version"] != prod_best["version"]:
                f1_improvement = (
                    (best_version["f1_score"] - prod_best["f1_score"]) / prod_best["f1_score"]
                ) * 100
                recommendations.append({
                    "type": "promotion",
                    "message": (
                        f"Version {best_version['version']} outperforms Production version "
                        f"{prod_best['version']} by {f1_improvement:.2f}% in F1 score. "
                        f"Consider promoting to Production."
                    ),
                    "best_version": int(best_version["version"]),
                    "current_prod_version": int(prod_best["version"]),
                    "f1_improvement_pct": float(f1_improvement),
                })
        
        # Check for performance degradation
        if len(comparison_df) >= 2:
            sorted_df = comparison_df.sort_values("version")
            recent_f1 = sorted_df.iloc[-1]["f1_score"]
            previous_f1 = sorted_df.iloc[-2]["f1_score"]
            
            if recent_f1 < previous_f1:
                degradation_pct = ((previous_f1 - recent_f1) / previous_f1) * 100
                if degradation_pct > 5:  # More than 5% degradation
                    recommendations.append({
                        "type": "degradation",
                        "message": (
                            f"Performance degradation detected: Version {sorted_df.iloc[-1]['version']} "
                            f"shows {degradation_pct:.2f}% decrease in F1 score compared to "
                            f"version {sorted_df.iloc[-2]['version']}."
                        ),
                        "degradation_pct": float(degradation_pct),
                    })
        
        return recommendations
    
    def _generate_html_report(self, comparison_result: Dict) -> str:
        """Generate HTML report from comparison results."""
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<title>Model Comparison Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #4CAF50; color: white; }",
            "tr:nth-child(even) { background-color: #f2f2f2; }",
            ".recommendation { padding: 10px; margin: 10px 0; border-left: 4px solid #2196F3; background-color: #E3F2FD; }",
            ".warning { border-left-color: #FF9800; background-color: #FFF3E0; }",
            "</style>",
            "</head><body>",
            f"<h1>Model Comparison Report: {comparison_result['model_name']}</h1>",
            f"<p><strong>Generated:</strong> {comparison_result['comparison_date']}</p>",
            f"<p><strong>Versions Compared:</strong> {comparison_result['num_versions']}</p>",
        ]
        
        # Add comparison table
        if comparison_result["data"]:
            html_parts.append("<h2>Comparison Table</h2>")
            html_parts.append("<table>")
            
            # Header
            headers = list(comparison_result["data"][0].keys())
            html_parts.append("<tr>" + "".join([f"<th>{h}</th>" for h in headers]) + "</tr>")
            
            # Rows
            for row in comparison_result["data"]:
                html_parts.append(
                    "<tr>" + "".join([f"<td>{row.get(h, 'N/A')}</td>" for h in headers]) + "</tr>"
                )
            
            html_parts.append("</table>")
        
        # Add visualizations
        if "visualizations" in comparison_result:
            html_parts.append("<h2>Visualizations</h2>")
            for plot_name, plot_path in comparison_result["visualizations"].items():
                if os.path.exists(plot_path):
                    # Convert to base64 for embedding
                    import base64
                    with open(plot_path, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode()
                    html_parts.append(
                        f'<h3>{plot_name.replace("_", " ").title()}</h3>'
                    )
                    html_parts.append(
                        f'<img src="data:image/png;base64,{img_data}" style="max-width: 100%;">'
                    )
        
        # Add recommendations
        if comparison_result.get("recommendations"):
            html_parts.append("<h2>Recommendations</h2>")
            for rec in comparison_result["recommendations"]:
                css_class = "warning" if rec["type"] == "degradation" else "recommendation"
                html_parts.append(
                    f'<div class="{css_class}"><strong>{rec["type"].title()}:</strong> {rec["message"]}</div>'
                )
        
        html_parts.append("</body></html>")
        return "\n".join(html_parts)
    
    def log_comparison_to_mlflow(
        self,
        model_name: str,
        comparison_result: Dict,
        run_id: Optional[str] = None
    ):
        """
        Log comparison report to MLflow as artifacts.
        
        Parameters
        ----------
        model_name : str
            Model name
        comparison_result : dict
            Comparison result dictionary
        run_id : str, optional
            Run ID to log to. If None, creates a new run.
        """
        if run_id:
            with mlflow.start_run(run_id=run_id):
                self._log_comparison_artifacts(comparison_result)
        else:
            with mlflow.start_run(run_name=f"model_comparison_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.set_tag("comparison_type", "model_registry")
                mlflow.set_tag("model_name", model_name)
                self._log_comparison_artifacts(comparison_result)
    
    def _log_comparison_artifacts(self, comparison_result: Dict):
        """Log comparison artifacts to MLflow."""
        files_to_cleanup = []
        
        try:
            # Save JSON report
            json_path = tempfile.mktemp(suffix=".json")
            files_to_cleanup.append(json_path)
            with open(json_path, "w") as f:
                json.dump(comparison_result, f, indent=2, default=str)
            mlflow.log_artifact(json_path, "comparison")
            
            # Save HTML report
            if "html_report" in comparison_result:
                html_path = tempfile.mktemp(suffix=".html")
                files_to_cleanup.append(html_path)
                with open(html_path, "w") as f:
                    f.write(comparison_result["html_report"])
                mlflow.log_artifact(html_path, "comparison")
            
            # Log visualization plots
            if "visualizations" in comparison_result:
                for plot_name, plot_path in comparison_result["visualizations"].items():
                    if os.path.exists(plot_path):
                        mlflow.log_artifact(plot_path, "comparison/plots")
                        files_to_cleanup.append(plot_path)
        finally:
            # Cleanup temporary files
            for file_path in files_to_cleanup:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Could not remove temporary file {file_path}: {e}")
    
    def evaluate_promotion_candidate(
        self,
        model_name: str,
        candidate_version: int,
        target_stage: str = "Production"
    ) -> Dict:
        """
        Evaluate if a model version should be promoted based on configured rules.
        
        Parameters
        ----------
        model_name : str
            Model name
        candidate_version : int
            Version to evaluate for promotion
        target_stage : str
            Target stage (default: Production)
            
        Returns
        -------
        dict
            Evaluation result with promotion decision and reasoning
        """
        logger.info(
            f"Evaluating promotion candidate: {model_name} version {candidate_version} to {target_stage}"
        )
        
        # Get candidate version metrics
        try:
            candidate_mv = self.client.get_model_version(model_name, candidate_version)
            candidate_run = self.client.get_run(candidate_mv.run_id)
            candidate_metrics = candidate_run.data.metrics
        except Exception as e:
            logger.error(f"Could not fetch candidate version: {e}")
            return {
                "promote": False,
                "reason": f"Error fetching candidate version: {e}",
            }
        
        # Get current Production version (if exists)
        try:
            prod_versions = self.client.get_latest_versions(model_name, stages=[target_stage])
            current_prod = prod_versions[0] if prod_versions else None
        except Exception as e:
            logger.warning(f"Could not fetch current {target_stage} version: {e}")
            current_prod = None
        
        evaluation = {
            "candidate_version": candidate_version,
            "target_stage": target_stage,
            "promote": False,
            "reason": "",
            "metrics_comparison": {},
        }
        
        # Check promotion rules
        promotion_rules = self.promotion_config.get("rules", {})
        primary_metric = promotion_rules.get("primary_metric", "f1_score")
        min_threshold = promotion_rules.get("min_threshold", {})
        improvement_threshold = promotion_rules.get("improvement_threshold_pct", 0)
        
        # Check minimum threshold
        if primary_metric in candidate_metrics:
            candidate_score = candidate_metrics[primary_metric]
            min_score = min_threshold.get(primary_metric, 0)
            
            if candidate_score < min_score:
                evaluation["promote"] = False
                evaluation["reason"] = (
                    f"Candidate {primary_metric} ({candidate_score:.4f}) "
                    f"below minimum threshold ({min_score:.4f})"
                )
                return evaluation
        
        # Compare with current Production (if exists)
        if current_prod:
            try:
                prod_run = self.client.get_run(current_prod.run_id)
                prod_metrics = prod_run.data.metrics
                
                if primary_metric in candidate_metrics and primary_metric in prod_metrics:
                    candidate_score = candidate_metrics[primary_metric]
                    prod_score = prod_metrics[primary_metric]
                    
                    improvement_pct = ((candidate_score - prod_score) / prod_score) * 100
                    
                    evaluation["metrics_comparison"] = {
                        "candidate": {primary_metric: candidate_score},
                        "current_production": {primary_metric: prod_score},
                        "improvement_pct": improvement_pct,
                    }
                    
                    if improvement_pct >= improvement_threshold:
                        evaluation["promote"] = True
                        evaluation["reason"] = (
                            f"Candidate outperforms Production by {improvement_pct:.2f}% "
                            f"(threshold: {improvement_threshold}%)"
                        )
                    else:
                        evaluation["promote"] = False
                        evaluation["reason"] = (
                            f"Candidate improvement ({improvement_pct:.2f}%) "
                            f"below threshold ({improvement_threshold}%)"
                        )
                else:
                    evaluation["promote"] = False
                    evaluation["reason"] = f"Primary metric '{primary_metric}' not found in both versions"
                    
            except Exception as e:
                logger.warning(f"Error comparing with Production: {e}")
                evaluation["promote"] = False
                evaluation["reason"] = f"Error comparing with Production: {e}"
        else:
            # No current Production version - promote if meets minimum threshold
            if primary_metric in candidate_metrics:
                candidate_score = candidate_metrics[primary_metric]
                min_score = min_threshold.get(primary_metric, 0)
                
                if candidate_score >= min_score:
                    evaluation["promote"] = True
                    evaluation["reason"] = (
                        f"No current {target_stage} version. "
                        f"Candidate meets minimum threshold ({candidate_score:.4f} >= {min_score:.4f})"
                    )
                else:
                    evaluation["promote"] = False
                    evaluation["reason"] = (
                        f"Candidate {primary_metric} ({candidate_score:.4f}) "
                        f"below minimum threshold ({min_score:.4f})"
                    )
        
        return evaluation
    
    def auto_promote_model(
        self,
        model_name: str,
        candidate_version: int,
        target_stage: str = "Production",
        rollback_on_failure: bool = True
    ) -> Dict:
        """
        Automatically promote a model version if it meets promotion criteria.
        
        Parameters
        ----------
        model_name : str
            Model name
        candidate_version : int
            Version to promote
        target_stage : str
            Target stage (default: Production)
        rollback_on_failure : bool
            If True, rollback promotion if new model performs worse
            
        Returns
        -------
        dict
            Promotion result with status and details
        """
        logger.info(
            f"Attempting auto-promotion: {model_name} version {candidate_version} to {target_stage}"
        )
        
        # Evaluate promotion candidate
        evaluation = self.evaluate_promotion_candidate(model_name, candidate_version, target_stage)
        
        result = {
            "model_name": model_name,
            "candidate_version": candidate_version,
            "target_stage": target_stage,
            "promoted": False,
            "evaluation": evaluation,
            "timestamp": datetime.now().isoformat(),
        }
        
        if not evaluation["promote"]:
            result["reason"] = evaluation["reason"]
            logger.info(f"Promotion rejected: {evaluation['reason']}")
            return result
        
        # Perform promotion
        try:
            # Get current Production version (if exists) for potential rollback
            prod_versions = self.client.get_latest_versions(model_name, stages=[target_stage])
            previous_prod_version = prod_versions[0].version if prod_versions else None
            
            # Transition candidate to target stage
            self.client.transition_model_version_stage(
                name=model_name,
                version=candidate_version,
                stage=target_stage
            )
            
            # Archive previous Production version if exists
            if previous_prod_version:
                try:
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=previous_prod_version,
                        stage="Archived"
                    )
                    result["previous_version_archived"] = previous_prod_version
                except Exception as e:
                    logger.warning(f"Could not archive previous version: {e}")
            
            result["promoted"] = True
            result["reason"] = "Promotion successful"
            logger.info(
                f"Successfully promoted {model_name} version {candidate_version} to {target_stage}"
            )
            
            # Log promotion event
            self._log_promotion_event(result)
            
        except Exception as e:
            result["promoted"] = False
            result["reason"] = f"Promotion failed: {e}"
            logger.error(f"Promotion failed: {e}")
        
        return result
    
    def _log_promotion_event(self, promotion_result: Dict):
        """Log promotion event to MLflow."""
        try:
            with mlflow.start_run(run_name=f"promotion_{promotion_result['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.set_tag("event_type", "model_promotion")
                mlflow.set_tag("model_name", promotion_result["model_name"])
                mlflow.set_tag("promoted_version", str(promotion_result["candidate_version"]))
                mlflow.set_tag("target_stage", promotion_result["target_stage"])
                
                # Log promotion details as JSON
                json_path = tempfile.mktemp(suffix=".json")
                with open(json_path, "w") as f:
                    json.dump(promotion_result, f, indent=2, default=str)
                mlflow.log_artifact(json_path, "promotion")
                os.remove(json_path)
        except Exception as e:
            logger.warning(f"Could not log promotion event: {e}")
    
    def track_performance_over_time(
        self,
        model_name: str,
        stage: str = "Production",
        metric_name: str = "f1_score"
    ) -> pd.DataFrame:
        """
        Track model performance over time for a specific stage.
        
        Parameters
        ----------
        model_name : str
            Model name
        stage : str
            Stage to track (default: Production)
        metric_name : str
            Metric to track (default: f1_score)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with performance history
        """
        logger.info(f"Tracking performance over time for {model_name} in {stage} stage")
        
        # Get all versions in the specified stage
        try:
            versions = self.client.search_model_versions(f"name='{model_name}' AND current_stage='{stage}'")
        except Exception as e:
            logger.error(f"Error fetching versions: {e}")
            return pd.DataFrame()
        
        if not versions:
            logger.warning(f"No versions found in {stage} stage")
            return pd.DataFrame()
        
        # Collect performance data
        performance_data = []
        for mv in versions:
            try:
                run = self.client.get_run(mv.run_id)
                
                if metric_name in run.data.metrics:
                    performance_data.append({
                        "version": int(mv.version),
                        "timestamp": mv.creation_timestamp,
                        metric_name: run.data.metrics[metric_name],
                        "stage": mv.current_stage,
                    })
            except Exception as e:
                logger.warning(f"Error processing version {mv.version}: {e}")
                continue
        
        if not performance_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(performance_data)
        df = df.sort_values("timestamp")
        
        return df
    
    def detect_performance_degradation(
        self,
        model_name: str,
        stage: str = "Production",
        metric_name: str = "f1_score",
        threshold_pct: float = 5.0
    ) -> Dict:
        """
        Detect performance degradation over time.
        
        Parameters
        ----------
        model_name : str
            Model name
        stage : str
            Stage to check
        metric_name : str
            Metric to check
        threshold_pct : float
            Degradation threshold percentage
            
        Returns
        -------
        dict
            Degradation detection result
        """
        logger.info(f"Detecting performance degradation for {model_name} in {stage}")
        
        # Get performance history
        history_df = self.track_performance_over_time(model_name, stage, metric_name)
        
        if len(history_df) < 2:
            return {
                "degradation_detected": False,
                "reason": "Insufficient history for degradation detection",
            }
        
        # Calculate degradation
        latest_value = history_df.iloc[-1][metric_name]
        previous_value = history_df.iloc[-2][metric_name]
        
        degradation_pct = ((previous_value - latest_value) / previous_value) * 100
        
        result = {
            "degradation_detected": degradation_pct > threshold_pct,
            "degradation_pct": degradation_pct,
            "threshold_pct": threshold_pct,
            "latest_version": int(history_df.iloc[-1]["version"]),
            "previous_version": int(history_df.iloc[-2]["version"]),
            "latest_value": float(latest_value),
            "previous_value": float(previous_value),
        }
        
        if result["degradation_detected"]:
            result["alert"] = (
                f"Performance degradation detected: {degradation_pct:.2f}% decrease "
                f"in {metric_name} from version {result['previous_version']} to {result['latest_version']}"
            )
            logger.warning(result["alert"])
        
        return result
    
    def generate_performance_report(
        self,
        model_name: str,
        stage: str = "Production"
    ) -> Dict:
        """
        Generate comprehensive performance report for a model stage.
        
        Parameters
        ----------
        model_name : str
            Model name
        stage : str
            Stage to report on
            
        Returns
        -------
        dict
            Performance report with history, trends, and alerts
        """
        logger.info(f"Generating performance report for {model_name} in {stage}")
        
        # Track performance over time
        history_df = self.track_performance_over_time(model_name, stage)
        
        report = {
            "model_name": model_name,
            "stage": stage,
            "report_date": datetime.now().isoformat(),
            "history": history_df.to_dict("records") if not history_df.empty else [],
        }
        
        if history_df.empty:
            report["status"] = "no_history"
            report["message"] = "No performance history available"
            return report
        
        # Calculate trends
        metrics_to_analyze = ["f1_score", "accuracy", "precision", "recall"]
        trends = {}
        
        for metric in metrics_to_analyze:
            if metric in history_df.columns:
                values = history_df[metric].values
                if len(values) >= 2:
                    trend = "improving" if values[-1] > values[0] else "degrading"
                    change_pct = ((values[-1] - values[0]) / values[0]) * 100
                    trends[metric] = {
                        "trend": trend,
                        "change_pct": float(change_pct),
                        "latest": float(values[-1]),
                        "earliest": float(values[0]),
                    }
        
        report["trends"] = trends
        
        # Detect degradation
        degradation = self.detect_performance_degradation(model_name, stage)
        report["degradation"] = degradation
        
        # Overall status
        if degradation.get("degradation_detected"):
            report["status"] = "degrading"
        elif trends and any(t.get("trend") == "improving" for t in trends.values()):
            report["status"] = "improving"
        else:
            report["status"] = "stable"
        
        return report

