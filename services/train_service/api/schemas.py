"""
Pydantic schemas for the train service API.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    """Request payload to trigger training."""

    models: Optional[List[str]] = Field(
        default=None,
        description="Model identifiers to train (e.g., xgboost, random_forest). Defaults to config enabled models.",
    )
    grid_search: Optional[bool] = Field(
        default=None,
        description="Override grid search flag. If omitted, uses config value.",
    )
    compare: bool = Field(default=True, description="Generate comparison report and best model selection.")
    config_path: str = Field(
        default="src/config/model_config.yaml",
        description="Path to training configuration file.",
    )
