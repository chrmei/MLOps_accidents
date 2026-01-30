"""
FastAPI routes for the train service.
"""

import asyncio
import json
import logging
import os
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from services.common.config import Settings
from services.common.dependencies import AdminUser, SettingsDep
from services.common.job_store import JobStatus as StoreJobStatus
from services.common.job_store import JobType, job_store
from services.common.models import JobResponse, JobStatus as ApiJobStatus

from ..core.trainer import run_training
from .schemas import TrainRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/train", tags=["train"])


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _job_to_response(job) -> JobResponse:
    return JobResponse(
        job_id=job.id,
        status=ApiJobStatus(job.status.value),
        job_type=job.job_type,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        result=job.result,
        error=job.error,
        progress=job.progress,
    )


async def _run_training_job(job_id: str, request: TrainRequest, settings: Settings):
    await job_store.update_job(
        job_id,
        status=StoreJobStatus.RUNNING,
        progress=5.0,
        message="Training started",
    )

    try:
        result = await asyncio.to_thread(
            run_training,
            request.models,
            request.grid_search,
            request.compare,
            request.config_path,
        )
        await job_store.update_job(
            job_id,
            status=StoreJobStatus.COMPLETED,
            progress=100.0,
            message="Training completed",
            result=result,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Training job failed")
        await job_store.update_job(
            job_id,
            status=StoreJobStatus.FAILED,
            progress=100.0,
            message="Training failed",
            error=str(exc),
        )


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@router.post("/", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_training(
    request: TrainRequest,
    current_user: AdminUser,
    settings: SettingsDep,
):
    """Trigger model training (admin-only)."""
    job = await job_store.create_job(
        job_type=JobType.TRAINING.value,
        created_by=current_user.username,
        parameters=request.model_dump(),
    )

    asyncio.create_task(_run_training_job(job.id, request, settings))
    return _job_to_response(job)


@router.get("/status/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str, current_user: AdminUser):
    job = await job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return _job_to_response(job)


@router.get("/jobs", response_model=List[JobResponse])
async def list_jobs(
    current_user: AdminUser,
    job_type: Optional[str] = Query(None, description="Filter by job type"),
    status_filter: Optional[ApiJobStatus] = Query(None, alias="status", description="Filter by job status"),
):
    status_value = None
    if status_filter:
        status_value = StoreJobStatus(status_filter.value)

    jobs = await job_store.list_jobs(
        job_type=job_type,
        status=status_value,
        created_by=current_user.username,
    )
    return [_job_to_response(job) for job in jobs]


@router.get("/metrics/{model_type}")
async def get_model_metrics(model_type: str, current_user: AdminUser, settings: SettingsDep):
    """Return saved metrics for a given model type."""
    metrics_dir = os.path.join(settings.DATA_DIR, "metrics")
    metrics_path = os.path.join(metrics_dir, f"{model_type}_metrics.json")

    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Metrics not found for model type '{model_type}'",
        )

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return {"model_type": model_type, "metrics": metrics}
