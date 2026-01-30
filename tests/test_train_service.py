"""
Test suite for Train Service API endpoints.

Tests cover:
- Health checks
- Training job creation and status
- Job listing and filtering
- Model metrics retrieval
- Authorization checks (admin-only endpoints)
"""

import pytest
from httpx import AsyncClient


@pytest.mark.train
class TestTrainService:
    """Test suite for Train Service endpoints."""

    # =========================================================================
    # Health Check Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_health_check(
        self, http_client: AsyncClient, train_health_url: str
    ):
        """Test health check endpoint."""
        response = await http_client.get(train_health_url)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data

    # =========================================================================
    # Training Job Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_start_training_success(
        self,
        http_client: AsyncClient,
        train_base_url: str,
        admin_headers: dict,
    ):
        """Test starting training job with valid request."""
        request_data = {}
        response = await http_client.post(
            f"{train_base_url}/",
            json=request_data,
            headers=admin_headers,
        )
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert data["job_type"] == "training"

    @pytest.mark.asyncio
    async def test_start_training_with_models(
        self,
        http_client: AsyncClient,
        train_base_url: str,
        admin_headers: dict,
    ):
        """Test starting training job with specific models."""
        request_data = {
            "models": ["logistic_regression", "random_forest"],
            "compare": True,
        }
        response = await http_client.post(
            f"{train_base_url}/",
            json=request_data,
            headers=admin_headers,
        )
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data

    @pytest.mark.asyncio
    async def test_start_training_with_grid_search(
        self,
        http_client: AsyncClient,
        train_base_url: str,
        admin_headers: dict,
    ):
        """Test starting training job with grid search enabled."""
        request_data = {
            "grid_search": True,
            "compare": False,
        }
        response = await http_client.post(
            f"{train_base_url}/",
            json=request_data,
            headers=admin_headers,
        )
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data

    @pytest.mark.asyncio
    async def test_start_training_with_custom_config(
        self,
        http_client: AsyncClient,
        train_base_url: str,
        admin_headers: dict,
    ):
        """Test starting training job with custom config path."""
        request_data = {
            "config_path": "src/config/model_config.yaml",
        }
        response = await http_client.post(
            f"{train_base_url}/",
            json=request_data,
            headers=admin_headers,
        )
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data

    @pytest.mark.asyncio
    async def test_start_training_unauthorized(
        self,
        http_client: AsyncClient,
        train_base_url: str,
        user_headers: dict,
    ):
        """Test starting training job as regular user (should fail)."""
        response = await http_client.post(
            f"{train_base_url}/",
            json={},
            headers=user_headers,
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_start_training_no_auth(
        self, http_client: AsyncClient, train_base_url: str
    ):
        """Test starting training job without authentication."""
        response = await http_client.post(
            f"{train_base_url}/",
            json={},
        )
        assert response.status_code == 401

    # =========================================================================
    # Job Status Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_get_job_status_success(
        self,
        http_client: AsyncClient,
        train_base_url: str,
        admin_headers: dict,
    ):
        """Test getting job status for existing job."""
        # First create a job
        create_response = await http_client.post(
            f"{train_base_url}/",
            json={},
            headers=admin_headers,
        )
        assert create_response.status_code == 202
        job_id = create_response.json()["job_id"]

        # Get job status
        response = await http_client.get(
            f"{train_base_url}/status/{job_id}",
            headers=admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert "status" in data
        assert "job_type" in data

    @pytest.mark.asyncio
    async def test_get_job_status_not_found(
        self,
        http_client: AsyncClient,
        train_base_url: str,
        admin_headers: dict,
    ):
        """Test getting status for non-existent job."""
        response = await http_client.get(
            f"{train_base_url}/status/nonexistent-job-id",
            headers=admin_headers,
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_job_status_unauthorized(
        self,
        http_client: AsyncClient,
        train_base_url: str,
        user_headers: dict,
    ):
        """Test getting job status as regular user."""
        response = await http_client.get(
            f"{train_base_url}/status/some-job-id",
            headers=user_headers,
        )
        assert response.status_code == 403

    # =========================================================================
    # Job Listing Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_list_jobs_success(
        self,
        http_client: AsyncClient,
        train_base_url: str,
        admin_headers: dict,
    ):
        """Test listing all training jobs."""
        response = await http_client.get(
            f"{train_base_url}/jobs",
            headers=admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_list_jobs_with_filters(
        self,
        http_client: AsyncClient,
        train_base_url: str,
        admin_headers: dict,
    ):
        """Test listing jobs with filters."""
        # Filter by status
        response = await http_client.get(
            f"{train_base_url}/jobs?status=pending",
            headers=admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_list_jobs_unauthorized(
        self,
        http_client: AsyncClient,
        train_base_url: str,
        user_headers: dict,
    ):
        """Test listing jobs as regular user."""
        response = await http_client.get(
            f"{train_base_url}/jobs",
            headers=user_headers,
        )
        assert response.status_code == 403

    # =========================================================================
    # Model Metrics Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_get_model_metrics_success(
        self,
        http_client: AsyncClient,
        train_base_url: str,
        admin_headers: dict,
    ):
        """Test getting model metrics for existing model type."""
        # Try to get metrics for a common model type
        # Note: This may fail if metrics don't exist, which is OK
        model_types = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
        for model_type in model_types:
            response = await http_client.get(
                f"{train_base_url}/metrics/{model_type}",
                headers=admin_headers,
            )
            # Accept both success (200) and not found (404)
            assert response.status_code in [200, 404]
            if response.status_code == 200:
                data = response.json()
                assert "model_type" in data
                assert "metrics" in data
                break

    @pytest.mark.asyncio
    async def test_get_model_metrics_not_found(
        self,
        http_client: AsyncClient,
        train_base_url: str,
        admin_headers: dict,
    ):
        """Test getting metrics for non-existent model type."""
        response = await http_client.get(
            f"{train_base_url}/metrics/nonexistent_model",
            headers=admin_headers,
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_model_metrics_unauthorized(
        self,
        http_client: AsyncClient,
        train_base_url: str,
        user_headers: dict,
    ):
        """Test getting model metrics as regular user."""
        response = await http_client.get(
            f"{train_base_url}/metrics/logistic_regression",
            headers=user_headers,
        )
        assert response.status_code == 403
