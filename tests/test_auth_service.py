"""
Test suite for Auth Service API endpoints.

Tests cover:
- Successful authentication flows
- Failed authentication scenarios
- Token refresh
- User management (admin operations)
- Authorization checks
"""

import pytest
from httpx import AsyncClient


@pytest.mark.auth
class TestAuthService:
    """Test suite for Auth Service endpoints."""

    # =========================================================================
    # Health Check Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_health_check(
        self, http_client: AsyncClient, auth_health_url: str
    ):
        """Test health check endpoint."""
        response = await http_client.get(auth_health_url)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert "timestamp" in data

    # =========================================================================
    # Login Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_login_success(
        self,
        http_client: AsyncClient,
        auth_base_url: str,
        admin_credentials: dict,
    ):
        """Test successful login with valid credentials."""
        response = await http_client.post(
            f"{auth_base_url}/login",
            json=admin_credentials,
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
        assert isinstance(data["expires_in"], int)
        assert data["expires_in"] > 0

    @pytest.mark.asyncio
    async def test_login_invalid_username(
        self, http_client: AsyncClient, auth_base_url: str
    ):
        """Test login with invalid username."""
        response = await http_client.post(
            f"{auth_base_url}/login",
            json={"username": "nonexistent", "password": "password"},
        )
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_login_invalid_password(
        self,
        http_client: AsyncClient,
        auth_base_url: str,
        admin_credentials: dict,
    ):
        """Test login with invalid password."""
        response = await http_client.post(
            f"{auth_base_url}/login",
            json={
                "username": admin_credentials["username"],
                "password": "wrong_password",
            },
        )
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_login_missing_fields(
        self, http_client: AsyncClient, auth_base_url: str
    ):
        """Test login with missing required fields."""
        # Missing password
        response = await http_client.post(
            f"{auth_base_url}/login",
            json={"username": "admin"},
        )
        assert response.status_code == 422

        # Missing username
        response = await http_client.post(
            f"{auth_base_url}/login",
            json={"password": "password"},
        )
        assert response.status_code == 422

        # Empty body
        response = await http_client.post(f"{auth_base_url}/login", json={})
        assert response.status_code == 422

    # =========================================================================
    # Token Refresh Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_refresh_token_success(
        self,
        http_client: AsyncClient,
        auth_base_url: str,
        admin_credentials: dict,
    ):
        """Test successful token refresh."""
        # First, login to get refresh token
        login_response = await http_client.post(
            f"{auth_base_url}/login",
            json=admin_credentials,
        )
        assert login_response.status_code == 200
        refresh_token = login_response.json()["refresh_token"]

        # Refresh the token
        response = await http_client.post(
            f"{auth_base_url}/refresh",
            json={"refresh_token": refresh_token},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data

    @pytest.mark.asyncio
    async def test_refresh_token_invalid(
        self, http_client: AsyncClient, auth_base_url: str, invalid_token: str
    ):
        """Test token refresh with invalid refresh token."""
        response = await http_client.post(
            f"{auth_base_url}/refresh",
            json={"refresh_token": invalid_token},
        )
        assert response.status_code in [401, 422]

    @pytest.mark.asyncio
    async def test_refresh_token_missing(
        self, http_client: AsyncClient, auth_base_url: str
    ):
        """Test token refresh with missing refresh token."""
        response = await http_client.post(
            f"{auth_base_url}/refresh",
            json={},
        )
        assert response.status_code == 422

    # =========================================================================
    # Current User Info Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_get_current_user_success(
        self,
        http_client: AsyncClient,
        auth_base_url: str,
        admin_token: str,
    ):
        """Test getting current user info with valid token."""
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = await http_client.get(
            f"{auth_base_url}/me",
            headers=headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "username" in data
        assert "role" in data
        assert data["role"] == "admin"

    @pytest.mark.asyncio
    async def test_get_current_user_no_token(
        self, http_client: AsyncClient, auth_base_url: str
    ):
        """Test getting current user info without token."""
        response = await http_client.get(f"{auth_base_url}/me")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(
        self,
        http_client: AsyncClient,
        auth_base_url: str,
        invalid_token: str,
    ):
        """Test getting current user info with invalid token."""
        headers = {"Authorization": f"Bearer {invalid_token}"}
        response = await http_client.get(
            f"{auth_base_url}/me",
            headers=headers,
        )
        assert response.status_code == 401

    # =========================================================================
    # User Management Tests (Admin Only)
    # =========================================================================

    @pytest.mark.asyncio
    async def test_list_users_success(
        self,
        http_client: AsyncClient,
        auth_base_url: str,
        admin_headers: dict,
    ):
        """Test listing all users as admin."""
        response = await http_client.get(
            f"{auth_base_url}/users",
            headers=admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # Should have at least admin user
        assert len(data) > 0
        assert any(user["username"] == "admin" for user in data)

    @pytest.mark.asyncio
    async def test_list_users_unauthorized(
        self,
        http_client: AsyncClient,
        auth_base_url: str,
        user_headers: dict,
    ):
        """Test listing users as regular user (should fail)."""
        response = await http_client.get(
            f"{auth_base_url}/users",
            headers=user_headers,
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_create_user_success(
        self,
        http_client: AsyncClient,
        auth_base_url: str,
        admin_headers: dict,
    ):
        """Test creating a new user as admin."""
        import uuid
        # Use unique username to avoid conflicts from previous test runs
        unique_id = str(uuid.uuid4())[:8]
        user_data = {
            "username": f"newuser_{unique_id}",
            "password": "NewUser@123",
            "email": f"newuser_{unique_id}@example.com",
            "full_name": "New User",
            "role": "user",
        }
        response = await http_client.post(
            f"{auth_base_url}/users",
            json=user_data,
            headers=admin_headers,
        )
        # Accept both 200 (created) and 201 (created) as success
        assert response.status_code in (200, 201), f"Expected 200/201, got {response.status_code}: {response.text}"
        data = response.json()
        assert data["username"] == user_data["username"]
        assert data["email"] == user_data["email"]
        assert data["role"] == user_data["role"]
        assert "id" in data

    @pytest.mark.asyncio
    async def test_create_user_duplicate(
        self,
        http_client: AsyncClient,
        auth_base_url: str,
        admin_headers: dict,
    ):
        """Test creating a user with duplicate username."""
        user_data = {
            "username": "admin",  # Already exists
            "password": "Password@123",
            "email": "admin2@example.com",
            "role": "user",
        }
        response = await http_client.post(
            f"{auth_base_url}/users",
            json=user_data,
            headers=admin_headers,
        )
        # Should fail with 400 or 409
        assert response.status_code in [400, 409, 422]

    @pytest.mark.asyncio
    async def test_create_user_invalid_data(
        self,
        http_client: AsyncClient,
        auth_base_url: str,
        admin_headers: dict,
    ):
        """Test creating user with invalid data."""
        # Missing required fields
        response = await http_client.post(
            f"{auth_base_url}/users",
            json={"username": "test"},
            headers=admin_headers,
        )
        assert response.status_code == 422

        # Invalid email format
        response = await http_client.post(
            f"{auth_base_url}/users",
            json={
                "username": "testuser2",
                "password": "Test@123",
                "email": "invalid-email",
                "role": "user",
            },
            headers=admin_headers,
        )
        assert response.status_code == 422

        # Password too short
        response = await http_client.post(
            f"{auth_base_url}/users",
            json={
                "username": "testuser3",
                "password": "short",
                "email": "test@example.com",
                "role": "user",
            },
            headers=admin_headers,
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_create_user_unauthorized(
        self,
        http_client: AsyncClient,
        auth_base_url: str,
        user_headers: dict,
    ):
        """Test creating user as regular user (should fail)."""
        user_data = {
            "username": "unauthorized_user",
            "password": "Password@123",
            "email": "unauthorized@example.com",
            "role": "user",
        }
        response = await http_client.post(
            f"{auth_base_url}/users",
            json=user_data,
            headers=user_headers,
        )
        assert response.status_code == 403
