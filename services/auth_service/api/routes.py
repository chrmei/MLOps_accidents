"""
FastAPI routes for the authentication service.
"""

from datetime import timedelta
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from services.common.auth import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    create_user,
    get_all_users,
    verify_token,
)
from services.common.config import settings
from services.common.dependencies import ActiveUser, AdminUser
from services.common.models import (
    LoginRequest,
    LoginResponse,
    Token,
    TokenRefreshRequest,
    UserCreate,
    UserResponse,
)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Authenticate user and return access token."""
    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role.value},
        expires_delta=access_token_expires,
    )

    refresh_token = create_refresh_token(
        data={"sub": user.username, "role": user.role.value}
    )

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=int(access_token_expires.total_seconds()),
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(request: TokenRefreshRequest):
    """Refresh an expired access token using a refresh token."""
    token_data = verify_token(request.refresh_token, token_type="refresh")

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    new_access_token = create_access_token(
        data={"sub": token_data.username, "role": token_data.role.value},
        expires_delta=access_token_expires,
    )

    return Token(
        access_token=new_access_token,
        token_type="bearer",
        expires_in=int(access_token_expires.total_seconds()),
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: ActiveUser):
    """Get current authenticated user info."""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        is_active=current_user.is_active,
    )


@router.get("/users", response_model=List[UserResponse])
async def list_users(current_user: AdminUser):
    """List all users (admin-only)."""
    all_users = get_all_users()
    return [
        UserResponse(
            id=u.id,
            username=u.username,
            email=u.email,
            full_name=u.full_name,
            role=u.role,
            is_active=u.is_active,
        )
        for u in all_users
    ]


@router.post("/users", response_model=UserResponse)
async def create_new_user(
    user_data: UserCreate,
    current_user: AdminUser,
):
    """Create a new user (admin-only)."""
    new_user = create_user(
        username=user_data.username,
        password=user_data.password,
        email=user_data.email,
        full_name=user_data.full_name,
        role=user_data.role,
    )

    return UserResponse(
        id=new_user.id,
        username=new_user.username,
        email=new_user.email,
        full_name=new_user.full_name,
        role=new_user.role,
        is_active=new_user.is_active,
    )
