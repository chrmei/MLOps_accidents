"""JWT-aware HTTP client for backend API."""

from __future__ import annotations

import httpx
import streamlit as st

from ..auth import TOKEN_KEY, get_api_base


def get_client() -> httpx.Client | None:
    """
    Return an httpx client with Bearer token from session.
    Returns None if not authenticated. Caller should use for a single request and not hold.
    """
    token = st.session_state.get(TOKEN_KEY)
    if not token:
        return None
    return httpx.Client(
        base_url=get_api_base(),
        timeout=30.0,
        headers={"Authorization": f"Bearer {token}"},
    )


def post(path: str, **kwargs) -> httpx.Response | None:
    """POST to API. Returns response or None if not authenticated."""
    client = get_client()
    if client is None:
        return None
    with client:
        return client.post(path, **kwargs)


def get(path: str, **kwargs) -> httpx.Response | None:
    """GET from API. Returns response or None if not authenticated."""
    client = get_client()
    if client is None:
        return None
    with client:
        return client.get(path, **kwargs)


def put(path: str, **kwargs) -> httpx.Response | None:
    """PUT to API. Returns response or None if not authenticated."""
    client = get_client()
    if client is None:
        return None
    with client:
        return client.put(path, **kwargs)


def delete(path: str, **kwargs) -> httpx.Response | None:
    """DELETE to API. Returns response or None if not authenticated."""
    client = get_client()
    if client is None:
        return None
    with client:
        return client.delete(path, **kwargs)
