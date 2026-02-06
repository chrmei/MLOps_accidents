"""Minimal header component (empty - Streamlit's native menu handles UI controls)."""
import streamlit as st

from ..auth import USER_KEY


def render_header():
    """Render minimal header (empty - Streamlit's native three-dots menu is used instead)."""
    # Header is intentionally empty - Streamlit's native menu (top right three dots)
    # provides Rerun, Settings, Print, Record screencast, Clear Cache
    # No custom menu items needed
    pass
