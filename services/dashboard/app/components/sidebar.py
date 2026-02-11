"""Sidebar navigation and logout."""

import streamlit as st

from ..auth import USER_KEY, is_admin
from ..utils.session import clear_session
from ..config import BROWSER_BASE_URL

PAGE_KEY = "dashboard_page"


def render_sidebar():
    """Render role-based sidebar: Prediction for all; Admin gets Data Ops, ML Ops, User Management."""
    user = st.session_state.get(USER_KEY)
    if not user:
        return
    role = user.get("role", "")
    st.sidebar.markdown(f"**{user.get('username', '')}** ({role})")
    st.sidebar.markdown("---")
    if st.sidebar.button(
        "Control Center (Prediction)", key="nav_pred", use_container_width=True
    ):
        st.session_state[PAGE_KEY] = "prediction"
        st.rerun()
    if is_admin(role):
        if st.sidebar.button("Data Ops", key="nav_data", use_container_width=True):
            st.session_state[PAGE_KEY] = "data_ops"
            st.rerun()
        if st.sidebar.button("ML Ops", key="nav_ml", use_container_width=True):
            st.session_state[PAGE_KEY] = "ml_ops"
            st.rerun()
        st.sidebar.link_button(
            "Ml Flow",
            "https://dagshub.com/chrmei/MLOps_accidents.mlflow/#/experiments/0/runs?searchFilter=&orderByKey=metrics.%60f1_score%60&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D",
            use_container_width=True,
        )
        if st.sidebar.button(
            "User Management", key="nav_users", use_container_width=True
        ):
            st.session_state[PAGE_KEY] = "user_mgmt"
            st.rerun()
        
        # Documentation menu item (Admin only) - using expander for dropdown effect
        st.sidebar.markdown("---")
        
        # Use BROWSER_BASE_URL for browser-accessible links (defaults to localhost)
        # This is separate from API_BASE_URL which uses internal Docker service names
        
        # Documentation dropdown using expander
        with st.sidebar.expander("Documentation", expanded=False):
            st.link_button("Docs", f"{BROWSER_BASE_URL}/docs", use_container_width=True)
            st.link_button("OpenAPI", f"{BROWSER_BASE_URL}/openapi.json", use_container_width=True)
            st.link_button("redoc", f"{BROWSER_BASE_URL}/redoc", use_container_width=True)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", type="secondary"):
        clear_session()
        st.rerun()
