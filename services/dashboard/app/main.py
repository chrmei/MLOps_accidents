"""Streamlit dashboard entry: auth check, then Control Center or login."""
import streamlit as st

from app.utils.session import ensure_authenticated
from app.components.sidebar import PAGE_KEY, render_sidebar
from app.views.login import render_login
from app.views.user_prediction import render as render_prediction
from app.views.admin_data_ops import render as render_data_ops
from app.views.admin_ml_ops import render as render_ml_ops
from app.views.admin_user_mgmt import render as render_user_mgmt
from app.auth import USER_KEY, is_admin

st.set_page_config(
    page_title="Accident Severity Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not ensure_authenticated():
    render_login()
else:
    render_sidebar()
    page = st.session_state.get(PAGE_KEY, "prediction")
    user = st.session_state.get(USER_KEY, {})
    role = user.get("role", "")
    if page == "data_ops" and is_admin(role):
        render_data_ops()
    elif page == "ml_ops" and is_admin(role):
        render_ml_ops()
    elif page == "user_mgmt" and is_admin(role):
        render_user_mgmt()
    else:
        st.session_state[PAGE_KEY] = "prediction"
        render_prediction()
