"""Streamlit dashboard entry: auth check, then Control Center or login."""
import streamlit as st

from app.utils.session import ensure_authenticated
from app.components.sidebar import PAGE_KEY, render_sidebar
from app.components.header import render_header
from app.views.login import render_login
from app.views.forgot_password import render as render_forgot_password
from app.views.user_prediction import render as render_prediction
from app.views.admin_data_ops import render as render_data_ops
from app.views.admin_ml_ops import render as render_ml_ops
from app.views.admin_user_mgmt import render as render_user_mgmt
from app.views.presentation.introduction import render as render_presentation_intro
from app.views.presentation.architecture import render as render_presentation_architecture
from app.views.presentation.ml_pipeline import render as render_presentation_ml_pipeline
from app.views.presentation.frontend import render as render_presentation_frontend
from app.views.presentation.monitoring import render as render_presentation_monitoring
from app.views.presentation.devops import render as render_presentation_devops
from app.views.presentation.conclusion import render as render_presentation_conclusion
from app.auth import USER_KEY, is_admin

st.set_page_config(
    page_title="Accident Severity Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide "Show password" eye icon globally so passwords are never revealed
st.markdown(
    """
    <style>
    [data-testid="stTextInput"] button[title="Show password text"],
    [data-testid="stTextInput"] button[aria-label="Show password text"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not ensure_authenticated():
    if st.session_state.get("show_forgot_password"):
        render_forgot_password()
    else:
        render_login()
else:
    render_sidebar()
    render_header()
    page = st.session_state.get(PAGE_KEY, "prediction")
    user = st.session_state.get(USER_KEY, {})
    role = user.get("role", "")
    
    # Presentation pages (accessible to all authenticated users)
    if page == "presentation_intro":
        render_presentation_intro()
    elif page == "presentation_architecture":
        render_presentation_architecture()
    elif page == "presentation_ml_pipeline":
        render_presentation_ml_pipeline()
    elif page == "presentation_frontend":
        render_presentation_frontend()
    elif page == "presentation_monitoring":
        render_presentation_monitoring()
    elif page == "presentation_devops":
        render_presentation_devops()
    elif page == "presentation_conclusion":
        render_presentation_conclusion()
    # Admin pages
    elif page == "data_ops" and is_admin(role):
        render_data_ops()
    elif page == "ml_ops" and is_admin(role):
        render_ml_ops()
    elif page == "user_mgmt" and is_admin(role):
        render_user_mgmt()
    else:
        st.session_state[PAGE_KEY] = "prediction"
        render_prediction()
