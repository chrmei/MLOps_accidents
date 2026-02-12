"""Conclusion & Future Work - Presentation Page (Rafael)."""

import streamlit as st

def green_tag(text):
    return f'<span style="background-color:#066206; padding:0px 2px; border-radius:4px;"><b>{text}</b></span>'

def yellow_tag(text):
    return f'<span style="background-color:#8e7800; padding:0px 2px; border-radius:4px;"><b>{text}</b></span>'

def render():
    """Render Conclusion & Future Work page."""
    st.title("Conclusion & Future Work")
    st.markdown("**Duration: 1-2 minutes** | **Presenter: Rafael**")
    st.markdown("---")

    st.header("Achievements")
    st.markdown(f"""
        - Complete {green_tag("microservices")} architecture
        - Production-ready {green_tag("ML pipeline")} with {green_tag("versioning")}
        - {green_tag("User Interface")} with {green_tag("Role Based Access")}
        - {green_tag("Monitoring")} and {green_tag("Alerts")}
        - DevOps practices with {green_tag("CI/CD")} pipelines
        """, unsafe_allow_html=True)

    st.header("Future Enhancements")
    st.markdown(f"""
        - {yellow_tag("K3s")}/{yellow_tag("Kubernetes")} deployment for better scalability and management
        - {yellow_tag("Evidently AI")} integration for data and model drift reporting
        - Enhanced evaluation and alerting procedure
        - Enhanced CI/CD pipelines
        - Production optimizations like {yellow_tag("scaling")} and {yellow_tag("caching")}
        """, unsafe_allow_html=True)
