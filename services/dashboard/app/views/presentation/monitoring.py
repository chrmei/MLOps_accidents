"""Monitoring & Observability - Presentation Page (Rafael)."""

import streamlit as st


def render():
    """Render Monitoring & Observability page."""
    st.title("Monitoring & Observability")
    st.markdown("**Duration: 2-3 minutes** | **Presenter: Rafael**")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.header("Prometheus")
    with col2:
        st.header("Grafana")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("API metrics")
        st.markdown(
            """
            - API request total
            - API request duration
            """
        )
    with col2:
        st.image("/app/app/assets/images/grafana_api_dashboard.png", caption="Example API Dashboard in Grafana")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model and Data Quality Metrics")
        st.markdown(
            """
            - Model accuracy
            - Model precision
            - Model recall
            - Model F1 score
            - Column drift share (from Evidently)
            """
        )
    with col2:
        st.image("/app/app/assets/images/grafana_model_dashboard.png", caption="Example Model Dashboard in Grafana")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Resource Metrics")
        st.markdown(
            """ 
            - Exposed by Node Exporter
            """
        )
    with col2:
        st.image("/app/app/assets/images/grafana_resource_dashboard.png", caption="Example Resource Dashboard in Grafana")


    col1, col2 = st.columns(2)
    with col1:
        st.header("Alerts")
        st.subheader("🚨 **Service Health** Alerts")
        st.markdown(
            """
            - Predict Service Down
            """
        )
        st.subheader("🔍 **Model and Data Quality** Alerts")
        st.markdown(
            """
            - High Column Drift Share
            """
        )
    with col2:
        st.header("Health Checks")
        st.markdown(
            """
            All services expose `/health` endpoints.
            """
        )
