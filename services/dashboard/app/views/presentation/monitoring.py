"""Monitoring & Observability - Presentation Page (Rafael)."""

import streamlit as st


def render():
    """Render Monitoring & Observability page."""
    st.title("Monitoring & Observability")
    st.markdown("**Duration: 2-3 minutes** | **Presenter: Rafael**")
    st.markdown("---")

    st.header("Prometheus")
    st.markdown(
        """
        - Metrics collection from Predict Service:
            - General metrics:
                - API request total
                - API request duration
            - On evaluation endpoint:
                - Model accuracy
                - Model precision
                - Model recall
                - Model F1 score
                - Column drift share (from Evidently)
        - Node exporter for system metrics
        """
    )

    st.header("Grafana")

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Dashboards")
        st.image("/app/app/assets/images/grafana_api_dashboard.png", caption="Example API Dashboard in Grafana")
        st.image("/app/app/assets/images/grafana_model_dashboard.png", caption="Example Model Dashboard in Grafana")
        st.image("/app/app/assets/images/grafana_resource_dashboard.png", caption="Example Resource Dashboard in Grafana")

    with col2:
        st.subheader("Alerts")
        st.markdown(
            """
            - **Service Health** Alerts:
                - Predict Service Down
            - **Model and Data Quality** Alerts:
                - High Column Drift Share
            """
        )

    st.header("Health Checks")
    st.markdown(
        """
        All services expose `/health` endpoints.
        """
    )
