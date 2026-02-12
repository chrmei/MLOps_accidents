"""Monitoring & Observability - Presentation Page (Rafael)."""

import streamlit as st


def render():
    """Render Monitoring & Observability page."""
    st.title("Monitoring & Observability")
    st.markdown("**Duration: 2-3 minutes**")
    st.markdown("---")

    st.header("Prometheus (port 9090)")
    st.markdown(
        """
        - Metrics collection from Predict Service
        - Node exporter for system metrics
        - Custom metrics: prediction latency, request count, error rate
        """
    )

    st.header("Grafana (port 3000)")
    st.markdown(
        """
        - **API Dashboard**: Request rate, latency, error rate
        - **Model Dashboard**: Model performance metrics over time
        - **Resource Dashboard**: CPU, RAM, disk usage
        - **Alerting**: Configured alert rules for high error rates and latency
        """
    )

    st.header("Health Checks")
    st.markdown(
        """
        All services expose `/health` endpoints.
        """
    )
