"""Conclusion & Future Work - Presentation Page (Rafael)."""

import streamlit as st


def render():
    """Render Conclusion & Future Work page."""
    st.title("Conclusion & Future Work")
    st.markdown("**Duration: 1-2 minutes**")
    st.markdown("---")

    st.header("Achievements")
    st.markdown(
        """
        - Complete microservices architecture
        - Production-ready ML pipeline with versioning
        - Monitoring and observability stack
        - User-facing dashboard
        """
    )

    st.header("Future Enhancements")
    st.markdown(
        """
        - K3s/Kubernetes deployment (already exists in deploy/k3s/)
        - Evidently AI for drift detection
        - Enhanced CI/CD pipelines
        - Production optimizations (scaling, caching)
        """
    )
