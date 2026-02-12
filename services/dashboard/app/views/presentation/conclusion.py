"""Conclusion & Future Work - Presentation Page (Rafael)."""

import streamlit as st


def render():
    """Render Conclusion & Future Work page."""
    st.title("Conclusion & Future Work")
    st.markdown("**Duration: 1-2 minutes** | **Presenter: Rafael**")
    st.markdown("---")

    st.header("Achievements")
    st.markdown(
        """
        - Production-ready ML pipeline with versioning
        - Complete microservices architecture
        - Monitoring and observability
        - User Interface with Role Based Access
        """
    )

    st.header("Future Enhancements")
    st.markdown(
        """
        - K3s/Kubernetes deployment for better scalability and management
        - Evidently AI integration for data and model drift reporting
        - Enhanced evaluation and alerting procedure
        - Enhanced CI/CD pipelines
        - Production optimizations (scaling, caching)
        """
    )
