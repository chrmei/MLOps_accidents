"""System Architecture - Presentation Page (Surya)."""

import streamlit as st


def render():
    """Render System Architecture page."""
    st.title("System Architecture")
    st.markdown("**Duration: 4-5 minutes**")
    st.markdown("---")

    st.header("Microservices Architecture")
    st.markdown(
        """
        - **Auth Service** (port 8004): JWT authentication, user management, role-based access control
        - **Data Service** (port 8001): Data preprocessing and feature engineering (async jobs)
        - **Train Service** (port 8002): Model training with MLflow integration (async jobs)
        - **Predict Service** (port 8003): Real-time inference API with Prometheus metrics
        - **Geocode Service** (port 8005): Address to coordinates conversion
        - **Weather Service** (port 8006): Weather data integration
        - **Docs Service** (port 8010): Unified OpenAPI documentation aggregator
        """
    )

    st.header("Infrastructure Components")
    st.markdown(
        """
        - **Nginx**: Reverse proxy/API Gateway (port 80) with rate limiting and JWT forwarding
        - **PostgreSQL**: Persistent storage for users and job logs
        - **Docker Compose**: Orchestration of all services
        - **K3s/Kubernetes**: Deployment option (deploy/k3s/ directory exists)
        """
    )

    st.header("Design Patterns")
    st.markdown(
        """
        - Async job pattern for long-running operations (preprocessing, training)
        - Shared volumes for data/models
        - Service mesh communication via Docker network
        """
    )
