"""System Architecture - Presentation Page (Surya)."""

import streamlit as st


def render():
    """Render System Architecture page."""
    st.title("System Architecture")
    st.markdown("**Duration: 1-2 minutes**")
    st.markdown("---")

    st.header("Microservices Catalog")
    st.markdown(
        """
        **Core ML services**
        - **Data Service** (8001): Loads raw datasets, cleans and encodes features, persists
            interim datasets, and exposes async preprocessing jobs with status tracking.
        - **Train Service** (8002): Trains models using processed data, logs metrics and artifacts
            to MLflow, and writes trained models to the shared models volume.
        - **Predict Service** (8003): Low-latency inference API that loads the latest model,
            validates inputs, and emits Prometheus metrics (latency, errors, throughput).

        **Security and access**
        - **Auth Service** (8004): Issues JWTs, enforces RBAC for admin/user routes,
            manages user lifecycle, and rate-limits login/refresh attempts.

        **External data enrichment**
        - **Geocode Service** (8005): Converts address to coordinates with caching and
            provider throttling.
        - **Weather Service** (8006): Fetches weather features for a location/time window
            and returns normalized inputs for prediction.

        **User interface and docs**
        - **Dashboard** (8501): Streamlit control center for predictions, admin ops,
            and the presentation itself.
        - **Docs Service** (8010): Aggregates OpenAPI specs across services into one UI.

        **Simulation and QA**
        - **Sim Traffic Service**: Generates synthetic traffic scenarios for what-if analyses.
        - **Sim Eval Service**: Evaluates scenarios against models and produces reports.
        - **Test Service**: Runs automated API tests in an isolated container.
        """
    )

    st.header("Service Flow")
    st.markdown(
        """
        - **Dashboard/UI** -> **Nginx** -> **Auth** (login, JWT issuance)
        - **Predict** serves inference; enriches requests via **Geocode/Weather**
        - **Data** and **Train** run long jobs asynchronously and log status
        - **Docs** aggregates OpenAPI specs for quick API discovery
        - **Test/Sim** services validate APIs and simulation scenarios
        """
    )

    st.header("Compact Diagram")
    st.markdown(
        """
        ```text
        User
            |
            v
        Dashboard (8501)
            |
            v
        Nginx / API Gateway (80)
            |----> Auth (8004) ----> JWT
            |----> Predict (8003) --+--> Geocode (8005)
            |                       +--> Weather (8006)
            |----> Data (8001) ----> Job Store
            |----> Train (8002) ---> MLflow
            |----> Docs (8010)
            |----> Test / Sim Services
        ```
        """
    )

    st.header("Infrastructure")
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
