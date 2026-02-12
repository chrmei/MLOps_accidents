"""DevOps & Automation - Presentation Page (Surya)."""

import streamlit as st


def render():
    """Render DevOps & Automation page."""
    st.title("DevOps & Automation")
    st.markdown("**Duration: 1-2 minutes**")
    st.markdown("---")

    st.header("CI/CD")
    st.markdown(
        """
        - GitHub Actions runs on push/PR to keep main stable
        - Linting and formatting checks prevent style drift
        - Unit and API tests gate merges before deployment
        - Build steps validate Docker images and dependency lock
        """
    )

    st.header("Testing")
    st.markdown(
        """
        - Pytest covers auth, data, train, and predict endpoints
        - Health and readiness endpoints are exercised for uptime
        - JWT flow is validated (login, refresh, protected routes)
        - Negative cases verify rate limits and auth failures
        - Dedicated test container runs the suite in isolation
        """
    )

    st.header("Containerization")
    st.markdown(
        """
        - Multi-stage Dockerfiles keep runtime images small
        - Separate targets for dev, training, and production serving
        - Docker Compose wires services, networks, and volumes locally
        - Environment-driven config makes parity with CI and prod
        """
    )

    st.header("Ops Automation")
    st.markdown(
        """
        - Makefile wraps common workflows (build, up, test, clean)
        - One-command setup reduces onboarding time and errors
        - K3s manifests mirror Compose services for cluster deploys
        - Declarative YAML supports repeatable, versioned rollouts
        """
    )
