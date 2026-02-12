"""Introduction & Problem Statement - Presentation Page (Rafael)."""

import streamlit as st


def render():
    """Render Introduction & Problem Statement page."""
    st.title("Introduction & Problem Statement")
    st.markdown("**Duration: 2-3 minutes**")
    st.markdown("---")

    st.header("Problem")
    st.markdown(
        """
        Predicting road accident severity using French road accident data.
        """
    )

    st.header("Objective")
    st.markdown(
        """
        Build an MLOps system for reproducible, scalable accident prediction.
        """
    )

    st.header("Key Requirements")
    st.markdown(
        """
        - Reproducible ML pipeline
        - Model versioning and tracking
        - Production-ready API serving
        - Monitoring and observability
        """
    )
