"""Introduction & Problem Statement - Presentation Page (Rafael)."""

import streamlit as st


def render():
    """Render Introduction & Problem Statement page."""
    st.title("Introduction & Overview")
    st.markdown("**Duration: 2-3 minutes** | **Presenter: Rafael**")
    st.markdown("---")



    st.header("Problem Statement")
    st.markdown(
        """
        - Predicting road accident severity using French road accident data
        - **Data** provided by the [French government](https://www.data.gouv.fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2024),
        also available in [Kaggle](https://www.kaggle.com/datasets/ahmedlahlou/accidents-in-france-from-2005-to-2016)
        - Consists of 4 CSV files:
            - **Victim** data (security, position in vehicule, ...)
            - **Accident** data (time, intersection, coordinates, ...)
            - **Location** data (road type, surface, ...)
            - **Vehicule** data (type, maneuver, ...)
        - **Target** derived from severity levels given for each victim:
            - 0: Victims only slightly injured
            - 1: At least one victim seriously injured or dead
        """
    )

    st.header("Objective")
    st.markdown(
        """
        Build a **reproducible**, **scalable**, and **maintainable** ML product for accident severity prediction
        """
    )

    st.subheader("Key Requirements")
    st.markdown(
        """
        - Reproducible ML pipeline
        - Model and Data versioning and tracking
        - Production-ready API serving
        - Monitoring and observability
        """
    )

    st.header("Overview")
    st.markdown(
        """
        - System Architecture (Surya)
        - Data Science & ML Pipeline (Christian)
        - Frontend & User Interface (Christian)
        - Monitoring & Observability (Rafael)
        - DevOps & Automation (Surya)
        - Conclusion & Future Work (Rafael)
        """
    )
