"""Control Center: prediction form, map, result."""
from __future__ import annotations

import streamlit as st
from streamlit_folium import st_folium
import folium

from ..components.api_client import post
from ..components.prediction_form import render_prediction_form


def render():
    st.title("Control Center — Accident Severity")
    st.markdown("Enter incident details and get severity prediction.")

    features = render_prediction_form()

    # Initialize session state for prediction results if not exists
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "prediction_features" not in st.session_state:
        st.session_state.prediction_features = None

    if st.button("Predict", type="primary"):
        resp = post("/api/v1/predict/", json={"features": features})
        if resp is None:
            st.error("Not authenticated.")
            st.session_state.prediction_result = None
            st.session_state.prediction_features = None
            return
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", "Prediction failed")
            except Exception:
                detail = "Prediction failed"
            st.error(detail)
            st.session_state.prediction_result = None
            st.session_state.prediction_features = None
            return
        data = resp.json()
        pred = data.get("prediction")
        prob = data.get("probability", 0.0)
        model_type = data.get("model_type", "")
        severity_label = "Severe (1)" if pred == 1 else "Not severe (0)"
        # Store results in session state
        st.session_state.prediction_result = {
            "prediction": pred,
            "probability": prob,
            "model_type": model_type,
            "severity_label": severity_label,
        }
        st.session_state.prediction_features = features.copy()

    # Display results from session state (persists across reruns)
    if st.session_state.prediction_result is not None:
        result = st.session_state.prediction_result
        st.success(
            f"**Severity:** {result['severity_label']}  |  **Risk probability:** {result['probability']:.2%}  |  Model: {result['model_type']}"
        )
        if (
            st.session_state.prediction_features
            and "lat" in st.session_state.prediction_features
            and "long" in st.session_state.prediction_features
        ):
            m = folium.Map(
                location=[
                    float(st.session_state.prediction_features["lat"]),
                    float(st.session_state.prediction_features["long"]),
                ],
                zoom_start=12,
            )
            folium.Marker(
                [
                    float(st.session_state.prediction_features["lat"]),
                    float(st.session_state.prediction_features["long"]),
                ],
                tooltip="Incident",
            ).add_to(m)
            st_folium(m, width=700, height=400)
