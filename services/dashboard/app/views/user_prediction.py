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

    if st.button("Predict", type="primary"):
        resp = post("/api/v1/predict/", json={"features": features})
        if resp is None:
            st.error("Not authenticated.")
            return
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", "Prediction failed")
            except Exception:
                detail = "Prediction failed"
            st.error(detail)
            return
        data = resp.json()
        pred = data.get("prediction")
        prob = data.get("probability", 0.0)
        model_type = data.get("model_type", "")
        severity_label = "Severe (1)" if pred == 1 else "Not severe (0)"
        st.success(
            f"**Severity:** {severity_label}  |  **Risk probability:** {prob:.2%}  |  Model: {model_type}"
        )
        if "lat" in features and "long" in features:
            m = folium.Map(
                location=[float(features["lat"]), float(features["long"])],
                zoom_start=12,
            )
            folium.Marker(
                [float(features["lat"]), float(features["long"])],
                tooltip="Incident",
            ).add_to(m)
            st_folium(m, width=700, height=400)
