"""Admin ML Ops: trigger training, view config and metrics."""
import streamlit as st
import json

from ..components.api_client import get, post, put


def render():
    st.title("ML Ops")
    st.markdown("Trigger training; view and edit config; view metrics.")

    if st.button("Start Training", type="primary"):
        resp = post("/api/v1/train/", json={"compare": True})
        if resp is None:
            st.error("Not authenticated.")
        elif resp.status_code == 202:
            job = resp.json()
            st.success(f"Training job started: {job.get('job_id', '')}")
        else:
            try:
                st.error(resp.json().get("detail", "Failed"))
            except Exception:
                st.error("Failed")

    st.markdown("---")
    st.subheader("Training jobs")
    resp = get("/api/v1/train/jobs")
    if resp is None:
        st.warning("Not authenticated.")
    elif resp.status_code == 200:
        jobs = resp.json()
        if not jobs:
            st.info("No jobs yet.")
        else:
            for j in jobs[:20]:
                st.markdown(
                    f"**{j.get('job_id', '')}** | {j.get('status', '')} | {j.get('created_at', '')}"
                )
                if j.get("error"):
                    st.caption(f"Error: {j['error']}")

    st.markdown("---")
    st.subheader("Config")
    resp = get("/api/v1/train/config")
    if resp and resp.status_code == 200:
        config = resp.json()
        with st.expander("View / Edit config"):
            edited = st.text_area("YAML-like config (JSON)", value=json.dumps(config, indent=2), height=200)
            if st.button("Save config"):
                try:
                    body = json.loads(edited)
                    r = put("/api/v1/train/config", json=body)
                    if r and r.status_code == 200:
                        st.success("Config saved.")
                    else:
                        st.error("Save failed.")
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")
    else:
        st.warning("Could not load config.")

    st.markdown("---")
    st.subheader("Metrics")
    model_type = st.selectbox("Model type", ["lightgbm", "xgboost", "random_forest", "logistic_regression"])
    if st.button("Load metrics"):
        resp = get(f"/api/v1/train/metrics/{model_type}")
        if resp and resp.status_code == 200:
            data = resp.json()
            st.json(data.get("metrics", {}))
        else:
            st.info("No metrics for this model.")
