"""Admin Data Ops: trigger pipelines, view job logs."""
import streamlit as st

from ..components.api_client import get, post


def render():
    st.title("Data Ops")
    st.markdown("Trigger preprocessing and feature engineering; view job logs.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run Preprocessing", type="primary"):
            resp = post("/api/v1/data/preprocess", json={})
            if resp is None:
                st.error("Not authenticated.")
            elif resp.status_code == 202:
                job = resp.json()
                st.success(f"Job started: {job.get('job_id', '')}")
            else:
                try:
                    st.error(resp.json().get("detail", "Failed"))
                except Exception:
                    st.error("Failed")
    with col2:
        if st.button("Run Feature Engineering", type="primary"):
            resp = post("/api/v1/data/build-features", json={})
            if resp is None:
                st.error("Not authenticated.")
            elif resp.status_code == 202:
                job = resp.json()
                st.success(f"Job started: {job.get('job_id', '')}")
            else:
                try:
                    st.error(resp.json().get("detail", "Failed"))
                except Exception:
                    st.error("Failed")

    st.markdown("---")
    st.subheader("Job logs")
    resp = get("/api/v1/data/jobs")
    if resp is None:
        st.warning("Not authenticated.")
    elif resp.status_code == 200:
        jobs = resp.json()
        if not jobs:
            st.info("No jobs yet.")
        else:
            for j in jobs[:20]:
                st.markdown(
                    f"**{j.get('job_id', '')}** | {j.get('job_type', '')} | "
                    f"{j.get('status', '')} | {j.get('created_at', '')}"
                )
                if j.get("error"):
                    st.caption(f"Error: {j['error']}")
    else:
        st.error("Failed to load jobs.")
