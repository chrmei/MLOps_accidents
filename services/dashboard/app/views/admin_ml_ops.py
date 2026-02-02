"""Admin ML Ops: trigger training, view and edit config."""
import streamlit as st
import json

from ..components.api_client import get, post, put


def render():
    st.title("ML Ops")
    st.markdown("Trigger training; view and edit config.")

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
    st.subheader("Training jobs")

    @st.fragment(run_every=4)
    def training_jobs_section():
        st.markdown(
            """
            <style>
            div[data-testid="stExpander"] {
                border: none !important;
                box-shadow: none !important;
            }
            div[data-testid="stExpander"] > div {
                border: none !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        resp = get("/api/v1/train/jobs")
        if resp is None:
            st.warning("Not authenticated.")
        elif resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text or "Failed to load jobs.")
                st.error(detail)
            except Exception:
                st.error("Failed to load training jobs.")
        else:
            jobs = resp.json()
            if not jobs:
                st.info("No training jobs yet. Use **Start Training** above to run one.")
            else:
                for j in jobs[:20]:
                    msg = j.get("message") or ""
                    prog = j.get("progress")
                    prog_str = f" · {prog:.0f}%" if prog is not None else ""
                    line_text = (
                        f"{j.get('job_type', '')} | {j.get('status', '')} | "
                        f"{j.get('created_at', '')} | {j.get('job_id', '')}"
                        + (f" | {msg}{prog_str}" if msg or prog_str else "")
                    )
                    with st.expander(line_text, expanded=False):
                        if j.get("error"):
                            st.caption(f"**Error:** {j['error']}")
                        logs = j.get("logs") or []
                        if logs:
                            st.text("\n".join(logs))
                        result = j.get("result")
                        if result:
                            st.json(result)

    training_jobs_section()
