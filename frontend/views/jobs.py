"""Jobs: status of every upload job, the rejected-row report, and retry (admins)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import api_client as api

PAGE_SIZE = 20
STATUS_ICON = {"queued": "🕒", "running": "⚙️", "succeeded": "✅", "failed": "❌"}


def _job_table() -> list[dict]:
    c1, c2 = st.columns(2)
    status = c1.selectbox("Status", ["all", "queued", "running", "succeeded", "failed"])
    page = c2.number_input("Page", min_value=1, value=1, step=1)
    params = {"limit": PAGE_SIZE, "offset": (page - 1) * PAGE_SIZE}
    if status != "all":
        params["status"] = status
    jobs = api.get("/jobs", params, cached=False)
    if not jobs:
        st.info("No jobs yet." if status == "all" and page == 1 else "No jobs match.")
        return []
    df = pd.DataFrame(jobs)
    df["status"] = df["status"].map(lambda s: f"{STATUS_ICON.get(s, '')} {s}")
    st.dataframe(df[["job_id", "status", "stage", "rows_in", "rows_loaded", "rows_rejected",
                     "retry_count", "created_at", "finished_at", "error"]],
                 width="stretch", hide_index=True)
    return jobs


def _detail(job: dict) -> None:
    st.subheader(f"Job {job['job_id'][:8]}")
    st.write(f"**{STATUS_ICON.get(job['status'], '')} {job['status']}**"
             f"{' - stage ' + job['stage'] if job.get('stage') else ''}; "
             f"{job['rows_loaded']} loaded, {job['rows_rejected']} rejected of {job['rows_in']} rows in.")
    if job.get("error"):
        st.error(job["error"])
    if api.is_admin() and job["status"] == "failed":
        if st.button("Retry this job", type="primary"):
            try:
                api.post(f"/jobs/{job['job_id']}/retry")
                st.success("Queued again.")
            except api.ApiError as exc:
                st.error(str(exc))
    if job["rows_rejected"] == 0 and job["status"] != "failed":
        return
    st.markdown("**Rejected rows**")
    page = st.number_input("Rejected-rows page", min_value=1, value=1, step=1)
    try:
        rows = api.get(f"/jobs/{job['job_id']}/rejected-rows", {"limit": 100, "offset": (page - 1) * 100})
    except api.ApiError as exc:
        st.error(str(exc))
        return
    if not rows:
        st.caption("No rejected rows recorded.")
        return
    df = pd.DataFrame(rows)
    df["raw"] = df["raw"].map(lambda r: "" if r is None else str(r))
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("Download this page (CSV)", df.to_csv(index=False), f"rejected_{job['job_id'][:8]}.csv")


@st.fragment(run_every="5s")
def _live() -> None:
    try:
        jobs = _job_table()
    except api.ApiError as exc:
        st.error(str(exc))
        return
    st.session_state["jobs_seen"] = {j["job_id"]: j for j in jobs}


def render() -> None:
    st.title("Jobs")
    st.caption("Refreshes every 5 seconds.")
    _live()
    seen = st.session_state.get("jobs_seen", {})
    if seen:
        st.divider()
        pick = st.selectbox("Inspect a job", list(seen), format_func=lambda j: f"{j[:8]} - {seen[j]['status']}")
        _detail(seen[pick])
