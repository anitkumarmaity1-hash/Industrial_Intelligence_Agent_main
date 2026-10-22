"""Agent investigations: ask a question, see findings and citations, browse history."""

from __future__ import annotations

import streamlit as st

import api_client as api
from ui import RISK_BADGE, render_result

INVESTIGATE_TIMEOUT_SECONDS = 120
HISTORY_PAGE = 10


def _ask() -> None:
    st.subheader("Ask the agent")
    with st.form("ask"):
        question = st.text_area("Operations question", height=90, max_chars=2000,
                                placeholder='e.g. "Why is PUMP-A101 running hot?" or '
                                            '"Which machines currently show abnormal behavior?"')
        machine = st.text_input("Machine id (optional)",
                                help="Leave empty for a fleet-wide question or when the question names the machine.")
        go = st.form_submit_button("Investigate", type="primary")
    if not go:
        return
    if not question.strip():
        st.warning("Enter a question first.")
        return
    payload = {"question": question.strip(), "machine_id": machine.strip() or None}
    with st.spinner("Running the agent - gathering evidence, assessing risk..."):
        try:
            st.session_state["latest_result"] = api.post(
                "/investigate", payload, timeout=INVESTIGATE_TIMEOUT_SECONDS)
        except api.ApiError as exc:
            st.session_state.pop("latest_result", None)
            st.error(str(exc))


def _history() -> None:
    st.subheader("History")
    c1, c2 = st.columns([2, 1])
    machine = c1.text_input("Filter by machine id", key="hist_machine")
    page = c2.number_input("Page", min_value=1, value=1, step=1, key="hist_page")
    params = {"limit": HISTORY_PAGE, "offset": (page - 1) * HISTORY_PAGE}
    if machine.strip():
        params["machine_id"] = machine.strip()
    try:
        rows = api.get("/investigations", params)
    except api.ApiError as exc:
        st.error(str(exc))
        return
    if not rows:
        st.info("No investigations yet." if page == 1 and not machine.strip() else "Nothing on this page.")
        return
    for row in rows:
        when = str(row["created_at"])[:16].replace("T", " ")
        label = (f"{when} - {row['created_by_label']} - {row['question'][:80]} - "
                 f"{RISK_BADGE.get(row['risk_level'], row['risk_level'] or '-')}")
        with st.expander(label):
            if st.button("Show result", key=f"show-{row['id']}"):
                st.session_state["open_investigation"] = row["id"]
            if st.session_state.get("open_investigation") == row["id"]:
                try:
                    render_result(api.get(f"/investigations/{row['id']}")["result"])
                except api.ApiError as exc:
                    st.error(str(exc))


def render() -> None:
    st.title("Agent investigations")
    _ask()
    if st.session_state.get("latest_result"):
        st.divider()
        st.markdown("**Latest result**")
        render_result(st.session_state["latest_result"])
    st.divider()
    _history()
