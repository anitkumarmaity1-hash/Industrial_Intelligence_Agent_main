"""Sign-in page."""

from __future__ import annotations

import streamlit as st

import api_client as api


def render() -> None:
    st.title("🏭 Industrial Intelligence Agent")
    if st.session_state.get("notice"):
        st.info(st.session_state["notice"])
    st.subheader("Sign in")
    with st.form("login"):
        tenant = st.text_input("Company (tenant id)", placeholder="acme")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        go = st.form_submit_button("Sign in", type="primary")
    if go:
        if not (tenant.strip() and email.strip() and password):
            st.warning("Enter the company, email and password.")
            return
        try:
            api.sign_in(tenant, email, password)
        except api.ApiError as exc:
            if exc.status == 401:
                st.error("Invalid credentials.")
            elif exc.status == 503:
                st.error("Login is unavailable (the service may not be configured yet).")
            else:
                st.error(str(exc))
            return
        st.rerun()
