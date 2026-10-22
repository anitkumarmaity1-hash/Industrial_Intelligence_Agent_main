"""Industrial Intelligence Agent - dashboard entry point.

A thin client of the FastAPI backend: it signs in with POST /auth/login, keeps the JWT in
st.session_state, and calls the API only (never the database). Which tenant you see is
decided by the token, never by anything chosen here. Viewers get no write pages; the API
refuses their writes regardless.

Run:  streamlit run frontend/streamlit_app.py     (API_BASE_URL defaults to http://localhost:8000)
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = str(Path(__file__).resolve().parent)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import streamlit as st  # noqa: E402

import api_client as api  # noqa: E402
import navigation  # noqa: E402
from views import admin, fleet, investigate, jobs, login, upload  # noqa: E402

st.set_page_config(page_title="Industrial Intelligence Agent", page_icon="🏭", layout="wide")


def _pages() -> list:
    renderers = {"fleet": fleet.render, "investigate": investigate.render, "upload": upload.render,
                 "jobs": jobs.render, "admin": admin.render}
    return [st.Page(renderers[mod], title=title, icon=icon, url_path=path, default=(mod == "fleet"))
            for mod, title, icon, path, _admin in navigation.specs_for(st.session_state.get("role"))]


if not api.token():
    st.navigation([st.Page(login.render, title="Sign in", url_path="login")], position="hidden").run()
else:
    with st.sidebar:
        st.markdown(f"**{st.session_state.get('email')}**  \n{st.session_state.get('tenant')} · {st.session_state.get('role')}")
        if st.button("Refresh data"):
            api._cached_get.clear()
            st.rerun()
        if st.button("Sign out"):
            api.sign_out()
            st.rerun()
    st.navigation(_pages()).run()
