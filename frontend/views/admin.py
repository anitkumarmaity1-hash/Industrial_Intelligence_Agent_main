"""Admin: sensor registry editor, connectors, users. Admin role only (the API enforces it too)."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import api_client as api


def _call(label: str, fn, rerun: bool = False) -> bool:
    """Run an API write. On success show `label` (kept across a rerun when the page must refresh)."""
    try:
        fn()
    except api.ApiError as exc:
        st.error(str(exc))
        return False
    if rerun:
        st.session_state["flash"] = label
        st.rerun()
    st.success(label)
    return True


# ------------------------------------------------------------------ sensors

def _sensors() -> None:
    try:
        sensors = api.get("/sensors")
    except api.ApiError as exc:
        st.error(str(exc))
        return
    if not sensors:
        st.info("No sensors declared yet - they are created by an upload's mapping.")
        return
    st.dataframe(pd.DataFrame(sensors), width="stretch", hide_index=True)
    st.caption("Edits apply to detection runs after the change; anomalies already stored are not recomputed.")
    spec = st.selectbox("Sensor to edit", sensors, format_func=lambda s: s["sensor_name"], key="reg_pick")
    name = spec["sensor_name"]
    c1, c2 = st.columns(2)
    use_min = c1.checkbox("Has a normal minimum", spec["normal_min"] is not None, key=f"rmin_on_{name}")
    lo = c1.number_input("Normal min", value=float(spec["normal_min"] or 0.0), disabled=not use_min, key=f"rmin_{name}")
    use_max = c2.checkbox("Has a normal maximum", spec["normal_max"] is not None, key=f"rmax_on_{name}")
    hi = c2.number_input("Normal max", value=float(spec["normal_max"] or 0.0), disabled=not use_max, key=f"rmax_{name}")
    use_z = st.checkbox("Override the z-score threshold", spec["z_score_threshold"] is not None, key=f"rz_on_{name}")
    z = st.number_input("z-score threshold", min_value=0.1, max_value=20.0,
                        value=float(spec["z_score_threshold"] or 3.0), disabled=not use_z, key=f"rz_{name}")
    enabled = st.checkbox("Enabled", spec["enabled"], key=f"ren_{name}")
    if st.button("Save sensor", type="primary"):
        payload = {"normal_min": lo if use_min else None, "normal_max": hi if use_max else None,
                   "z_score_threshold": z if use_z else None, "enabled": enabled}
        _call(f"Saved {name}.", lambda: api.post(f"/sensors/{name}", payload), rerun=True)


# --------------------------------------------------------------- connectors

def _connector_actions(c: dict) -> None:
    cid = c["connector_id"]
    a, b, d, e = st.columns(4)
    if a.button("Disable" if c["enabled"] else "Enable", key=f"tog_{cid}"):
        _call("Updated.", lambda: api.post(f"/connectors/{cid}/{'disable' if c['enabled'] else 'enable'}"), rerun=True)
    if b.button("Run now", key=f"run_{cid}"):
        _call("Run requested; the worker picks it up on its next tick.", lambda: api.post(f"/connectors/{cid}/run"))
    if d.button("Test access", key=f"test_{cid}"):
        try:
            out = api.post(f"/connectors/{cid}/test")
            (st.success if out["ok"] else st.error)(out["message"])
        except api.ApiError as exc:
            st.error(str(exc))
    sure = e.checkbox("Confirm delete", key=f"sure_{cid}")
    if e.button("Delete", key=f"del_{cid}", disabled=not sure):
        _call("Deleted.", lambda: api.post(f"/connectors/{cid}/delete"), rerun=True)
    try:
        runs = api.get(f"/connectors/{cid}/runs", {"limit": 10})
    except api.ApiError as exc:
        st.error(str(exc))
        return
    if runs:
        st.dataframe(pd.DataFrame(runs)[["started_at", "trigger", "status", "objects_seen", "objects_queued",
                                         "objects_failed", "message"]], width="stretch", hide_index=True)
    else:
        st.caption("No runs yet.")


def _connectors() -> None:
    try:
        conns = api.get("/connectors")
    except api.ApiError as exc:
        st.error(str(exc))
        return
    if not conns:
        st.info("No connectors yet.")
    for c in conns:
        with st.expander(f"{c['name']} - gs://{c['bucket']}/{c['prefix']} - {c['schedule']} - "
                         f"{'enabled' if c['enabled'] else 'disabled'} - last: {c['last_status']}"):
            if c.get("last_message"):
                st.caption(c["last_message"])
            _connector_actions(c)
    st.markdown("**New GCS connector**")
    try:
        uploads = [u for u in api.get("/uploads", {"limit": 50}) if u["has_mapping"]]
    except api.ApiError:
        uploads = []
    with st.form("new_connector"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Name")
        bucket = c2.text_input("Bucket")
        c3, c4, c5 = st.columns(3)
        prefix = c3.text_input("Prefix", "")
        pattern = c4.text_input("File pattern", "*.csv")
        schedule = c5.text_input("Schedule (cron, UTC)", "*/15 * * * *")
        secret_ref = st.text_input("Secret name (blank only for the demo emulator)")
        source = st.selectbox("Mapping from", ["(paste JSON)"] + [u["upload_id"] for u in uploads],
                              format_func=lambda x: x if x == "(paste JSON)" else
                              next(f"{u['filename']} ({x[:8]})" for u in uploads if u["upload_id"] == x))
        mapping_json = st.text_area("Mapping JSON (when pasting)", height=120)
        go = st.form_submit_button("Create connector", type="primary")
    if not go:
        return
    payload: dict = {"name": name, "bucket": bucket, "prefix": prefix, "pattern": pattern, "schedule": schedule,
                     "secret_ref": secret_ref.strip() or None}
    if source == "(paste JSON)":
        try:
            payload["mapping"] = json.loads(mapping_json)
        except ValueError:
            st.error("The mapping is not valid JSON.")
            return
    else:
        payload["mapping_from_upload_id"] = source
    _call("Connector created.", lambda: api.post("/connectors", payload), rerun=True)


# -------------------------------------------------------------------- users

def _users() -> None:
    try:
        users = api.get("/users")
    except api.ApiError as exc:
        st.error(str(exc))
        return
    st.dataframe(pd.DataFrame(users)[["email", "role", "is_active", "last_login_at", "created_at"]],
                 width="stretch", hide_index=True)
    u = st.selectbox("User", users, format_func=lambda x: f"{x['email']} ({x['role']}{'' if x['is_active'] else ', inactive'})")
    c1, c2 = st.columns(2)
    new_role = c1.selectbox("Role", ["viewer", "admin"], index=["viewer", "admin"].index(u["role"]), key="u_role")
    if c1.button("Change role") and new_role != u["role"]:
        _call("Role changed.", lambda: api.post(f"/users/{u['id']}/role", {"role": new_role}), rerun=True)
    if c2.button("Deactivate" if u["is_active"] else "Reactivate"):
        _call("Updated.", lambda: api.post(f"/users/{u['id']}/active", {"active": not u["is_active"]}), rerun=True)
    st.markdown("**New user**")
    with st.form("new_user", clear_on_submit=True):
        email = st.text_input("Email")
        password = st.text_input("Initial password", type="password")
        role = st.selectbox("Role", ["viewer", "admin"], key="nu_role")
        go = st.form_submit_button("Create user", type="primary")
    if go:
        _call("User created.", lambda: api.post("/users", {"email": email, "password": password, "role": role}), rerun=True)


def render() -> None:
    st.title("Admin")
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))
    if not api.is_admin():
        st.warning("Only admins can open this page.")
        return
    t_sensors, t_conn, t_users = st.tabs(["Sensor registry", "Connectors", "Users"])
    with t_sensors:
        _sensors()
    with t_conn:
        _connectors()
    with t_users:
        _users()
