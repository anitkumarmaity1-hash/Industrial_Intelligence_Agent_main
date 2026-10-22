"""Upload + mapping wizard (admins): file -> preview -> map sensors and units -> validate -> submit.

Wide-format files (one column per sensor) are mapped with widgets; any mapping the API
accepts (including long format) can be pasted as JSON instead. The API owns all validation:
this page only builds the mapping, shows the dry-run report and submits.
"""

from __future__ import annotations

import json
import re

import pandas as pd
import streamlit as st

import api_client as api

STEPS = ["1. File", "2. Preview", "3. Map", "4. Validate & submit"]


def _wiz() -> dict:
    return st.session_state.setdefault("wizard", {"step": 0})


def _reset() -> None:
    st.session_state["wizard"] = {"step": 0}
    for key in [k for k in st.session_state if str(k).startswith("map_")]:
        del st.session_state[key]


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", str(name).lower()).strip("_")
    return (s if s[:1].isalpha() else f"s_{s}")[:64] or "sensor"


def _step_file(w: dict) -> None:
    st.markdown("Upload a CSV with one row per timestamp and machine, and one column per sensor.")
    f = st.file_uploader("CSV file", type=["csv"])
    if f is None:
        return
    if st.button("Upload file", type="primary"):
        try:
            up = api.post("/uploads", {"filename": f.name, "content_type": "text/csv"})
            api.post(f"/uploads/{up['upload_id']}/content", data=f.getvalue(),
                     headers={"Content-Type": "text/csv"}, timeout=120)
            w.update(upload_id=up["upload_id"], filename=f.name, step=1)
            w.pop("preview", None)
            w.pop("dry_run", None)
            st.rerun()
        except api.ApiError as exc:
            st.error(str(exc))


def _step_preview(w: dict) -> None:
    if "preview" not in w:
        try:
            w["preview"] = api.post(f"/uploads/{w['upload_id']}/preview", {"rows": 10})
        except api.ApiError as exc:
            st.error(str(exc))
            return
    pv = w["preview"]
    st.markdown(f"**{w['filename']}** - first {pv['rows_returned']} rows")
    st.dataframe(pd.DataFrame(pv["rows"]), width="stretch", hide_index=True)
    st.dataframe(pd.DataFrame(pv["columns"]).rename(columns={"detected_type": "detected type"}),
                 width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    if c1.button("Back"):
        _reset()
        st.rerun()
    if c2.button("Next: map columns", type="primary"):
        w["step"] = 2
        st.rerun()


def _units() -> list[str]:
    try:
        return [u["unit"] for u in api.get("/units")]
    except api.ApiError:
        return []


def _build_mapping(w: dict) -> dict | None:
    cols = [c["name"] for c in w["preview"]["columns"]]
    numeric = [c["name"] for c in w["preview"]["columns"] if c["detected_type"] == "number"]
    guess_ts = next((c["name"] for c in w["preview"]["columns"] if c["detected_type"] == "timestamp"), cols[0])
    c1, c2, c3 = st.columns(3)
    machine = c1.selectbox("Machine id column", cols, key="map_machine")
    ts = c2.selectbox("Timestamp column", cols, index=cols.index(guess_ts), key="map_ts")
    line = c3.selectbox("Production line column (optional)", ["(none)"] + cols, key="map_line")
    ts_fmt = st.text_input("Timestamp format (optional, strptime; default ISO 8601)", key="map_tsfmt")
    chosen = st.multiselect("Sensor columns", [c for c in numeric if c not in (machine, ts)],
                            default=[c for c in numeric if c not in (machine, ts)], key="map_sensors")
    units = _units()
    sensors: dict = {}
    for col in chosen:
        st.markdown(f"**{col}**")
        a, b, c, d, e = st.columns([2, 1, 1, 1, 1])
        name = a.text_input("Sensor name", _slug(col), key=f"map_name_{col}")
        unit = b.selectbox("File unit", units, key=f"map_unit_{col}")
        target = c.selectbox("Store as", units, index=units.index(unit) if unit in units else 0,
                             key=f"map_target_{col}_{unit}")
        lo = d.text_input("Normal min", key=f"map_lo_{col}")
        hi = e.text_input("Normal max", key=f"map_hi_{col}")
        spec: dict = {"source": col, "unit": unit, "target_unit": target}
        try:
            if lo.strip():
                spec["normal_min"] = float(lo)
            if hi.strip():
                spec["normal_max"] = float(hi)
        except ValueError:
            st.error(f"Normal min/max for {col} must be numbers.")
            return None
        sensors[name] = spec
    mapping: dict = {"dataset_kind": "sensor_readings", "format": "wide", "machine_id": machine,
                     "timestamp": ts, "sensors": sensors}
    if line != "(none)":
        mapping["production_line"] = line
    if ts_fmt.strip():
        mapping["timestamp_format"] = ts_fmt.strip()
    with st.expander("Advanced: paste a mapping as JSON instead (long format, anomaly settings, ...)"):
        pasted = st.text_area("Mapping JSON (overrides the widgets above when not empty)", key="map_json", height=160)
    if pasted.strip():
        try:
            parsed = json.loads(pasted)
        except ValueError:
            st.error("That is not valid JSON.")
            return None
        if not isinstance(parsed, dict):
            st.error("The mapping must be a JSON object.")
            return None
        return parsed
    if not sensors:
        st.info("Select at least one sensor column.")
        return None
    return mapping


def _step_map(w: dict) -> None:
    mapping = _build_mapping(w)
    c1, c2 = st.columns(2)
    if c1.button("Back", key="map_back"):
        w["step"] = 1
        st.rerun()
    if mapping is not None and c2.button("Validate on a sample", type="primary"):
        try:
            res = api.post(f"/uploads/{w['upload_id']}/mapping", {"mapping": mapping, "confirm": False})
        except api.ApiError as exc:
            st.error(str(exc))
            return
        w.update(mapping=mapping, dry_run=res, step=3)
        st.rerun()


def _step_submit(w: dict) -> None:
    res, rep = w["dry_run"], w["dry_run"]["report"]
    (st.success if res["valid"] else st.error)(
        f"Sample of {res['sampled_rows']} rows: {rep['readings_accepted']} readings accepted, "
        f"{rep['total_errors']} problem(s)." + ("" if res["valid"] else " This mapping cannot be applied."))
    if rep["counts"]:
        st.dataframe(pd.DataFrame(sorted(rep["counts"].items()), columns=["problem", "count"]),
                     width="stretch", hide_index=True)
        st.dataframe(pd.DataFrame(rep["errors"][:50]), width="stretch", hide_index=True)
        st.caption("Bad rows are dropped and listed in the job's rejected-row report; the whole file is validated by the job.")
    with st.expander("Mapping that will be submitted"):
        st.json(w["mapping"])
    if w.get("job"):
        job = w["job"]
        st.success(f"Job {job['job_id'][:8]} queued ({job['status']}). Follow it on the Jobs page.")
        if st.button("Upload another file"):
            _reset()
            st.rerun()
        return
    c1, c2 = st.columns(2)
    if c1.button("Back", key="sub_back"):
        w["step"] = 2
        st.rerun()
    if c2.button("Confirm and start the job", type="primary", disabled=not res["valid"]):
        try:
            out = api.post(f"/uploads/{w['upload_id']}/mapping", {"mapping": w["mapping"], "confirm": True})
        except api.ApiError as exc:
            st.error(str(exc))
            return
        w["job"] = out["job"]
        st.rerun()


def render() -> None:
    st.title("Upload sensor data")
    if not api.is_admin():
        st.warning("Only admins can upload data.")
        return
    w = _wiz()
    st.progress((w["step"] + 1) / len(STEPS), text=STEPS[w["step"]])
    [_step_file, _step_preview, _step_map, _step_submit][w["step"]](w)
