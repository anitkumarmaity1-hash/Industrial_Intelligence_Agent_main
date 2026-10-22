"""Fleet health: per-machine risk, a sensor trend against the tenant's normal range,
and the machine's anomaly list."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import api_client as api
from ui import HEALTH_BADGE, RISK_BADGE, SEVERITY_BADGE

PAGE_SIZE = 25


def _trend(machine_id: str) -> None:
    st.subheader(f"{machine_id} - sensor trend")
    try:
        sensors = [s for s in api.get("/sensors") if s["enabled"]]
    except api.ApiError as exc:
        st.error(str(exc))
        return
    if not sensors:
        st.info("This company has no enabled sensors yet. Upload a file on the Upload page.")
        return
    c1, c2 = st.columns(2)
    spec = c1.selectbox("Sensor", sensors, format_func=lambda s: f"{s['sensor_name']} ({s['unit']})")
    days = c2.select_slider("Window (days, ending at the newest reading)", [1, 3, 7, 14, 31], value=7)
    try:
        series = api.get(f"/machines/{machine_id}/series",
                         {"sensor": spec["sensor_name"], "days": days, "max_points": 300})
    except api.ApiError as exc:
        st.error(str(exc))
        return
    if not series["points"]:
        st.info(f"No {spec['sensor_name']} readings for {machine_id} yet.")
        return
    df = pd.DataFrame(series["points"])
    df["ts"] = pd.to_datetime(df["ts"])
    x = alt.X("ts:T", title=None)
    layers = [
        alt.Chart(df).mark_area(opacity=0.2).encode(x=x, y=alt.Y("min:Q", title=series["unit"]), y2="max:Q"),
        alt.Chart(df).mark_line().encode(x=x, y="avg:Q"),
    ]
    limits = [v for v in (series["normal_min"], series["normal_max"]) if v is not None]
    if limits:
        rules = pd.DataFrame({"limit": limits})
        layers.append(alt.Chart(rules).mark_rule(color="red", strokeDash=[6, 4]).encode(y="limit:Q"))
    st.altair_chart(alt.layer(*layers).interactive(), width="stretch")
    over = (df["max"] > series["normal_max"]) if series["normal_max"] is not None else pd.Series(False, index=df.index)
    under = (df["min"] < series["normal_min"]) if series["normal_min"] is not None else pd.Series(False, index=df.index)
    outside = int((over | under).sum())
    rng = " to ".join("-" if v is None else f"{v:g}" for v in (series["normal_min"], series["normal_max"]))
    st.caption(f"Line = average, band = min-max per {series['bucket_seconds'] // 60} min bucket. "
               f"Normal range from the sensor registry: {rng} {series['unit']}. "
               f"Buckets touching a limit: {outside} of {len(df)}.")


def _anomalies(machine_id: str) -> None:
    st.subheader(f"{machine_id} - anomalies")
    n = st.slider("Most recent N", 5, 200, 25, step=5)
    try:
        rows = api.get(f"/machines/{machine_id}/anomalies", {"limit": n})
    except api.ApiError as exc:
        st.error(str(exc))
        return
    if not rows:
        st.info("No flagged anomalies for this machine.")
        return
    df = pd.DataFrame(rows)
    df["severity"] = df["severity"].map(lambda s: SEVERITY_BADGE.get(s, s))
    st.dataframe(df.rename(columns={"detected_at": "Detected at", "anomaly_score": "Score",
                                    "severity": "Severity", "triggered_reasons": "Reasons"}),
                 width="stretch", hide_index=True)


def render() -> None:
    st.title("Fleet health")
    page = st.number_input("Page", min_value=1, value=1, step=1)
    try:
        data = api.get("/fleet/summary", {"limit": PAGE_SIZE, "offset": (page - 1) * PAGE_SIZE})
    except api.ApiError as exc:
        st.error(str(exc))
        return
    items = data["items"]
    if data["total"] == 0:
        st.info("No machines yet. An admin can upload a sensor file on the Upload page.")
        return
    if not items:
        st.info(f"No machines on page {page}; the fleet has {data['total']} machine(s).")
        return
    pages = -(-data["total"] // PAGE_SIZE)
    st.caption(f"{data['total']} machine(s); page {page} of {pages}. Risk is the sustained anomalous-reading "
               "rate over the last 24 hourly windows (the same rule the agent uses).")
    counts = pd.Series([i["risk_level"] for i in items]).value_counts()
    cols = st.columns(4)
    for col, level in zip(cols, ("HIGH", "MEDIUM", "LOW", "UNKNOWN")):
        col.metric(RISK_BADGE[level], int(counts.get(level, 0)))
    df = pd.DataFrame(items)
    df["risk_level"] = df["risk_level"].map(lambda s: RISK_BADGE.get(s, s))
    df["health_status"] = df["health_status"].map(lambda s: HEALTH_BADGE.get(s, s) if s else "-")
    st.dataframe(df[["machine_id", "production_line", "risk_level", "health_status", "anomalous_reading_count",
                     "reading_count", "window_end"]].rename(columns={
                         "machine_id": "Machine", "production_line": "Line", "risk_level": "Risk",
                         "health_status": "Latest window", "anomalous_reading_count": "Anomalous readings",
                         "reading_count": "Readings", "window_end": "As of"}),
                 width="stretch", hide_index=True)
    st.divider()
    machine = st.selectbox("Machine", [i["machine_id"] for i in items])
    _trend(machine)
    st.divider()
    _anomalies(machine)
