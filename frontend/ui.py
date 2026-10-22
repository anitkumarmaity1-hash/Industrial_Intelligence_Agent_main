"""Small shared rendering helpers (badges, investigation result)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

RISK_BADGE = {"LOW": "🟢 LOW", "MEDIUM": "🟡 MEDIUM", "HIGH": "🔴 HIGH", "UNKNOWN": "⚪ UNKNOWN"}
HEALTH_BADGE = {"HEALTHY": "🟢 HEALTHY", "WATCH": "🟡 WATCH", "AT_RISK": "🔴 AT_RISK"}
SEVERITY_BADGE = {"MEDIUM": "🟡 MEDIUM", "HIGH": "🔴 HIGH"}


def strip_sections(narrative: str, drop: set[str]) -> str:
    """Drop narrative sections that the structured widgets already show."""
    parts = narrative.split("\n\n## ")
    kept = [parts[0]] + [p for p in parts[1:] if p.split("\n", 1)[0].strip() not in drop]
    return "\n\n## ".join(kept)


def render_result(result: dict) -> None:
    """One investigation result, as returned by POST /investigate or stored in history."""
    bits = [f"**Intent:** `{result.get('intent', '?')}`"]
    if result.get("machine_id"):
        bits.append(f"**Machine:** `{result['machine_id']}`")
    if result.get("risk_level"):
        bits.append(f"**Risk:** {RISK_BADGE.get(result['risk_level'], result['risk_level'])}")
    st.markdown(" &nbsp;|&nbsp; ".join(bits))
    if result.get("limitations_note"):
        st.warning(f"⚠️ {result['limitations_note']}")
    st.markdown(strip_sections(result.get("narrative", ""), {"Possible causes", "Supporting evidence"}))
    left, right = st.columns(2)
    with left:
        st.markdown("**Possible causes** _(hypotheses, not confirmed diagnoses)_")
        causes = result.get("root_cause_candidates") or []
        for c in causes:
            st.markdown(f"- {c}")
        if not causes:
            st.caption("None identified from current evidence.")
    with right:
        st.markdown("**Supporting citations**")
        cites = result.get("supporting_citations") or []
        for c in cites:
            st.markdown(f"- {c}")
        if not cites:
            st.caption("No document citations for this investigation.")
    if result.get("fleet_ranking"):
        st.markdown("**Fleet ranking**")
        df = pd.DataFrame(result["fleet_ranking"])
        for col, badges in (("health_status", HEALTH_BADGE), ("risk_level", RISK_BADGE)):
            if col in df.columns:
                df[col] = df[col].map(lambda s, b=badges: b.get(s, s))
        st.dataframe(df, width="stretch", hide_index=True)
    if result.get("truncated"):
        st.caption("This stored result was truncated because it was very large.")
