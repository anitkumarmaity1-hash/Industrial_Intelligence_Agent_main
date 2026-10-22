"""
Phase 5 tests: the LangGraph investigation agent.

Same philosophy as test_rag.py — this layer is tested against fakes, not
a live Postgres/Pinecone, so it runs offline and fast. The database and
retrieval layers already have their own tests (test_postgres.py,
test_rag.py) against real backends; this file tests the graph's own
logic (routing, evidence assembly, deterministic risk/root-cause,
graceful degradation on a tool failure) in isolation from them, exactly
as the master prompt's DEVELOPMENT WORKFLOW asks: "First make the graph
work with deterministic/mock tool outputs ... Then connect real tools."
Real-tool wiring is covered separately by test_api.py's /investigate
tests against a live database.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.agents.graph import run_investigation
from app.agents.risk import assess_risk
from app.agents.synthesis import build_root_cause_candidates
from app.rag.retriever import RetrievedChunk

AT_RISK_HEALTH = {
    "machine_id": "M-04",
    "window_start": datetime(2026, 1, 1, 10, 0),
    "window_end": datetime(2026, 1, 1, 11, 0),
    "health_status": "AT_RISK",
    "max_anomaly_score": 92,
    "anomalous_reading_count": 40,
    "reading_count": 60,
}
HEALTHY_HEALTH = {**AT_RISK_HEALTH, "health_status": "HEALTHY",
                  "max_anomaly_score": 2, "anomalous_reading_count": 1}
HIGH_ANOMALY = {"detected_at": datetime(
    2026, 1, 1, 10, 30), "anomaly_score": 92, "severity": "HIGH", "triggered_reasons": "OSF"}
MEDIUM_ANOMALY = {"detected_at": datetime(
    2026, 1, 1, 9, 30), "anomaly_score": 55, "severity": "MEDIUM", "triggered_reasons": "OSF"}


class _FakeRetriever:
    backend_name = "fake"

    def __init__(self, hits: list[RetrievedChunk] | None = None):
        self._hits = hits or []

    def search(self, query, top_k=5, filters=None):
        return self._hits


ONE_HIT = [
    RetrievedChunk(
        chunk_id="c1", doc_id="TRB-400", title="Troubleshooting Guide", doc_type="troubleshooting",
        section="OSF", text="...", score=0.9, backend="fake",
    )
]


# ---------------------------------------------------------------------
# risk.py — deterministic, no graph needed
# ---------------------------------------------------------------------

def test_at_risk_status_is_always_high_risk():
    assert assess_risk(AT_RISK_HEALTH, []) == "HIGH"


def test_high_severity_anomaly_is_high_risk_even_with_healthy_status():
    assert assess_risk(HEALTHY_HEALTH, [HIGH_ANOMALY]) == "HIGH"


def test_watch_status_is_medium_risk():
    assert assess_risk(
        {**HEALTHY_HEALTH, "health_status": "WATCH"}, []) == "MEDIUM"


def test_no_evidence_at_all_is_unknown_risk():
    """Audit finding F2 (Phase 9.1): no sensor snapshot, no trend and no
    anomaly log used to fall through to LOW — indistinguishable from
    "checked everything, it's healthy". It must come back UNKNOWN."""
    assert assess_risk(None, []) == "UNKNOWN"
    assert assess_risk(None, [], []) == "UNKNOWN"


def test_partial_evidence_still_uses_the_degraded_mode_rule():
    """Only *all three* evidence sources being empty is UNKNOWN. A real
    snapshot or a non-empty anomaly log with no trend is the existing,
    already-calibrated degraded mode (few/no trend windows), not a
    missing-evidence one, and must not regress to UNKNOWN."""
    assert assess_risk(HEALTHY_HEALTH, []) == "LOW"
    assert assess_risk(None, [HIGH_ANOMALY]) == "HIGH"
    assert assess_risk(None, [MEDIUM_ANOMALY]) == "MEDIUM"


def test_lazy_planner_skipping_sensor_and_anomaly_tools_is_unknown_not_low():
    """Reproduces the audit's exact F2 repro: a planner that skips both
    the sensor and anomaly tools must not report a machine as LOW risk
    just because nothing was fetched — even though the machine's real
    last-24h trend (not visible to assess_risk here, by construction) was
    100% anomalous in the audit's example."""
    assert assess_risk(sensor_evidence=None,
                       anomaly_metrics=[], sensor_trend=[]) == "UNKNOWN"


# ---------------------------------------------------------------------
# synthesis.build_root_cause_candidates — deterministic, no graph needed
# ---------------------------------------------------------------------

def test_root_cause_candidates_are_labeled_possible_not_confirmed():
    """No matching documentation -> hedged as an unconfirmed observation,
    never asserted as a proven cause (AI SAFETY / DECISION SUPPORT)."""
    candidates = build_root_cause_candidates([HIGH_ANOMALY], [])
    assert len(candidates) == 1
    assert candidates[0].startswith("Possible cause:")
    assert "unconfirmed" in candidates[0].lower()
    assert "this caused" not in candidates[0].lower()


def test_no_anomalies_yields_no_candidate_causes():
    candidates = build_root_cause_candidates([], [])
    assert "insufficient evidence" in candidates[0].lower()


def test_matching_documentation_is_cited_in_the_candidate():
    docs = [{"doc_id": "TRB-400", "title": "x", "doc_type": "troubleshooting",
             "section": "OSF", "text": "", "score": 0.9, "citation": "...",
             "failure_modes": ["OSF"]}]
    candidates = build_root_cause_candidates([HIGH_ANOMALY], docs)
    assert "TRB-400" in candidates[0]


def test_documentation_for_a_different_failure_mode_is_not_cited():
    """Audit F3: retrieval can return a real, on-topic chunk that just
    doesn't cover this reading's failure mode — e.g. an HDF (heat
    dissipation) doc retrieved alongside an OSF (overstrain) anomaly
    because the query happened to share vocabulary. That must not be
    credited as "documentation describes related failure patterns";
    every prior chunk being cited regardless of topic was the bug."""
    docs = [{"doc_id": "SOP-102", "title": "Heat Dissipation and Cooling", "doc_type": "sop",
             "section": "HDF", "text": "", "score": 0.9, "citation": "...",
             "failure_modes": ["HDF"]}]
    candidates = build_root_cause_candidates([HIGH_ANOMALY], docs)
    assert "SOP-102" not in candidates[0]
    assert "unconfirmed" in candidates[0].lower()


def test_documentation_with_no_failure_modes_is_not_cited():
    """A chunk from a doc that declares no failure_modes at all (e.g.
    SAF-300) must never be credited as supporting a specific cause."""
    docs = [{"doc_id": "SAF-300", "title": "Isolation and Safety", "doc_type": "safety",
             "section": "1", "text": "", "score": 0.9, "citation": "...",
             "failure_modes": []}]
    candidates = build_root_cause_candidates([HIGH_ANOMALY], docs)
    assert "SAF-300" not in candidates[0]


def test_statistical_anomaly_with_no_named_failure_mode_matches_rnf_docs():
    """A z-score/statistical trigger names no AI4I mechanism, so it must
    not be credited against an OSF/HDF/PWF/TWF-only doc — only a doc that
    also covers RNF (the AI4I 'no discernible cause' label)."""
    stat_anomaly = {**HIGH_ANOMALY, "triggered_reasons":
                    "Torque statistically unlike this machine's recent history"}
    twf_only_doc = [{"doc_id": "SOP-101", "title": "x", "doc_type": "sop", "section": "3",
                     "text": "", "score": 0.9, "citation": "...", "failure_modes": ["TWF", "OSF"]}]
    rnf_doc = [{"doc_id": "TRB-400", "title": "x", "doc_type": "troubleshooting", "section": "5",
                "text": "", "score": 0.9, "citation": "...", "failure_modes": ["TWF", "HDF", "PWF", "OSF", "RNF"]}]

    not_cited = build_root_cause_candidates([stat_anomaly], twf_only_doc)
    assert "SOP-101" not in not_cited[0]

    cited = build_root_cause_candidates([stat_anomaly], rnf_doc)
    assert "TRB-400" in cited[0]


def test_ai4i_reference_rates_add_real_world_context_when_matched():
    """P2 cleanup: ai4i_reference_rates (queries.get_ai4i_failure_mode_rates)
    is now wired into build_root_cause_candidates, so the real UCI AI4I
    dataset stops being decorative. The rate must be clearly labeled as
    dataset-wide context, never as support for this specific machine."""
    rates = {"OSF": {"failure_count": 78, "share_of_real_failures": 0.227,
                     "total_real_rows": 10000, "total_real_failures": 339}}
    candidates = build_root_cause_candidates(
        [HIGH_ANOMALY], [], ai4i_reference_rates=rates)
    assert "23%" in candidates[0]
    assert "real UCI AI4I 2020 dataset" in candidates[0]
    assert "78/339" in candidates[0]


def test_ai4i_reference_rates_omitted_when_not_provided():
    """Backward-compatible default: every existing call site (and every
    test above) keeps passing unchanged when the new argument is omitted."""
    candidates = build_root_cause_candidates([HIGH_ANOMALY], [])
    assert "AI4I" not in candidates[0]


def test_ai4i_reference_rates_omitted_for_mode_missing_from_rates():
    """The candidate's inferred mode (RNF, for a bare statistical trigger
    with no named AI4I mechanism) not being present in the supplied rates
    dict — e.g. a partial/degraded fetch — is handled the same way as no
    rates at all: silently no sentence, not a KeyError."""
    stat_anomaly = {**HIGH_ANOMALY, "triggered_reasons":
                    "Torque statistically unlike this machine's recent history"}
    rates = {"OSF": {"failure_count": 78, "share_of_real_failures": 0.227,
                     "total_real_rows": 10000, "total_real_failures": 339}}
    candidates = build_root_cause_candidates(
        [stat_anomaly], [], ai4i_reference_rates=rates)
    assert "AI4I" not in candidates[0]


# ---------------------------------------------------------------------
# graph.run_investigation — machine_investigation branch
# ---------------------------------------------------------------------

def test_machine_investigation_routes_correctly_and_assesses_high_risk():
    with patch("app.database.queries.get_machine_health", return_value=AT_RISK_HEALTH), \
            patch("app.database.queries.get_sensor_summary", return_value=[AT_RISK_HEALTH]), \
            patch("app.database.queries.get_maintenance_records", return_value=[]), \
            patch("app.database.queries.get_machine_anomalies", return_value=[HIGH_ANOMALY]), \
            patch("app.database.queries.get_ai4i_failure_mode_rates", return_value={}):
        result = run_investigation("Why is M-04 underperforming?",
                                   "M-04", conn=object(), retriever=_FakeRetriever(ONE_HIT))

    assert result["intent"] == "machine_investigation"
    assert result["risk_level"] == "HIGH"
    assert result["citations"] == [
        "TRB-400 § OSF — Troubleshooting Guide (synthetic document)"]
    assert result["narrative"].startswith("## Machine\nM-04")
    assert result["errors"] == []


def test_healthy_machine_with_no_anomalies_is_low_risk_with_no_causes():
    with patch("app.database.queries.get_machine_health", return_value=HEALTHY_HEALTH), \
            patch("app.database.queries.get_sensor_summary", return_value=[HEALTHY_HEALTH]), \
            patch("app.database.queries.get_maintenance_records", return_value=[]), \
            patch("app.database.queries.get_machine_anomalies", return_value=[]), \
            patch("app.database.queries.get_ai4i_failure_mode_rates", return_value={}):
        result = run_investigation(
            "How is M-16 doing?", "M-16", conn=object(), retriever=_FakeRetriever())

    assert result["risk_level"] == "LOW"
    assert "insufficient evidence" in result["root_cause_candidates"][0].lower(
    )


def test_a_failed_tool_degrades_the_investigation_instead_of_raising():
    """A dropped DB connection mid-investigation must not crash the graph
    — it should show up in `errors` and the rest of the run continues
    with whatever evidence *was* gathered."""
    with patch("app.database.queries.get_machine_health", side_effect=SQLAlchemyError("connection reset")), \
            patch("app.database.queries.get_sensor_summary", return_value=[]), \
            patch("app.database.queries.get_maintenance_records", return_value=[]), \
            patch("app.database.queries.get_machine_anomalies", return_value=[MEDIUM_ANOMALY]), \
            patch("app.database.queries.get_ai4i_failure_mode_rates", return_value={}):
        result = run_investigation(
            "Why is M-07 underperforming?", "M-07", conn=object(), retriever=_FakeRetriever())

    assert result["sensor_evidence"] is None
    assert any("connection reset" in e for e in result["errors"])
    # Anomaly evidence still made it through despite the sensor tool failing.
    assert result["risk_level"] == "MEDIUM"


def test_all_evidence_tools_failing_is_unknown_not_low():
    """Audit F2, end-to-end: if the sensor AND anomaly tools both fail
    (e.g. a DB outage mid-investigation), the fixed chain must report
    UNKNOWN, not silently read as a healthy machine."""
    with patch("app.database.queries.get_machine_health",
               side_effect=SQLAlchemyError("connection reset")), \
            patch("app.database.queries.get_sensor_summary", return_value=[]), \
            patch("app.database.queries.get_maintenance_records", return_value=[]), \
            patch("app.database.queries.get_machine_anomalies",
                  side_effect=SQLAlchemyError("connection reset")), \
            patch("app.database.queries.get_ai4i_failure_mode_rates", return_value={}):
        result = run_investigation(
            "Why is M-04 underperforming?", "M-04", conn=object(), retriever=_FakeRetriever())

    assert result["risk_level"] == "UNKNOWN"
    assert "could not be assessed" in result["recommendation"].lower()
    assert any("connection reset" in e for e in result["errors"])


# ---------------------------------------------------------------------
# graph.run_investigation — fleet_scan branch
# ---------------------------------------------------------------------

def test_no_machine_id_routes_to_fleet_scan():
    fleet = [
        {"machine_id": "M-04", "production_line": "LINE-1", "window_start": "t",
            "health_status": "AT_RISK", "max_anomaly_score": 92, "anomalous_reading_count": 40},
        {"machine_id": "M-09", "production_line": "LINE-2", "window_start": "t",
            "health_status": "HEALTHY", "max_anomaly_score": 1, "anomalous_reading_count": 0},
    ]
    with patch("app.database.queries.get_fleet_status", return_value=fleet), \
            patch("app.database.queries.get_fleet_sensor_trend", return_value={}), \
            patch("app.database.queries.get_fleet_recent_high_anomalies", return_value=set()):
        result = run_investigation(
            "Which machines currently show abnormal behavior?", None, conn=object())

    assert result["intent"] == "fleet_scan"
    assert "M-04" in result["recommendation"]
    # healthy machine shouldn't be flagged
    assert "M-09" not in result["recommendation"]
    assert result["narrative"].startswith("## Fleet status")


def test_fleet_scan_with_nothing_abnormal_says_so():
    fleet = [{"machine_id": "M-09", "production_line": "LINE-2", "window_start": "t",
              "health_status": "HEALTHY", "max_anomaly_score": 1, "anomalous_reading_count": 0}]
    with patch("app.database.queries.get_fleet_status", return_value=fleet), \
            patch("app.database.queries.get_fleet_sensor_trend", return_value={}), \
            patch("app.database.queries.get_fleet_recent_high_anomalies", return_value=set()):
        result = run_investigation(
            "Which machines currently show abnormal behavior?", None, conn=object())

    assert "no machines" in result["recommendation"].lower()


# ---------------------------------------------------------------------
# Phase 9 regression tests — fleet scan must agree with a single-machine
# investigation.
#
# Bug: fleet_scan flagged a machine purely off get_fleet_status's single
# latest hourly window (health_status != HEALTHY). A one-off WATCH hour is
# exactly the statistical noise risk.py's own rate rule exists to ignore —
# so a machine the fleet scan flagged could, investigated individually,
# come back LOW. fleet_scan now runs each machine's trend through the same
# risk.assess_risk() a single-machine investigation uses.
# ---------------------------------------------------------------------

def test_fleet_scan_does_not_flag_a_lone_watch_hour_when_trend_is_clean():
    """M-09/M-18-like: WATCH in the latest hour only, clean 24h trend.
    Must NOT be flagged — matches what investigating it directly would say."""
    fleet = [{"machine_id": "M-09", "production_line": "LINE-2", "window_start": "t",
              "health_status": "WATCH", "max_anomaly_score": 1, "anomalous_reading_count": 1}]
    trend = _trend(anomalous_per_window=0)
    trend[0] = _window(23, anomalous=1, status="WATCH")  # the one noisy hour
    with patch("app.database.queries.get_fleet_status", return_value=fleet), \
            patch("app.database.queries.get_fleet_sensor_trend", return_value={"M-09": trend}), \
            patch("app.database.queries.get_fleet_recent_high_anomalies", return_value=set()):
        result = run_investigation(
            "Which machines currently show abnormal behavior?", None, conn=object())

    assert "no machines" in result["recommendation"].lower()
    assert "M-09" not in result["recommendation"]


def test_fleet_scan_flags_sustained_fault_using_rate_not_latest_status():
    """M-04-like: latest hour only says WATCH, but the 24h trend is a
    sustained near-100% anomalous rate. Must be flagged HIGH, same as a
    direct investigation of that machine would report."""
    fleet = [{"machine_id": "M-04", "production_line": "LINE-1", "window_start": "t",
              "health_status": "WATCH", "max_anomaly_score": 1, "anomalous_reading_count": 12}]
    trend = _trend(anomalous_per_window=12, status="WATCH")
    with patch("app.database.queries.get_fleet_status", return_value=fleet), \
            patch("app.database.queries.get_fleet_sensor_trend", return_value={"M-04": trend}), \
            patch("app.database.queries.get_fleet_recent_high_anomalies", return_value=set()):
        result = run_investigation(
            "Which machines currently show abnormal behavior?", None, conn=object())

    assert "M-04 (HIGH)" in result["recommendation"]


# ---------------------------------------------------------------------
# Phase 9 regression tests — risk must discriminate between machines.
#
# Bug: assess_risk treated "any MEDIUM anomaly in the latest 20 events" as
# MEDIUM risk. Baseline noise (~1% of readings) makes that true for every
# machine, so all 18 machines in the fleet came back MEDIUM — including the
# one machine (M-04) with a genuinely active fault. Risk is now driven by
# the *sustained anomalous-reading rate* over the recent hourly windows.
# ---------------------------------------------------------------------

def _window(hour: int, anomalous: int, status: str = "HEALTHY", readings: int = 12) -> dict:
    """One hourly sensor_summary row, most-recent-first ordering is up to the caller."""
    return {
        "machine_id": "M-X",
        "window_start": datetime(2026, 3, 1, hour, 0),
        "window_end": datetime(2026, 3, 1, hour, 59),
        "health_status": status,
        "max_anomaly_score": 1 if anomalous else 0,
        "anomalous_reading_count": anomalous,
        "reading_count": readings,
    }


def _trend(anomalous_per_window: int, status: str = "HEALTHY", n: int = 24) -> list[dict]:
    """n hourly windows, most recent first (matches queries.get_sensor_summary)."""
    return [_window(h % 24, anomalous_per_window, status) for h in range(n)][::-1]


def _medium_anomalies(n: int = 20) -> list[dict]:
    return [
        {"detected_at": datetime(2026, 3, 1, 12, 0), "anomaly_score": 1,
         "severity": "MEDIUM", "triggered_reasons": "Overstrain"}
        for _ in range(n)
    ]


def test_sustained_anomalies_are_high_risk_even_without_high_severity_events():
    """M-04-like: every reading anomalous for 24h, only MEDIUM-severity events."""
    trend = _trend(anomalous_per_window=12, status="WATCH")
    assert assess_risk(trend[0], _medium_anomalies(), trend) == "HIGH"


def test_baseline_noise_with_medium_anomalies_is_low_risk():
    """The regression: healthy machine, ~1% noise, 20 MEDIUM events -> LOW, not MEDIUM."""
    trend = _trend(anomalous_per_window=0)
    trend[3] = _window(9, anomalous=1)  # ~0.3% noise
    assert assess_risk(trend[0], _medium_anomalies(), trend) == "LOW"


def test_transient_watch_snapshot_alone_is_not_medium_when_trend_is_clean():
    trend = _trend(anomalous_per_window=0)
    trend[0] = _window(23, anomalous=1, status="WATCH")
    assert assess_risk(trend[0], [], trend) == "LOW"


def test_moderate_sustained_rate_is_medium():
    # 1 anomalous reading in 12 per window ~= 8% >> ~1% fleet noise floor, < 25%.
    trend = _trend(anomalous_per_window=1)
    assert assess_risk(trend[0], _medium_anomalies(), trend) == "MEDIUM"


def test_stale_high_severity_event_outside_trend_window_is_ignored():
    """A HIGH event from weeks ago (a repaired fault) must not make today HIGH."""
    trend = _trend(anomalous_per_window=0)
    old_high = {**HIGH_ANOMALY, "detected_at": datetime(2026, 1, 15, 10, 0)}
    assert assess_risk(trend[0], [old_high], trend) == "LOW"


def test_recent_high_severity_event_inside_trend_window_is_high():
    trend = _trend(anomalous_per_window=0)
    recent_high = {**HIGH_ANOMALY, "detected_at": datetime(2026, 3, 1, 20, 0)}
    assert assess_risk(trend[0], [recent_high], trend) == "HIGH"


def test_at_risk_snapshot_is_still_high_with_clean_trend():
    trend = _trend(anomalous_per_window=0)
    assert assess_risk(AT_RISK_HEALTH, [], trend) == "HIGH"


def test_too_few_trend_windows_falls_back_to_severity_rule():
    """Degraded mode: not enough windows to measure a rate -> old conservative rule."""
    trend = _trend(anomalous_per_window=0, n=2)
    assert assess_risk(trend[0], [MEDIUM_ANOMALY], trend) == "MEDIUM"


def test_graph_distinguishes_faulty_machine_from_healthy_one():
    """End-to-end through LangGraph: same MEDIUM-heavy anomaly log, different trend."""
    def run(trend):
        with patch("app.database.queries.get_machine_health", return_value=trend[0]), \
                patch("app.database.queries.get_sensor_summary", return_value=trend), \
                patch("app.database.queries.get_maintenance_records", return_value=[]), \
                patch("app.database.queries.get_machine_anomalies", return_value=_medium_anomalies()), \
                patch("app.database.queries.get_ai4i_failure_mode_rates", return_value={}):
            return run_investigation("Why is it underperforming?", "M-X",
                                     conn=object(), retriever=_FakeRetriever())["risk_level"]

    assert run(_trend(anomalous_per_window=12, status="WATCH")) == "HIGH"
    assert run(_trend(anomalous_per_window=0)) == "LOW"


# ---------------------------------------------------------------------
# Phase 9 regression tests — report layer (synthesis.py)
# ---------------------------------------------------------------------

def test_duplicate_retrieved_chunks_are_cited_once():
    """Two chunks from the same doc section used to be listed twice under
    'Supporting evidence' and in citations."""
    dup = RetrievedChunk(
        chunk_id="c1", doc_id="INC-500", title="Incident Reports", doc_type="incident",
        section="INC-2025-014", text="...", score=0.9, backend="fake",
    )
    dup2 = RetrievedChunk(
        chunk_id="c2", doc_id="INC-500", title="Incident Reports", doc_type="incident",
        section="INC-2025-014", text="...", score=0.8, backend="fake",
    )
    trend = _trend(anomalous_per_window=12, status="WATCH")
    with patch("app.database.queries.get_machine_health", return_value=trend[0]), \
            patch("app.database.queries.get_sensor_summary", return_value=trend), \
            patch("app.database.queries.get_maintenance_records", return_value=[]), \
            patch("app.database.queries.get_machine_anomalies", return_value=_medium_anomalies()), \
            patch("app.database.queries.get_ai4i_failure_mode_rates", return_value={}):
        result = run_investigation("why?", "M-X", conn=object(),
                                   retriever=_FakeRetriever([dup, dup2]))

    assert len(result["citations"]) == 1
    assert result["narrative"].count("INC-500 § INC-2025-014") == 1


def test_high_risk_recommendation_names_sustained_anomalies_as_a_basis():
    trend = _trend(anomalous_per_window=12, status="WATCH")
    with patch("app.database.queries.get_machine_health", return_value=trend[0]), \
            patch("app.database.queries.get_sensor_summary", return_value=trend), \
            patch("app.database.queries.get_maintenance_records", return_value=[]), \
            patch("app.database.queries.get_machine_anomalies", return_value=_medium_anomalies()), \
            patch("app.database.queries.get_ai4i_failure_mode_rates", return_value={}):
        result = run_investigation(
            "why?", "M-X", conn=object(), retriever=_FakeRetriever())

    assert result["risk_level"] == "HIGH"
    assert "sustained anomalous readings" in result["recommendation"]


def test_fleet_ranking_breaks_status_ties_by_anomalous_reading_count():
    """Three WATCH machines tie on max_anomaly_score; the one with the most
    anomalous readings must rank first regardless of machine_id order."""
    def row(mid, anomalous):
        return {"machine_id": mid, "production_line": "L", "health_status": "WATCH",
                "max_anomaly_score": 1, "anomalous_reading_count": anomalous, "reading_count": 12}

    fleet = [row("M-01", 1), row("M-02", 1), row("M-18", 12)]
    with patch("app.database.queries.get_fleet_status", return_value=fleet), \
            patch("app.database.queries.get_fleet_sensor_trend", return_value={}), \
            patch("app.database.queries.get_fleet_recent_high_anomalies", return_value=set()):
        result = run_investigation(
            "Which machines are abnormal?", None, conn=object())

    assert result["recommendation"].index(
        "M-18") < result["recommendation"].index("M-01")


# ---------------------------------------------------------------------
# Audit F6: rule-based ("retrieval weak -> reformulate -> retry once") loop
# ---------------------------------------------------------------------

class _RetryTrackingRetriever:
    """Empty on the first call (simulating a query that missed), a real
    hit on the second — records every (query, filters) pair it was asked
    so tests can assert both HOW MANY times it was called and what
    changed between the two calls."""
    backend_name = "fake"

    def __init__(self, second_call_hits=None):
        self.calls: list[tuple[str, dict | None]] = []
        self._second_call_hits = second_call_hits if second_call_hits is not None else ONE_HIT

    def search(self, query, top_k=5, filters=None):
        self.calls.append((query, filters))
        return [] if len(self.calls) == 1 else self._second_call_hits


def test_empty_first_search_triggers_exactly_one_reformulated_retry():
    retriever = _RetryTrackingRetriever()
    with patch("app.database.queries.get_machine_health", return_value=AT_RISK_HEALTH), \
            patch("app.database.queries.get_sensor_summary", return_value=[AT_RISK_HEALTH]), \
            patch("app.database.queries.get_maintenance_records", return_value=[]), \
            patch("app.database.queries.get_machine_anomalies", return_value=[HIGH_ANOMALY]), \
            patch("app.database.queries.get_ai4i_failure_mode_rates", return_value={}):
        result = run_investigation(
            "Why is M-04 underperforming?", "M-04", conn=object(), retriever=retriever)

    assert len(retriever.calls) == 2  # bounded to exactly one retry
    # the retry's hit made it into state
    assert len(result["maintenance_evidence"]) == 1

    first_query, first_filters = retriever.calls[0]
    retry_query, retry_filters = retriever.calls[1]
    assert "M-04" in first_query
    # Reformulation actually changed something, not just a duplicate call:
    assert "M-04" not in retry_query  # audit F6: machine ID token dropped
    # HIGH_ANOMALY's triggered_reasons="OSF" implicates exactly one AI4I
    # mode, so the retry adds a failure_mode filter the first attempt didn't have.
    assert (first_filters or {}).get("failure_mode") is None
    assert retry_filters == {"failure_mode": "OSF"}


def test_retry_does_not_loop_forever_when_still_empty():
    """A machine with genuinely no relevant documentation must not retry
    more than once, and an empty result after the retry is still not an
    error — see tools.search_maintenance_documents's own docstring."""
    retriever = _RetryTrackingRetriever(second_call_hits=[])
    with patch("app.database.queries.get_machine_health", return_value=HEALTHY_HEALTH), \
            patch("app.database.queries.get_sensor_summary", return_value=[HEALTHY_HEALTH]), \
            patch("app.database.queries.get_maintenance_records", return_value=[]), \
            patch("app.database.queries.get_machine_anomalies", return_value=[]), \
            patch("app.database.queries.get_ai4i_failure_mode_rates", return_value={}):
        result = run_investigation(
            "How is M-16 doing?", "M-16", conn=object(), retriever=retriever)

    assert len(retriever.calls) == 2
    assert result["maintenance_evidence"] == []
    assert result["errors"] == []


def test_non_empty_first_search_never_triggers_a_retry():
    class _AlwaysHitRetriever(_RetryTrackingRetriever):
        def search(self, query, top_k=5, filters=None):
            self.calls.append((query, filters))
            return ONE_HIT

    tracker = _AlwaysHitRetriever()
    with patch("app.database.queries.get_machine_health", return_value=AT_RISK_HEALTH), \
            patch("app.database.queries.get_sensor_summary", return_value=[AT_RISK_HEALTH]), \
            patch("app.database.queries.get_maintenance_records", return_value=[]), \
            patch("app.database.queries.get_machine_anomalies", return_value=[HIGH_ANOMALY]), \
            patch("app.database.queries.get_ai4i_failure_mode_rates", return_value={}):
        result = run_investigation(
            "Why is M-04 underperforming?", "M-04", conn=object(), retriever=tracker)

    # no retry when the first attempt already hit
    assert len(tracker.calls) == 1
    assert len(result["maintenance_evidence"]) == 1


# ---------------------------------------------------------------------
# Audit F8 regression tests — synthesis.compute_recent_vs_baseline
# ---------------------------------------------------------------------

from app.agents.synthesis import (  # noqa: E402
    RECENT_WINDOW_COUNT,
    BASELINE_WINDOW_COUNT,
    build_root_cause_candidates as _build_root_cause_candidates,
    compute_recent_vs_baseline,
)


def _baseline_window(hour_offset: int, anomalous: int, defect_rate: float,
                     readings: int = 60) -> dict:
    return {
        "window_start": datetime(2026, 3, 1, 0, 0),
        "avg_defect_rate": defect_rate,
        "anomalous_reading_count": anomalous,
        "reading_count": readings,
    }


def _flat_baseline_trend(recent_anomalous: int, baseline_anomalous: int,
                         recent_defect: float = 0.02, baseline_defect: float = 0.02,
                         recent_n: int = RECENT_WINDOW_COUNT,
                         baseline_n: int = BASELINE_WINDOW_COUNT) -> list[dict]:
    """Most-recent-first: `recent_n` windows at the "last 24h" rate,
    followed by `baseline_n` windows at the "prior baseline" rate."""
    recent = [_baseline_window(h, recent_anomalous, recent_defect)
              for h in range(recent_n)]
    baseline = [_baseline_window(h, baseline_anomalous, baseline_defect)
                for h in range(baseline_n)]
    return recent + baseline


def test_too_few_baseline_windows_returns_none():
    """Not enough prior-baseline history yet (e.g. a new machine) ->
    None, so callers fall back to the raw newest-vs-oldest delta rather
    than compute a baseline comparison from a handful of hours."""
    short_trend = _flat_baseline_trend(
        recent_anomalous=1, baseline_anomalous=1, baseline_n=5)
    assert compute_recent_vs_baseline(short_trend) is None
    assert compute_recent_vs_baseline([]) is None


def test_sustained_spike_produces_an_explicit_multiplier():
    """The audit's exact complaint (M-04: raw defect-rate noise missed
    that the anomalous rate was ~8x the machine's own recent baseline) —
    a sustained last-24h anomalous rate well above the prior baseline
    must show up as a multiplier >= 1.5, not just a small absolute delta."""
    # last 24h: 10/60 anomalous per window (~16.7%); baseline: 1/60 (~1.7%)
    trend = _flat_baseline_trend(recent_anomalous=10, baseline_anomalous=1)
    comparison = compute_recent_vs_baseline(trend)

    assert comparison is not None
    assert comparison["recent_24h"]["anomalous_rate_pct"] == "16.7%"
    assert comparison["prior_baseline"]["anomalous_rate_pct"] == "1.7%"
    assert comparison["anomalous_rate_multiplier"] >= 8.0


def test_flat_trend_yields_no_multiplier_worth_reporting():
    """Genuinely flat recent-vs-baseline (the M-04 "0.172 -> 0.170" case
    from the audit) must not be dressed up as a meaningful change."""
    trend = _flat_baseline_trend(
        recent_anomalous=1, baseline_anomalous=1, recent_defect=0.172, baseline_defect=0.170)
    comparison = compute_recent_vs_baseline(trend)

    assert comparison is not None
    assert comparison["anomalous_rate_multiplier"] == 1.0
    # ~1.01x is real but not the kind of swing worth flagging as a spike/drop.
    assert 0.67 < comparison["defect_rate_multiplier"] < 1.5


def test_zero_baseline_rate_does_not_produce_a_divide_by_zero_multiplier():
    """A genuinely zero prior-baseline anomalous rate makes '8x' undefined
    (not infinite) -- must degrade to None, not raise or fabricate a number."""
    trend = _flat_baseline_trend(recent_anomalous=5, baseline_anomalous=0)
    comparison = compute_recent_vs_baseline(trend)

    assert comparison is not None
    assert comparison["prior_baseline"]["anomalous_rate_pct"] == "0.0%"
    assert comparison["anomalous_rate_multiplier"] is None


def test_what_changed_report_uses_baseline_comparison_when_available():
    """End-to-end: build_report's 'What changed?' section should surface
    the multiplier language, not the old bare two-point delta, once a
    long-enough sensor_trend_baseline is present in state."""
    from app.agents.synthesis import build_report

    trend = _flat_baseline_trend(recent_anomalous=10, baseline_anomalous=1)
    state = {
        "machine_id": "M-04",
        "risk_level": "HIGH",
        "sensor_evidence": None,
        "sensor_trend": [],
        "sensor_trend_baseline": trend,
        "anomaly_metrics": [],
        "root_cause_candidates": [],
        "errors": [],
    }
    _, narrative, _ = build_report(state)
    assert "Anomalous-reading rate, last 24h" in narrative
    assert "x the machine's own recent baseline" in narrative


# ---------------------------------------------------------------------
# Audit F8 regression test — root-cause wording no longer contradicts a
# LOW-risk recommendation (finding 2)
# ---------------------------------------------------------------------

def test_matching_cause_no_longer_claims_it_warrants_inspection():
    """Per-candidate text used to say 'warrants inspection' unconditionally,
    which could directly contradict a LOW-risk report's own 'no inspection
    priority indicated' recommendation line. The candidate text should
    describe evidentiary support only, not recommend an action."""
    docs = [{"doc_id": "TRB-400", "title": "x", "doc_type": "troubleshooting",
             "section": "OSF", "text": "", "score": 0.9, "citation": "...",
             "failure_modes": ["OSF"]}]
    candidates = _build_root_cause_candidates([HIGH_ANOMALY], docs)
    assert "warrants inspection" not in candidates[0].lower()
    assert "TRB-400" in candidates[0]
    assert "consistent with" in candidates[0].lower()
