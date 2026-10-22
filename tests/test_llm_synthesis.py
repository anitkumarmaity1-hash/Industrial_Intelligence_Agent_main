"""
Phase 6 tests: Gemini synthesis (app/agents/llm.py, synthesis.build_report_with_llm,
nodes.build_recommendation's synthesizer branch).

Same philosophy as test_agent.py: no live Vertex AI call, no credentials.
GeminiSynthesizer is replaced with a fake that returns/raises what each
test needs, so this runs offline and fast. What's under test is the
WIRING - prompt content, success path, graceful fallback, and that
fleet_scan never touches Gemini at all - not Gemini's actual output
quality, which can't be asserted on deterministically anyway.
"""

from __future__ import annotations

import pytest

from app.agents.llm import GeminiReportSections, _build_prompt
from app.agents.nodes import build_recommendation
from app.agents.state import AgentState

from datetime import datetime

BASE_STATE: AgentState = {
    "user_query": "Why is M-04 underperforming?",
    "machine_id": "M-04",
    "intent": "machine_investigation",
    "sensor_evidence": {
        "machine_id": "M-04",
        "window_start": datetime(2026, 1, 1, 10, 0),
        "window_end": datetime(2026, 1, 1, 11, 0),
        "health_status": "AT_RISK",
        "max_anomaly_score": 92,
        "anomalous_reading_count": 40,
        "reading_count": 60,
    },
    "sensor_trend": [
        {"window_start": datetime(
            2026, 1, 1, 10, 0), "avg_defect_rate": 0.08, "avg_production_rate": 40.0},
        {"window_start": datetime(
            2026, 1, 1, 9, 0), "avg_defect_rate": 0.02, "avg_production_rate": 55.0},
    ],
    "historical_evidence": [
        {"event_date": datetime(2025, 12, 1), "event_type": "REPAIR",
         "technician_notes": "Replaced bearing", "resolved": True}
    ],
    "maintenance_evidence": [
        {"doc_id": "TRB-400", "title": "Troubleshooting Guide", "doc_type": "troubleshooting",
         "section": "OSF", "text": "Check tool wear.", "score": 0.9, "citation": "TRB-400 §OSF"}
    ],
    "anomaly_metrics": [
        {"detected_at": datetime(2026, 1, 1, 10, 30), "anomaly_score": 92,
         "severity": "HIGH", "triggered_reasons": "OSF"}
    ],
    "risk_level": "HIGH",
    "root_cause_candidates": ["Possible cause: OSF. Documentation describes related failure patterns."],
    "errors": [],
}

FLEET_STATE: AgentState = {
    "user_query": "Which machines are at risk?",
    "machine_id": None,
    "intent": "fleet_scan",
    "fleet_ranking": [
        {"machine_id": "M-04", "production_line": "Line-1", "window_start": datetime(2026, 1, 1, 10, 0),
         "health_status": "AT_RISK", "max_anomaly_score": 92, "anomalous_reading_count": 40},
    ],
    "errors": [],
}

VALID_SECTIONS = GeminiReportSections(
    what_changed=["Anomaly score rose to 92 (HIGH severity)."],
    investigation_summary="Sensor and maintenance evidence both point to tool wear.",
    possible_causes_prose=[
        "Tool wear consistent with the OSF pattern seen in sensor data."],
    recommended_action="Schedule a tool-wear inspection before the next shift.",
    confidence_limitations="Based on one anomaly event and one prior repair record.",
)


class _FakeSynthesizer:
    """Stands in for GeminiSynthesizer. `to_return`/`to_raise` control
    what .synthesize() does, so success and failure paths are both
    driven from ordinary Python objects, not a mocked network call."""

    def __init__(self, to_return: GeminiReportSections | None = None, to_raise: Exception | None = None):
        self._to_return = to_return
        self._to_raise = to_raise
        self.calls: list[AgentState] = []

    def synthesize(self, state: AgentState) -> GeminiReportSections:
        self.calls.append(state)
        if self._to_raise is not None:
            raise self._to_raise
        assert self._to_return is not None
        return self._to_return


# 1. Prompt content -----------------------------------------------------

def test_prompt_includes_evidence_and_excludes_bookkeeping():
    prompt = _build_prompt(BASE_STATE)

    # Evidence Gemini is allowed to see:
    assert "M-04" in prompt
    assert "AT_RISK" in prompt
    assert "OSF" in prompt
    assert "TRB-400" in prompt

    # Instructed not to add candidates beyond what nodes.py already found:
    assert "root_cause_candidates_already_identified" in prompt

    # Bookkeeping fields (intent, errors) are routing/internal, not evidence:
    assert '"intent"' not in prompt
    assert '"errors"' not in prompt


# 2. Success path ---------------------------------------------------------

def test_success_path_uses_gemini_sections_and_returns_no_error():
    synthesizer = _FakeSynthesizer(to_return=VALID_SECTIONS)
    node = build_recommendation(synthesizer)

    result = node(BASE_STATE)

    assert result["recommendation"] == VALID_SECTIONS.recommended_action
    assert "Schedule a tool-wear inspection" in result["narrative"]
    assert "Anomaly score rose to 92" in result["narrative"]
    assert result["errors"] == []  # no fallback needed
    assert len(synthesizer.calls) == 1


# 3. Failure fallback -------------------------------------------------------

def test_gemini_failure_falls_back_to_deterministic_template():
    synthesizer = _FakeSynthesizer(to_raise=RuntimeError(
        "Gemini returned an empty response."))
    node = build_recommendation(synthesizer)

    result = node(BASE_STATE)

    # deterministic template still produced something
    assert result["recommendation"]
    assert result["narrative"]
    assert any("Gemini synthesis unavailable" in e for e in result["errors"])
    assert "Gemini returned an empty response." in result["errors"][-1]


# 4. No-synthesizer parity ---------------------------------------------------

def test_no_synthesizer_matches_phase5_deterministic_output():
    node_with_none = build_recommendation(None)
    node_default = build_recommendation()  # default arg is also None

    result_none = node_with_none(BASE_STATE)
    result_default = node_default(BASE_STATE)

    assert result_none == result_default
    assert result_none["errors"] == []


# 5. fleet_scan never calls Gemini -------------------------------------------

def test_fleet_scan_never_invokes_synthesizer():
    synthesizer = _FakeSynthesizer(to_return=VALID_SECTIONS)
    node = build_recommendation(synthesizer)

    result = node(FLEET_STATE)

    assert synthesizer.calls == []  # never called for fleet_scan
    assert result["recommendation"]  # deterministic fleet report still built


# 6. Schema validation on malformed Gemini output ----------------------------

def test_gemini_schema_mismatch_falls_back_cleanly():
    class _BadShapeError(Exception):
        pass

    synthesizer = _FakeSynthesizer(
        to_raise=RuntimeError(
            "Gemini response did not match the expected schema: ...")
    )
    node = build_recommendation(synthesizer)

    result = node(BASE_STATE)

    assert result["recommendation"]
    assert any("schema" in e.lower() or "unavailable" in e.lower()
               for e in result["errors"])


# ---------------------------------------------------------------------
# Audit F7 regression tests — llm._validate_report_against_evidence
# ---------------------------------------------------------------------

from app.agents.llm import _build_evidence_payload, _validate_report_against_evidence  # noqa: E402

BASE_PAYLOAD = _build_evidence_payload(BASE_STATE)


def test_valid_sections_pass_the_guardrail():
    """VALID_SECTIONS' one cause and its numbers (92) both trace back to
    BASE_STATE's evidence -- must not raise."""
    _validate_report_against_evidence(VALID_SECTIONS, BASE_PAYLOAD)  # no raise


def test_wrong_cause_count_is_rejected():
    """BASE_STATE has exactly one root_cause_candidates entry;
    possible_causes_prose with two must fail -- 'cover exactly these
    candidates' (audit F7) wasn't enforced before this."""
    bad = VALID_SECTIONS.model_copy(update={
        "possible_causes_prose": [
            "Tool wear consistent with the OSF pattern.",
            "A second, unrequested cause Gemini made up.",
        ]
    })
    with pytest.raises(ValueError, match="possible_causes_prose"):
        _validate_report_against_evidence(bad, BASE_PAYLOAD)


def test_hallucinated_number_is_rejected():
    """A statistic that never appeared anywhere in the evidence JSON
    (BASE_STATE's anomaly score is 92, not 88) must be caught."""
    bad = VALID_SECTIONS.model_copy(update={
        "what_changed": ["Anomaly score rose to 88 (HIGH severity)."],
    })
    with pytest.raises(ValueError, match="88"):
        _validate_report_against_evidence(bad, BASE_PAYLOAD)


def test_number_reused_from_evidence_is_accepted():
    """92 (the real anomaly score) appearing in prose must NOT be flagged
    -- the guardrail checks traceability to evidence, not novelty."""
    ok = VALID_SECTIONS.model_copy(update={
        "what_changed": ["Anomaly score rose to 92 (HIGH severity), consistent with sensor data."],
    })
    _validate_report_against_evidence(ok, BASE_PAYLOAD)  # no raise


def test_single_digit_numbers_are_not_flagged():
    """Small standalone digits (counts like 'one prior repair') are
    excluded by design (see llm._NUMBER_TOKEN_PATTERN's docstring) --
    otherwise nearly every real report would trip the guardrail on
    harmless prose."""
    ok = VALID_SECTIONS.model_copy(update={
        "investigation_summary": "Reviewed 1 prior repair record and 2 evidence sources.",
    })
    _validate_report_against_evidence(ok, BASE_PAYLOAD)  # no raise


def test_date_components_are_not_flagged_as_invented_statistics():
    """A timestamp string like BASE_STATE's 2026-01-01 must not make its
    component digits ('01', '2026') look like unsupported statistics --
    they're already literally present in the evidence JSON as part of the
    serialized datetime anyway, but this also guards against the pattern
    misfiring on a *different* date Gemini might format differently."""
    ok = VALID_SECTIONS.model_copy(update={
        "confidence_limitations": "Based on evidence gathered around 10:30 on 2026-01-01.",
    })
    _validate_report_against_evidence(ok, BASE_PAYLOAD)  # no raise


def test_recent_vs_baseline_figures_are_available_to_reference():
    """When sensor_trend_baseline has enough history, the payload carries
    the same computed baseline numbers the deterministic template renders
    (audit F8/F7 tie-in) -- so a Gemini claim using them is verifiable."""
    from datetime import datetime as _dt

    trend = [
        {"window_start": _dt(2026, 1, 1, h % 24, 0), "avg_defect_rate": 0.05,
         "anomalous_reading_count": 10, "reading_count": 60}
        for h in range(24)
    ] + [
        {"window_start": _dt(2026, 1, 1, h % 24, 0), "avg_defect_rate": 0.01,
         "anomalous_reading_count": 1, "reading_count": 60}
        for h in range(168)
    ]
    state = {**BASE_STATE, "sensor_trend_baseline": trend}
    payload = _build_evidence_payload(state)
    assert payload["recent_vs_baseline"] is not None
    assert payload["recent_vs_baseline"]["recent_24h"]["anomalous_rate_pct"] == "16.7%"

    claiming_it = VALID_SECTIONS.model_copy(update={
        "what_changed": ["Anomalous-reading rate over the last 24h is 16.7%."],
    })
    _validate_report_against_evidence(claiming_it, payload)  # no raise
