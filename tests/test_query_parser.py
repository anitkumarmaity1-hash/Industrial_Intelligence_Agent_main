"""
Audit F1 regression tests: machine and intent extraction from the question.

Two layers, both offline (no Postgres, no LLM):

  * app/agents/query_parser.py is pure, so it's tested directly.
  * The /investigate and /chat routes are tested with the fleet list and
    run_investigation stubbed, asserting on what the AGENT IS HANDED —
    that's the actual F1 failure (the agent silently got machine_id=None
    or the wrong machine), which a response-shape test would miss.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agents.query_parser import (
    AmbiguousMachineError,
    UnknownMachineError,
    parse_question,
    resolve_target,
)
from app.api import routes
from app.api.dependencies import (
    get_conn,
    get_evidence_planner_llm,
    get_gemini_synthesizer,
    get_search_retriever,
)
from app.main import app

FLEET = {f"M-{i:02d}" for i in range(1, 19)}


# ---------------------------------------------------------------------
# parse_question
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("Why is M-17 underperforming?", ("M-17",)),
        ("what changed before m17 declined?", ("M-17",)),
        ("check M_04 please", ("M-04",)),
        ("check M\u201304 please", ("M-04",)),          # en dash from copy/paste
        # zero-pad to fleet format
        ("Is M-7 okay?", ("M-07",)),
        ("Why is machine 17 slow?", ("M-17",)),
        ("machine #3 is noisy", ("M-03",)),
        ("Machine No. 12 status", ("M-12",)),
        ("M-17 and machine 17 again", ("M-17",)),      # de-duplicated
        ("compare M-04 with M-17", ("M-04", "M-17")),  # order of appearance
    ],
)
def test_parse_finds_machine_references(text, expected):
    assert parse_question(text).machine_ids == expected


@pytest.mark.parametrize(
    "text",
    [
        "Which machines currently show abnormal behavior?",
        "ABM-5 bracket torque spec",       # embedded in a longer token
        "M-17A firmware",                  # trailing alphanumeric
        "M-1000 is not a machine id",      # 4 digits: not a fleet-format ID
        "the 5 M 3 ratio",                 # bare-space form is not accepted
        "What maintenance procedure applies to tool wear?",
    ],
)
def test_parse_ignores_non_machine_text(text):
    assert parse_question(text).machine_ids == ()


@pytest.mark.parametrize(
    "text, fleet_wide",
    [
        ("Which machines currently show abnormal behavior?", True),
        ("Which machines should the maintenance team inspect first?", True),
        ("Anything unusual across the plant?", True),
        ("Is any machine overheating?", True),
        ("Why is this underperforming?", False),
        ("Why is M-17 underperforming?", False),
    ],
)
def test_parse_detects_fleet_wide_wording(text, fleet_wide):
    assert parse_question(text).fleet_wide is fleet_wide


# ---------------------------------------------------------------------
# resolve_target
# ---------------------------------------------------------------------

def test_machine_in_text_is_used_when_no_explicit_id():
    """The F1 headline bug: this used to be a fleet scan."""
    t = resolve_target("Why is M-17 underperforming?", None, FLEET)
    assert (t.machine_id, t.intent, t.notes) == (
        "M-17", "machine_investigation", ())


def test_explicit_id_matching_text_is_fine():
    t = resolve_target("Why is M-04 underperforming?", "M-04", FLEET)
    assert t.machine_id == "M-04" and t.intent == "machine_investigation"


def test_explicit_id_conflicting_with_text_is_rejected_not_guessed():
    with pytest.raises(AmbiguousMachineError) as exc:
        resolve_target("Why is M-17 underperforming?", "M-04", FLEET)
    assert exc.value.status_code == 422
    assert "M-17" in str(exc.value) and "M-04" in str(exc.value)


def test_several_machines_in_text_is_rejected():
    with pytest.raises(AmbiguousMachineError) as exc:
        resolve_target("Compare M-04 and M-17", None, FLEET)
    assert exc.value.status_code == 422


def test_unknown_machine_in_text_is_404_not_a_fleet_scan():
    with pytest.raises(UnknownMachineError) as exc:
        resolve_target("Why is M-99 underperforming?", None, FLEET)
    assert exc.value.status_code == 404
    assert "M-99" in str(exc.value)


def test_unknown_machine_in_text_is_404_even_with_explicit_id():
    with pytest.raises(UnknownMachineError):
        resolve_target("Why is M-99 underperforming?", "M-04", FLEET)


def test_explicit_id_used_when_text_names_nothing():
    """The Streamlit 'scope to selected machine' flow."""
    t = resolve_target("Why is this machine underperforming?", "M-04", FLEET)
    assert (t.machine_id, t.intent, t.notes) == (
        "M-04", "machine_investigation", ())


def test_fleet_wording_with_explicit_id_investigates_but_says_so():
    t = resolve_target(
        "Which machines currently show abnormal behavior?", "M-04", FLEET)
    assert t.machine_id == "M-04"
    assert len(t.notes) == 1 and "fleet-wide" in t.notes[0]


def test_fleet_question_without_machine_is_a_quiet_fleet_scan():
    t = resolve_target(
        "Which machines currently show abnormal behavior?", None, FLEET)
    assert (t.machine_id, t.intent, t.notes) == (None, "fleet_scan", ())


def test_vague_question_without_machine_falls_back_with_a_note():
    t = resolve_target("Why is it underperforming?", None, FLEET)
    assert t.machine_id is None and t.intent == "fleet_scan"
    assert len(t.notes) == 1 and "No machine was identified" in t.notes[0]


# ---------------------------------------------------------------------
# Routes: assert on what the agent is handed
# ---------------------------------------------------------------------

@pytest.fixture()
def client(monkeypatch):
    seen: dict = {}

    def fake_run(question, machine_id, **_kwargs):
        seen["machine_id"] = machine_id
        return {
            "machine_id": machine_id,
            "intent": "machine_investigation" if machine_id else "fleet_scan",
            "recommendation": "rec",
            "narrative": "narr",
            "errors": ["retrieval degraded"],
        }

    monkeypatch.setattr(routes, "run_investigation", fake_run)
    monkeypatch.setattr(
        routes.queries, "list_machines",
        lambda conn, **_kw: [{"machine_id": m} for m in sorted(FLEET)])
    for dep in (get_conn, get_search_retriever,
                get_gemini_synthesizer, get_evidence_planner_llm):
        app.dependency_overrides[dep] = lambda: None
    try:
        with TestClient(app) as c:
            c.seen = seen  # type: ignore[attr-defined]
            yield c
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("path, key", [("/investigate", "question"), ("/chat", "message")])
class TestRoutes:
    def test_machine_named_in_text_reaches_the_agent(self, client, path, key):
        r = client.post(path, json={key: "Why is M-17 underperforming?"})
        assert r.status_code == 200
        assert client.seen["machine_id"] == "M-17"
        body = r.json()
        assert body["machine_id"] == "M-17"
        assert body["intent"] == "machine_investigation"

    def test_conflicting_explicit_id_is_422_and_agent_never_runs(self, client, path, key):
        r = client.post(
            path, json={"machine_id": "M-04", key: "Why is M-17 underperforming?"})
        assert r.status_code == 422
        assert "M-17" in r.json()["detail"]
        assert "machine_id" not in client.seen

    def test_unknown_machine_in_text_is_404(self, client, path, key):
        r = client.post(path, json={key: "Why is M-99 underperforming?"})
        assert r.status_code == 404
        assert "machine_id" not in client.seen

    def test_unknown_explicit_id_is_still_404(self, client, path, key):
        r = client.post(path, json={"machine_id": "M-99", key: "Why?"})
        assert r.status_code == 404
        assert r.json()["detail"] == "Machine 'M-99' not found."

    def test_fleet_question_still_runs_fleet_scan(self, client, path, key):
        r = client.post(
            path, json={key: "Which machines currently show abnormal behavior?"})
        assert r.status_code == 200
        assert client.seen["machine_id"] is None
        assert r.json()["intent"] == "fleet_scan"

    def test_resolution_notes_precede_tool_errors_in_limitations(self, client, path, key):
        r = client.post(
            path, json={"machine_id": "M-04",
                        key: "Which machines currently show abnormal behavior?"})
        assert r.status_code == 200
        note = r.json()["limitations_note"]
        assert note.startswith("The question reads as fleet-wide")
        assert note.endswith("retrieval degraded")
