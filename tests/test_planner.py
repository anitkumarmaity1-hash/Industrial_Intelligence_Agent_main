"""
Audit F5 tests: the Gemini planner adapter and the Gemini 2.5 -> 3 move.

The planner (app/agents/planner.py) previously had zero tests. These run
offline against a FAKE google-genai client that enforces the two request
rules Google documents for Gemini 3 function calling, so a regression
fails here instead of as a 400 on the first live tool round:

  1. Every replayed model turn containing function calls must carry the
     thought_signature the model returned on the first functionCall part
     (Vertex returns 400 INVALID_ARGUMENT otherwise).
  2. The next user turn must hold exactly one function_response part per
     function call in that model turn.

What this does NOT prove: that the real Vertex endpoint accepts our
requests. That needs one live call (see the F5 notes) — the fake encodes
Google's documented rules, not Google's server.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import pytest

pytest.importorskip("google.genai")
pytest.importorskip("langchain_core")

from google.genai import types  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool  # noqa: E402

from app.agents import planner, tools  # noqa: E402
from app.agents.graph import run_investigation  # noqa: E402
from app.agents.planner import _GenAIToolCallingChat  # noqa: E402
from app.core.config import Settings  # noqa: E402


# ---------------------------------------------------------------------
# Fake Gemini 3 client
# ---------------------------------------------------------------------

def _fc(name: str, args: dict | None = None, sig: bytes | None = None) -> types.Part:
    return types.Part(
        function_call=types.FunctionCall(name=name, args=args or {}),
        thought_signature=sig,
    )


def _resp(*parts: types.Part) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(
            content=types.Content(role="model", parts=list(parts)))]
    )


def _text(t: str) -> types.Part:
    return types.Part.from_text(text=t)


class Gemini3FakeClient:
    """Stands in for genai.Client: `client.models.generate_content(...)`."""

    def __init__(self, script: list[types.GenerateContentResponse]) -> None:
        self.models = self
        self._script = list(script)
        self.calls: list[dict] = []

    def generate_content(self, model, contents, config):
        self.calls.append(
            {"model": model, "contents": contents, "config": config})
        self._validate(contents)
        return self._script.pop(0)

    @staticmethod
    def _validate(contents) -> None:
        for i, content in enumerate(contents):
            if content.role != "model":
                continue
            calls = [p for p in content.parts if p.function_call is not None]
            if not calls:
                continue
            if not calls[0].thought_signature:
                raise RuntimeError(
                    f"400 INVALID_ARGUMENT: function call {calls[0].function_call.name} "
                    f"in content block {i} is missing a thought_signature")
            nxt = contents[i + 1] if i + 1 < len(contents) else None
            n_resp = 0 if nxt is None else sum(
                p.function_response is not None for p in nxt.parts)
            if nxt is None or nxt.role != "user" or n_resp != len(calls):
                raise RuntimeError(
                    f"400 INVALID_ARGUMENT: {len(calls)} function call parts in block "
                    f"{i} but {n_resp} function response parts in the next block")


def _noop_tools() -> list:
    return [
        StructuredTool.from_function(
            func=lambda: "ok", name="get_historical_evidence", description="history"),
        StructuredTool.from_function(
            func=lambda query="": "ok", name="search_maintenance_documents", description="docs"),
    ]


def _tool_msg(name: str, call_id: str) -> ToolMessage:
    return ToolMessage(content="{}", name=name, tool_call_id=call_id)


# ---------------------------------------------------------------------
# Thought-signature round trip
# ---------------------------------------------------------------------

def test_signature_and_parallel_responses_survive_the_round_trip():
    """Parallel calls: only the FIRST functionCall part carries a
    signature (Google's rule). Both calls' responses must come back in
    ONE user turn."""
    client = Gemini3FakeClient([
        _resp(_fc("get_sensor_evidence", sig=b"SIG-A"),
              _fc("get_anomaly_metrics")),
        _resp(_text("Evidence gathering complete.")),
    ])
    chat = _GenAIToolCallingChat(
        client, "gemini-3.6-flash").bind_tools(_noop_tools())

    history: list = [SystemMessage(content="sys"), HumanMessage(content="go")]
    ai = chat.invoke(history)
    assert [c["name"] for c in ai.tool_calls] == [
        "get_sensor_evidence", "get_anomaly_metrics"]

    history += [ai,
                _tool_msg("get_sensor_evidence", ai.tool_calls[0]["id"]),
                _tool_msg("get_anomaly_metrics", ai.tool_calls[1]["id"])]
    final = chat.invoke(history)  # fake raises 400 if the rules are broken
    assert not final.tool_calls

    sent = client.calls[1]["contents"]
    assert sent[1].role == "model"
    assert sent[1].parts[0].thought_signature == b"SIG-A"
    assert sent[2].role == "user"
    assert len(sent[2].parts) == 2 and all(
        p.function_response is not None for p in sent[2].parts)


def test_raw_content_is_json_serialisable():
    """So a LangGraph checkpointer could persist the messages."""
    client = Gemini3FakeClient(
        [_resp(_fc("get_sensor_evidence", sig=b"\x00\xffSIG"))])
    ai = _GenAIToolCallingChat(client, "m").bind_tools(
        _noop_tools()).invoke([HumanMessage(content="go")])
    restored = types.Content.model_validate(
        json.loads(json.dumps(ai.additional_kwargs[planner.RAW_CONTENT_KEY])))
    assert restored.parts[0].thought_signature == b"\x00\xffSIG"


def test_rebuilding_a_call_without_the_signature_is_rejected_by_gemini_3(caplog):
    """The pre-F5 behaviour, kept as a fallback. This test documents WHY
    it can't be the primary path: the Gemini 3 rules reject it."""
    legacy = AIMessage(content="", tool_calls=[
        {"name": "get_sensor_evidence", "args": {}, "id": "call_0"}])
    history = [HumanMessage(content="go"), legacy,
               _tool_msg("get_sensor_evidence", "call_0")]
    client = Gemini3FakeClient([_resp(_text("done"))])
    chat = _GenAIToolCallingChat(client, "m").bind_tools(_noop_tools())

    with caplog.at_level(logging.WARNING, logger=planner.logger.name):
        with pytest.raises(RuntimeError, match="thought_signature"):
            chat.invoke(history)
    assert "thought_signature" in caplog.text


def test_pre_gemini_3_style_history_still_converts_without_a_signature():
    """No raw content and no tool calls (e.g. plain text turns) must not
    break or warn."""
    _, contents = planner._messages_to_genai_contents(
        [SystemMessage(content="s"), HumanMessage(content="q"),
         AIMessage(content="Evidence gathering complete.")])
    assert [c.role for c in contents] == ["user", "model"]


def test_thought_summary_parts_are_not_treated_as_the_answer():
    thought = types.Part(text="internal reasoning summary", thought=True)
    ai = planner._genai_response_to_ai_message(
        _resp(thought, _text("Evidence gathering complete.")))
    assert ai.content == "Evidence gathering complete."


# ---------------------------------------------------------------------
# Temperature: omitted by default (Gemini 3 guidance)
# ---------------------------------------------------------------------

@pytest.mark.parametrize("configured, expected", [(None, None), (0.5, 0.5)])
def test_temperature_is_only_sent_when_explicitly_configured(configured, expected):
    client = Gemini3FakeClient([_resp(_text("done"))])
    _GenAIToolCallingChat(client, "m", temperature=configured).bind_tools(
        _noop_tools()).invoke([HumanMessage(content="go")])
    assert client.calls[0]["config"].temperature == expected


def test_defaults_target_gemini_3_global_with_no_forced_temperature(monkeypatch):
    for var in ("GEMINI_MODEL", "PLANNER_MODEL", "GEMINI_LOCATION",
                "GEMINI_TEMPERATURE", "PLANNER_TEMPERATURE"):
        monkeypatch.delenv(var, raising=False)
    s = Settings()
    assert s.gemini_model == s.planner_model == "gemini-3.6-flash"
    assert s.gemini_location == "global"
    assert s.gemini_temperature is None and s.planner_temperature is None


def test_temperature_env_override_and_junk_value(monkeypatch):
    monkeypatch.setenv("PLANNER_TEMPERATURE", "0")
    assert Settings().planner_temperature == 0.0
    monkeypatch.setenv("PLANNER_TEMPERATURE", "warm")
    with pytest.raises(ValueError):
        Settings()


# ---------------------------------------------------------------------
# Audit F6: feature-detected default (was opt-in-off-by-default)
# ---------------------------------------------------------------------

def test_agentic_routing_auto_activates_whenever_gcp_project_is_set(monkeypatch):
    """The old default was False (opt-in) — this is the actual audit F6
    fix at the config level: agentic_routing_enabled now defaults to
    True, mirroring gemini_synthesis_enabled, so agentic_routing_configured
    tracks GOOGLE_CLOUD_PROJECT alone unless someone explicitly opts out."""
    monkeypatch.delenv("AGENTIC_ROUTING_ENABLED", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "some-project")
    settings = Settings()
    assert settings.agentic_routing_enabled is True
    assert settings.agentic_routing_configured is True


def test_agentic_routing_kill_switch_still_works_with_credentials(monkeypatch):
    """The explicit env var remains a real opt-out, not just a formality —
    a reproducible, fully-deterministic demo run must be possible without
    unsetting GOOGLE_CLOUD_PROJECT (which would also disable Gemini
    synthesis)."""
    monkeypatch.setenv("AGENTIC_ROUTING_ENABLED", "false")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "some-project")
    settings = Settings()
    assert settings.agentic_routing_configured is False


def test_agentic_routing_not_configured_without_credentials(monkeypatch):
    """No GOOGLE_CLOUD_PROJECT (a credential-less run, e.g. CI) must stay
    on the deterministic path regardless of the enabled flag's default."""
    monkeypatch.delenv("AGENTIC_ROUTING_ENABLED", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    settings = Settings()
    assert settings.agentic_routing_configured is False


# ---------------------------------------------------------------------
# End to end: real graph + ToolNode + real planner tools, fake Gemini 3
# ---------------------------------------------------------------------

def test_agentic_graph_completes_multi_round_gemini_3_tool_calling(monkeypatch):
    """Sensor/anomaly evidence is gathered deterministically BEFORE this
    planner ever runs (audit F6) — get_sensor_evidence/get_anomaly_metrics
    are no longer tools the fake LLM can call at all; it only sees
    get_historical_evidence and search_maintenance_documents. The mocked
    tools.get_machine_health/get_machine_anomalies below feed the
    mandatory deterministic nodes, not the planner."""
    health = {
        "machine_id": "M-04", "window_start": datetime(2026, 1, 1, 10),
        "window_end": datetime(2026, 1, 1, 11), "health_status": "AT_RISK",
        "max_anomaly_score": 92, "anomalous_reading_count": 40, "reading_count": 60,
    }
    anomaly = {"detected_at": datetime(2026, 1, 1, 10, 30), "anomaly_score": 92,
               "severity": "HIGH", "triggered_reasons": "OSF"}
    monkeypatch.setattr(tools, "get_machine_health", lambda c, m, e, **kw: health)
    # `limit=None` accepted here (not just c, m, e): nodes.py's F8 fix added
    # a second get_sensor_summary call with a `limit=` kwarg for the
    # deterministic baseline-trend comparison. This mock predates that and
    # would TypeError on it — previously silent because these two
    # langchain-core/google-genai-gated tests were always skipped without
    # the optional cloud SDKs installed. Found and fixed incidentally while
    # validating the F9/F10/F11 changes with those SDKs installed.
    monkeypatch.setattr(tools, "get_sensor_summary", lambda c, m, e, limit=None, **kw: [])
    monkeypatch.setattr(tools, "get_machine_anomalies",
                        lambda c, m, e, **kw: [anomaly])
    monkeypatch.setattr(tools, "get_ai4i_failure_mode_rates",
                        lambda c, e, **kw: {})
    monkeypatch.setattr(tools, "query_machine_history",
                        lambda c, m, e, **kw: [{"event_date": "2026-01-01",
                                          "event_type": "bearing replaced", "resolved": True}])
    monkeypatch.setattr(
        tools, "search_maintenance_documents",
        lambda q, e, retriever=None, top_k=None, failure_mode=None: [
            {"doc_id": "TRB-400", "title": "x", "doc_type": "troubleshooting",
             "section": "OSF", "text": "...", "score": 0.9, "citation": "...",
             "failure_modes": ["OSF"]},
        ] if failure_mode is None else [],
    )

    client = Gemini3FakeClient([
        # round 1: history + docs, in parallel, signature on the first only
        _resp(_fc("get_historical_evidence", sig=b"SIG-1"),
              _fc("search_maintenance_documents", {"query": "overstrain"})),
        _resp(_text("Evidence gathering complete.")),
    ])
    llm = _GenAIToolCallingChat(client, "gemini-3.6-flash")

    result = run_investigation(
        "Why is M-04 underperforming?", "M-04", conn=None, planner_llm=llm)

    assert len(client.calls) == 2  # every round passed the Gemini 3 rules

    # The model must see the WHOLE conversation every round. Regression:
    # AgentState.messages had no reducer, so ToolNode's output replaced the
    # history and from round 2 the model got only function responses.
    roles = [[c.role for c in call["contents"]] for call in client.calls]
    assert roles == [["user"], ["user", "model", "user"]]
    assert client.calls[1]["contents"][1].parts[0].thought_signature == b"SIG-1"

    # Sensor/anomaly evidence came from the mandatory deterministic nodes,
    # not from anything the fake LLM was asked to fetch — it was never
    # offered those tools at all.
    assert result["risk_level"] == "HIGH"
    assert result["sensor_evidence"] == health
    assert result["anomaly_metrics"] == [anomaly]
    assert result["historical_evidence"] == [
        {"event_date": "2026-01-01", "event_type": "bearing replaced", "resolved": True}]
    assert not any("did not fetch" in e for e in result["errors"])


def test_lazy_planner_that_calls_nothing_still_gets_high_risk(monkeypatch):
    """Audit F6, replaying the auditor's own F2 attack against the NEW
    design: a scripted planner that immediately confirms 'done' without
    calling get_historical_evidence or search_maintenance_documents at
    all. Before this fix (see planner.py's module docstring), the
    equivalent lazy planner could also skip get_sensor_evidence/
    get_anomaly_metrics and turn a 100%-anomalous machine into a LOW-risk
    report. Now those two are not choices the planner has — it cannot
    reach this failure mode even by doing nothing, because it was never
    given the tools that caused it."""
    health = {
        "machine_id": "M-04", "window_start": datetime(2026, 1, 1, 10),
        "window_end": datetime(2026, 1, 1, 11), "health_status": "AT_RISK",
        "max_anomaly_score": 98, "anomalous_reading_count": 58, "reading_count": 60,
    }
    anomaly = {"detected_at": datetime(2026, 1, 1, 10, 30), "anomaly_score": 98,
               "severity": "HIGH", "triggered_reasons": "OSF"}
    monkeypatch.setattr(tools, "get_machine_health", lambda c, m, e, **kw: health)
    monkeypatch.setattr(tools, "get_sensor_summary", lambda c, m, e, limit=None, **kw: [])
    monkeypatch.setattr(tools, "get_machine_anomalies",
                        lambda c, m, e, **kw: [anomaly])
    monkeypatch.setattr(tools, "get_ai4i_failure_mode_rates",
                        lambda c, e, **kw: {})
    monkeypatch.setattr(tools, "search_maintenance_documents",
                        lambda q, e, retriever=None, top_k=None, failure_mode=None: [])

    # The lazy/broken/adversarial planner: replies "done" on round one,
    # no tool calls, no matter what it's shown.
    client = Gemini3FakeClient([_resp(_text("Evidence gathering complete."))])
    llm = _GenAIToolCallingChat(client, "gemini-3.6-flash")

    result = run_investigation(
        "Why is M-04 underperforming?", "M-04", conn=None, planner_llm=llm)

    assert len(client.calls) == 1  # the planner really did call nothing
    # not LOW, not UNKNOWN — real evidence exists
    assert result["risk_level"] == "HIGH"
    assert result["sensor_evidence"] == health
    assert result["anomaly_metrics"] == [anomaly]
    assert result["historical_evidence"] == []
    assert result["maintenance_evidence"] == []


def test_planner_is_never_offered_sensor_or_anomaly_tools():
    """Direct check on build_planner_tools' contract (audit F6): the
    supplementary planner's tool list must be exactly the two
    supplementary tools, by name — a future regression that accidentally
    re-adds get_sensor_evidence/get_anomaly_metrics here would silently
    reopen the exact hole test_lazy_planner_that_calls_nothing_still_
    gets_high_risk above exists to close."""
    tool_names = {
        t.name for t in planner.build_planner_tools(
            conn=None, retriever=None, machine_id="M-04", errors=[], collected={})
    }
    assert tool_names == {"get_historical_evidence",
                          "search_maintenance_documents"}
