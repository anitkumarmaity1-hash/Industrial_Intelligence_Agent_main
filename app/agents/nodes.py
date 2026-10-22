"""
LangGraph node functions — audit F6 revision adds mandatory sensor/anomaly
evidence gathering plus a rule-based maintenance-doc retry loop; see below
and app/agents/graph.py's module docstring for the current graph shape.

Every node has the same shape: `(state) -> dict` returning only the keys
it changes (LangGraph merges the return into AgentState). `conn` and
`retriever` are bound into each node as closures at graph-build time
(see graph.py) rather than threaded through state — they are
request-scoped resources, not investigation data, and don't belong in a
dict that the synthesizer will eventually hand to an LLM.

Every node reads the *current* errors list and returns the extended one,
rather than relying on a reducer — one explicit list per run is enough
for a 2-day MVP and keeps AgentState a plain TypedDict (see state.py).

gather_sensor_evidence and gather_anomaly_metrics are now ALWAYS run,
unconditionally, for a machine_investigation — audit F6: a planner (real
or scripted-broken) can no longer leave risk.assess_risk with nothing to
reason about, because it is never given the choice. gather_historical_
evidence and gather_maintenance_evidence remain the deterministic
fallback for the two SUPPLEMENTARY evidence types, used whenever the
Gemini-backed planner (app/agents/planner.py) isn't configured; when it
is, plan_supplementary_evidence / finalize_supplementary_evidence (near
the bottom of this file) take their place. Either way,
retry_maintenance_evidence runs next: a rule-based (no LLM) reformulate-
and-retry-once step for when the first maintenance-doc search came back
empty.
"""
from __future__ import annotations
import logging
from app.rag.retriever import DocumentRetriever
from app.agents.state import AgentState
from app.agents.risk import DEFAULT_RISK_THRESHOLDS, RiskThresholds, assess_risk
from app.agents import synthesis, tools
from sqlalchemy.engine import Connection
from typing import Any
from app.agents.llm import GeminiSynthesizer
from app.core.tenancy import DEFAULT_TENANT_ID
logger = logging.getLogger(__name__)


def route_intent(state: AgentState) -> dict[str, Any]:
    """Deterministic routing — no LLM call. A machine_id present means a
    single-machine investigation; its absence means a fleet-wide question
    ("which machines currently show abnormal behavior?"). Vertex/Gemini
    isn't wired up until Phase 6, and even once it is, this decision is
    a fact about the request (was a machine named or not), not something
    that benefits from being inferred rather than checked directly.
    """
    intent = "machine_investigation" if state.get(
        "machine_id") else "fleet_scan"
    return {"intent": intent, "errors": list(state.get("errors", []))}


def fleet_scan(
    conn: Connection,
    tenant_id: str = DEFAULT_TENANT_ID,
    thresholds: RiskThresholds | None = None,
):
    def _node(state: AgentState) -> dict[str, Any]:
        errors = list(state.get("errors", []))
        ranking = tools.get_fleet_status(conn, errors, tenant_id=tenant_id)
        trend_by_machine = tools.get_fleet_sensor_trend(
            conn, errors, tenant_id=tenant_id)
        high_anomaly_machines = tools.get_fleet_recent_high_anomalies(
            conn, errors, tenant_id=tenant_id)

        # Phase 9 fix: attach a risk_level computed by the exact same rule
        # a single-machine investigation uses (risk.assess_risk), instead
        # of letting synthesis.py flag machines off get_fleet_status's
        # single latest hourly window alone. Without this, a machine whose
        # one-off WATCH hour is noise gets flagged fleet-wide while its own
        # /investigate correctly reports LOW.
        for row in ranking:
            machine_id = row.get("machine_id")
            trend = trend_by_machine.get(machine_id, [])
            anomaly_metrics = (
                [{"severity": "HIGH", "detected_at": None}]
                if machine_id in high_anomaly_machines else []
            )
            row["risk_level"] = assess_risk(
                row, anomaly_metrics, trend, thresholds=thresholds)

        return {"fleet_ranking": ranking, "errors": errors}

    return _node


def gather_sensor_evidence(conn: Connection, tenant_id: str = DEFAULT_TENANT_ID):
    def _node(state: AgentState) -> dict[str, Any]:
        errors = list(state.get("errors", []))
        machine_id = state["machine_id"]
        latest = tools.get_machine_health(
            conn, machine_id, errors, tenant_id=tenant_id)
        # Unchanged: exactly what risk.assess_risk's sustained-rate rule is
        # calibrated against (24 hourly windows — see risk.py). Do not widen
        # this call; see synthesis.RECENT_WINDOW_COUNT / BASELINE_WINDOW_COUNT
        # for the separate, longer fetch used only for "What changed?".
        trend = tools.get_sensor_summary(
            conn, machine_id, errors, tenant_id=tenant_id)
        # Audit F8: a second, longer fetch purely for the deterministic
        # last-24h-vs-prior-baseline comparison (synthesis.
        # compute_recent_vs_baseline) — never fed to risk.assess_risk, so
        # widening it cannot change a risk_level. One extra query per
        # investigation, same pattern as fleet_scan's multi-query reads.
        baseline_trend = tools.get_sensor_summary(
            conn, machine_id, errors, limit=synthesis.TREND_WINDOWS_FOR_BASELINE,
            tenant_id=tenant_id,
        )
        return {
            "sensor_evidence": latest,
            "sensor_trend": trend,
            "sensor_trend_baseline": baseline_trend,
            "errors": errors,
        }

    return _node


def gather_historical_evidence(conn: Connection, tenant_id: str = DEFAULT_TENANT_ID):
    def _node(state: AgentState) -> dict[str, Any]:
        errors = list(state.get("errors", []))
        records = tools.query_machine_history(
            conn, state["machine_id"], errors, tenant_id=tenant_id)
        return {"historical_evidence": records, "errors": errors}

    return _node


def gather_anomaly_metrics(conn: Connection, tenant_id: str = DEFAULT_TENANT_ID):
    def _node(state: AgentState) -> dict[str, Any]:
        errors = list(state.get("errors", []))
        anomalies = tools.get_machine_anomalies(
            conn, state["machine_id"], errors, tenant_id=tenant_id)
        # P2 cleanup: ai4i_reference (the real UCI AI4I 2020 dataset) was
        # loaded and indexed but no tool ever queried it. One fleet-wide,
        # dataset-wide lookup — cheap enough to gather unconditionally
        # alongside anomaly evidence — lets identify_root_causes attach a
        # real base rate to each candidate instead of only ever reasoning
        # over the synthetic fleet. See synthesis.build_root_cause_candidates.
        ai4i_rates = tools.get_ai4i_failure_mode_rates(conn, errors)
        return {
            "anomaly_metrics": anomalies,
            "ai4i_reference_rates": ai4i_rates,
            "errors": errors,
        }

    return _node


def _primary_maintenance_query(state: AgentState) -> str:
    """Build the first-attempt maintenance-doc query the same way the
    fixed chain always has: prefer the anomaly log's own triggered_reasons
    over the raw user question (a better index into the maintenance docs
    than a possibly vague natural-language question), prefixed with the
    machine ID. Falls back to the user's question when no anomaly reasons
    are available.
    """
    machine_id = state.get("machine_id") or ""
    anomaly_metrics = state.get("anomaly_metrics") or []
    reasons = [a.get("triggered_reasons")
               for a in anomaly_metrics if a.get("triggered_reasons")]
    body = " ".join(reasons) if reasons else state.get("user_query", "")
    return f"{machine_id} {body}".strip()


def _reformulated_maintenance_query(state: AgentState) -> tuple[str, str | None]:
    """Rule-based (no LLM) fallback query for retry_maintenance_evidence,
    used only when the first attempt came back with nothing — see that
    node. Deliberately different from _primary_maintenance_query in two
    ways, not just a retry of the same search:

      1. Drops the machine-ID token. Most of the maintenance docs (SOPs,
         the equipment manual, troubleshooting guide) don't mention
         machine IDs at all — see documents/*.md — so on a lexical (BM25)
         backend that token is dead weight for those doc types; only the
         incident-report log names machines, and it stays reachable via
         the failure-mode filter (and its own text) below, not the ID.
      2. Adds a failure_mode filter when the anomaly log implicates
         exactly one AI4I mode (via synthesis.infer_failure_modes) — a
         structured-metadata filter the first attempt didn't use, for
         precision instead of just broader recall. Left unset when zero
         or multiple modes are implicated: with zero, there is nothing to
         filter on; with multiple, one anomaly log entry doesn't clearly
         point at a single mode, and a wrong filter would be worse than
         none (see app.agents.synthesis's own reasoning for the same
         "don't guess a single mode" discipline in root-cause matching).
    """
    anomaly_metrics = state.get("anomaly_metrics") or []
    reasons = [a.get("triggered_reasons")
               for a in anomaly_metrics if a.get("triggered_reasons")]
    modes: set[str] = set()
    for reason in reasons:
        modes |= synthesis.infer_failure_modes(reason)
    failure_mode = next(iter(modes)) if len(modes) == 1 else None
    body = " ".join(reasons) if reasons else state.get("user_query", "")
    return body.strip(), failure_mode


def gather_maintenance_evidence(retriever: DocumentRetriever | None):
    def _node(state: AgentState) -> dict[str, Any]:
        errors = list(state.get("errors", []))
        query = _primary_maintenance_query(state)
        docs = tools.search_maintenance_documents(
            query, errors, retriever=retriever)
        return {"maintenance_evidence": docs, "errors": errors}

    return _node


def check_maintenance_retrieval(state: AgentState) -> dict[str, Any]:
    """No-op passthrough node — exists purely as a named, traced routing
    junction (see graph.py) that both the deterministic
    gather_maintenance_evidence and the planner's
    finalize_supplementary_evidence feed into, so
    maintenance_retrieval_weak's conditional edge only has to be wired
    once regardless of which branch gathered evidence. Shows up in the
    node-execution trace log (graph.py's _traced wrapper) as its own
    step, which is the point: "was retrieval checked and, if so, was it
    retried" should be an observable fact about a specific investigation,
    not something inferred from timing.
    """
    return {}


def maintenance_retrieval_weak(state: AgentState) -> str:
    """Conditional-edge selector (audit F6's 'retrieval weak ->
    reformulate query -> retry once' loop). "Weak" here means empty:
    search_maintenance_documents already applies a score floor (audit
    F3) and returns [] rather than irrelevant top-k padding, so an empty
    result is a clean, unambiguous signal — no threshold-guessing needed
    here. Bounded to exactly one retry via maintenance_retry_used, set by
    retry_maintenance_evidence — this function itself has no memory of
    its own, so the bound holds even if this were ever reached twice in
    one run.
    """
    if state.get("maintenance_retry_used"):
        return "continue"
    return "retry" if not state.get("maintenance_evidence") else "continue"


def retry_maintenance_evidence(retriever: DocumentRetriever | None):
    """Rule-based (no LLM, no credentials needed) reformulate-and-retry —
    see _reformulated_maintenance_query for the two reformulation rules
    and maintenance_retrieval_weak for when this runs at all. Deliberately
    does not itself loop or re-check: it always sets
    maintenance_retry_used=True so the conditional edge that routed here
    cannot route here again, bounding this to exactly one extra search
    per investigation regardless of the outcome. An empty result even
    after reformulation is not treated as an error — see
    tools.search_maintenance_documents's own docstring: "empty when
    nothing relevant is found" is a valid, reportable outcome, not a
    failure.
    """

    def _node(state: AgentState) -> dict[str, Any]:
        errors = list(state.get("errors", []))
        query, failure_mode = _reformulated_maintenance_query(state)
        docs = tools.search_maintenance_documents(
            query, errors, retriever=retriever, failure_mode=failure_mode
        ) if query else []
        return {
            "maintenance_evidence": docs,
            "maintenance_retry_used": True,
            "errors": errors,
        }

    return _node


def assess_risk_node(thresholds: RiskThresholds | None = None):
    """Deterministic — see app/agents/risk.py. Not a node worth binding
    conn/retriever into; it only reads evidence already in state. Now a
    factory (production-readiness fix 11) so build_graph can hand it this
    tenant's calibrated thresholds once, instead of every call reading the
    demo-fleet module constants."""
    def _node(state: AgentState) -> dict[str, Any]:
        risk_level = assess_risk(state.get("sensor_evidence"),
                                 state.get("anomaly_metrics") or [],
                                 state.get("sensor_trend") or [],
                                 thresholds=thresholds)
        return {"risk_level": risk_level}

    return _node


def identify_root_causes(state: AgentState) -> dict[str, Any]:
    """Deterministic — see synthesis.build_root_cause_candidates. Produces
    hypotheses labeled as such, never a confirmed cause (AI SAFETY /
    DECISION SUPPORT)."""
    candidates = synthesis.build_root_cause_candidates(
        state.get("anomaly_metrics") or [], state.get(
            "maintenance_evidence") or [],
        ai4i_reference_rates=state.get("ai4i_reference_rates"),
    )
    return {"root_cause_candidates": candidates}


def build_recommendation(synthesizer: "GeminiSynthesizer | None" = None):
    """Assembles the final recommendation + full narrative report.

    fleet_scan always uses the deterministic template — see
    synthesis.build_report_with_llm's docstring for why that branch was
    kept out of scope for Gemini. For machine_investigation, if a
    synthesizer was supplied and evidence exists, this tries Gemini
    first and falls back to the deterministic template on ANY failure
    (network, schema mismatch, empty response, whatever) — the person
    asking a question about a machine should never see a 500 because an
    LLM call failed when a perfectly good template answer is available.
    The fallback is logged into state['errors'] so it's visible in the
    response's confidence_limitations, not silently swallowed.
    """

    def _node(state: AgentState) -> dict[str, Any]:
        errors = list(state.get("errors", []))

        if synthesizer is not None and state.get("intent") == "machine_investigation":
            try:
                sections = synthesizer.synthesize(state)
                recommendation, narrative, citations = synthesis.build_report_with_llm(
                    state, sections)
                return {"recommendation": recommendation, "narrative": narrative, "citations": citations, "errors": errors}
            except Exception as exc:  # noqa: BLE001 - any Gemini failure degrades to the template, never crashes
                logger.warning(
                    "Gemini synthesis failed, falling back to deterministic template: %s", exc)
                errors.append(
                    f"Gemini synthesis unavailable, used deterministic report instead: {exc}")

        recommendation, narrative, citations = synthesis.build_report(state)
        return {"recommendation": recommendation, "narrative": narrative, "citations": citations, "errors": errors}

    return _node


# --- Audit F6: supplementary evidence-gathering planner (see app/agents/planner.py) ---


def _evidence_summary(state: AgentState) -> str:
    """Plain-text summary of the sensor/anomaly evidence the mandatory
    deterministic nodes already gathered, handed to the planner LLM so it
    can decide whether supplementary evidence would help — it never sees
    this data as a tool result (there is no tool for it; see
    plan_supplementary_evidence) because it cannot change what it does
    with it: risk_level is computed from the real evidence in state by
    app.agents.risk.assess_risk regardless of what the planner reads
    here or decides to do next.
    """
    sensor = state.get("sensor_evidence") or {}
    trend = state.get("sensor_trend") or []
    anomalies = state.get("anomaly_metrics") or []
    severities = [a.get("severity") for a in anomalies]
    if not sensor and not trend and not anomalies:
        return (
            "Sensor and anomaly data could not be retrieved for this machine "
            "(see errors) — investigate with whatever supplementary evidence "
            "is available."
        )
    return (
        f"Already gathered deterministically (not your decision, not "
        f"re-fetchable): latest health_status="
        f"{sensor.get('health_status', 'unknown')}, "
        f"{len(trend)} recent hourly sensor window(s), "
        f"{len(anomalies)} anomaly event(s) with severities {severities}."
    )


def plan_supplementary_evidence(planner_llm: Any, planner_tools: list[Any]):
    """LLM-driven decision over the two SUPPLEMENTARY evidence tools
    (get_historical_evidence, search_maintenance_documents) — audit F6
    removed get_sensor_evidence/get_anomaly_metrics from what this can
    even call; see app/agents/planner.py's module docstring.

    `planner_llm` (a _GenAIToolCallingChat, see planner.get_planner_llm)
    and `planner_tools` (see planner.build_planner_tools) are both built
    once per request in app.agents.graph.build_graph and closed over here
    — `planner_tools` closes over the same `conn`/`retriever`/`collected`
    objects across every round of this node's invocation, which is how
    evidence a tool call gathers in round 1 is still visible when
    nodes.finalize_supplementary_evidence runs after round 3.

    Builds the initial system+user message on the first call (state has
    no 'messages' yet) — the user message includes _evidence_summary's
    text, since the whole point of restricting this planner to
    supplementary tools is that it should decide FROM the evidence
    already gathered, not in ignorance of it. Otherwise continues the
    existing conversation. Stops calling the LLM once
    state['planner_rounds'] hits the hard cap
    (app.agents.planner.MAX_PLANNER_TOOL_ROUNDS) regardless of what the
    model wants next — a confused model must never hang a request.
    """

    def _node(state: AgentState) -> dict[str, Any]:
        from langchain_core.messages import HumanMessage, SystemMessage

        from app.agents import planner as planner_module

        messages = list(state.get("messages") or [])
        # `messages` is an append-reducer field (see state.py), so this
        # node returns only what is NEW this call, never the whole list.
        new_messages: list[Any] = []
        if not messages:
            messages = [
                SystemMessage(
                    content=planner_module.PLANNER_SYSTEM_INSTRUCTION),
                HumanMessage(
                    content=(
                        f"Machine: {state.get('machine_id')}\n"
                        f"Operator question: {state.get('user_query', '')}\n"
                        f"{_evidence_summary(state)}\n"
                        "Decide whether maintenance history and/or documentation "
                        "search would add value here, call whichever supplementary "
                        "tools you need (or none), then confirm you are done."
                    )
                ),
            ]
            new_messages = list(messages)

        rounds = state.get("planner_rounds", 0)
        if rounds >= planner_module.MAX_PLANNER_TOOL_ROUNDS:
            logger.warning(
                "planner round cap (%d) reached for machine=%s, stopping without "
                "further tool calls",
                planner_module.MAX_PLANNER_TOOL_ROUNDS,
                state.get("machine_id"),
            )
            return {"messages": new_messages, "planner_done": True, "planner_rounds": rounds}

        bound_llm = planner_llm.bind_tools(planner_tools)
        try:
            response = bound_llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - a dead/slow planner LLM degrades the plan, never the request
            # The genai client carries a timeout and a bounded retry
            # (app/core/genai_client.py); if the call still fails, stop
            # planning instead of letting the exception 500 the whole
            # investigation. Mandatory sensor/anomaly evidence is already
            # in state, so risk is unaffected. Whatever the planner did
            # not collect stays empty, and the rule-based retrieval retry
            # (check_maintenance_retrieval) still gets its shot at the docs.
            logger.warning(
                "planner LLM call failed for machine=%s, continuing without "
                "further supplementary planning: %s",
                state.get("machine_id"), exc,
            )
            errors = list(state.get("errors", []))
            errors.append(
                f"Supplementary evidence planner unavailable ({type(exc).__name__}); "
                "maintenance history was not gathered by the planner.")
            return {
                "messages": new_messages,
                "planner_done": True,
                "planner_rounds": rounds,
                "errors": errors,
            }
        done = not getattr(response, "tool_calls", None)
        return {
            "messages": new_messages + [response],
            "planner_rounds": rounds + 1,
            "planner_done": done,
        }

    return _node


def planner_should_continue(state: AgentState) -> str:
    """Conditional-edge selector for graph.py: "continue" routes to the
    ToolNode that executes the planner's requested tool calls; "done"
    routes to finalize_supplementary_evidence."""
    return "done" if state.get("planner_done") else "continue"


def finalize_supplementary_evidence(collected: dict[str, Any], planner_errors: list[str]):
    """Merge the supplementary evidence the planner's tool calls
    collected (see planner.build_planner_tools) into the same AgentState
    keys the deterministic fallback (gather_historical_evidence,
    gather_maintenance_evidence) populates, so
    retry_maintenance_evidence / assess_risk / identify_root_causes /
    build_recommendation don't need to know which path gathered the
    evidence — they read state the same way either way.

    `collected` and `planner_errors` are the same dict/list objects
    app.agents.graph.build_graph passed into planner.build_planner_tools,
    mutated in place across every planner round — this node just reads
    their final contents once the loop ends.

    A tool the planner never called leaves its AgentState key empty
    ([]) rather than erroring — that's a real, and now low-stakes,
    behavioural difference from calling both unconditionally (audit F6:
    it used to also be able to leave sensor_evidence/anomaly_metrics
    empty, which is what made this high-stakes; those two are no longer
    reachable from here at all — see plan_supplementary_evidence). No
    errors entry is appended for a skipped historical/maintenance call:
    unlike the pre-F6 version of this function, there is nothing to warn
    about here, because nothing here can affect risk_level.
    """

    def _node(state: AgentState) -> dict[str, Any]:
        errors = list(state.get("errors", [])) + planner_errors
        return {
            "historical_evidence": collected.get("historical_evidence", []),
            "maintenance_evidence": collected.get("maintenance_evidence", []),
            "errors": errors,
        }

    return _node
