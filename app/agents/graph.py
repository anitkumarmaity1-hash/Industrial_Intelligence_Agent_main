"""
LangGraph assembly — audit F6 revision.

Current graph shape — sensor and anomaly evidence are now gathered
deterministically and unconditionally for every machine_investigation,
never a tool either the deterministic fallback or the planner can skip.
What varies is only the two SUPPLEMENTARY evidence types (maintenance
history, doc search): a Gemini-backed planner decides whether/how to
gather them when configured (feature-detected — see
app/core/config.py's agentic_routing_configured), otherwise a
deterministic fallback gathers both unconditionally, the same way the
mandatory pair always does. Either branch then passes through the same
rule-based maintenance-doc retry loop before risk/root-cause/synthesis,
which are identical either way and must stay deterministic (or, for
synthesis, its own separately-gated Gemini call) regardless of how
evidence was gathered:

    START
      |
    route_intent
      |
      +-- fleet_scan --------------------------------------> build_recommendation --> END
      |
      +-- machine_investigation
             |
           gather_sensor_evidence        (mandatory, deterministic — audit F2/F6)
             |
           gather_anomaly_metrics        (mandatory, deterministic — audit F2/F6)
             |
             +-- supplementary planner configured (agentic_routing_configured) --------+
             |                                                                         |
             |   plan_supplementary_evidence  <---------------+   (LLM decides among   |
             |     |                                          |    2 tools; see        |
             |     +-- continue --> execute_supplementary_tools-+   app/agents/planner) |
             |     |                                                                   |
             |     +-- done                                                            |
             |           |                                                             |
             |         finalize_supplementary_evidence                                 |
             |                                                                         |
             +-- not configured --------------------------------------+                |
             |                                                        |                |
             |   gather_historical_evidence                           |                |
             |     |                                                  |                |
             |   gather_maintenance_evidence (query from anomaly log) |                |
             |                                                        |                |
             +--------------------------------------------------------+----------------+
                   |
                 check_maintenance_retrieval        (routing junction, both branches land here)
                   |
                   +-- retrieval weak (empty) & not yet retried --> retry_maintenance_evidence --+
                   |                                                                              |
                   +-- otherwise ----------------------------------------------------------------+
                         |
                       assess_risk_node              (deterministic, risk.py)
                         |
                       identify_root_causes          (deterministic, synthesis.py)
                         |
                       build_recommendation          (deterministic template, or a separately-
                         |                             gated Gemini call — app/agents/llm.py)
                       END

Which of the two evidence-gathering branches gets built is decided once
per request in build_graph, based on whether a planner LLM was actually
constructed (see app.agents.planner.get_planner_llm) — never a runtime
branch inside the compiled graph, so a request either runs the
LLM-planned supplementary path end-to-end or the deterministic fallback
end-to-end, never a mix. The mandatory pair and the retry loop are not
part of that choice at all — they run the same way regardless.

`conn` and `retriever` are request-scoped (a pooled DB connection, a
possibly-cached retriever), so the graph is built fresh per investigation
in `build_graph()` rather than compiled once at import time and reused —
building a LangGraph StateGraph is cheap (it's wiring function references,
not opening connections), so this costs nothing measurable and avoids
carrying request-scoped state across requests.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from sqlalchemy.engine import Connection

from app.agents import nodes
from app.agents.risk import RiskThresholds
from app.agents.state import AgentState, Intent
from app.core.config import get_settings
from app.core.tenancy import DEFAULT_TENANT_ID
from app.rag.retriever import DocumentRetriever

logger = logging.getLogger(__name__)

NodeFn = Callable[[AgentState], dict[str, Any]]


def _traced(node_name: str, fn: NodeFn) -> NodeFn:
    """Wrap a node function so every node's execution is observable without
    touching each node's own body (nodes.py stays deterministic-logic-only).

    Logs, per node, per investigation: which node ran, the machine_id (or
    "fleet" for fleet-wide questions) it ran for, how long it took, and
    whether it raised. This is the "which agent node executed" /
    "how long important operations took" requirement from the master
    prompt's OBSERVABILITY section — nodes.py and tools.py already log
    their own tool-level failures; this adds the node-level trace around
    them.
    """

    def _wrapped(state: AgentState) -> dict[str, Any]:
        machine_id = state.get("machine_id") or "fleet"
        started = time.perf_counter()
        logger.info("node start: %s machine=%s", node_name, machine_id)
        try:
            result = fn(state)
        except Exception:
            elapsed = time.perf_counter() - started
            logger.exception(
                "node failed: %s machine=%s after %.3fs",
                node_name, machine_id, elapsed,
            )
            raise
        elapsed = time.perf_counter() - started
        logger.info(
            "node done: %s machine=%s in %.3fs", node_name, machine_id, elapsed,
        )
        return result

    return _wrapped


def _route_from_intent(state: AgentState) -> Intent:
    """Conditional-edge selector reads the routing decision route_intent
    already wrote into state — it does not recompute it."""
    return state["intent"]


def build_graph(
    conn: Connection,
    retriever: DocumentRetriever | None = None,
    synthesizer=None,
    machine_id: str | None = None,
    planner_llm: Any | None = None,
    tenant_id: str = DEFAULT_TENANT_ID,
    risk_thresholds: RiskThresholds | None = None,
):
    """Construct and compile the investigation graph for one request.

    tenant_id scopes every database read the graph makes. The routes always
    pass the authenticated tenant explicitly; the default exists so direct
    callers and tests keep working, and it fails safe — it can only ever
    read the demo tenant's rows, never another tenant's. (The retriever is
    scoped separately, by construction: see app.rag.retriever.get_retriever.)

    risk_thresholds: this tenant's calibrated RiskThresholds (production-
    readiness fix 11), or None to use the demo-fleet defaults — see
    app.core.tenant_settings.load_tenant_calibration, which routes.py
    calls once per request and passes straight through here.

    machine_id and planner_llm are both optional — every existing caller
    that only passes conn/retriever/synthesizer still gets the full
    deterministic path (mandatory sensor/anomaly evidence, deterministic
    historical+maintenance-doc fallback, retry loop). The supplementary
    LLM-planned path (see module docstring) is only built when BOTH
    machine_id is given (fleet_scan has no single machine to plan around)
    AND planner_llm is not None (i.e. app.agents.planner.get_planner_llm()
    actually returned a usable chat client — see that function's
    degrade-to-None contract). Note what this choice does NOT affect:
    gather_sensor_evidence and gather_anomaly_metrics are added and wired
    identically either way — see module docstring.
    """
    graph = StateGraph(AgentState)
    use_supplementary_planner = planner_llm is not None and machine_id is not None

    graph.add_node("route_intent", _traced("route_intent", nodes.route_intent))
    graph.add_node("fleet_scan", _traced(
        "fleet_scan", nodes.fleet_scan(conn, tenant_id, risk_thresholds)))
    graph.add_node("assess_risk", _traced(
        "assess_risk", nodes.assess_risk_node(risk_thresholds)))
    graph.add_node("identify_root_causes", _traced(
        "identify_root_causes", nodes.identify_root_causes))
    graph.add_node("build_recommendation", _traced(
        "build_recommendation", nodes.build_recommendation(synthesizer)))

    # Audit F2/F6: mandatory, deterministic, unconditional — neither the
    # planner branch below nor the deterministic fallback can skip these;
    # they are simply not part of that choice. See module docstring.
    graph.add_node("gather_sensor_evidence", _traced(
        "gather_sensor_evidence", nodes.gather_sensor_evidence(conn, tenant_id)))
    graph.add_node("gather_anomaly_metrics", _traced(
        "gather_anomaly_metrics", nodes.gather_anomaly_metrics(conn, tenant_id)))

    # Audit F6: rule-based (no LLM) retrieval retry — shared routing
    # junction both evidence-gathering branches feed into below.
    graph.add_node("check_maintenance_retrieval", _traced(
        "check_maintenance_retrieval", nodes.check_maintenance_retrieval))
    graph.add_node("retry_maintenance_evidence", _traced(
        "retry_maintenance_evidence", nodes.retry_maintenance_evidence(retriever)))

    graph.add_edge(START, "route_intent")
    graph.add_edge("fleet_scan", "build_recommendation")
    graph.add_edge("gather_sensor_evidence", "gather_anomaly_metrics")
    graph.add_conditional_edges(
        "check_maintenance_retrieval",
        nodes.maintenance_retrieval_weak,
        {"retry": "retry_maintenance_evidence", "continue": "assess_risk"},
    )
    graph.add_edge("retry_maintenance_evidence", "assess_risk")
    graph.add_edge("assess_risk", "identify_root_causes")
    graph.add_edge("identify_root_causes", "build_recommendation")
    graph.add_edge("build_recommendation", END)

    graph.add_conditional_edges(
        "route_intent",
        _route_from_intent,
        {"fleet_scan": "fleet_scan",
            "machine_investigation": "gather_sensor_evidence"},
    )

    if use_supplementary_planner:
        # LLM-driven supplementary evidence gathering. `collected` and
        # `planner_errors` are created once here, per request, and closed
        # over by both build_planner_tools (which the LLM calls, round
        # after round) and finalize_supplementary_evidence (which reads
        # the final result once the loop ends) — see planner.py's module
        # docstring for why this needed to be build-time-scoped state
        # rather than threaded through AgentState like everything else.
        from langgraph.prebuilt import ToolNode

        from app.agents import planner as planner_module

        collected: dict[str, Any] = {}
        planner_errors: list[str] = []
        planner_tools = planner_module.build_planner_tools(
            conn, retriever, machine_id, planner_errors, collected,
            tenant_id=tenant_id,
        )

        graph.add_node("plan_supplementary_evidence", _traced(
            "plan_supplementary_evidence", nodes.plan_supplementary_evidence(planner_llm, planner_tools)))
        graph.add_node("execute_supplementary_tools", ToolNode(
            planner_tools, messages_key="messages"))
        graph.add_node("finalize_supplementary_evidence", _traced(
            "finalize_supplementary_evidence", nodes.finalize_supplementary_evidence(collected, planner_errors)))

        graph.add_edge("gather_anomaly_metrics", "plan_supplementary_evidence")
        graph.add_conditional_edges(
            "plan_supplementary_evidence",
            nodes.planner_should_continue,
            {"continue": "execute_supplementary_tools",
                "done": "finalize_supplementary_evidence"},
        )
        graph.add_edge("execute_supplementary_tools",
                       "plan_supplementary_evidence")
        graph.add_edge("finalize_supplementary_evidence",
                       "check_maintenance_retrieval")
    else:
        # Deterministic fallback for the two supplementary evidence types
        # (default when the planner isn't configured).
        graph.add_node("gather_historical_evidence", _traced(
            "gather_historical_evidence", nodes.gather_historical_evidence(conn, tenant_id)))
        graph.add_node("gather_maintenance_evidence", _traced(
            "gather_maintenance_evidence", nodes.gather_maintenance_evidence(retriever)))

        # Anomaly metrics are gathered before maintenance evidence because
        # gather_maintenance_evidence builds its retrieval query from the
        # anomaly log's triggered_reasons (see nodes.py) — order here is a
        # real data dependency, not just documentation order.
        graph.add_edge("gather_anomaly_metrics", "gather_historical_evidence")
        graph.add_edge("gather_historical_evidence",
                       "gather_maintenance_evidence")
        graph.add_edge("gather_maintenance_evidence",
                       "check_maintenance_retrieval")

    return graph.compile()


def run_investigation(
    question: str,
    machine_id: str | None,
    conn: Connection,
    retriever: DocumentRetriever | None = None,
    synthesizer=None,
    planner_llm: Any | None = None,
    tenant_id: str = DEFAULT_TENANT_ID,
    risk_thresholds: RiskThresholds | None = None,
) -> AgentState:
    """Entry point used by the /investigate and /chat endpoints
    (app/api/routes.py) and by tests. Builds the graph, runs it to
    completion, and returns the final state as a plain dict.

    planner_llm is optional — omitting it (the default) preserves the
    exact deterministic behaviour (mandatory sensor/anomaly evidence,
    deterministic historical+maintenance-doc fallback) every existing
    caller and test relies on. Pass app.agents.planner.get_planner_llm()'s
    result to use the LLM-planned supplementary-evidence path when it's
    configured.

    tenant_id: see build_graph. risk_thresholds: see build_graph
    (production-readiness fix 11). Raises langgraph's GraphRecursionError
    if the graph exceeds Settings.graph_recursion_limit supersteps.
    """
    compiled = build_graph(conn, retriever, synthesizer,
                           machine_id, planner_llm, tenant_id, risk_thresholds)
    initial_state: AgentState = {
        "user_query": question,
        "machine_id": machine_id,
        "errors": [],
    }
    started = time.perf_counter()
    logger.info("investigation start: machine=%s question=%r",
                machine_id or "fleet", question)
    # Hard superstep ceiling (Settings.graph_recursion_limit), independent
    # of the planner's own round cap: the longest legitimate run is ~18
    # supersteps, so hitting this means a wiring bug, and it surfaces as a
    # GraphRecursionError (mapped to a clean 500 in app/api/routes.py)
    # instead of an unbounded loop.
    result: dict[str, Any] = compiled.invoke(
        initial_state,
        config={"recursion_limit": get_settings().graph_recursion_limit},
    )
    elapsed = time.perf_counter() - started
    logger.info(
        "investigation done: machine=%s risk=%s errors=%d in %.3fs",
        machine_id or "fleet",
        result.get("risk_level"),
        len(result.get("errors", [])),
        elapsed,
    )
    return result  # type: ignore[return-value]
