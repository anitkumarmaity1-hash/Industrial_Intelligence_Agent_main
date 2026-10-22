"""
LangGraph state for the investigation agent — Phase 5.

A TypedDict, not a Pydantic model: LangGraph merges each node's return
value into this dict key-by-key on every superstep, which is the update
model StateGraph is built around. Validation of the *inputs* (machine_id,
question) already happens one layer up, in the Phase 4 Pydantic schemas
(app/api/schemas.py) before the graph ever runs — this state does not
re-validate them.

Every field here maps directly onto a concept the master prompt names
explicitly in its AGENT STATE section. Nothing is added "just in case":

    user_query, machine_id          -> input
    intent                          -> routing decision (deterministic, see nodes.py)
    sensor_evidence                 -> get_machine_health (latest snapshot)
    sensor_trend                    -> get_sensor_summary (recent hourly windows)
    historical_evidence             -> get_maintenance_records
    maintenance_evidence            -> search_maintenance_documents results
    anomaly_metrics                 -> get_machine_anomalies
    fleet_ranking                   -> fleet-wide questions only (no single machine_id)
    ai4i_reference_rates            -> get_ai4i_failure_mode_rates (real-dataset base rates)
    risk_level                      -> deterministic, computed in app/agents/risk.py
    root_cause_candidates           -> deterministic, computed in nodes.py
    recommendation, citations       -> output
    errors                          -> per-tool failures, so a partial DB/RAG
                                        outage degrades the investigation
                                        instead of crashing it (see nodes.py)

Audit F6 revision. Evidence-gathering is no longer a strict either/or
between "the Phase 5 fixed chain" and "the Phase 10 planner decides
everything" — see app/agents/graph.py's module docstring for the current
shape. Two things changed in AgentState because of it:

  * sensor_evidence / sensor_trend / anomaly_metrics are now ALWAYS
    populated by mandatory deterministic nodes before either the
    planner or the deterministic fallback for historical/maintenance
    evidence runs — a planner (real or scripted-broken) can no longer
    leave risk.assess_risk with nothing to reason about.
  * maintenance_retry_used (new field, see below) tracks the rule-based
    "retrieval came back empty -> reformulate -> retry once" loop that
    now runs after historical/maintenance evidence is gathered, whether
    that gathering was done by the fixed chain or the planner.

The messages / planner_rounds / planner_done trio below is used ONLY
when the supplementary-evidence planner actually runs (Gemini
configured and AGENTIC_ROUTING_ENABLED isn't explicitly set to false —
see app/core/config.py's agentic_routing_configured). When it doesn't,
these are never set and the deterministic historical+maintenance path
runs instead:

    messages          -> the planner LLM's tool-calling conversation
                          (LangChain BaseMessage objects). The ONE reducer
                          field (`add_messages`, append-by-id): nodes
                          return only the NEW messages. It has to be,
                          because LangGraph's ToolNode returns just its
                          own ToolMessages — without a reducer those
                          REPLACED the history, so from round 2 the model
                          saw only function responses (no question, no
                          function-call turn, no thought signatures).
    planner_rounds     -> how many planner LLM calls have happened this
                          investigation, so the hard round cap
                          (planner.MAX_PLANNER_TOOL_ROUNDS) is enforceable
                          from state rather than a closure variable that
                          wouldn't survive a resumed/replayed run. This
                          only applies to the round *count*: the evidence
                          the planner's tools write as they run
                          (`collected`, in planner.build_planner_tools)
                          is still a per-request closure, same as nodes.py
                          uses for conn/retriever — it's read back into
                          state once by finalize_supplementary_evidence,
                          not carried across a resume, so it doesn't need
                          the same replay-safety planner_rounds does.
    planner_done        -> True once the planner LLM stopped requesting
                          tool calls (or hit the round cap). Read by
                          graph.py's conditional edge out of
                          plan_supplementary_evidence.

Audit F8 fix (finding 1: "What changed?" led with defect-rate noise and
missed the real signal). sensor_trend stays exactly as it was — it's
what risk.assess_risk's sustained-rate calibration is tuned against
(see risk.py's module docstring), and widening it would silently change
that calibration. sensor_trend_baseline is a second, longer fetch
(app.agents.nodes.gather_sensor_evidence, app.agents.synthesis's
RECENT_WINDOW_COUNT / BASELINE_WINDOW_COUNT) used ONLY to build the
deterministic last-24h-vs-prior-baseline comparison
(synthesis.compute_recent_vs_baseline) that both build_report and the
Gemini prompt (app/agents/llm.py) render into the "What changed?"
section. Same most-recent-first ordering as sensor_trend.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages

Intent = Literal["machine_investigation", "fleet_scan"]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]


class AgentState(TypedDict, total=False):
    # --- input ------------------------------------------------------
    user_query: str
    machine_id: str | None

    # --- routing ------------------------------------------------------
    intent: Intent

    # --- evidence gathered by tool nodes -------------------------------
    # latest snapshot (get_machine_health)
    sensor_evidence: dict[str, Any] | None
    # recent hourly windows, most-recent-first (get_sensor_summary) — feeds
    # risk.assess_risk; leave the window this covers alone (see above)
    sensor_trend: list[dict[str, Any]]
    # longer hourly history, most-recent-first, for the "What changed?"
    # baseline comparison only (audit F8) — never read by risk.assess_risk
    sensor_trend_baseline: list[dict[str, Any]]
    historical_evidence: list[dict[str, Any]]
    maintenance_evidence: list[dict[str, Any]]
    anomaly_metrics: list[dict[str, Any]]
    fleet_ranking: list[dict[str, Any]]
    # real-world (UCI AI4I 2020) per-failure-mode incidence rates — P2
    # cleanup, see app.database.queries.get_ai4i_failure_mode_rates and
    # synthesis.build_root_cause_candidates. Fleet-wide, not per-machine;
    # unset for fleet_scan, which doesn't build root-cause candidates.
    ai4i_reference_rates: dict[str, dict[str, Any]]

    # --- analysis: deterministic, code-computed (never LLM-guessed) ----
    risk_level: RiskLevel | None
    root_cause_candidates: list[str]

    # --- output ---------------------------------------------------------
    recommendation: str
    narrative: str
    citations: list[str]

    # --- error tracking ---------------------------------------------------
    errors: list[str]

    # --- Audit F6: rule-based maintenance-doc retry -----------------------
    # Set once the reformulate-and-retry step (app.agents.nodes.
    # retry_maintenance_evidence) has run, so the conditional edge that
    # triggers it (maintenance_retrieval_weak) fires at most once per
    # investigation regardless of which branch gathered evidence.
    maintenance_retry_used: bool

    # --- supplementary-evidence planner only (see module docstring) --------
    # LangChain BaseMessage objects; Any to avoid a hard langchain_core import here
    messages: Annotated[list[Any], add_messages]
    planner_rounds: int
    planner_done: bool
