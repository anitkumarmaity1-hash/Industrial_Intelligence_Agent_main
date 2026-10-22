"""
API routes.

The read endpoints (Phase 4) stayed exactly as written: thin wrappers
that validate/parse the request, call one `app.database.queries`
function, map `None`/`[]` to the right HTTP status, serialize with the
matching Pydantic model. No query logic or business logic lives here —
that discipline is what let Phase 5 add /investigate without touching
any of them.

/investigate (Phase 5) is the one endpoint that isn't a thin wrapper: it
runs the LangGraph agent (app.agents.graph.run_investigation) and maps
its final state onto InvestigateResponse. All the actual investigation
logic — evidence gathering, risk assessment, root-cause candidates,
report synthesis — lives in app/agents/, not here; this function is
still just request validation in, agent state out.

Tenant isolation (production-readiness fixes 1 and 3): every endpoint here
takes `tenant: TenantContext = Depends(get_tenant)` and passes
`tenant.tenant_id` to every query and to the agent. The tenant comes from
the caller's API key only; there is no request field that names one. The
two LLM-backed endpoints additionally run `rate_limit_llm`.

/chat (Phase 9 fix — was a Phase 4 stub) runs the same LangGraph agent
/investigate does, single-turn: no session/history state, no follow-up
memory. That's a deliberate scope call, not an oversight — true
multi-turn conversation (tracking what "it" refers to across messages,
re-using earlier evidence) needs session storage this MVP has nowhere to
put and the master prompt's example user journeys are all single-shot
questions anyway, so it would be exactly the unnecessary complexity the
master prompt says to avoid. /chat exists, and does real evidence-backed
investigation, it just doesn't remember the previous message. Revisit
only if the portfolio demo specifically needs multi-turn follow-ups.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from langgraph.errors import GraphRecursionError
from sqlalchemy.engine import Connection

from app.api.dependencies import (
    get_conn,
    get_evidence_planner_llm,
    get_gemini_synthesizer,
    get_search_retriever,
    get_tenant,
    get_tenant_calibration,
    rate_limit_llm,
)
from app.api.schemas import (
    AnomalyOut,
    ChatRequest,
    ChatResponse,
    HealthOut,
    InspectionRankingOut,
    InvestigateRequest,
    InvestigateResponse,
    MachineOut,
    ReadingOut,
    SensorPointOut,
    SensorRegistryOut,
    Severity,
)
from app.core.tenancy import TenantContext
from app.core.tenant_settings import TenantCalibration
from app.core.request_context import get_request_id
from app.database import queries

from app.agents.graph import run_investigation
from app.agents.query_parser import (
    MachineIdScheme, QueryResolutionError, ResolvedTarget, resolve_target,
)
from app.agents.risk import RiskThresholds
from app.rag.retriever import DocumentRetriever
from app.agents.llm import GeminiSynthesizer
from app.agents.risk import assess_risk
from app.agents import tools as agent_tools
router = APIRouter()

logger = logging.getLogger(__name__)


def _get_machine_or_404(conn: Connection, machine_id: str, tenant_id: str) -> dict:
    """Shared existence check so every {machine_id} route 404s the same way.
    Scoped to the tenant, so another tenant's machine id is indistinguishable
    from one that doesn't exist."""
    machine = queries.get_machine(conn, machine_id, tenant_id=tenant_id)
    if machine is None:
        raise HTTPException(
            status_code=404, detail=f"Machine {machine_id!r} not found.")
    return machine


def _resolve_target(
    conn: Connection,
    question: str,
    machine_id: str | None,
    tenant_id: str,
    scheme: MachineIdScheme | None = None,
) -> ResolvedTarget:
    """Decide which machine (or the whole fleet) a question is about.

    Audit F1: the target used to come only from the optional machine_id
    field, so the question text was never read. Now the question is
    parsed too (app/agents/query_parser.py) and the two are reconciled:
    an unknown machine is a 404, and a conflicting or ambiguous request
    is a 422 rather than a silent guess. One query (the 18-row fleet
    list) serves both the explicit-id check and the text validation.

    scheme: this tenant's machine-ID scheme (production-readiness fix 12),
    or None to use the demo fleet's "M-NN" convention.
    """
    known = {m["machine_id"]
             for m in queries.list_machines(conn, tenant_id=tenant_id)}
    if machine_id is not None and machine_id not in known:
        raise HTTPException(
            status_code=404, detail=f"Machine {machine_id!r} not found.")
    try:
        return resolve_target(question, machine_id, known, scheme)
    except QueryResolutionError as exc:
        raise HTTPException(status_code=exc.status_code,
                            detail=str(exc)) from exc


@router.get("/machines", response_model=list[MachineOut])
def list_machines(
    conn: Connection = Depends(get_conn),
    tenant: TenantContext = Depends(get_tenant),
) -> list[dict]:
    """All machines in the caller's fleet."""
    return queries.list_machines(conn, tenant_id=tenant.tenant_id)


@router.get("/machines/rank-for-inspection", response_model=list[InspectionRankingOut])
def rank_machines_for_inspection(
    since: datetime = Query(
        description="Rank by anomalies detected at or after this timestamp."),
    top_n: int = Query(default=5, ge=1, le=18),
    conn: Connection = Depends(get_conn),
    tenant: TenantContext = Depends(get_tenant),
) -> list[dict]:
    """Deterministic "inspect first" ranking by anomaly volume/severity
    since a given time (P2 cleanup — see app.database.queries.
    rank_machines_for_inspection's own docstring: written and tested in
    Phase 2, never exposed by any endpoint or called by the agent).

    Registered ahead of GET /machines/{machine_id} below so this literal
    path is matched first — FastAPI resolves routes in registration
    order, and {machine_id} would otherwise swallow "rank-for-inspection"
    as a machine_id.

    Deliberately separate from the agent's fleet_scan (POST /investigate
    or /chat with no machine_id), not a replacement for it: fleet_scan
    flags machines with the same risk_level rule a single-machine
    investigation uses (app.agents.risk.assess_risk, audit F9), which is
    the right rule for "does this machine need investigating". This
    endpoint answers a narrower, cheaper question — raw recent anomaly
    volume/severity, e.g. for a maintenance queue — without running the
    LangGraph agent at all.
    """
    return queries.rank_machines_for_inspection(
        conn, since=since, top_n=top_n, tenant_id=tenant.tenant_id)


@router.get("/machines/{machine_id}", response_model=MachineOut)
def get_machine(
    machine_id: str,
    conn: Connection = Depends(get_conn),
    tenant: TenantContext = Depends(get_tenant),
) -> dict:
    """Single machine's static metadata."""
    return _get_machine_or_404(conn, machine_id, tenant.tenant_id)


@router.get("/machines/{machine_id}/health", response_model=HealthOut)
def get_machine_health(
    machine_id: str,
    conn: Connection = Depends(get_conn),
    tenant: TenantContext = Depends(get_tenant),
) -> dict:
    """Latest hourly sensor_summary snapshot for one machine, plus its
    risk_level.

    Two distinct 404 cases, reported with different detail so a caller
    can tell them apart: the machine doesn't exist at all, vs. it exists
    but sensor_summary has no rows for it yet.

    Audit F8 (finding 3): risk_level is computed here by the exact same
    function, over the exact same evidence, that a machine_investigation
    uses (app.agents.risk.assess_risk fed by app.agents.nodes.
    gather_sensor_evidence / gather_anomaly_metrics's own default
    windows) — not a second, differently-tuned notion of "is this machine
    okay". Always a concrete LOW/MEDIUM/HIGH/UNKNOWN here (never the
    schema's None case): assess_risk only returns UNKNOWN, not None, and
    `health` being non-None at this point (we already 404'd otherwise)
    means assess_risk always has at least the snapshot to reason about.
    A dropped sensor_trend/anomalies query degrades gracefully instead —
    assess_risk falls back to its conservative snapshot-only rule (see
    its own docstring) rather than raising, and the endpoint's primary
    payload (the health snapshot) still succeeds; failures are logged,
    same as every other tools.* call's degrade-not-crash contract.
    """
    tenant_id = tenant.tenant_id
    _get_machine_or_404(conn, machine_id, tenant_id)
    health = queries.get_machine_health(conn, machine_id, tenant_id=tenant_id)
    if health is None:
        raise HTTPException(
            status_code=404, detail=f"No sensor data for machine {machine_id!r} yet.")

    errors: list[str] = []
    # Same default windows nodes.gather_sensor_evidence / gather_anomaly_metrics
    # use for a machine_investigation's mandatory evidence — matching them is
    # what makes this risk_level agree with /investigate's, not just resemble it.
    sensor_trend = agent_tools.get_sensor_summary(
        conn, machine_id, errors, tenant_id=tenant_id)
    anomaly_metrics = agent_tools.get_machine_anomalies(
        conn, machine_id, errors, tenant_id=tenant_id)
    if errors:
        logger.warning(
            "risk_level computation for %s degraded: %s", machine_id, "; ".join(errors))

    return {**health, "risk_level": assess_risk(health, anomaly_metrics, sensor_trend)}


@router.get("/machines/{machine_id}/anomalies", response_model=list[AnomalyOut])
def get_machine_anomalies(
    machine_id: str,
    since: datetime | None = Query(
        default=None, description="Only anomalies detected at or after this timestamp."),
    severity: Severity | None = Query(
        default=None, description="Restrict to MEDIUM or HIGH."),
    limit: int = Query(default=100, ge=1, le=1000),
    conn: Connection = Depends(get_conn),
    tenant: TenantContext = Depends(get_tenant),
) -> list[dict]:
    """Flagged anomaly events for one machine, most recent first."""
    _get_machine_or_404(conn, machine_id, tenant.tenant_id)
    return queries.get_machine_anomalies(
        conn, machine_id, since=since, severity=severity, limit=limit,
        tenant_id=tenant.tenant_id)


@router.get("/machines/{machine_id}/sensors", response_model=list[SensorPointOut])
def get_machine_sensors(
    machine_id: str,
    since: datetime | None = Query(
        default=None, description="Only windows starting at or after this timestamp."),
    until: datetime | None = Query(
        default=None, description="Only windows starting at or before this timestamp."),
    limit: int = Query(
        default=168, ge=1, le=2000, description="Max hourly windows, most recent first (default: 7 days)."),
    conn: Connection = Depends(get_conn),
    tenant: TenantContext = Depends(get_tenant),
) -> list[dict]:
    """Hourly sensor_summary history for one machine (audit F11: backs the
    Streamlit trend chart — see app.database.queries.get_sensor_summary,
    which already existed and was already used internally by the agent's
    evidence-gathering tools, but had no endpoint of its own)."""
    _get_machine_or_404(conn, machine_id, tenant.tenant_id)
    return queries.get_sensor_summary(
        conn, machine_id, since=since, until=until, limit=limit,
        tenant_id=tenant.tenant_id)


def _run_agent(question: str, machine_id: str | None, conn: Connection,
               retriever: DocumentRetriever, synthesizer, planner_llm,
               tenant_id: str, risk_thresholds: RiskThresholds | None = None) -> dict:
    """run_investigation with the one failure mode the graph can raise on
    its own (exceeding its superstep ceiling) mapped to a clean 500."""
    try:
        return run_investigation(
            question, machine_id, conn=conn, retriever=retriever,
            synthesizer=synthesizer, planner_llm=planner_llm, tenant_id=tenant_id,
            risk_thresholds=risk_thresholds,
        )
    except GraphRecursionError as exc:
        logger.error("investigation exceeded its step budget: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="The investigation exceeded its step budget and was stopped.",
        ) from exc


def _audit(conn: Connection, tenant: TenantContext, endpoint: str,
           question: str, target: ResolvedTarget, result: dict) -> None:
    """Best-effort audit trail (production-readiness fix 19) — who queried
    what, with what result. Logged and swallowed on failure: an audit
    insert must never turn a successful investigation into a 500."""
    try:
        queries.insert_investigation_audit(
            conn,
            tenant_id=tenant.tenant_id,
            request_id=get_request_id(),
            endpoint=endpoint,
            machine_id=target.machine_id,
            intent=result.get("intent"),
            risk_level=result.get("risk_level"),
            authenticated=tenant.authenticated,
            question=question,
        )
        conn.commit()
    except Exception:  # noqa: BLE001 - audit logging is best-effort
        logger.exception("Failed to write investigation audit log.")


MAX_STORED_RESULT_BYTES = 64 * 1024


def _store_investigation(conn: Connection, tenant: TenantContext, endpoint: str,
                         question: str, target: ResolvedTarget, response: dict) -> None:
    """Keep the full result for the dashboard's history (Phase 6). Best-effort like
    _audit: a failed insert is logged and swallowed. The stored JSON is capped; an
    oversized narrative is truncated and flagged rather than dropping the row."""
    try:
        result = dict(response)
        if len(json.dumps(result, default=str).encode()) > MAX_STORED_RESULT_BYTES:
            result["narrative"] = str(result.get("narrative", ""))[:20000] + "\n\n[truncated]"
            result["truncated"] = True
            if len(json.dumps(result, default=str).encode()) > MAX_STORED_RESULT_BYTES:
                result["fleet_ranking"] = None
        user_id = getattr(tenant, "user_id", None)
        label = getattr(tenant, "email", None) or getattr(tenant, "kind", None) or "unknown"
        queries.insert_investigation(
            conn, str(uuid.uuid4()), created_by=user_id, created_by_label=label,
            endpoint=endpoint, question=question, machine_id=target.machine_id,
            intent=response.get("intent"), risk_level=response.get("risk_level"),
            result=result, tenant_id=tenant.tenant_id)
        conn.commit()
    except Exception:  # noqa: BLE001 - history is best-effort
        logger.exception("Failed to store the investigation result.")
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass


@router.post("/investigate", response_model=InvestigateResponse, dependencies=[Depends(rate_limit_llm)])
def investigate(
    payload: InvestigateRequest,
    conn: Connection = Depends(get_conn),
    tenant: TenantContext = Depends(get_tenant),
    calibration: TenantCalibration = Depends(get_tenant_calibration),
    retriever: DocumentRetriever = Depends(get_search_retriever),
    synthesizer: GeminiSynthesizer | None = Depends(get_gemini_synthesizer),
    planner_llm=Depends(get_evidence_planner_llm),
) -> dict:
    target = _resolve_target(
        conn, payload.question, payload.machine_id, tenant.tenant_id,
        calibration.machine_id_scheme)

    result = _run_agent(
        payload.question, target.machine_id, conn, retriever,
        synthesizer, planner_llm, tenant.tenant_id, calibration.risk_thresholds,
    )
    _audit(conn, tenant, "investigate", payload.question, target, result)

    response = {
        "machine_id": result.get("machine_id"),
        "question": payload.question,
        "intent": result["intent"],
        "risk_level": result.get("risk_level"),
        "root_cause_candidates": result.get("root_cause_candidates", []),
        "supporting_citations": result.get("citations", []),
        "fleet_ranking": result.get("fleet_ranking"),
        "recommendation": result["recommendation"],
        "narrative": result["narrative"],
        "limitations_note": "; ".join([*target.notes, *result.get("errors", [])]),
    }
    _store_investigation(conn, tenant, "investigate", payload.question, target, response)
    return response


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(rate_limit_llm)])
def chat(
    payload: ChatRequest,
    conn: Connection = Depends(get_conn),
    tenant: TenantContext = Depends(get_tenant),
    calibration: TenantCalibration = Depends(get_tenant_calibration),
    retriever: DocumentRetriever = Depends(get_search_retriever),
    synthesizer: GeminiSynthesizer | None = Depends(get_gemini_synthesizer),
    planner_llm=Depends(get_evidence_planner_llm),
) -> dict:
    """Single-turn conversational access to the same agent /investigate
    runs — evidence gathering, risk assessment and synthesis are
    identical, only the request/response shape is chat-flavored. See
    this module's docstring for why it's single-turn."""
    target = _resolve_target(
        conn, payload.message, payload.machine_id, tenant.tenant_id,
        calibration.machine_id_scheme)

    result = _run_agent(
        payload.message, target.machine_id, conn, retriever,
        synthesizer, planner_llm, tenant.tenant_id, calibration.risk_thresholds,
    )
    _audit(conn, tenant, "chat", payload.message, target, result)

    response = {
        "machine_id": result.get("machine_id"),
        "reply": result["recommendation"],
        "narrative": result["narrative"],
        "intent": result["intent"],
        "risk_level": result.get("risk_level"),
        "supporting_citations": result.get("citations", []),
        "limitations_note": "; ".join([*target.notes, *result.get("errors", [])]),
    }
    _store_investigation(
        conn, tenant, "chat", payload.message, target,
        {"machine_id": response["machine_id"], "question": payload.message,
         "intent": response["intent"], "risk_level": response["risk_level"],
         "root_cause_candidates": result.get("root_cause_candidates", []),
         "supporting_citations": response["supporting_citations"],
         "fleet_ranking": result.get("fleet_ranking"),
         "recommendation": response["reply"], "narrative": response["narrative"],
         "limitations_note": response["limitations_note"]})
    return response


@router.get("/sensors", response_model=list[SensorRegistryOut])
def list_sensors(
    conn: Connection = Depends(get_conn),
    tenant: TenantContext = Depends(get_tenant),
) -> list[dict]:
    """The caller's declared sensors (sensor_registry): unit, normal range,
    z-score override, enabled. Only ever this tenant's rows."""
    return queries.list_sensor_registry(conn, tenant_id=tenant.tenant_id)


@router.get("/machines/{machine_id}/readings", response_model=list[ReadingOut])
def get_machine_readings(
    machine_id: str,
    sensor: str | None = Query(default=None, description="Restrict to one sensor_name."),
    since: datetime | None = Query(default=None),
    until: datetime | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
    conn: Connection = Depends(get_conn),
    tenant: TenantContext = Depends(get_tenant),
) -> list[dict]:
    """Raw long-format readings for one machine, newest first (404 for a
    machine that is not in the caller's own fleet)."""
    _get_machine_or_404(conn, machine_id, tenant.tenant_id)
    return queries.get_sensor_readings(
        conn, machine_id, sensor_name=sensor, since=since, until=until, limit=limit,
        tenant_id=tenant.tenant_id)
