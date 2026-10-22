"""
Pydantic schemas for the Phase 4 API.

Every field here mirrors a column that already exists in `sql/schema.sql`
and a key already returned by `app/database/queries.py` — nothing is
invented. Types are tightened where the schema has a CHECK constraint
(`health_status`, `severity`) so an invalid value is a 500 that surfaces a
real data bug, not a string FastAPI would silently pass through.

These are output/query models only. `/investigate` gets its own request
model here too; `/chat`'s request/response models are below it.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

HealthStatus = Literal["HEALTHY", "WATCH", "AT_RISK"]
Severity = Literal["MEDIUM", "HIGH"]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]


class MachineOut(BaseModel):
    """Static machine metadata. Backs GET /machines and GET /machines/{id}."""

    machine_id: str
    production_line: str
    type: str | None = None   # NULL for fleets with no AI4I quality variant (migration 0005)
    name: str | None = None
    install_date: date | None = None
    location: str | None = None


class SensorRegistryOut(BaseModel):
    """One sensor a tenant has declared (GET /sensors)."""

    sensor_name: str
    unit: str
    normal_min: float | None = None
    normal_max: float | None = None
    z_score_threshold: float | None = None
    enabled: bool


class ReadingOut(BaseModel):
    """One raw long-format reading (GET /machines/{id}/readings)."""

    sensor_name: str
    ts: datetime
    value: float


class HealthOut(BaseModel):
    """Latest hourly sensor_summary snapshot for one machine, plus the
    same sustained-rate risk_level a machine_investigation would compute.

    `health_status` is still the raw latest window, not a time-smoothed
    verdict — a single snapshot can show a transient WATCH from
    statistical noise (a known Phase 2 finding) — kept as-is rather than
    hidden, since it's a real (if noisy) fact about the most recent hour.

    `risk_level` (audit F8, finding 3) is new: previously this endpoint
    exposed only `health_status`, while the Streamlit Overview tab's
    badge came straight from it and the Investigate tab's badge came from
    app.agents.risk.assess_risk — two different rules for "how healthy is
    this machine", which could and did disagree (M-04 showing 🟡 WATCH on
    Overview and 🔴 HIGH on Investigate for the same underlying data).
    risk_level here is computed by that exact same function
    (app/api/routes.py:get_machine_health), so the two tabs now agree by
    construction rather than by coincidence. None only when there isn't
    enough evidence to assess at all (assess_risk's UNKNOWN case).
    """

    machine_id: str
    window_start: datetime
    window_end: datetime
    health_status: HealthStatus
    max_anomaly_score: int
    anomalous_reading_count: int
    reading_count: int
    avg_defect_rate: float | None = None
    tool_wear_min: float | None = None
    risk_level: RiskLevel | None = None


class AnomalyOut(BaseModel):
    """One flagged anomaly event. Backs GET /machines/{id}/anomalies."""

    detected_at: datetime
    anomaly_score: int
    severity: Severity
    triggered_reasons: str | None = None


class SensorPointOut(BaseModel):
    """One hourly sensor_summary window. Backs GET /machines/{id}/sensors
    (audit F11: the Streamlit Overview tab showed window start/end
    timestamps as "key metrics" with no way to see a trend, because no
    endpoint exposed more than the single latest window that
    GET /machines/{id}/health already returns)."""

    window_start: datetime
    window_end: datetime
    avg_air_temp_k: float | None = None
    avg_process_temp_k: float | None = None
    avg_rotational_speed_rpm: float | None = None
    avg_torque_nm: float | None = None
    tool_wear_min: float | None = None
    avg_production_rate: float | None = None
    avg_energy_consumption_kwh: float | None = None
    avg_defect_rate: float | None = None
    reading_count: int
    anomalous_reading_count: int
    max_anomaly_score: int
    health_status: HealthStatus


Intent = Literal["machine_investigation", "fleet_scan"]


class InspectionRankingOut(BaseModel):
    """One ranked entry. Backs GET /machines/rank-for-inspection (P2
    cleanup: app.database.queries.rank_machines_for_inspection was
    written and tested in Phase 2 but no caller — API or agent — ever
    used it; the agent's own fleet_scan flags machines by a different,
    risk_level-based rule (app.agents.risk.assess_risk, Phase 9), which
    is deliberately NOT replaced here — this endpoint exposes the
    simpler, purely anomaly-volume/severity ranking as its own thing,
    e.g. for a maintenance-queue view, not as a duplicate of fleet_scan)."""

    machine_id: str
    anomaly_count: int
    high_severity_count: int
    most_recent_anomaly: datetime


class InvestigateRequest(BaseModel):
    """Request contract for POST /investigate.

    `machine_id` is optional because the master prompt's example queries
    ("which machines currently show abnormal behavior?") aren't always
    about one machine. The target is resolved from BOTH fields (audit F1,
    app/agents/query_parser.py): a machine named in `question` is used
    even when `machine_id` is omitted, a machine_id that contradicts the
    question is a 422, and an unknown machine is a 404. With neither, the
    graph runs the fleet-wide branch (see app/agents/nodes.py:route_intent).
    """

    machine_id: str | None = Field(
        default=None, description="Machine to investigate, if the question is machine-specific.")
    question: str = Field(
        min_length=1, max_length=2000, description="Natural-language operations question.")


class InvestigateResponse(BaseModel):
    """Response for POST /investigate — the LangGraph agent's final state,
    mapped onto the master prompt's RESPONSE FORMAT.

    `narrative` carries the full markdown report (## Machine / ## Risk /
    etc.); the other fields expose the same information structured, for
    a caller that wants to render its own UI (e.g. the Phase 7 Streamlit
    dashboard) rather than display the markdown as-is.
    """

    machine_id: str | None
    question: str
    intent: Intent
    risk_level: RiskLevel | None = None
    root_cause_candidates: list[str] = Field(default_factory=list)
    supporting_citations: list[str] = Field(default_factory=list)
    fleet_ranking: list[dict] | None = None
    recommendation: str
    narrative: str
    limitations_note: str = Field(
        default="",
        description="Set when one or more evidence sources failed during this investigation.",
    )


class ChatRequest(BaseModel):
    """Request contract for POST /chat. Same shape and target-resolution rules as InvestigateRequest.

    `max_length` on both this and InvestigateRequest.question (audit F10):
    unbounded input meant a 200 KB body previously sailed through as a 200,
    and every extra character in a Gemini-backed request is billed cost and
    added latency. 2000 characters is generous for a natural-language
    operations question with room to paste a few lines of context.
    """

    message: str = Field(min_length=1, max_length=2000)
    machine_id: str | None = None


class ChatResponse(BaseModel):
    """Response for POST /chat (Phase 9 fix).

    /chat runs the same LangGraph agent /investigate does
    (app.agents.graph.run_investigation) — see app/api/routes.py's module
    docstring for why it's single-turn (no session/history state) rather
    than true multi-turn conversational memory. `reply` is the same
    deterministic recommendation text /investigate calls
    `recommendation`; `narrative` is the full markdown report, for a
    client that wants to render more than the one-line reply.
    """

    machine_id: str | None
    reply: str
    narrative: str
    intent: Intent
    risk_level: RiskLevel | None = None
    supporting_citations: list[str] = Field(default_factory=list)
    limitations_note: str = Field(
        default="",
        description="Set when the target was inferred/fell back, or an evidence source failed.",
    )
