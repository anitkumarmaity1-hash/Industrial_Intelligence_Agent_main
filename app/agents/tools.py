"""
Agent tools — Phase 5.

Per the master prompt's TOOLS section, six named tools:
get_machine_health, get_sensor_summary, get_machine_anomalies,
query_machine_history, search_maintenance_documents, get_maintenance_records.

Deliberate decision: query_machine_history and get_maintenance_records
would be the same query (maintenance_records for one machine — there is
no separate "history" table). Rather than write the same SQL twice under
two names, query_machine_history is a thin alias documented as such below.
Five underlying data sources, six names, zero duplicated logic.

Tenant isolation: every tool that reads company data takes a required
keyword-only `tenant_id` and passes it straight to app.database.queries,
which filters on it. (search_maintenance_documents needs no tenant_id: the
retriever it is handed is already scoped to one tenant's documents — see
app.rag.retriever.get_retriever.)

Everything else here is intentionally NOT a new abstraction over
app/database/queries.py and app/rag/retriever.py — those modules are
already the tested tool contracts (queries.py's own docstring says as
much: "the Phase 4 FastAPI endpoints and Phase 5 LangGraph tools will
call [these]"). This module's only job is uniform error handling: every
function here catches its own failures and returns an empty result
instead of raising, so one dead tool degrades an investigation instead of
crashing the whole graph run. Callers (nodes.py) still see clean return
types; failures are pushed onto the caller-supplied `errors` list instead
of propagating as exceptions, which is the contract nodes.py relies on.
"""

from __future__ import annotations

import functools
import logging
import time
from datetime import datetime
from typing import Any, Callable

from sqlalchemy.engine import Connection
from sqlalchemy.exc import SQLAlchemyError

from app.database import queries
from app.rag.retriever import DocumentRetriever
from app.rag.retriever import search_maintenance_documents as _search_maintenance_documents

logger = logging.getLogger(__name__)


def _log_tool_call(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Log "which tool was called" / timing for every tool in this module,
    without duplicating a log line inside each function's own try/except
    (those already log *failures* with the actual exception; this decorator
    adds the call/duration trace uniformly, success or not, and reports the
    size of whatever was returned so a silently-empty result is visible
    without being an error).
    """

    @functools.wraps(fn)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        # Every tool here takes either (conn, machine_id, errors, ...) or
        # (query, errors, ...) — machine_id/query, when present, is always
        # the first positional arg after any connectable.
        label = kwargs.get("machine_id") or (
            args[1] if len(args) > 1 and isinstance(args[1], str) else
            args[0] if args and isinstance(args[0], str) else "-"
        )
        started = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - started
        size = len(result) if isinstance(result, (list, dict, set, tuple)) else (
            1 if result is not None else 0)
        logger.info("tool call: %s target=%s result_size=%d in %.3fs",
                    fn.__name__, label, size, elapsed)
        return result

    return _wrapped


@_log_tool_call
def get_machine_health(
    conn: Connection, machine_id: str, errors: list[str], *, tenant_id: str
) -> dict[str, Any] | None:
    """Latest sensor_summary snapshot. Feeds risk assessment directly."""
    try:
        return queries.get_machine_health(conn, machine_id, tenant_id=tenant_id)
    except SQLAlchemyError as exc:
        logger.warning("get_machine_health failed for %s: %s", machine_id, exc)
        errors.append(
            f"Could not retrieve health snapshot for {machine_id}: {exc}")
        return None


@_log_tool_call
def get_sensor_summary(
    conn: Connection, machine_id: str, errors: list[str], limit: int = 24, *, tenant_id: str
) -> list[dict[str, Any]]:
    """Recent hourly sensor windows. Default limit is 24 (last day at the
    hourly grain this pipeline produces) — enough to show a trend without
    sending the LLM a week of redundant history in Phase 6."""
    try:
        return queries.get_sensor_summary(
            conn, machine_id, limit=limit, tenant_id=tenant_id)
    except SQLAlchemyError as exc:
        logger.warning("get_sensor_summary failed for %s: %s", machine_id, exc)
        errors.append(
            f"Could not retrieve sensor history for {machine_id}: {exc}")
        return []


@_log_tool_call
def get_machine_anomalies(
    conn: Connection, machine_id: str, errors: list[str], limit: int = 20, *, tenant_id: str
) -> list[dict[str, Any]]:
    """Flagged anomaly events. Drives both risk_level and root-cause candidates."""
    try:
        return queries.get_machine_anomalies(
            conn, machine_id, limit=limit, tenant_id=tenant_id)
    except SQLAlchemyError as exc:
        logger.warning(
            "get_machine_anomalies failed for %s: %s", machine_id, exc)
        errors.append(
            f"Could not retrieve anomaly history for {machine_id}: {exc}")
        return []


@_log_tool_call
def get_maintenance_records(
    conn: Connection, machine_id: str, errors: list[str], *, tenant_id: str
) -> list[dict[str, Any]]:
    """Maintenance/repair event log for one machine."""
    try:
        return queries.get_maintenance_records(
            conn, machine_id, tenant_id=tenant_id)
    except SQLAlchemyError as exc:
        logger.warning(
            "get_maintenance_records failed for %s: %s", machine_id, exc)
        errors.append(
            f"Could not retrieve maintenance records for {machine_id}: {exc}")
        return []


def query_machine_history(
    conn: Connection, machine_id: str, errors: list[str], *, tenant_id: str
) -> list[dict[str, Any]]:
    """Alias for get_maintenance_records — see module docstring."""
    return get_maintenance_records(conn, machine_id, errors, tenant_id=tenant_id)


@_log_tool_call
def get_fleet_status(conn: Connection, errors: list[str], *, tenant_id: str) -> list[dict[str, Any]]:
    """Latest health window per machine, fleet-wide. Backs "which machines
    currently show abnormal behavior?" — no machine_id required."""
    try:
        return queries.get_fleet_status(conn, tenant_id=tenant_id)
    except SQLAlchemyError as exc:
        logger.warning("get_fleet_status failed: %s", exc)
        errors.append(f"Could not retrieve fleet status: {exc}")
        return []


@_log_tool_call
def get_fleet_sensor_trend(
    conn: Connection, errors: list[str], *, tenant_id: str
) -> dict[str, list[dict[str, Any]]]:
    """Last 24 hourly windows per machine, fleet-wide. Feeds fleet_scan's
    per-machine risk assessment (Phase 9 fix) so it uses the same
    sustained-rate rule as a single-machine investigation."""
    try:
        return queries.get_fleet_sensor_trend(conn, tenant_id=tenant_id)
    except SQLAlchemyError as exc:
        logger.warning("get_fleet_sensor_trend failed: %s", exc)
        errors.append(f"Could not retrieve fleet sensor trend: {exc}")
        return {}


@_log_tool_call
def get_fleet_recent_high_anomalies(conn: Connection, errors: list[str], *, tenant_id: str) -> set[str]:
    """machine_ids with a HIGH anomaly inside their own trend window,
    fleet-wide. Feeds fleet_scan's per-machine risk assessment alongside
    get_fleet_sensor_trend."""
    try:
        return queries.get_fleet_recent_high_anomalies(
            conn, tenant_id=tenant_id)
    except SQLAlchemyError as exc:
        logger.warning("get_fleet_recent_high_anomalies failed: %s", exc)
        errors.append(f"Could not retrieve fleet anomaly data: {exc}")
        return set()


@_log_tool_call
def get_ai4i_failure_mode_rates(conn: Connection, errors: list[str]) -> dict[str, dict]:
    """Real-world (UCI AI4I 2020) incidence rate per failure mode — see
    queries.get_ai4i_failure_mode_rates. No machine_id: this is one
    fleet-wide, dataset-wide lookup, not per-machine, so it's cheap
    enough to gather unconditionally alongside anomaly evidence."""
    try:
        return queries.get_ai4i_failure_mode_rates(conn)
    except SQLAlchemyError as exc:
        logger.warning("get_ai4i_failure_mode_rates failed: %s", exc)
        errors.append(f"Could not retrieve AI4I reference rates: {exc}")
        return {}


@_log_tool_call
def search_maintenance_documents(
    query: str,
    errors: list[str],
    retriever: DocumentRetriever | None = None,
    top_k: int | None = None,
    failure_mode: str | None = None,
) -> list[dict[str, Any]]:
    """Maintenance-document RAG. Returns plain dicts (not RetrievedChunk
    objects) so nodes.py and the API layer don't import the RAG module's
    internal types — the citation string is precomputed here so callers
    never re-derive it."""
    try:
        results = _search_maintenance_documents(
            query, top_k=top_k, failure_mode=failure_mode, retriever=retriever
        )
    except Exception as exc:  # noqa: BLE001 - any backend failure degrades, doesn't crash
        logger.warning("search_maintenance_documents failed: %s", exc)
        errors.append(f"Maintenance documentation search failed: {exc}")
        return []
    return [
        {
            "doc_id": r.doc_id,
            "title": r.title,
            "doc_type": r.doc_type,
            "section": r.section,
            "text": r.text,
            "score": r.score,
            "citation": r.citation(),
            "failure_modes": r.failure_modes,
        }
        for r in results
    ]
