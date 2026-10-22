"""General audit trail (Phase 4): login, permission denials, role changes, writes.

Best-effort like the investigation audit: a failed audit insert is logged and
swallowed so it can never turn a good request into a 500. Needs the connection's
tenant to be bound (app.api.dependencies.get_principal does that); without it RLS
rejects the insert, which is also swallowed. Events with no known tenant (a login
for a tenant that does not exist) go to the application log only.
"""

from __future__ import annotations

import logging

from sqlalchemy.engine import Connection

from app.core.request_context import get_request_id
from app.database import queries

logger = logging.getLogger(__name__)


def record(conn: Connection, principal, action: str, outcome: str, *,
           resource: str | None = None, detail: dict | None = None,
           actor: str | None = None) -> None:
    kind = getattr(principal, "kind", "anonymous")
    try:
        queries.insert_audit_log(
            conn,
            tenant_id=principal.tenant_id,
            request_id=get_request_id(),
            principal_kind=kind,
            principal_id=getattr(principal, "user_id", None),
            actor=actor or getattr(principal, "email", None) or kind,
            action=action, resource=(resource or None) and resource[:200],
            outcome=outcome, detail=detail)
        conn.commit()
    except Exception:  # noqa: BLE001 - audit logging is best-effort
        logger.exception("Failed to write audit log (%s %s).", action, outcome)
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
