"""The connector scheduler: a tick inside the worker loop, state in Postgres.

Each connector row has `next_run_at`. A tick claims due connectors through the SECURITY DEFINER
function iia_claim_due_connectors(): one atomic UPDATE .. FOR UPDATE SKIP LOCKED that stamps a
lease (`claimed_at`) and returns only ids. Two worker processes ticking at the same moment get
disjoint sets; a crashed process's lease simply expires. After a run, `next_run_at` is set from
the cron expression. "Run now" is just `next_run_at = now` (+ a flag so the run is labelled manual).

The claim is the one cross-tenant step; everything after it runs through run_pull with that
connector's tenant bound for RLS, as a system principal.
"""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.connectors.base import StoreFactory
from app.connectors.puller import run_pull
from app.connectors.settings import connector_settings
from app.core.secrets_provider import SecretsProvider
from app.storage import StorageProvider

logger = logging.getLogger(__name__)


def claim_due(engine: Engine, lease_seconds: int, limit: int) -> list[tuple[str, str, bool]]:
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT connector_id, tenant_id, manual FROM iia_claim_due_connectors(:l, :n)"),
            {"l": lease_seconds, "n": limit}).all()
    return [(str(r[0]), r[1], bool(r[2])) for r in rows]


def tick(engine: Engine, *, storage: StorageProvider | None = None,
         store_factory: StoreFactory | None = None, secrets: SecretsProvider | None = None,
         now: datetime | None = None) -> int:
    """Claim and run due connectors. Returns how many were run."""
    cfg = connector_settings()
    claimed = claim_due(engine, cfg.lease_seconds, cfg.claim_batch)
    for connector_id, tenant_id, manual in claimed:
        try:
            run_pull(engine, connector_id, tenant_id, trigger="manual" if manual else "schedule",
                     storage=storage, store_factory=store_factory, secrets=secrets, now=now)
        except Exception as exc:  # noqa: BLE001 - one connector never stops the others
            logger.warning("connector %s: run failed (%s)", connector_id, type(exc).__name__)
    return len(claimed)
