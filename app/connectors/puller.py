"""One pull of one connector: list new files, ledger them, hand each to the Phase 3 pipeline.

Runs in the worker as the `iia_worker` DB role and a SYSTEM principal (no human, no API key).
Every database call goes through `tenant_conn`, which binds the connector's tenant for
Postgres RLS: the scheduler never touches another tenant's rows.

How a file becomes a job (and why nothing is ingested twice):
  * connector_objects UNIQUE (tenant, connector, key, version) is the ledger. A listed
    (key, version) already 'queued' or 'rejected' is skipped.
  * For a new one the file is downloaded and stored like a manual upload, then ONE
    transaction creates the `uploads` row (with the connector's mapping snapshot), the
    `jobs` row and the ledger row. If another scheduler won the race the ledger insert
    reports it, the transaction rolls back and the stored copy is deleted: no duplicate
    upload, job or row. The Phase 3 pipeline is idempotent on top of that.
  * The listing watermark advances only past files whose job is accepted (or that are
    permanently rejected). A transiently failed or too-young file holds it, so it is
    seen again; a permanently bad file does not block the ones after it.

Every message stored or logged here is fixed text: no exception text is ever copied.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy.engine import Engine

from app.connectors.base import (
    ConnectorError,
    ObjectInfo,
    ObjectStore,
    StoreFactory,
)
from app.connectors.settings import connector_settings
from app.connectors.validation import matches, next_run
from app.core import audit
from app.core.ingest_settings import ingest_settings
from app.core.principal import Principal
from app.core.secrets_provider import SecretError, SecretsProvider, get_secrets_provider
from app.database import queries
from app.database.tenant_context import bind_tenant
from app.jobs.pipeline import tenant_conn
from app.storage import StorageError, StorageProvider, get_storage, new_upload_id, upload_key

logger = logging.getLogger(__name__)

SPOOL_BYTES = 8 * 1024 * 1024
LOOKUP_CHUNK = 500


@dataclass
class PullResult:
    status: str                 # ok | partial | failed
    seen: int = 0
    new: int = 0
    queued: int = 0
    failed: int = 0
    message: str | None = None


class _RaceLost(Exception):
    """Another scheduler already queued this object version."""


def system_principal(tenant_id: str) -> Principal:
    return Principal(tenant_id=tenant_id, authenticated=True, role=None, kind="system")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _hash_and_rewind(spool) -> str:
    digest = hashlib.sha256()
    spool.seek(0)
    for chunk in iter(lambda: spool.read(1024 * 1024), b""):
        digest.update(chunk)
    spool.seek(0)
    return digest.hexdigest()


def _record_failure(engine, c, o: ObjectInfo, tenant_id: str, error: str, permanent: bool) -> str:
    with tenant_conn(engine, tenant_id) as conn:
        status = queries.record_connector_object_failure(
            conn, str(c["id"]), object_key=o.key, object_version=o.version, size_bytes=o.size,
            object_updated=o.updated, error=error, permanent=permanent,
            max_attempts=connector_settings().object_max_attempts, tenant_id=tenant_id)
    return status or "rejected"


def _ingest_object(engine: Engine, store: ObjectStore, storage: StorageProvider, c: dict,
                   o: ObjectInfo, tenant_id: str) -> str:
    """-> 'queued' | 'retry' | 'rejected' | 'duplicate'. Raises only a fatal ConnectorError."""
    max_bytes = ingest_settings().upload_max_bytes
    if o.size > max_bytes:
        return _record_failure(engine, c, o, tenant_id, "the file exceeds the size limit", True)
    upload_id = new_upload_id()
    key = upload_key(tenant_id, upload_id)
    try:
        with tempfile.SpooledTemporaryFile(max_size=SPOOL_BYTES) as spool:
            store.download(o, spool, max_bytes)
            size = spool.tell()
            if size == 0:
                return _record_failure(engine, c, o, tenant_id, "the file is empty", True)
            digest = _hash_and_rewind(spool)
            storage.put(key, spool, tenant_id=tenant_id)
    except ConnectorError as exc:
        if exc.fatal:
            raise
        return _record_failure(engine, c, o, tenant_id, exc.message, exc.permanent)
    except StorageError:
        return _record_failure(engine, c, o, tenant_id, "could not store the downloaded file", False)
    except Exception as exc:  # noqa: BLE001 - outside the store contract; never block the other files
        logger.warning("connector %s: unexpected error fetching a file (%s)", c["id"], type(exc).__name__)
        return _record_failure(engine, c, o, tenant_id, "unexpected error while fetching the file", False)

    job_id = str(uuid.uuid4())
    try:
        with tenant_conn(engine, tenant_id) as conn:
            queries.create_upload(conn, upload_id, (os.path.basename(o.key) or "object.csv")[:255],
                                  "text/csv", key, tenant_id=tenant_id)
            queries.set_upload_content(conn, upload_id, size, digest, tenant_id=tenant_id)
            queries.set_upload_mapping(conn, upload_id, c["mapping"], tenant_id=tenant_id)
            queries.create_job(conn, job_id, upload_id, ingest_settings().job_max_retries,
                               tenant_id=tenant_id)
            if not queries.mark_connector_object_queued(
                    conn, str(c["id"]), object_key=o.key, object_version=o.version,
                    size_bytes=size, object_updated=o.updated, upload_id=upload_id,
                    job_id=job_id, tenant_id=tenant_id):
                raise _RaceLost()
    except _RaceLost:
        _discard(storage, key, tenant_id)
        return "duplicate"
    except Exception:  # noqa: BLE001 - e.g. database briefly unavailable: leave it to the next run
        _discard(storage, key, tenant_id)
        raise
    return "queued"


def _discard(storage: StorageProvider, key: str, tenant_id: str) -> None:
    try:
        storage.delete(key, tenant_id=tenant_id)
    except Exception:  # noqa: BLE001 - an orphaned file is harmless (random key, never referenced)
        logger.warning("could not remove an unreferenced stored file")


def _fail(message: str) -> tuple[PullResult, datetime | None]:
    return PullResult("failed", message=message), None


def _pull(engine: Engine, c: dict, tenant_id: str, storage: StorageProvider,
          store_factory: StoreFactory, secrets: SecretsProvider,
          now: datetime) -> tuple[PullResult, datetime | None]:
    cfg = connector_settings()
    secret = None
    if c["secret_ref"]:
        try:
            secret = secrets.get(c["secret_ref"])
        except SecretError:
            return _fail("the configured secret is not available")
    try:
        store = store_factory(c, secret)
        listed = list(store.list_objects(c["config"].get("prefix", "")))
    except ConnectorError as exc:
        return _fail(exc.message)
    except Exception as exc:  # noqa: BLE001
        logger.warning("connector %s: listing failed (%s)", c["id"], type(exc).__name__)
        return _fail("unexpected error while listing objects")

    wm = c["list_watermark"]
    floor = wm - timedelta(seconds=cfg.overlap_seconds) if wm else None
    eligible = sorted((o for o in listed if matches(c["config"], o.key)
                       and (floor is None or o.updated >= floor)),
                      key=lambda o: (o.updated, o.key))
    result = PullResult("ok", seen=len(eligible))

    known: dict[tuple[str, str], dict] = {}
    keys = [o.key for o in eligible]
    with tenant_conn(engine, tenant_id) as conn:
        for i in range(0, len(keys), LOOKUP_CHUNK):
            for r in queries.get_connector_objects(conn, str(c["id"]), keys[i:i + LOOKUP_CHUNK],
                                                   tenant_id=tenant_id):
                known[(r["object_key"], r["object_version"])] = r

    new_wm: datetime | None = None
    blocked = False
    attempted = 0
    deadline = time.monotonic() + cfg.run_max_seconds
    too_young = now - timedelta(seconds=cfg.min_age_seconds)
    for o in eligible:
        row = known.get((o.key, o.version))
        if row and row["status"] in ("queued", "rejected"):
            resolved = True
        elif o.updated > too_young or attempted >= cfg.max_objects_per_run \
                or time.monotonic() > deadline:
            resolved = False                      # not yet / not this run: holds the watermark
        else:
            attempted += 1
            result.new += 1
            try:
                outcome = _ingest_object(engine, store, storage, c, o, tenant_id)
            except ConnectorError as exc:         # fatal: credentials revoked mid-run, etc.
                result.status, result.message = "failed", exc.message
                return result, new_wm
            if outcome == "queued":
                result.queued += 1
            elif outcome in ("retry", "rejected"):
                result.failed += 1
            resolved = outcome != "retry"
            with tenant_conn(engine, tenant_id) as conn:
                queries.touch_connector_claim(conn, str(c["id"]), tenant_id=tenant_id)
        if not resolved:
            blocked = True
        elif not blocked:
            new_wm = o.updated

    if result.failed:
        result.status = "partial"
        result.message = f"{result.queued} file(s) queued, {result.failed} file(s) failed"
    elif result.queued:
        result.message = f"{result.queued} file(s) queued"
    else:
        result.message = "no new files"
    return result, new_wm


def run_pull(engine: Engine, connector_id: str, tenant_id: str, *, trigger: str = "schedule",
             storage: StorageProvider | None = None, store_factory: StoreFactory | None = None,
             secrets: SecretsProvider | None = None, now: datetime | None = None
             ) -> PullResult | None:
    """Pull one connector. Records a run, updates the connector, writes an audit row.
    Returns None if the connector no longer exists. Never raises for connector-level problems."""
    from app.connectors.gcs import make_gcs_store
    storage = storage or get_storage()
    store_factory = store_factory or make_gcs_store
    secrets = secrets or get_secrets_provider()
    now = now or _utcnow()
    run_id = str(uuid.uuid4())

    with tenant_conn(engine, tenant_id) as conn:
        c = queries.get_connector(conn, connector_id, tenant_id=tenant_id)
        if c is None:
            return None
        queries.create_connector_run(conn, run_id, connector_id, trigger, tenant_id=tenant_id)

    try:
        result, wm = _pull(engine, c, tenant_id, storage, store_factory, secrets, now)
    except Exception as exc:  # noqa: BLE001 - a database problem mid-run; the lease will expire
        logger.warning("connector %s: pull aborted (%s)", connector_id, type(exc).__name__)
        result, wm = PullResult("failed", message="the pull was interrupted by an internal error"), None

    try:
        nxt = next_run(c["schedule"], now)
    except ValueError:
        nxt = None
    with tenant_conn(engine, tenant_id) as conn:
        queries.finish_connector_run(
            conn, run_id, status=result.status, objects_seen=result.seen, objects_new=result.new,
            objects_queued=result.queued, objects_failed=result.failed, message=result.message,
            tenant_id=tenant_id)
        queries.finish_connector_state(
            conn, connector_id, last_status=result.status, last_message=result.message,
            next_run_at=nxt, list_watermark=wm, tenant_id=tenant_id)

    with engine.connect() as aconn:           # audit.record commits by itself
        bind_tenant(aconn, tenant_id)
        audit.record(aconn, system_principal(tenant_id), "connector.pull",
                     "failure" if result.status == "failed" else "success",
                     resource=connector_id, actor="scheduler",
                     detail={"trigger": trigger, "status": result.status, "seen": result.seen,
                             "queued": result.queued, "failed": result.failed})
    logger.info("connector %s (tenant %s): %s - %s", connector_id, tenant_id, result.status,
                result.message)
    return result
