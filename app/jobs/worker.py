"""Background worker: a Postgres-backed queue (SELECT .. FOR UPDATE SKIP LOCKED).

Phase 5: the same loop also ticks the connector scheduler (app/connectors/scheduler.py) every
CONNECTOR_TICK_SECONDS; `--once` drains jobs only and does not run connectors.

    python -m app.jobs.worker            # poll forever (docker-compose `worker`)
    python -m app.jobs.worker --once     # drain what is queued, then exit

* Connects as `iia_worker` (WORKER_DATABASE_URL): a non-owner role subject to
  RLS. Starting on a role that bypasses RLS is refused when RLS_REQUIRED
  (default on for a remote database), exactly like the API.
* Claiming is the ONE cross-tenant step. It goes through the SECURITY DEFINER
  function iia_claim_job() (migration 0007), which returns only
  (job_id, tenant_id). Everything after that runs with that tenant bound.
* Failure handling: a PermanentJobError fails the job at once with its
  tenant-safe message. Any other exception is transient: the job is re-queued
  with exponential backoff (base * 2^retry_count, capped at 5 min) until
  `max_retries` is used up, then fails. The stored error text is generic for
  transient failures (the detail, which can name internal hosts, is logged only).
* A worker that dies mid-job leaves status='running' with a stale heartbeat
  (jobs.locked_at, refreshed at every stage); iia_reap_stale_jobs() re-queues
  or fails such jobs after JOB_STALE_SECONDS.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.core.config import get_settings
from app.connectors.scheduler import tick as connector_tick
from app.connectors.settings import connector_settings
from app.core.ingest_settings import ingest_settings
from app.core.logging_config import setup_logging
from app.database import queries
from app.database.tenant_context import bypasses_rls, install
from app.jobs.pipeline import PermanentJobError, run_job, tenant_conn
from app.storage import StorageProvider, get_storage

logger = logging.getLogger(__name__)

MAX_BACKOFF_SECONDS = 300.0
_stop = False


def make_engine() -> Engine:
    s, cfg = ingest_settings(), get_settings()
    url = s.worker_database_url or cfg.database_url
    if not url:
        raise RuntimeError("WORKER_DATABASE_URL (or DATABASE_URL) is not set")
    engine = install(create_engine(url, pool_pre_ping=True))
    if bypasses_rls(engine):
        msg = ("The worker's database role bypasses row-level security (superuser or BYPASSRLS). "
               "Set WORKER_DATABASE_URL to the iia_worker role (scripts/provision_app_role.py).")
        if cfg.rls_required:
            raise RuntimeError(msg)
        logger.warning(msg)
    return engine


def claim_job(engine: Engine) -> tuple[str, str] | None:
    with engine.begin() as conn:
        row = conn.execute(text("SELECT job_id, tenant_id FROM iia_claim_job()")).first()
    return (str(row[0]), row[1]) if row else None


def reap_stale(engine: Engine, stale_seconds: int) -> int:
    with engine.begin() as conn:
        return conn.execute(text("SELECT iia_reap_stale_jobs(:s)"), {"s": stale_seconds}).scalar() or 0


def backoff_seconds(retry_count: int, base: float) -> float:
    return min(MAX_BACKOFF_SECONDS, base * (2 ** retry_count))


def record_failure(engine: Engine, job_id: str, tenant_id: str, exc: BaseException) -> str:
    """Re-queue or fail the job. Returns 'requeued' or 'failed'."""
    s = ingest_settings()
    permanent = isinstance(exc, PermanentJobError)
    with tenant_conn(engine, tenant_id) as conn:
        job = queries.get_job(conn, job_id, tenant_id=tenant_id)
        if job is None:
            return "failed"
        if not permanent and job["retry_count"] < job["max_retries"]:
            delay = backoff_seconds(job["retry_count"], s.job_backoff_base_seconds)
            queries.requeue_job(
                conn, job_id, delay_seconds=delay, tenant_id=tenant_id,
                error=f"transient error ({type(exc).__name__}); retry {job['retry_count'] + 1} "
                      f"of {job['max_retries']} scheduled")
            return "requeued"
        message = str(exc)[:1000] if permanent else (
            f"transient error ({type(exc).__name__}); gave up after {job['retry_count']} retr"
            f"{'y' if job['retry_count'] == 1 else 'ies'}")
        queries.finish_job(conn, job_id, status="failed", error=message, tenant_id=tenant_id)
        return "failed"


def process_one(engine: Engine, storage: StorageProvider | None = None, spark_factory=None) -> bool:
    """Claim and run one job. Returns False when the queue was empty."""
    claimed = claim_job(engine)
    if claimed is None:
        return False
    job_id, tenant_id = claimed
    logger.info("claimed job %s (tenant %s)", job_id, tenant_id)
    try:
        run_job(engine, job_id, tenant_id, storage=storage, spark_factory=spark_factory)
    except Exception as exc:  # noqa: BLE001 - every failure is recorded on the job
        outcome = record_failure(engine, job_id, tenant_id, exc)
        log = logger.warning if isinstance(exc, PermanentJobError) else logger.exception
        log("job %s %s: %s", job_id, outcome, exc)
    return True


def _handle_signal(signum, _frame) -> None:
    global _stop
    _stop = True
    logger.info("signal %s received; finishing the current job, then exiting", signum)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", help="process the queue until empty, then exit")
    args = parser.parse_args(argv)
    setup_logging()
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    s = ingest_settings()
    engine = make_engine()
    storage = get_storage()
    logger.info("worker started (storage=%s, poll=%ss)", s.storage_local_dir, s.worker_poll_seconds)

    last_reap = 0.0
    last_tick = float("-inf")
    tick_every = connector_settings().tick_seconds
    while not _stop:
        if time.monotonic() - last_reap > max(30.0, s.job_stale_seconds / 4):
            n = reap_stale(engine, s.job_stale_seconds)
            if n:
                logger.warning("re-queued/failed %d stale job(s)", n)
            last_reap = time.monotonic()
        if not args.once and time.monotonic() - last_tick >= tick_every:
            last_tick = time.monotonic()
            try:
                connector_tick(engine, storage=storage)
            except Exception:  # noqa: BLE001 - never stop the job loop for a connector problem
                logger.exception("connector tick error")
        try:
            worked = process_one(engine, storage)
        except Exception:  # noqa: BLE001 - e.g. database briefly unreachable; keep the loop alive
            logger.exception("worker loop error")
            worked = False
            time.sleep(s.worker_poll_seconds)
        if not worked:
            if args.once:
                break
            time.sleep(s.worker_poll_seconds)
    engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())