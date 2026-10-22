"""Connector settings, read fresh from the environment on every call (like ingest_settings)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.config import _env


def _int(name: str, default: int) -> int:
    raw = _env(name)
    return int(raw) if raw is not None else default


@dataclass(frozen=True)
class ConnectorSettings:
    tick_seconds: float          # how often the worker looks for due connectors
    lease_seconds: int           # a claimed connector is not claimed again for this long
    claim_batch: int             # connectors claimed per tick
    max_objects_per_run: int     # new files ingested per pull (the rest wait for the next run)
    min_age_seconds: int         # skip objects younger than this (may still be uploading)
    overlap_seconds: int         # re-list this far before the watermark (clock skew)
    object_max_attempts: int     # transient failures per file before it is rejected
    min_interval_seconds: int    # shortest allowed gap between scheduled runs
    run_max_seconds: int         # time budget of one pull
    max_per_tenant: int          # connectors a tenant may configure
    secrets_dir: Path


def connector_settings() -> ConnectorSettings:
    return ConnectorSettings(
        tick_seconds=float(_env("CONNECTOR_TICK_SECONDS", "30") or 30),
        lease_seconds=_int("CONNECTOR_LEASE_SECONDS", 900),
        claim_batch=_int("CONNECTOR_CLAIM_BATCH", 5),
        max_objects_per_run=_int("CONNECTOR_MAX_OBJECTS_PER_RUN", 25),
        min_age_seconds=_int("CONNECTOR_MIN_AGE_SECONDS", 60),
        overlap_seconds=_int("CONNECTOR_OVERLAP_SECONDS", 3600),
        object_max_attempts=_int("CONNECTOR_OBJECT_MAX_ATTEMPTS", 5),
        min_interval_seconds=_int("CONNECTOR_MIN_INTERVAL_SECONDS", 300),
        run_max_seconds=_int("CONNECTOR_RUN_MAX_SECONDS", 300),
        max_per_tenant=_int("CONNECTOR_MAX_PER_TENANT", 5),
        secrets_dir=Path(_env("CONNECTOR_SECRETS_DIR", "/run/secrets/connectors")),
    )
