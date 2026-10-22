"""Validation of what a tenant admin may configure on a connector.

The config is built SERVER-SIDE from three validated fields (bucket, prefix, pattern); there is
no free-form config blob, so no endpoint URL and no credential-shaped key can be stored.
"""

from __future__ import annotations

import fnmatch
import re
from datetime import datetime, timezone

from cronsim import CronSim, CronSimError

BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]$")
PREFIX_RE = re.compile(r"^[A-Za-z0-9._/ -]{0,512}$")
PATTERN_RE = re.compile(r"^[A-Za-z0-9._*?/\[\]-]{1,100}$")
MAX_NAME = 100


def build_gcs_config(bucket: str, prefix: str, pattern: str) -> dict[str, str]:
    if not BUCKET_RE.match(bucket or "") or ".." in bucket:
        raise ValueError("bucket is not a valid GCS bucket name")
    prefix = prefix or ""
    if not PREFIX_RE.match(prefix) or prefix.startswith("/") or ".." in prefix:
        raise ValueError("prefix may use letters, digits and . _ - / space; no leading '/', no '..'")
    if not PATTERN_RE.match(pattern or "") or ".." in pattern:
        raise ValueError("pattern is a glob of letters, digits and . _ - / * ? [ ] (no '..')")
    return {"bucket": bucket, "prefix": prefix, "pattern": pattern}


def matches(config: dict, key: str) -> bool:
    """Does this object key belong to the connector (prefix + glob on the part after it)?"""
    prefix = config.get("prefix", "")
    if not key.startswith(prefix) or key.endswith("/"):
        return False
    return fnmatch.fnmatchcase(key[len(prefix):], config.get("pattern", "*.csv"))


def _occurrences(expr: str, after: datetime, n: int) -> list[datetime]:
    try:
        it = CronSim(expr, after.replace(tzinfo=timezone.utc))
        return [next(it).astimezone(timezone.utc).replace(tzinfo=None) for _ in range(n)]
    except (CronSimError, StopIteration, ValueError, OverflowError):
        raise ValueError("schedule is not a valid 5-field cron expression") from None


def validate_schedule(expr: str, min_interval_seconds: int) -> None:
    if not isinstance(expr, str) or len(expr) > 100 or len(expr.split()) != 5:
        raise ValueError("schedule must be a 5-field cron expression (minute hour day month weekday), UTC")
    times = _occurrences(expr, datetime.now(timezone.utc).replace(tzinfo=None), 6)
    gap = min((b - a).total_seconds() for a, b in zip(times, times[1:]))
    if gap < min_interval_seconds:
        raise ValueError(f"schedule runs more often than every {min_interval_seconds} seconds")


def next_run(expr: str, after: datetime) -> datetime:
    """Next scheduled time strictly after `after` (naive UTC)."""
    return _occurrences(expr, after, 1)[0]
