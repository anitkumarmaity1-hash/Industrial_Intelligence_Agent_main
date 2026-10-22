"""
Sliding-window rate limiter for the two billed, LLM-backed endpoints
(/investigate and /chat).

Deliberately in-process and dependency-free (no Redis, per the project's
"don't over-engineer" rule): a deque of recent request timestamps per key,
guarded by a lock. The consequences are documented, not hidden:

  * The counters live in one worker process. With N workers/replicas the
    effective limit is N x the configured one. Fine for the single-container
    deployment this repo ships; put a gateway limiter in front if you scale
    out.
  * Counters reset on restart.

Keyed by tenant for authenticated callers, and by (tenant, client IP) for
unauthenticated demo traffic, so one anonymous client can't burn the shared
demo allowance for everyone else.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque


class SlidingWindowRateLimiter:
    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str, limit: int, window_seconds: float = 60.0) -> float | None:
        """Record one request for `key`.

        Returns None if the request is allowed, otherwise the number of
        seconds until it would be (the Retry-After value). `limit <= 0`
        disables limiting entirely.
        """
        if limit <= 0:
            return None
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            cutoff = now - window_seconds
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= limit:
                return max(0.0, hits[0] + window_seconds - now)
            hits.append(now)
            # Opportunistic cleanup so idle keys don't accumulate forever.
            if len(self._hits) > 10_000:
                for stale in [k for k, v in self._hits.items() if not v or v[-1] <= cutoff]:
                    del self._hits[stale]
            return None

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


# Process-wide instance used by app.api.dependencies.rate_limit_llm.
limiter = SlidingWindowRateLimiter()


def retry_after_header(wait_seconds: float) -> str:
    return str(max(1, math.ceil(wait_seconds)))
