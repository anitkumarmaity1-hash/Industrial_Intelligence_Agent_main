"""
Request/trace-ID propagation — production-readiness fix 14.

Structured per-node/tool logging already existed (app/core/logging_config.py,
app/agents/graph.py's `_traced`, app/agents/tools.py's `_log_tool_call`), but
none of it carried anything that let you pull every log line for ONE
request back out of a shared process's interleaved output — the audit's
own "no correlation/request ID threading" finding.

`app.main`'s middleware sets a request ID here (generated, or taken from
an inbound X-Request-ID so a caller's own trace ID survives the hop) as
early as possible in the request lifecycle. A contextvar, not a thread-
local: FastAPI/Starlette run request handling on asyncio tasks, and
contextvars — unlike threading.local — are correctly isolated per task
even when multiple requests interleave on the same thread. Everything
downstream — deep in the agent graph, in a tool call, in the audit-log
insert in app/api/routes.py — reads it back with `get_request_id()`
without needing it threaded through every function signature.
"""

from __future__ import annotations

import contextvars

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


def set_request_id(value: str | None) -> None:
    _request_id.set(value)


def get_request_id() -> str | None:
    return _request_id.get()


class RequestIdLogFilter:
    """Attached to the root handler by logging_config.setup_logging() so
    every LogRecord gets a `request_id` attribute the format string can
    reference — "-" outside any request (startup, a background script)."""

    def filter(self, record) -> bool:  # noqa: A003 - logging.Filter's own name
        record.request_id = get_request_id() or "-"
        return True
