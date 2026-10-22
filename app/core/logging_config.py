"""
Central logging configuration — Phase 9 observability fix.

Phase 5-8 code was already calling `logger.info(...)` / `logger.warning(...)`
throughout app/agents, app/rag and app/main.py (see e.g. tools.py's failure
logs, retriever.py's backend-selection logs), but nothing ever called
`logging.basicConfig()` or attached a handler. Python's root logger has no
handler by default, so every one of those calls was silently dropped except
for the rare case a warning happened to hit the interpreter's last-resort
handler. The master prompt's OBSERVABILITY section requires being able to
see which node executed, which tool ran, what machine was analyzed, whether
retrieval succeeded, and operation timings — none of that is possible
without a configured handler, regardless of how many logger.info() calls
exist in the code.

This module is the single place that turns those existing calls into
visible output. It does not add a logging framework or change any log
call site's meaning — it just gives the calls somewhere to go.

Usage: call `setup_logging()` once, as early as possible in the process
(app/main.py does this at import time so it applies under `uvicorn
app.main:app`; a standalone script should call it at the top of __main__).
Calling it more than once is safe — later calls are a no-op unless
force=True.
"""

from __future__ import annotations

import logging
import os

from app.core.request_context import RequestIdLogFilter

_CONFIGURED = False

# Concise, greppable, single-line format. Deliberately no timezone/millis
# bikeshedding — %(asctime)s default is "good enough" for a 2-day MVP and
# matches what `docker compose logs` timestamps look like anyway.
# Production-readiness fix 14: %(request_id)s ties every line to one HTTP
# request (see app.core.request_context) — "-" for anything logged outside
# a request (startup, a standalone script).
_FORMAT = "%(asctime)s %(levelname)-8s [req=%(request_id)s] %(name)s: %(message)s"

# Third-party libraries that are chatty at INFO and drown out app logs.
# Their own errors/warnings still come through.
_QUIET_LOGGERS = ("httpx", "httpcore", "urllib3", "watchfiles")


def setup_logging(level: str | None = None, force: bool = False) -> None:
    """Configure the root logger once for the whole process.

    Args:
        level: Overrides the LOG_LEVEL env var (default "INFO") when given.
            Never read from anywhere secret — this is just a verbosity knob.
        force: Reconfigure even if setup_logging() already ran. Tests that
            need a clean handler set can use this; normal app startup should
            leave it False so a second import doesn't duplicate handlers.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    resolved_level = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()

    logging.basicConfig(
        level=resolved_level,
        format=_FORMAT,
        force=force,  # drop any accidental handlers a library added first
    )

    # Every handler on the root logger gets the filter, not just the one
    # basicConfig() just attached — a test or a library that adds its own
    # handler still gets %(request_id)s populated instead of a KeyError.
    id_filter = RequestIdLogFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(id_filter)

    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger(__name__).info(
        "Logging configured (level=%s).", resolved_level)
