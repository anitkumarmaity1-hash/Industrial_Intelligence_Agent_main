"""Small test helpers shared across modules."""

from __future__ import annotations

import os
from contextlib import contextmanager

from app.core.config import get_settings


@contextmanager
def env(**values: str | None):
    """Temporarily set (or, with None, unset) environment variables AND
    rebuild the cached Settings, then restore both.

    Settings is cached process-wide (app.core.config.get_settings), so a
    plain monkeypatch.setenv isn't enough to change behaviour mid-test.
    """
    saved = {k: os.environ.get(k) for k in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings(refresh=True)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings(refresh=True)
