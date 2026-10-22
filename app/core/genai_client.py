"""
One place that builds google-genai (Vertex AI) clients.

Synthesis (app/agents/llm.py), the evidence planner (app/agents/planner.py)
and the embedder (app/rag/embeddings.py) all used to construct
`genai.Client(vertexai=True, ...)` with no timeout and no retry policy, so
a hung Vertex call would hold a request worker indefinitely and a single
transient 429/503 failed the whole call. They now share this constructor:

  * a per-request deadline (Settings.llm_timeout_seconds), and
  * a bounded retry with exponential backoff on the SDK's default
    retryable statuses — 408, 429 and 5xx (Settings.llm_max_attempts is
    the total attempt count, so 2 means one retry).

Failures that survive the retries still raise. What happens next is the
caller's existing degrade path: synthesis falls back to the deterministic
report (nodes.build_recommendation), the planner falls back to the
rule-based evidence path (nodes.plan_supplementary_evidence), and the
embedder's failure surfaces as a retrieval error the retriever already
turns into a clean 503 / BM25 fallback.

`google-genai` is imported lazily, same as everywhere else, so the app
still runs with the cloud extras uninstalled.
"""

from __future__ import annotations

from typing import Any

from app.core.config import Settings, get_settings


def build_http_options(settings: Settings | None = None) -> Any:
    """google.genai.types.HttpOptions carrying this app's timeout/retry policy."""
    from google.genai import types

    settings = settings or get_settings()
    timeout_ms = max(1, int(settings.llm_timeout_seconds * 1000))
    attempts = max(1, settings.llm_max_attempts)
    return types.HttpOptions(
        timeout=timeout_ms,
        retry_options=types.HttpRetryOptions(
            attempts=attempts,
            initial_delay=1.0,
            max_delay=8.0,
            exp_base=2.0,
            jitter=0.5,
        ),
    )


def build_genai_client(project: str, location: str, settings: Settings | None = None) -> Any:
    """A Vertex-backed genai client with the shared timeout/retry policy.

    Raises ImportError if google-genai isn't installed (callers already
    handle that as "feature unavailable").
    """
    from google import genai

    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=build_http_options(settings),
    )
