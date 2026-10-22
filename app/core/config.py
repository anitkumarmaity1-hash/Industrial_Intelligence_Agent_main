"""
Central configuration for the Industrial Intelligence Agent.

Everything that differs between a laptop, CI and a cloud deployment lives
here and is read from the environment. No secret is ever hardcoded or
given a working default — the defaults below are non-secret shapes
(paths, model names, region) only.

Phase 2 read DATABASE_URL directly from os.environ in its scripts. From
Phase 3 onwards the number of knobs grows (vector store, embedding model,
Vertex project), so they are collected in one place rather than scattered
across modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from app.core.tenancy import DEFAULT_TENANT_ID, validate_tenant_id

load_dotenv()

# Repository root, resolved from this file's location so scripts work
# regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable, treating empty strings as unset."""
    value = os.environ.get(name, default)
    if value is not None and value.strip() == "":
        return default
    return value


def _env_optional_float(name: str) -> float | None:
    """Float env var, or None when unset/blank. Raises ValueError on junk
    so a typo'd temperature fails loudly at startup, not silently."""
    raw = _env(name)
    return float(raw) if raw not in (None, "") else None


JWT_MIN_SECRET_CHARS = 32


def _env_optional_int(name: str) -> int | None:
    """Int env var, or None when unset/blank/not a number (treated as 'not configured')."""
    raw = _env(name)
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", ""})


def database_is_remote(url: str | None) -> bool:
    """True when DATABASE_URL points somewhere other than this machine.

    Local = no host (unix socket), localhost, or a loopback address. A bare
    docker-compose service name like `postgres` is NOT local: it is another
    container, which is exactly the "deployed" shape this default is for.
    Unset or unparseable -> not remote (nothing to protect; the API serves
    503s without a database anyway)."""
    if not url:
        return False
    try:
        from sqlalchemy.engine import make_url
        host = (make_url(url).host or "").lower()
    except Exception:  # noqa: BLE001 - a malformed URL is handled elsewhere
        return False
    if host in _LOCAL_HOSTS or host.startswith("127."):
        return False
    return True


def _env_csv(name: str) -> tuple[str, ...]:
    raw = _env(name) or ""
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass(frozen=True)
class Settings:
    """Runtime configuration. Construct via `get_settings()`."""

    # --- Phase 2: PostgreSQL -------------------------------------------
    # DATABASE_URL is the OWNER/admin connection (alembic, scripts, backups).
    database_url: str | None = field(
        default_factory=lambda: _env("DATABASE_URL"))
    # Phase 2 (RLS): the API itself connects as the least-privilege `iia_app`
    # role via APP_DATABASE_URL. Unset -> the API falls back to DATABASE_URL
    # (zero-config local demo); main.py then warns, or refuses on a remote
    # database, because an owner/superuser connection bypasses RLS.
    app_database_url: str | None = field(
        default_factory=lambda: _env("APP_DATABASE_URL"))
    # Refuse to start when the API's role bypasses RLS. Default: on for a
    # remote database, off (warning only) for a local one.
    rls_required: bool = field(default_factory=lambda: _env_bool(
        "RLS_REQUIRED", database_is_remote(_env("APP_DATABASE_URL") or _env("DATABASE_URL"))))

    @property
    def engine_url(self) -> str | None:
        """The URL the API's engine uses."""
        return self.app_database_url or self.database_url

    # --- Security ------------------------------------------------------
    # Authentication model (production-readiness fix 3):
    #
    #   * Each tenant has its own API key, stored hashed in the `tenants`
    #     table (create one with scripts/manage_tenants.py). The tenant a
    #     request acts as is derived from the key, so a caller can only
    #     ever reach their own tenant's rows and documents.
    #   * AUTH_REQUIRED=true  -> every tenant-scoped endpoint needs a valid
    #     X-API-Key; no key or a wrong key is a 401.
    #   * AUTH_REQUIRED unset -> the zero-config demo posture: a request
    #     with no key acts as the demo tenant ("default"); a request that
    #     sends a key must still send a valid one. Defaults to true
    #     whenever the legacy API_KEY below is set, so setting it keeps
    #     enforcing auth exactly as it did before.
    #   * Phase 1 security default: when AUTH_REQUIRED is unset AND
    #     DATABASE_URL points at a non-local host (see database_is_remote),
    #     auth is required too. A deployment pointed at a real database no
    #     longer serves unauthenticated demo traffic by default; running
    #     with AUTH_REQUIRED=false there is still possible but is explicit
    #     and logged as a warning at startup.
    #   * API_KEY (legacy, optional) is a bootstrap credential for the demo
    #     tenant only — handy for a single-tenant deployment that hasn't
    #     created a `tenants` row key yet. It can never select another
    #     tenant.
    api_key: str | None = field(default_factory=lambda: _env("API_KEY"))
    auth_required: bool = field(default_factory=lambda: _env_bool(
        "AUTH_REQUIRED", bool(_env("API_KEY")) or database_is_remote(
            _env("DATABASE_URL"))
        or database_is_remote(_env("APP_DATABASE_URL"))))

    # Phase 4: human login. Both are REQUIRED to enable /auth/login and Bearer
    # tokens and have no defaults (an unset secret must never mean a known one).
    # Missing/short secret or a non-positive lifetime -> login answers 503 and a
    # Bearer token is rejected; API keys and the demo posture are unaffected.
    jwt_secret: str | None = field(default_factory=lambda: _env("JWT_SECRET"))
    jwt_ttl_minutes: int | None = field(default_factory=lambda: _env_optional_int("JWT_TTL_MINUTES"))

    @property
    def jwt_enabled(self) -> bool:
        return bool(self.jwt_secret) and len(self.jwt_secret) >= JWT_MIN_SECRET_CHARS \
            and bool(self.jwt_ttl_minutes) and self.jwt_ttl_minutes > 0

    # CORS: explicit allow-list of browser origins, comma separated
    # (e.g. "https://ops.example.com,http://localhost:8501"). Empty (the
    # default) installs no CORS middleware at all, so browsers refuse
    # cross-origin calls — the safe default. The bundled Streamlit
    # frontend calls the API server-to-server and needs no CORS entry.
    cors_allow_origins: tuple[str, ...] = field(
        default_factory=lambda: _env_csv("CORS_ALLOW_ORIGINS"))

    # Rate limit for the two LLM-backed endpoints (/investigate, /chat),
    # per tenant (per client IP for unauthenticated demo traffic).
    # 0 disables. In-process, so it is per API worker — fine for a single
    # container; put a gateway limiter in front if you scale out.
    rate_limit_per_minute: int = field(default_factory=lambda: int(
        _env("RATE_LIMIT_PER_MINUTE", "30") or "30"))

    # --- LLM call resilience -------------------------------------------
    # Applied to every google-genai client this app builds (synthesis,
    # planner, embeddings) — see app/core/genai_client.py. The SDK's own
    # default is no client-side deadline.
    llm_timeout_seconds: float = field(default_factory=lambda: float(
        _env("LLM_TIMEOUT_SECONDS", "30") or "30"))
    # Total attempts including the first call (2 = one retry on
    # 408/429/5xx with exponential backoff). 1 disables retry.
    llm_max_attempts: int = field(default_factory=lambda: int(
        _env("LLM_MAX_ATTEMPTS", "2") or "2"))

    # Hard ceiling on LangGraph supersteps per investigation. The planner
    # already has its own round cap (planner.MAX_PLANNER_TOOL_ROUNDS); this
    # is the independent backstop underneath it, so a future wiring bug
    # (a cycle that never reaches END) fails fast instead of spinning.
    graph_recursion_limit: int = field(default_factory=lambda: int(
        _env("GRAPH_RECURSION_LIMIT", "40") or "40"))

    # --- Phase 3: RAG --------------------------------------------------
    documents_dir: Path = field(
        default_factory=lambda: Path(
            _env("DOCUMENTS_DIR", str(PROJECT_ROOT / "documents")))
    )
    chunks_path: Path = field(
        default_factory=lambda: Path(
            _env("CHUNKS_PATH", str(PROJECT_ROOT / "data" /
                 "processed" / "document_chunks.jsonl"))
        )
    )

    # Which retrieval backend to use.
    #   "pinecone" — Vertex embeddings + Pinecone vector search (production path)
    #   "local"    — BM25 lexical search over the same chunk file (offline path)
    #   "auto"     — pinecone if credentials are present, otherwise local
    rag_backend: str = field(default_factory=lambda: (
        _env("RAG_BACKEND", "auto") or "auto").lower())

    chunk_size: int = field(default_factory=lambda: int(
        _env("RAG_CHUNK_SIZE", "900")))
    chunk_overlap: int = field(default_factory=lambda: int(
        _env("RAG_CHUNK_OVERLAP", "150")))
    retrieval_top_k: int = field(
        default_factory=lambda: int(_env("RAG_TOP_K", "5")))

    # Minimum fraction of the query's distinct terms a chunk must match to
    # be returned at all — the score floor the retriever docstrings always
    # promised but never enforced (audit F3). Below the floor, a hit is
    # dropped rather than padded into the result, same as BM25Index
    # already does for zero-scoring chunks; an empty result stays a
    # valid, reportable outcome.
    #
    # This is a *coverage ratio*, not a raw BM25 score cutoff — an
    # absolute score floor was tried first and rejected. BM25 score scales
    # with query length and per-term IDF, so it isn't comparable across
    # queries of different lengths: a short, perfectly on-topic query like
    # "tool wear" (score ~2.9, both terms present) scored *lower* than a
    # long off-topic query that coincidentally shared one common word with
    # a chunk (score ~5.1, e.g. "quarterly" appearing in both "quarterly
    # earnings" and MAN-200's maintenance-interval table). No single
    # absolute number separated those two cases correctly.
    #
    # Coverage ratio doesn't have that problem, because it's normalized
    # against the query's own length. Calibrated against this project's
    # own eval set (app/rag/evaluation.py RELEVANCE_CASES) plus the
    # shorter/filtered queries used in tests/test_rag.py:
    #   - the lowest coverage any required hit needs is 0.33 (a
    #     failure_mode-filtered query matching 1 of its 3 terms)
    #   - a broad sweep of genuinely off-topic queries (recipes, weather,
    #     tax deadlines, tutorials, sports, movies...) tops out at 0.25
    #     coverage, from single coincidental term matches
    # 0.3 sits in that gap. It is not perfect: a query can still coincide
    # with a *real* use of a shared word in an unrelated maintenance
    # record (e.g. "M-17 quarterly earnings" still matches an incident
    # report's genuine "quarterly-check limit" phrase at 0.5 coverage) —
    # that residual case is a corpus-scale limitation of lexical
    # retrieval, not something a coverage threshold can fix, and is
    # documented as such in tests/test_rag.py rather than hidden.
    # Re-run this calibration before trusting the default against a
    # materially different corpus.
    rag_min_term_coverage_lexical: float = field(default_factory=lambda: float(
        _env("RAG_MIN_TERM_COVERAGE_LEXICAL", "0.3")))

    # Pinecone cosine similarity floor. Unlike the value above, this is
    # NOT calibrated against live traffic — this environment has no
    # Vertex/Pinecone credentials (see audit F3/F5), so 0.75 is a
    # conventional starting point for cosine similarity, not a measured
    # one. Cosine similarity doesn't have the query-length problem BM25
    # does (it's already normalized to the same [-1, 1] scale regardless
    # of query length), so an absolute floor is the right shape here —
    # only the specific number is unverified. Recalibrate against real
    # query/document pairs once the dense path is reachable, the same way
    # rag_min_term_coverage_lexical was.
    rag_min_score_dense: float = field(default_factory=lambda: float(
        _env("RAG_MIN_SCORE_DENSE", "0.75")))

    # --- Pinecone ------------------------------------------------------
    pinecone_api_key: str | None = field(
        default_factory=lambda: _env("PINECONE_API_KEY"))
    pinecone_index: str = field(
        default_factory=lambda: _env(
            "PINECONE_INDEX", "industrial-intelligence") or ""
    )
    pinecone_cloud: str = field(
        default_factory=lambda: _env("PINECONE_CLOUD", "aws") or "aws")
    pinecone_region: str = field(
        default_factory=lambda: _env(
            "PINECONE_REGION", "us-east-1") or "us-east-1"
    )
    pinecone_namespace: str = field(
        default_factory=lambda: _env(
            "PINECONE_NAMESPACE", "maintenance-docs") or ""
    )

    # --- Vertex AI (embeddings + Gemini synthesis, Phase 6) ------------
    gcp_project: str | None = field(
        default_factory=lambda: _env("GOOGLE_CLOUD_PROJECT"))
    gcp_location: str = field(
        default_factory=lambda: _env(
            "GOOGLE_CLOUD_LOCATION", "us-central1") or "us-central1"
    )
    embedding_model: str = field(
        default_factory=lambda: _env(
            "VERTEX_EMBEDDING_MODEL", "gemini-embedding-001") or ""
    )
    embedding_dimension: int = field(
        default_factory=lambda: int(_env("EMBEDDING_DIMENSION", "768")))

    # Gemini 2.5 Flash is being retired (Google's published dates for it
    # have moved more than once — Oct 16 vs Oct 20, 2026 on Vertex, and
    # "no earlier than" wording — audit F5), so the default is its named
    # successor, gemini-3.6-flash. Confirm the exact ID in your own Vertex
    # Model Garden before a live demo: Vertex model lifecycle moves faster
    # than this file. Override with GEMINI_MODEL.
    gemini_model: str = field(default_factory=lambda: _env(
        "GEMINI_MODEL", "gemini-3.6-flash") or "")

    # Region for the Gemini calls (synthesis + planner) ONLY. Separate from
    # gcp_location above on purpose: that one also drives the embedding
    # client, which is a different model with its own regional
    # availability. Google's Gemini 3 docs list some Gemini 3 models as
    # available on the global endpoint only, and "global" is always a valid
    # location for Gemini, so it's the safe default. Set GEMINI_LOCATION to
    # a region only if you need data residency and have confirmed the model
    # is served there.
    gemini_location: str = field(
        default_factory=lambda: _env("GEMINI_LOCATION", "global") or "global"
    )

    # Sampling temperature for the two Gemini calls. None (the default)
    # means "don't send one", i.e. use the model's own default. Google's
    # Gemini 3 guidance is to leave temperature at its default of 1.0 —
    # forcing it low (this project used 0.2 and 0.0 on 2.5) can cause
    # looping or degraded reasoning on Gemini 3. Set GEMINI_TEMPERATURE /
    # PLANNER_TEMPERATURE explicitly only if you pin a 2.x model.
    gemini_temperature: float | None = field(
        default_factory=lambda: _env_optional_float("GEMINI_TEMPERATURE"))
    planner_temperature: float | None = field(
        default_factory=lambda: _env_optional_float("PLANNER_TEMPERATURE"))

    # Report synthesis (app/agents/synthesis.py, app/agents/llm.py) is
    # deterministic-template by default even with credentials present —
    # this is a second, explicit switch, not just "gcp_project is set".
    # Rationale: swapping the recommendation node's output source is a
    # bigger behavioural change than embeddings silently using Vertex,
    # and a portfolio demo should be able to force the deterministic path
    # on for a reproducible run without unsetting credentials.
    # (P2 cleanup: this default used to say "true" here, contradicting
    # the paragraph above and only matching docker-compose.yml's explicit
    # GEMINI_SYNTHESIS_ENABLED:-false because compose overrode it. A bare
    # `uvicorn` / test run outside compose got the opposite of the
    # documented behavior. Default is "false" now, so compose's override
    # is a redundant-but-consistent restatement, not a silent fix.)
    gemini_synthesis_enabled: bool = field(
        default_factory=lambda: (
            _env("GEMINI_SYNTHESIS_ENABLED", "false") or "false").lower() == "true"
    )

    # --- Phase 10 (revised, audit F6): supplementary evidence-gathering
    # planner (app/agents/planner.py) ------------------------------------
    # Mirrors gemini_synthesis_enabled's shape exactly, not
    # gemini_synthesis_enabled's old sibling default: this used to be a
    # separate, default-OFF switch (an opt-in "demo mode" nobody saw
    # unless they flipped it) — the audit's F6 finding was precisely that
    # this made "LangGraph agent with real tool use" untrue of the
    # default graph. Since the redesign, the planner can no longer choose
    # risk-critical evidence (sensor + anomaly data is always gathered
    # deterministically first — see app/agents/graph.py's module
    # docstring), so letting it run automatically whenever Gemini is
    # configured is no longer the bigger behavioural risk Phase 10
    # treated it as: the worst a misbehaving planner can now do is skip
    # optional supplementary evidence (maintenance history, doc search),
    # which was already a documented, low-stakes degrade. So this
    # defaults to "on whenever credentials are present" — unlike
    # gemini_synthesis_enabled just above, which defaults OFF even with
    # credentials present (a bigger behavioural change, see its own
    # comment) — with the env var kept as an explicit kill-switch for a
    # reproducible, fully-deterministic demo run without unsetting
    # GOOGLE_CLOUD_PROJECT.
    agentic_routing_enabled: bool = field(
        default_factory=lambda: (
            _env("AGENTIC_ROUTING_ENABLED", "true") or "true").lower() == "true"
    )
    # Reuses the same model family as synthesis by default — a Gemini
    # tool-calling call for evidence-gathering planning is a distinct
    # call from the synthesis call in app/agents/llm.py, but there's no
    # reason to default it to a different model. Override independently
    # if you want a cheaper/faster model for planning than for prose.
    planner_model: str = field(default_factory=lambda: _env(
        "PLANNER_MODEL", "gemini-3.6-flash") or "")

    # --- Per-tenant storage locations ---------------------------------
    # The demo tenant keeps the original locations so existing ingested
    # data stays valid. Every other tenant gets its own directory/file
    # and its own Pinecone namespace; nothing is ever shared or fallen
    # back to across tenants.

    def documents_dir_for(self, tenant_id: str) -> Path:
        validate_tenant_id(tenant_id)
        if tenant_id == DEFAULT_TENANT_ID:
            return self.documents_dir
        # documents.load_documents globs "*.md" non-recursively, so this
        # subdirectory is invisible to the demo tenant's ingestion.
        return self.documents_dir / "tenants" / tenant_id

    def chunks_path_for(self, tenant_id: str) -> Path:
        validate_tenant_id(tenant_id)
        if tenant_id == DEFAULT_TENANT_ID:
            return self.chunks_path
        return self.chunks_path.parent / "tenants" / tenant_id / self.chunks_path.name

    def pinecone_namespace_for(self, tenant_id: str) -> str:
        """`<tenant>:<base>` for real tenants; the bare base namespace for
        the demo tenant (legacy vectors stay valid). ':' cannot appear in a
        tenant id, so this mapping is injective."""
        validate_tenant_id(tenant_id)
        if tenant_id == DEFAULT_TENANT_ID:
            return self.pinecone_namespace
        return f"{tenant_id}:{self.pinecone_namespace}"

    @property
    def gemini_configured(self) -> bool:
        """True when Gemini synthesis can actually run (credentials present
        AND the feature switch is on). Mirrors pinecone_configured's shape."""
        return bool(self.gcp_project) and self.gemini_synthesis_enabled

    @property
    def agentic_routing_configured(self) -> bool:
        """True when the supplementary evidence-gathering planner can
        actually run (credentials present AND the kill-switch isn't set
        to false). Mirrors gemini_configured's shape — and, like it,
        defaults to true whenever credentials are present. See
        app.agents.planner.get_planner_llm, which also requires
        google-genai to actually be installed."""
        return bool(self.gcp_project) and self.agentic_routing_enabled

    @property
    def pinecone_configured(self) -> bool:
        """True when both halves of the production path are configured.

        Pinecone alone is not enough: without a Vertex project there is no
        way to embed a query, so the dense path cannot serve a search.
        """
        return bool(self.pinecone_api_key) and bool(self.gcp_project)

    def resolved_backend(self) -> str:
        """Resolve `rag_backend`, expanding "auto" against what is configured."""
        if self.rag_backend == "auto":
            return "pinecone" if self.pinecone_configured else "local"
        return self.rag_backend


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Return the process-wide Settings instance.

    Cached because reading env vars and touching the filesystem on every
    call is pointless; `refresh=True` exists for tests that monkeypatch
    the environment.
    """
    global _settings
    if _settings is None or refresh:
        _settings = Settings()
    return _settings
