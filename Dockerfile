# Industrial Intelligence Agent — API image (Phase 8)
#
# Scope: serves app.main:app only. Does NOT include PySpark (spark_jobs/),
# the offline data pipeline, or pytest — those run outside this image, on
# the host or in CI, and their pre-computed output (data/processed/,
# already-loaded PostgreSQL) is what this container consumes. See
# requirements-runtime.txt for why.
#
# Single stage is enough here: requirements-runtime.txt has no packages
# that need a C build step beyond psycopg2-binary (which ships prebuilt
# wheels), so there's nothing multi-stage buys us for a 2-day MVP. A
# multi-stage build would be the right call if this grew a package with a
# real compile step (e.g. non-binary psycopg2, or a native extension).

FROM python:3.12-slim

WORKDIR /srv

# psycopg2-binary ships manylinux wheels for this base image, so no
# build-essential/libpq-dev is needed. If you switch to psycopg2
# (non-binary) later, add those here.

# Phase 9 fix: the Pinecone + Vertex AI path (RAG_BACKEND=pinecone,
# GEMINI_SYNTHESIS_ENABLED=true) needs requirements-cloud.txt installed,
# but that shouldn't be a default — most runs of this image are the local
# BM25 + deterministic-template demo, and google-genai/pinecone are dead
# weight for that path (see requirements-runtime.txt's own comment on why
# this image stays lean by default). INSTALL_CLOUD_DEPS is the switch:
#   docker compose build --build-arg INSTALL_CLOUD_DEPS=true
# or, via docker-compose.yml's build.args passthrough, just set
# INSTALL_CLOUD_DEPS=true in your shell/.env before `docker compose up
# --build`. Default false keeps today's lean image unchanged.
ARG INSTALL_CLOUD_DEPS=false

COPY requirements-runtime.txt requirements-cloud.txt ./
RUN pip install --no-cache-dir -r requirements-runtime.txt \
    && if [ "$INSTALL_CLOUD_DEPS" = "true" ]; then \
    pip install --no-cache-dir -r requirements-cloud.txt; \
    fi

# Application code.
COPY app/ ./app/

# Demo seeding (docker-compose `seed` service): the script and Company B's sample CSV + mapping.
COPY scripts/seed_demo.py ./scripts/seed_demo.py
COPY sample_data/ ./sample_data/

# app/core/anomaly_rules.py (imported by the upload routes) reads the thresholds in
# spark_jobs/config.py, which is plain Python. Only that module and the package marker
# are copied: the rest of spark_jobs/ imports PySpark, which this image does not have.
COPY spark_jobs/__init__.py spark_jobs/config.py ./spark_jobs/

# Runtime data the RAG retriever reads directly off disk (BM25 over the
# pre-chunked, pre-committed JSONL — see app/rag/retriever.py). Nothing
# else under data/ (raw/, sample/, the parquet dirs) is read at request
# time, so only this one file is copied in.
COPY data/processed/document_chunks.jsonl ./data/processed/document_chunks.jsonl

# Audit F10: the container ran as root by default (the python:3.12-slim
# base's implicit default). A non-root user is standard container hygiene
# — it limits blast radius if a dependency vulnerability or an injection
# in a future feature ever led to arbitrary code execution inside this
# container. /srv is chowned to it so app code (already COPYed as root
# above) is still readable. /srv/data/storage is created here, owned by that
# user, so the shared upload volume mounted there is writable (Phase 3).
RUN mkdir -p /srv/data/storage && useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /srv
USER appuser

# Un-set by default: app/core/config.py treats a missing DATABASE_URL as
# "database-backed endpoints return 503" rather than a crash, and
# GEMINI_SYNTHESIS_ENABLED without GOOGLE_CLOUD_PROJECT resolves to the
# deterministic template path (gemini_configured is False) — so the image
# builds and starts with zero required env vars. Real values come from
# docker-compose.yml / --env-file at run time; never bake secrets in here.

EXPOSE 8000

# / is a plain liveness endpoint (app/main.py) with no DB/RAG dependency,
# so it's a true "is the process up" check, not "is Postgres up".
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/', timeout=2)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]