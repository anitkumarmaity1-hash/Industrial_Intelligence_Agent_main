# Industrial Intelligence Agent — Streamlit frontend image (Phase 8)
#
# Build context is the PROJECT ROOT, not frontend/, because it needs
# requirements-runtime.txt from the root (see docker-compose.yml's
# `context: .` / `dockerfile: frontend/Dockerfile`). If you build this
# standalone with `docker build -f frontend/Dockerfile .`, run it from the
# project root for the same reason.

FROM python:3.12-slim

WORKDIR /srv

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY frontend/ ./frontend/

# Audit F10: same non-root hygiene fix as the API image's Dockerfile.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /srv
USER appuser

# API_BASE_URL defaults to http://localhost:8000 in streamlit_app.py, which
# is wrong inside Compose's network — docker-compose.yml overrides it to
# http://api:8000. Set explicitly here too so `docker run` alone (without
# compose) still points somewhere sane if the API happens to be on the
# host network; harmless either way since compose's env always wins.
ENV API_BASE_URL=http://localhost:8000

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health', timeout=2)" || exit 1

CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
