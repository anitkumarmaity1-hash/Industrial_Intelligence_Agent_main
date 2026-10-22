"""Phase 6: the whole demo flow in-process (FastAPI TestClient + the real worker + Spark).

Needs DATABASE_URL, a JVM for Spark, and the worker role (WORKER_DATABASE_URL / provisioned
iia_worker). Skips cleanly without them. The same flow runs against a live compose stack with
`python scripts/e2e_demo.py --api http://localhost:8000`."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app.api.rate_limit import limiter
from app.main import app
from tests.helpers import env

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("e2e_demo", ROOT / "scripts" / "e2e_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_demo_flow_end_to_end(tmp_path):
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")
    if not (os.environ.get("WORKER_DATABASE_URL") or os.environ.get("DATABASE_URL")):
        pytest.skip("no worker database URL")
    if shutil.which("java") is None:
        pytest.skip("no Java for Spark")
    from app.jobs import worker

    e2e = _load_script()
    engine = create_engine(os.environ["DATABASE_URL"])
    wengine = worker.make_engine()
    limiter.reset()
    result, created = None, []
    with env(JWT_SECRET="phase6-e2e-secret-" + "0123456789abcdef" * 2, JWT_TTL_MINUTES="60", API_KEY=None,
             AUTH_REQUIRED=None, STORAGE_LOCAL_DIR=str(tmp_path), STORAGE_BACKEND="local",
             UPLOAD_RATE_LIMIT_PER_MINUTE="1000"), TestClient(app) as client:
        def request(method, path, **kw):
            return client.request(method, path, **kw)

        def wait_for_job(job_id, headers):
            while worker.process_one(wengine):          # run the real worker over the queue, in-process
                pass
            return client.get(f"/jobs/{job_id}", headers=headers).json()

        try:
            result = e2e.run_flow(request, engine, wait_for_job, created)
        finally:
            if created:
                e2e.cleanup(engine, created)
    assert result["machines"] == 3
