"""Seed the demo: a second company ("Company B") with an admin and a viewer, plus an admin for
the bundled demo tenant ("default"), and queue Company B's sample CSV through the real upload API.

Idempotent: existing tenants/users are left alone (their passwords are NOT changed) and the
sample file is uploaded only when the tenant has no uploads yet. Passwords come from the
environment and are never printed or stored in the repo:

    DATABASE_URL           owner connection (tenants/users are created directly, as manage_*.py do)
    DEMO_ADMIN_PASSWORD    required, >= 12 chars
    DEMO_VIEWER_PASSWORD   required, >= 12 chars
    API_BASE_URL           default http://localhost:8000 (only used to queue the sample upload)

    python scripts/seed_demo.py [--no-sample]

Accounts created (log in with tenant / email):
    company-b / admin@company-b.example , viewer@company-b.example
    default   / admin@default.example      (the seeded AI4I demo fleet)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

import requests
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.security import check_password_policy, hash_password  # noqa: E402
from app.core.tenancy import DEFAULT_TENANT_ID, generate_api_key, hash_api_key, validate_tenant_id  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("seed")

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "sample_data" / "company_b"
COMPANY_B = "company-b"


def ensure_tenant(conn, tenant_id: str, name: str) -> bool:
    """Create the tenant if missing. Its API key is generated and discarded: the demo uses
    logins; rotate a key with scripts/manage_tenants.py if a machine client is needed."""
    validate_tenant_id(tenant_id)
    if conn.execute(text("SELECT 1 FROM tenants WHERE tenant_id = :t"), {"t": tenant_id}).first():
        return False
    conn.execute(text("INSERT INTO tenants (tenant_id, name, api_key_hash) VALUES (:t, :n, :h)"),
                 {"t": tenant_id, "n": name, "h": hash_api_key(generate_api_key())})
    return True


def ensure_user(conn, tenant_id: str, email: str, password: str, role: str) -> bool:
    if conn.execute(text("SELECT 1 FROM users WHERE tenant_id = :t AND email = :e"),
                    {"t": tenant_id, "e": email}).first():
        return False
    conn.execute(text("INSERT INTO users (id, tenant_id, email, password_hash, role) "
                      "VALUES (:i, :t, :e, :h, :r)"),
                 {"i": str(uuid.uuid4()), "t": tenant_id, "e": email, "h": hash_password(password), "r": role})
    return True


def wait_for_api(base_url: str, seconds: int = 120) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            if requests.get(f"{base_url}/readyz", timeout=3).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise SystemExit(f"The API at {base_url} did not become ready within {seconds}s.")


def queue_sample(base_url: str, tenant: str, email: str, password: str, csv_path: Path, mapping_path: Path) -> str:
    """Upload the CSV and confirm its mapping through the public API; returns the job id."""
    r = requests.post(f"{base_url}/auth/login", json={"tenant": tenant, "email": email, "password": password}, timeout=30)
    r.raise_for_status()
    auth = {"Authorization": f"Bearer {r.json()['access_token']}"}
    up = requests.post(f"{base_url}/uploads", json={"filename": csv_path.name}, headers=auth, timeout=30)
    up.raise_for_status()
    uid = up.json()["upload_id"]
    requests.post(f"{base_url}/uploads/{uid}/content", data=csv_path.read_bytes(),
                  headers={**auth, "Content-Type": "text/csv"}, timeout=120).raise_for_status()
    m = requests.post(f"{base_url}/uploads/{uid}/mapping",
                      json={"mapping": json.loads(mapping_path.read_text()), "confirm": True},
                      headers=auth, timeout=60)
    m.raise_for_status()
    return m.json()["job"]["job_id"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-sample", action="store_true", help="create accounts only; do not queue the sample upload")
    args = ap.parse_args(argv)

    admin_pw, viewer_pw = os.environ.get("DEMO_ADMIN_PASSWORD", ""), os.environ.get("DEMO_VIEWER_PASSWORD", "")
    for label, pw in (("DEMO_ADMIN_PASSWORD", admin_pw), ("DEMO_VIEWER_PASSWORD", viewer_pw)):
        problem = check_password_policy(pw)
        if problem:
            log.error("%s: %s", label, problem if pw else "not set")
            return 1
    url = os.environ.get("DATABASE_URL")
    if not url:
        log.error("DATABASE_URL not set.")
        return 1

    engine = create_engine(url)
    with engine.begin() as conn:
        if ensure_tenant(conn, COMPANY_B, "Company B (demo)"):
            log.info("Created tenant %s.", COMPANY_B)
        for tenant, email, pw, role in ((COMPANY_B, "admin@company-b.example", admin_pw, "admin"),
                                        (COMPANY_B, "viewer@company-b.example", viewer_pw, "viewer"),
                                        (DEFAULT_TENANT_ID, "admin@default.example", admin_pw, "admin")):
            if ensure_user(conn, tenant, email, pw, role):
                log.info("Created %s %s in %s.", role, email, tenant)
        has_uploads = conn.execute(text("SELECT COUNT(*) FROM uploads WHERE tenant_id = :t"), {"t": COMPANY_B}).scalar()
    engine.dispose()

    if args.no_sample or has_uploads:
        log.info("Sample upload skipped (%s).", "--no-sample" if args.no_sample else "tenant already has uploads")
        return 0
    base = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
    wait_for_api(base)
    job = queue_sample(base, COMPANY_B, "admin@company-b.example", admin_pw,
                       SAMPLE_DIR / "readings.csv", SAMPLE_DIR / "mapping.json")
    log.info("Queued the Company B sample upload as job %s; the worker will process it.", job)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
