"""
Provision the least-privilege database roles (Phase 2 RLS; worker role Phase 3):
`iia_app` for the API and `iia_worker` for the background worker.

    APP_DB_PASSWORD=... [WORKER_DB_PASSWORD=...] python scripts/provision_app_role.py
    python scripts/provision_app_role.py --grants-only      # no password change

Connects with DATABASE_URL (the OWNER/admin role; needs CREATEROLE or
superuser the first time), runs the idempotent iia_grant_app_privileges()
created by migrations 0006/0007 (grants for BOTH roles), and gives `iia_app` LOGIN +
the password from the required APP_DB_PASSWORD env var (no default), and `iia_worker`
LOGIN + WORKER_DB_PASSWORD when that is set (it stays NOLOGIN otherwise).
scripts/restore_db.py calls the grants function after a restore, because pg_restore
--no-privileges drops grants. Point the API at its role with APP_DATABASE_URL and the
worker at its role with WORKER_DATABASE_URL.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def provision(url: str, password: str | None, worker_password: str | None = None) -> None:
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("SELECT iia_grant_app_privileges()"))
        for role, pw in (("iia_app", password), ("iia_worker", worker_password)):
            if pw is None:
                continue
            stmt = conn.execute(
                text("SELECT format('ALTER ROLE %I LOGIN PASSWORD %L', CAST(:role AS text), "
                     "CAST(:pw AS text))"),
                {"role": role, "pw": pw}).scalar()
            conn.exec_driver_sql(stmt.replace("%", "%%"))
    engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grants-only", action="store_true",
                        help="Re-apply grants only; leave LOGIN/password alone.")
    args = parser.parse_args()

    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        logger.error("DATABASE_URL not set (the owner/admin connection).")
        return 1
    password = None
    if not args.grants_only:
        password = os.environ.get("APP_DB_PASSWORD")
        if not password:
            logger.error(
                "APP_DB_PASSWORD not set (required; there is no default).")
            return 1
    worker_password = None if args.grants_only else (os.environ.get("WORKER_DB_PASSWORD") or None)
    provision(url, password, worker_password)
    logger.info("iia_app/iia_worker grants applied%s%s.",
                "" if args.grants_only else "; iia_app LOGIN password set",
                "; iia_worker LOGIN password set" if worker_password else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
