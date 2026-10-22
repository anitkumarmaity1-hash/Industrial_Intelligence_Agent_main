"""
Restore a backup made by scripts/backup_db.py via `pg_restore`
(production-readiness fix 15).

Usage:
    python scripts/restore_db.py backups/20261101T030000Z.dump

Restores into DATABASE_URL's database. `--clean` drops existing objects
first (so this is safe to run against a database that already has the
old schema in it — the common case: restoring onto a fresh instance
after an incident); pass `--no-clean` to restore into a genuinely empty
database instead and skip that step.

This does not ask for confirmation before overwriting the target
database — by design, so it can be scripted/automated (a cron-driven DR
drill, a CI step). Point it at a throwaway/scratch database unless you
are deliberately restoring over production during an actual incident.

Reads DATABASE_URL from the environment / .env.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _database_url() -> str:
    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        logger.error("DATABASE_URL not set.")
        sys.exit(1)
    return url


def restore(dump_path: Path, database_url: str, clean: bool) -> None:
    pg_restore = shutil.which("pg_restore")
    if not pg_restore:
        logger.error(
            "pg_restore not found on PATH. Install the postgresql-client package.")
        sys.exit(1)
    if not dump_path.exists():
        logger.error("No such backup file: %s", dump_path)
        sys.exit(1)

    cmd = [pg_restore, f"--dbname={database_url}",
           "--no-owner", "--no-privileges"]
    if clean:
        cmd += ["--clean", "--if-exists"]
    cmd.append(str(dump_path))

    result = subprocess.run(cmd, capture_output=True, text=True)
    # pg_restore exits non-zero on warnings about things that don't exist
    # yet to --clean (expected on a first restore into an empty database);
    # only genuine errors in stderr indicate real trouble.
    if result.returncode != 0 and "error" in result.stderr.lower():
        logger.error("pg_restore failed:\n%s", result.stderr)
        sys.exit(result.returncode)
    if result.stderr.strip():
        logger.warning("pg_restore warnings:\n%s", result.stderr)

    logger.info("Restored %s into the target database.", dump_path)
    _regrant_app_role(database_url)


def _regrant_app_role(database_url: str) -> None:
    """pg_restore --no-privileges drops the iia_app grants; re-apply them via
    the idempotent iia_grant_app_privileges() the dump carries (Phase 2).
    Dumps from before migration 0006 lack it: warn and move on."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import SQLAlchemyError
    engine = create_engine(database_url)
    try:
        with engine.begin() as conn:
            conn.execute(text("SELECT iia_grant_app_privileges()"))
        logger.info("Re-applied iia_app grants.")
    except SQLAlchemyError as exc:
        logger.warning("Could not re-apply iia_app grants (%s). Run "
                       "scripts/provision_app_role.py --grants-only.", exc.__class__.__name__)
    finally:
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_file", type=Path)
    parser.add_argument("--no-clean", dest="clean", action="store_false",
                        help="Skip --clean (use for a genuinely empty target database).")
    args = parser.parse_args()

    restore(args.dump_file, _database_url(), args.clean)


if __name__ == "__main__":
    main()
