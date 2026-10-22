"""
Back up the database to a single file via `pg_dump --format=custom`
(production-readiness fix 15).

Usage:
    python scripts/backup_db.py                      # writes backups/<timestamp>.dump
    python scripts/backup_db.py --out backups/x.dump  # explicit path
    python scripts/backup_db.py --keep 7              # also prune, keeping the 7 newest

Custom format (`-Fc`) is used rather than plain SQL: it's compressed, it
restores with `pg_restore` (see scripts/restore_db.py) rather than
`psql`, and — the reason it matters here — it supports selective and
parallel restore, which plain `pg_dump > file.sql` does not.

This is the backup half of the audit's "no backup/restore strategy
beyond the Docker named volume" finding; see tests/test_backup_restore.py
for the half that actually proves a backup this script makes can be
restored (a backup that has never been restored is not a backup, it's an
unverified assumption).

Reads DATABASE_URL from the environment / .env.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_BACKUP_DIR = Path(__file__).resolve().parent.parent / "backups"


def _database_url() -> str:
    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        logger.error("DATABASE_URL not set.")
        sys.exit(1)
    return url


def backup(out_path: Path, database_url: str) -> Path:
    pg_dump = shutil.which("pg_dump")
    if not pg_dump:
        logger.error("pg_dump not found on PATH. Install the postgresql-client package.")
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [pg_dump, "--format=custom", f"--file={out_path}", database_url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error("pg_dump failed:\n%s", result.stderr)
        sys.exit(result.returncode)

    size_kb = out_path.stat().st_size / 1024
    logger.info("Wrote %s (%.1f KB).", out_path, size_kb)
    return out_path


def prune(backup_dir: Path, keep: int) -> None:
    dumps = sorted(backup_dir.glob("*.dump"), key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in dumps[keep:]:
        stale.unlink()
        logger.info("Pruned old backup %s.", stale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None,
                        help="Output file (default: backups/<UTC timestamp>.dump).")
    parser.add_argument("--keep", type=int, default=None,
                        help="After backing up, delete older *.dump files in the same "
                             "directory beyond this count.")
    args = parser.parse_args()

    out_path = args.out or (
        DEFAULT_BACKUP_DIR / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.dump")

    backup(out_path, _database_url())

    if args.keep is not None:
        prune(out_path.parent, args.keep)


if __name__ == "__main__":
    main()
