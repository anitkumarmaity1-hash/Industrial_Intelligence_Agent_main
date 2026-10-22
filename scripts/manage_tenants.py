"""
Create and manage tenants (companies) and their API keys.

Usage:
    python scripts/manage_tenants.py create acme --name "Acme Manufacturing"
    python scripts/manage_tenants.py rotate-key acme
    python scripts/manage_tenants.py deactivate acme      # / activate
    python scripts/manage_tenants.py list

`create` and `rotate-key` print the API key ONCE. Only its SHA-256 hash is
stored, so a lost key cannot be recovered — rotate to issue a new one (the
old key stops working immediately). Clients send it as `X-API-Key`.

Reads DATABASE_URL from the environment / .env.
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

from app.core.tenancy import generate_api_key, hash_api_key, validate_tenant_id  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _engine():
    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        logger.error("DATABASE_URL not set.")
        sys.exit(1)
    return create_engine(url)


def cmd_create(args) -> int:
    tenant_id = validate_tenant_id(args.tenant_id)
    key = generate_api_key()
    with _engine().begin() as conn:
        if conn.execute(text("SELECT 1 FROM tenants WHERE tenant_id = :t"), {"t": tenant_id}).first():
            logger.error("Tenant %r already exists (use rotate-key to issue a new key).", tenant_id)
            return 1
        conn.execute(
            text("INSERT INTO tenants (tenant_id, name, api_key_hash) VALUES (:t, :n, :h)"),
            {"t": tenant_id, "n": args.name or tenant_id, "h": hash_api_key(key)},
        )
    logger.info("Created tenant %s.\nAPI key (shown once, store it now):\n  %s", tenant_id, key)
    return 0


def cmd_rotate(args) -> int:
    tenant_id = validate_tenant_id(args.tenant_id)
    key = generate_api_key()
    with _engine().begin() as conn:
        res = conn.execute(
            text("UPDATE tenants SET api_key_hash = :h WHERE tenant_id = :t"),
            {"t": tenant_id, "h": hash_api_key(key)},
        )
    if res.rowcount == 0:
        logger.error("No such tenant: %s", tenant_id)
        return 1
    logger.info("New API key for %s (shown once; the old key is now invalid):\n  %s", tenant_id, key)
    return 0


def cmd_set_active(args, active: bool) -> int:
    tenant_id = validate_tenant_id(args.tenant_id)
    with _engine().begin() as conn:
        res = conn.execute(text("UPDATE tenants SET active = :a WHERE tenant_id = :t"),
                           {"a": active, "t": tenant_id})
    if res.rowcount == 0:
        logger.error("No such tenant: %s", tenant_id)
        return 1
    logger.info("Tenant %s is now %s.", tenant_id, "active" if active else "deactivated")
    return 0


def cmd_list(_args) -> int:
    with _engine().connect() as conn:
        rows = conn.execute(text(
            "SELECT tenant_id, name, active, api_key_hash IS NOT NULL AS has_key, created_at "
            "FROM tenants ORDER BY tenant_id")).all()
    for r in rows:
        logger.info("%-24s %-32s %-8s key=%s", r.tenant_id, r.name,
                    "active" if r.active else "INACTIVE", "yes" if r.has_key else "no")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create"); c.add_argument("tenant_id"); c.add_argument("--name")
    for name in ("rotate-key", "deactivate", "activate"):
        sub.add_parser(name).add_argument("tenant_id")
    sub.add_parser("list")
    args = parser.parse_args()
    try:
        return {"create": cmd_create, "rotate-key": cmd_rotate, "list": cmd_list,
                "deactivate": lambda a: cmd_set_active(a, False),
                "activate": lambda a: cmd_set_active(a, True)}[args.cmd](args)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
