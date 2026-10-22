"""
Create and manage a tenant's human users from the command line (Phase 4).

Usage:
    python scripts/manage_users.py create-admin acme admin@acme.example
    python scripts/manage_users.py list acme
    python scripts/manage_users.py set-password acme admin@acme.example
    python scripts/manage_users.py create-admin acme a@b.example --password-stdin < pw.txt

There is no open signup: this is how a tenant's FIRST admin is created; later users
are created by an admin through POST /users. `set-password` is an operator recovery
tool (there is no self-service password reset). The password is read from a hidden
prompt (asked twice) or, with --password-stdin, from the first line of stdin; it is
never taken from the command line, never printed, and only its argon2id hash is stored.

Uses DATABASE_URL (the owner/admin connection), like manage_tenants.py.
"""

from __future__ import annotations

import argparse
import getpass
import logging
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.security import check_password_policy, hash_password, normalize_email  # noqa: E402
from app.core.tenancy import validate_tenant_id  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _engine():
    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        logger.error("DATABASE_URL not set.")
        sys.exit(1)
    return create_engine(url)


def _read_password(args) -> str | None:
    if args.password_stdin:
        password = sys.stdin.readline().rstrip("\r\n")
    else:
        password = getpass.getpass("Password: ")
        if password != getpass.getpass("Repeat password: "):
            logger.error("Passwords do not match.")
            return None
    problem = check_password_policy(password)
    if problem:
        logger.error("%s", problem)
        return None
    return password


def _email(raw: str) -> str | None:
    email = normalize_email(raw)
    if email is None:
        logger.error("Not a valid email address: %r", raw)
    return email


def cmd_create_admin(args) -> int:
    tenant_id, email = validate_tenant_id(args.tenant_id), _email(args.email)
    if email is None:
        return 1
    password = _read_password(args)
    if password is None:
        return 1
    with _engine().begin() as conn:
        if not conn.execute(text("SELECT 1 FROM tenants WHERE tenant_id = :t"), {"t": tenant_id}).first():
            logger.error("No such tenant: %s", tenant_id)
            return 1
        if conn.execute(text("SELECT 1 FROM users WHERE tenant_id = :t AND email = :e"),
                        {"t": tenant_id, "e": email}).first():
            logger.error("User %s already exists in tenant %s (use set-password).", email, tenant_id)
            return 1
        conn.execute(
            text("INSERT INTO users (id, tenant_id, email, password_hash, role) "
                 "VALUES (:id, :t, :e, :h, 'admin')"),
            {"id": str(uuid.uuid4()), "t": tenant_id, "e": email, "h": hash_password(password)})
    logger.info("Created admin %s in tenant %s.", email, tenant_id)
    return 0


def cmd_set_password(args) -> int:
    tenant_id, email = validate_tenant_id(args.tenant_id), _email(args.email)
    if email is None:
        return 1
    password = _read_password(args)
    if password is None:
        return 1
    with _engine().begin() as conn:
        res = conn.execute(
            text("UPDATE users SET password_hash = :h WHERE tenant_id = :t AND email = :e"),
            {"h": hash_password(password), "t": tenant_id, "e": email})
    if res.rowcount == 0:
        logger.error("No such user: %s in tenant %s", email, tenant_id)
        return 1
    logger.info("Password updated for %s in tenant %s. Existing tokens stay valid until they expire.",
                email, tenant_id)
    return 0


def cmd_list(args) -> int:
    tenant_id = validate_tenant_id(args.tenant_id)
    with _engine().connect() as conn:
        rows = conn.execute(
            text("SELECT email, role, is_active, last_login_at FROM users "
                 "WHERE tenant_id = :t ORDER BY email"), {"t": tenant_id}).all()
    for r in rows:
        logger.info("%-40s %-8s %-8s last_login=%s", r.email, r.role,
                    "active" if r.is_active else "INACTIVE", r.last_login_at or "never")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("create-admin", "set-password"):
        p = sub.add_parser(name)
        p.add_argument("tenant_id")
        p.add_argument("email")
        p.add_argument("--password-stdin", action="store_true",
                       help="read the password from the first line of stdin instead of prompting")
    sub.add_parser("list").add_argument("tenant_id")
    args = parser.parse_args()
    try:
        return {"create-admin": cmd_create_admin, "set-password": cmd_set_password,
                "list": cmd_list}[args.cmd](args)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
