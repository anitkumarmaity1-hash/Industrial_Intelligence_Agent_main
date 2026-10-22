"""Tenant context for Postgres row-level security (Phase 2).

RLS policies compare each row's tenant_id with the transaction-local setting
`app.tenant_id`. This module is the single place that sets it:

  * `install(engine)` registers a `begin` listener. Every time a transaction
    starts on a Connection that was bound to a tenant, the setting is
    (re)applied with `set_config(..., true)` and a bound parameter. A commit
    in the middle of a request (routes._audit) therefore costs nothing: the
    next statement opens a new transaction and the listener re-applies it.
  * `bind_tenant(conn, tenant_id)` records the tenant on that Connection
    object only (an execution option), never on the pooled DBAPI connection,
    and applies it at once if a transaction is already open.

The setting is transaction-local, so it vanishes at commit/rollback; a pooled
connection can never carry a previous request's tenant. No context set means
`current_setting` is NULL/'' and the policies match no rows (fail closed).
"""

from __future__ import annotations

from sqlalchemy import event, text
from sqlalchemy.engine import Connection, Engine

from app.core.tenancy import validate_tenant_id

GUC = "app.tenant_id"
_OPTION = "rls_tenant_id"


def _apply_on_begin(conn: Connection) -> None:
    tenant_id = conn.get_execution_options().get(_OPTION)
    if tenant_id is None:
        return
    # Raw DBAPI cursor on purpose: running a Connection.execute inside the
    # `begin` event would try to begin again. psycopg2 binds the parameters.
    cursor = conn.connection.cursor()
    try:
        cursor.execute("SELECT set_config(%s, %s, true)", (GUC, tenant_id))
    finally:
        cursor.close()


def install(engine: Engine) -> Engine:
    """Attach the begin listener once per engine (idempotent)."""
    if not event.contains(engine, "begin", _apply_on_begin):
        event.listen(engine, "begin", _apply_on_begin)
    return engine


def bind_tenant(conn: object, tenant_id: str) -> None:
    """Make every transaction on `conn` run as `tenant_id`.

    Anything that is not a real Connection (tests stub get_conn with None)
    is ignored; such a stub cannot reach a database anyway.
    """
    if not isinstance(conn, Connection):
        return
    validate_tenant_id(tenant_id)
    conn.execution_options(**{_OPTION: tenant_id})
    if conn.in_transaction():
        conn.execute(text("SELECT set_config(:guc, :tid, true)"),
                     {"guc": GUC, "tid": tenant_id})


def lookup_tenant_by_key_hash(conn: Connection, api_key_hash: str) -> dict | None:
    """The active tenant whose key hashes to `api_key_hash`, or None.

    Goes through the SECURITY DEFINER function auth_lookup_tenant (migration
    0006): the API role cannot read `tenants` itself, so it cannot list
    tenants or their key hashes. Runs before any tenant is known.
    """
    row = conn.execute(
        text("SELECT tenant_id, name FROM auth_lookup_tenant(:h)"), {
            "h": api_key_hash}
    ).mappings().first()
    return dict(row) if row else None


def bypasses_rls(engine: Engine) -> bool:
    """True when the engine's role would ignore RLS (superuser or BYPASSRLS)."""
    with engine.connect() as conn:
        return bool(conn.execute(text(
            "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user"
        )).scalar())


def lookup_user_for_login(conn: Connection, tenant_id: str, email: str) -> dict | None:
    """Login lookup (Phase 4): None if the tenant does not exist or is inactive; else
    a row with the tenant id and, if the email matches a user, that user's id, hash,
    role and active flag (user fields None when there is no such user).

    SECURITY DEFINER auth_lookup_user (migration 0008): runs before any tenant is
    bound, and is the only path to users.password_hash."""
    row = conn.execute(
        text("SELECT tenant_id, user_id, password_hash, role, is_active "
             "FROM auth_lookup_user(:t, :e)"), {"t": tenant_id, "e": email}
    ).mappings().first()
    return dict(row) if row else None


def check_user_by_id(conn: Connection, tenant_id: str, user_id: str) -> dict | None:
    """Per-request JWT check: the user's CURRENT email/role/is_active, or None when the
    user does not exist in that tenant or the tenant is inactive (auth_check_user)."""
    row = conn.execute(
        text("SELECT user_id, email, role, is_active FROM auth_check_user(:t, CAST(:u AS uuid))"),
        {"t": tenant_id, "u": user_id}).mappings().first()
    return dict(row) if row else None
