"""Login and user management (Phase 4).

    POST /auth/login            tenant + email + password -> short-lived JWT (public)
    GET  /users                 list this tenant's users            (users:manage)
    POST /users                 create a user                       (users:manage)
    POST /users/{id}/role       change a user's role                (users:manage)
    POST /users/{id}/active     activate / deactivate a user        (users:manage)

All state changes are POSTs on purpose: CORS only has to allow GET/POST/OPTIONS.
There is no signup: the first admin comes from scripts/manage_users.py, later users
from an admin here. Management is confined to the caller's own tenant by the Principal
(RLS plus tenant_id predicates); a user id from another tenant is simply a 404.

Login does not reveal whether the tenant, the account or the password was wrong: every
failure is the same 401, and an unknown tenant/user still pays for one argon2 verify.
(One residual shape difference: a failed login for an EXISTING tenant also writes an
audit row. See README Limitations.)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.api.dependencies import get_conn, get_principal
from app.api.rate_limit import limiter, retry_after_header
from app.core import audit
from app.core.config import get_settings
from app.core.principal import Principal
from app.core.security import (
    check_password_policy, create_token, dummy_hash, hash_password, normalize_email,
    verify_password,
)
from app.core.tenancy import validate_tenant_id
from app.database import queries
from app.database.tenant_context import bind_tenant, lookup_user_for_login

logger = logging.getLogger(__name__)

login_router = APIRouter(tags=["auth"])
users_router = APIRouter(tags=["users"])

# Login throttling reuses the in-process sliding-window limiter. It counts ATTEMPTS (the
# limiter cannot count failures only), per (tenant, email) and per client IP.
LOGIN_PER_ACCOUNT_PER_MINUTE = 10
LOGIN_PER_IP_PER_MINUTE = 30


class LoginRequest(BaseModel):
    tenant: str = Field(max_length=64)
    email: str = Field(max_length=254)
    password: str = Field(max_length=1024)


class TokenOut(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int
    role: str


class UserOut(BaseModel):
    id: str
    email: str
    role: Literal["admin", "viewer"]
    is_active: bool
    created_at: datetime
    last_login_at: datetime | None = None


class UserCreate(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=1024)
    role: Literal["admin", "viewer"]


class RoleChange(BaseModel):
    role: Literal["admin", "viewer"]


class ActiveChange(BaseModel):
    active: bool


def _user_out(row: dict) -> dict:
    return {**row, "id": str(row["id"])}


def _throttle(key: str, limit: int) -> None:
    wait = limiter.check(key, limit, 60.0)
    if wait is not None:
        raise HTTPException(status_code=429, detail="Too many login attempts. Try again shortly.",
                            headers={"Retry-After": retry_after_header(wait)})


def _invalid_credentials() -> HTTPException:
    return HTTPException(status_code=401, detail="Invalid credentials.",
                         headers={"WWW-Authenticate": "Bearer"})


@login_router.post("/auth/login", response_model=TokenOut)
def login(body: LoginRequest, request: Request, conn: Connection = Depends(get_conn)) -> TokenOut:
    settings = get_settings()
    if not settings.jwt_enabled:
        raise HTTPException(status_code=503, detail="Login is not configured.")
    tenant_slug = body.tenant.strip().lower()
    email = normalize_email(body.email)
    client = request.client.host if request.client else "unknown"
    _throttle(f"login-ip|{client}", LOGIN_PER_IP_PER_MINUTE)
    _throttle(f"login|{tenant_slug}|{email or body.email.strip().lower()}", LOGIN_PER_ACCOUNT_PER_MINUTE)

    row = None
    if email is not None:
        try:
            validate_tenant_id(tenant_slug)
            row = lookup_user_for_login(conn, tenant_slug, email)
        except ValueError:
            row = None
        except SQLAlchemyError as exc:
            logger.error("Login lookup failed: %s", exc)
            raise HTTPException(status_code=503, detail="Could not verify credentials.") from exc

    user_id = row["user_id"] if row else None
    # Always one argon2 verify, against a real hash only if the account exists.
    password_ok = verify_password(row["password_hash"] if user_id else dummy_hash(), body.password)
    if not (password_ok and user_id and row["is_active"]):
        if row is not None:
            reason = ("unknown_user" if not user_id
                      else "inactive" if password_ok and not row["is_active"] else "bad_credentials")
            _audit_failed_login(conn, row["tenant_id"], email, reason)
        else:
            logger.warning("login failed (unknown tenant or malformed input)")
        raise _invalid_credentials()

    tenant_id = row["tenant_id"]
    principal = Principal(tenant_id=tenant_id, authenticated=True, role=row["role"], kind="jwt",
                          user_id=str(user_id), email=email)
    try:
        bind_tenant(conn, tenant_id)
        queries.touch_last_login(conn, str(user_id), tenant_id=tenant_id)
        conn.commit()
    except SQLAlchemyError as exc:      # last_login_at is informational; never block a good login
        logger.error("Could not update last_login_at: %s", exc)
        conn.rollback()
    audit.record(conn, principal, "auth.login", "success", resource="auth")
    token = create_token(user_id=str(user_id), tenant_id=tenant_id, secret=settings.jwt_secret,
                         ttl_minutes=settings.jwt_ttl_minutes)
    return TokenOut(access_token=token, expires_in=settings.jwt_ttl_minutes * 60, role=row["role"])


def _audit_failed_login(conn: Connection, tenant_id: str, email: str | None, reason: str) -> None:
    try:
        bind_tenant(conn, tenant_id)
    except ValueError:
        return
    audit.record(conn, Principal(tenant_id=tenant_id, kind="anonymous"), "auth.login", "failure",
                 resource="auth", actor=email, detail={"reason": reason})


# ---------------------------------------------------------------------
# User management (admin only; enforced by the router-level permission check)
# ---------------------------------------------------------------------

def _uuid_or_404(user_id: str) -> str:
    try:
        return str(uuid.UUID(user_id))
    except ValueError:
        raise HTTPException(status_code=404, detail="User not found.") from None


def _refuse_last_admin(conn: Connection, tenant_id: str, user_id: str) -> None:
    """Row-lock the tenant's active admins, then refuse if `user_id` is the only one."""
    admins = queries.lock_active_admins(conn, tenant_id=tenant_id)
    if admins == [user_id]:
        conn.rollback()
        raise HTTPException(
            status_code=409,
            detail="The last active admin of a tenant cannot be demoted or deactivated.")


@users_router.get("/users", response_model=list[UserOut])
def list_users(conn: Connection = Depends(get_conn),
               principal: Principal = Depends(get_principal)) -> list[dict]:
    return [_user_out(r) for r in queries.list_users(conn, tenant_id=principal.tenant_id)]


@users_router.post("/users", response_model=UserOut, status_code=201)
def create_user(body: UserCreate, conn: Connection = Depends(get_conn),
                principal: Principal = Depends(get_principal)) -> dict:
    email = normalize_email(body.email)
    if email is None:
        raise HTTPException(status_code=422, detail="A valid email address is required.")
    problem = check_password_policy(body.password)
    if problem:
        raise HTTPException(status_code=422, detail=problem)
    user_id = str(uuid.uuid4())
    try:
        row = queries.create_user(conn, user_id, email, hash_password(body.password), body.role,
                                  tenant_id=principal.tenant_id)
        conn.commit()
    except IntegrityError:
        conn.rollback()
        raise HTTPException(status_code=409,
                            detail="A user with that email already exists in this tenant.") from None
    audit.record(conn, principal, "user.create", "success", resource=f"user:{user_id}",
                 detail={"role": body.role})
    return _user_out(row)


@users_router.post("/users/{user_id}/role", response_model=UserOut)
def change_role(user_id: str, body: RoleChange, conn: Connection = Depends(get_conn),
                principal: Principal = Depends(get_principal)) -> dict:
    user_id = _uuid_or_404(user_id)
    tid = principal.tenant_id
    _lock_for_change(conn, tid, user_id, leaving=body.role != "admin")
    before = queries.get_user(conn, user_id, tenant_id=tid)
    if before is None:
        conn.rollback()
        raise HTTPException(status_code=404, detail="User not found.")
    row = queries.set_user_role(conn, user_id, body.role, tenant_id=tid)
    conn.commit()
    audit.record(conn, principal, "user.role_change", "success", resource=f"user:{user_id}",
                 detail={"from": before["role"], "to": body.role})
    return _user_out(row)


@users_router.post("/users/{user_id}/active", response_model=UserOut)
def set_active(user_id: str, body: ActiveChange, conn: Connection = Depends(get_conn),
               principal: Principal = Depends(get_principal)) -> dict:
    user_id = _uuid_or_404(user_id)
    tid = principal.tenant_id
    _lock_for_change(conn, tid, user_id, leaving=not body.active)
    row = queries.set_user_active(conn, user_id, body.active, tenant_id=tid)
    if row is None:
        conn.rollback()
        raise HTTPException(status_code=404, detail="User not found.")
    conn.commit()
    audit.record(conn, principal, "user.activate" if body.active else "user.deactivate", "success",
                 resource=f"user:{user_id}")
    return _user_out(row)


def _lock_for_change(conn: Connection, tenant_id: str, user_id: str, *, leaving: bool) -> None:
    """Serialise admin-set changes; if this change takes `user_id` out of the active-admin
    set (demote / deactivate), refuse when they are the last one."""
    if leaving:
        _refuse_last_admin(conn, tenant_id, user_id)
