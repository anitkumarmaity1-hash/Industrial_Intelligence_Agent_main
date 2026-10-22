"""Roles, permissions and the per-route permission table (Phase 4).

The whole authorization model is in this one module:

  * ROLE_MATRIX      role -> set of permissions
  * ROUTE_PERMISSIONS (METHOD, route path template) -> the permission the route needs
  * PUBLIC_ROUTES    routes that need no credential (explicit, not implied)

Enforcement is one router-level dependency (app.api.dependencies.
enforce_route_permission) that looks the matched route up in ROUTE_PERMISSIONS and
fails CLOSED (403) for a route that is not declared. tests/test_auth.py also walks
every registered route and fails if one is neither declared here nor public.

Roles:
  admin    human, full tenant control (uploads, mapping, retry, users, ...)
  viewer   human, read-only. Also what an anonymous demo-posture request is, for reads.
  api_key  machine credential (tenant API key / legacy API_KEY): read + ingest
           (upload, mapping). Cannot retry jobs, manage users or edit configuration
           (including connectors).
"""

from __future__ import annotations

from enum import Enum

ROLE_ADMIN = "admin"
ROLE_VIEWER = "viewer"
ROLE_API_KEY = "api_key"
USER_ROLES = (ROLE_ADMIN, ROLE_VIEWER)      # roles a users row may hold


class Permission(str, Enum):
    DATA_READ = "data:read"                  # machines, health, anomalies, sensors, readings
    INVESTIGATE_RUN = "investigate:run"      # /investigate, /chat
    UPLOADS_READ = "uploads:read"            # list/get uploads and jobs, rejected rows
    UPLOADS_WRITE = "uploads:write"          # create upload, content, preview (shows raw rows)
    MAPPING_WRITE = "mapping:write"          # validate / confirm a mapping (queues the job)
    JOBS_RETRY = "jobs:retry"
    USERS_MANAGE = "users:manage"            # create, list, change role, (de)activate users
    REGISTRY_WRITE = "registry:write"        # POST /sensors/{name}: edit a sensor's range/threshold/enabled
    CONNECTORS_READ = "connectors:read"      # list/get connectors and their runs (admin only)
    CONNECTORS_WRITE = "connectors:write"    # create/update/delete/enable/test/run a connector


P = Permission

_READ = frozenset({P.DATA_READ, P.INVESTIGATE_RUN, P.UPLOADS_READ})

ROLE_MATRIX: dict[str, frozenset[Permission]] = {
    ROLE_ADMIN: frozenset(Permission),
    ROLE_VIEWER: _READ,
    ROLE_API_KEY: _READ | {P.UPLOADS_WRITE, P.MAPPING_WRITE},
}

# Permissions whose use is a write: the request is audited (success or failure).
WRITE_PERMISSIONS = frozenset({
    P.UPLOADS_WRITE, P.MAPPING_WRITE, P.JOBS_RETRY, P.USERS_MANAGE,
    P.REGISTRY_WRITE, P.CONNECTORS_WRITE,
})   # CONNECTORS_READ is deliberately absent: reading config is not a write.


def role_allows(role: str | None, permission: Permission) -> bool:
    return role is not None and permission in ROLE_MATRIX.get(role, frozenset())


ROUTE_PERMISSIONS: dict[tuple[str, str], Permission] = {
    # --- app/api/routes.py
    ("GET", "/machines"): P.DATA_READ,
    ("GET", "/machines/rank-for-inspection"): P.DATA_READ,
    ("GET", "/machines/{machine_id}"): P.DATA_READ,
    ("GET", "/machines/{machine_id}/health"): P.DATA_READ,
    ("GET", "/machines/{machine_id}/anomalies"): P.DATA_READ,
    ("GET", "/machines/{machine_id}/sensors"): P.DATA_READ,
    ("GET", "/machines/{machine_id}/readings"): P.DATA_READ,
    ("GET", "/sensors"): P.DATA_READ,
    # --- app/api/dashboard.py (Phase 6)
    ("GET", "/fleet/summary"): P.DATA_READ,
    ("GET", "/machines/{machine_id}/series"): P.DATA_READ,
    ("GET", "/units"): P.DATA_READ,
    ("GET", "/investigations"): P.INVESTIGATE_RUN,
    ("GET", "/investigations/{investigation_id}"): P.INVESTIGATE_RUN,
    ("POST", "/sensors/{sensor_name}"): P.REGISTRY_WRITE,
    ("POST", "/investigate"): P.INVESTIGATE_RUN,
    ("POST", "/chat"): P.INVESTIGATE_RUN,
    # --- app/api/uploads.py
    ("POST", "/uploads"): P.UPLOADS_WRITE,
    ("POST", "/uploads/{upload_id}/content"): P.UPLOADS_WRITE,
    ("POST", "/uploads/{upload_id}/preview"): P.UPLOADS_WRITE,
    ("POST", "/uploads/{upload_id}/mapping"): P.MAPPING_WRITE,
    ("GET", "/uploads"): P.UPLOADS_READ,
    ("GET", "/uploads/{upload_id}"): P.UPLOADS_READ,
    ("GET", "/jobs"): P.UPLOADS_READ,
    ("GET", "/jobs/{job_id}"): P.UPLOADS_READ,
    ("GET", "/jobs/{job_id}/rejected-rows"): P.UPLOADS_READ,
    ("POST", "/jobs/{job_id}/retry"): P.JOBS_RETRY,
    # --- app/api/connectors.py (Phase 5; admin only, so a viewer or API key gets 403)
    ("GET", "/connectors"): P.CONNECTORS_READ,
    ("POST", "/connectors"): P.CONNECTORS_WRITE,
    ("GET", "/connectors/{connector_id}"): P.CONNECTORS_READ,
    ("POST", "/connectors/{connector_id}"): P.CONNECTORS_WRITE,
    ("POST", "/connectors/{connector_id}/enable"): P.CONNECTORS_WRITE,
    ("POST", "/connectors/{connector_id}/disable"): P.CONNECTORS_WRITE,
    ("POST", "/connectors/{connector_id}/delete"): P.CONNECTORS_WRITE,
    ("POST", "/connectors/{connector_id}/test"): P.CONNECTORS_WRITE,
    ("POST", "/connectors/{connector_id}/run"): P.CONNECTORS_WRITE,
    ("GET", "/connectors/{connector_id}/runs"): P.CONNECTORS_READ,
    # --- app/api/auth.py (user management; login itself is public)
    ("GET", "/users"): P.USERS_MANAGE,
    ("POST", "/users"): P.USERS_MANAGE,
    ("POST", "/users/{user_id}/role"): P.USERS_MANAGE,
    ("POST", "/users/{user_id}/active"): P.USERS_MANAGE,
}

# No credential needed. /auth/login is where a credential is obtained; the rest are
# liveness/readiness/landing. (FastAPI's /docs, /redoc and /openapi.json are not
# APIRoutes and are not part of this table.)
PUBLIC_ROUTES: frozenset[tuple[str, str]] = frozenset({
    ("GET", "/"), ("GET", "/healthz"), ("GET", "/readyz"), ("POST", "/auth/login"),
})
