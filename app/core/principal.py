"""Who a request acts as: tenant + role + how it authenticated (Phase 4).

`Principal` extends TenantContext, so every existing `tenant: TenantContext =
Depends(get_tenant)` route keeps working unchanged; get_tenant now simply returns the
Principal that app.api.dependencies.get_principal decided.

kind:  "jwt"      a human user (Authorization: Bearer)         role admin | viewer
       "api_key"  tenant API key or the legacy API_KEY         role api_key
       "demo"     no credential, AUTH_REQUIRED off             role viewer, authenticated False
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.tenancy import TenantContext


@dataclass(frozen=True)
class Principal(TenantContext):
    role: str | None = None
    kind: str = "demo"
    user_id: str | None = None
    email: str | None = None
