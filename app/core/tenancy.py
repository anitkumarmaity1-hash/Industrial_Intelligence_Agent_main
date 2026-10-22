"""
Tenant identity — the one place that defines what a "tenant" is.

Every row of company data (machines, sensor windows, anomalies,
maintenance records) and every retrieval index (BM25 chunk file, Pinecone
namespace) belongs to exactly one tenant. The tenant is derived from the
caller's credential in `app.api.dependencies.get_tenant`, never from a
request field, header or query parameter the caller can set freely.

The tenant id is used to build filesystem paths and Pinecone namespaces,
so it is restricted to a conservative charset (`validate_tenant_id`):
lowercase letters, digits, `_` and `-`. That rules out path traversal
(`../`), and it excludes `:`, which is the separator used to build
namespaces — so `namespace_for` is injective (two tenants can never share
a namespace).
"""

from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass

# The demo/seed tenant. Rows loaded by sql/seed.sql.gz and the documents
# under documents/ belong to it. It keeps the pre-tenancy layout on
# purpose (same chunk file, same Pinecone namespace) so nothing already
# ingested has to be rebuilt.
DEFAULT_TENANT_ID = "default"

_TENANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")

API_KEY_PREFIX = "iia_"


def validate_tenant_id(tenant_id: str) -> str:
    """Return `tenant_id` unchanged if it is well-formed, else raise ValueError."""
    if not isinstance(tenant_id, str) or not _TENANT_ID_RE.match(tenant_id):
        raise ValueError(
            f"Invalid tenant id {tenant_id!r}: use 1-48 chars of a-z, 0-9, '_' or '-', "
            "starting with a letter or digit."
        )
    return tenant_id


@dataclass(frozen=True)
class TenantContext:
    """Who the current request acts as.

    `authenticated` is False only in the zero-config demo posture
    (AUTH_REQUIRED unset, no X-API-Key sent), where the request falls back
    to the demo tenant — see app.api.dependencies.get_tenant.
    """

    tenant_id: str
    name: str | None = None
    authenticated: bool = False


def generate_api_key() -> str:
    """A new random credential. Shown to the operator once; only its hash is stored."""
    return API_KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    """SHA-256 hex digest of an API key.

    A fast hash is appropriate here (unlike for passwords): keys are 256-bit
    random values, so there is nothing to brute-force, and the lookup runs on
    every request.
    """
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()
