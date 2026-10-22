"""Where connector credentials come from (Phase 5).

The database never holds a credential: a connector row stores `secret_ref`, a NAME. At run
time a SecretsProvider turns the name into the secret value, which lives only in memory for
the duration of the call.

`EnvFileSecretsProvider` (the only implementation) looks, in order, at
  1. the environment variable  CONNECTOR_SECRET__<REF>   (REF upper-cased, every character
     outside A-Z0-9 replaced by `_`, so "acme-gcs" -> CONNECTOR_SECRET__ACME_GCS), then
  2. a file named <REF> inside CONNECTOR_SECRETS_DIR (default /run/secrets/connectors; a
     Docker/Kubernetes secret mount).

NOT implemented: AWS/GCP Secret Manager, Vault. A real secrets manager is the production
target; this provider is the stub that keeps the interface honest. Exceptions raised here
never contain the secret value.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping, Protocol

SECRET_REF_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,99}$")
ENV_PREFIX = "CONNECTOR_SECRET__"


class SecretError(Exception):
    """The secret cannot be resolved. The message names the reference, never a value."""


def validate_secret_ref(ref: str) -> str:
    if not isinstance(ref, str) or not SECRET_REF_RE.match(ref) or ".." in ref:
        raise SecretError("secret_ref must be 1-100 chars of a-z, 0-9, '_', '-' or '.', "
                          "starting with a letter or digit")
    return ref


class SecretsProvider(Protocol):
    def get(self, ref: str) -> str: ...


class EnvFileSecretsProvider:
    def __init__(self, env: Mapping[str, str] | None = None, directory: Path | None = None) -> None:
        self._env = os.environ if env is None else env
        self._dir = directory

    def get(self, ref: str) -> str:
        validate_secret_ref(ref)
        value = self._env.get(ENV_PREFIX + re.sub(r"[^A-Za-z0-9]", "_", ref).upper())
        if value and value.strip():
            return value.strip()
        directory = self._dir
        if directory is None:
            from app.connectors.settings import connector_settings
            directory = connector_settings().secrets_dir
        path = Path(directory) / ref
        try:
            if path.resolve().parent == Path(directory).resolve() and path.is_file():
                value = path.read_text().strip()
                if value:
                    return value
        except OSError:
            pass
        raise SecretError(f"secret {ref!r} is not available")


def get_secrets_provider() -> SecretsProvider:
    return EnvFileSecretsProvider()
