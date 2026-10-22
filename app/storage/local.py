"""Local-filesystem StorageProvider (the default, and what tests use)."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO

from app.storage import StorageError, assert_key_for_tenant


class LocalStorage:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def _path(self, key: str, tenant_id: str) -> Path:
        assert_key_for_tenant(key, tenant_id)
        path = (self.root / key).resolve()
        if self.root not in path.parents:          # belt and braces after the key check
            raise StorageError("resolved path escapes the storage root")
        return path

    def put(self, key: str, fileobj: BinaryIO, *, tenant_id: str) -> None:
        path = self._path(key, tenant_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".part-")
            try:
                with os.fdopen(fd, "wb") as out:
                    shutil.copyfileobj(fileobj, out, 1024 * 1024)
                os.replace(tmp, path)               # atomic: readers never see half a file
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        except OSError as exc:
            raise StorageError(f"local put failed: {exc}") from exc

    def open(self, key: str, *, tenant_id: str) -> BinaryIO:
        try:
            return open(self._path(key, tenant_id), "rb")
        except OSError as exc:
            raise StorageError(f"local open failed: {exc}") from exc

    def exists(self, key: str, *, tenant_id: str) -> bool:
        return self._path(key, tenant_id).is_file()

    def delete(self, key: str, *, tenant_id: str) -> None:
        self._path(key, tenant_id).unlink(missing_ok=True)
