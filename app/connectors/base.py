"""Connector interface and error types.

Every ConnectorError carries a FIXED, secret-free message: the text of an underlying
library exception (which can contain URLs, account names, even request bodies) is never
copied into it, so nothing a connector raises can leak a credential into a run, a job
error, an audit row or the logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import BinaryIO, Callable, Iterator, Protocol


class ConnectorError(Exception):
    message = "connector error"
    permanent = False       # permanent: retrying the same object version cannot succeed
    fatal = False           # fatal: the whole run must stop (every object would fail)

    def __init__(self) -> None:
        super().__init__(self.message)


class ConnectorAuthError(ConnectorError):
    message = "the storage service rejected the credentials"
    fatal = True


class ConnectorAccessError(ConnectorError):
    message = "the bucket was not found or access to it is denied"
    fatal = True


class ConnectorTransientError(ConnectorError):
    message = "the storage service is temporarily unavailable"


class ObjectGoneError(ConnectorError):
    message = "the object no longer exists at that version"
    permanent = True


class ObjectTooLargeError(ConnectorError):
    message = "the file exceeds the size limit"
    permanent = True


@dataclass(frozen=True)
class ObjectInfo:
    key: str
    version: str            # GCS generation: changes on every overwrite
    size: int
    updated: datetime       # naive UTC


class ObjectStore(Protocol):
    def check_access(self) -> None:
        """Raise a ConnectorError unless the bucket can be listed. Reads no object."""

    def list_objects(self, prefix: str) -> Iterator[ObjectInfo]: ...

    def download(self, info: ObjectInfo, fileobj: BinaryIO, max_bytes: int) -> None:
        """Write exactly this object version to `fileobj`."""


# (connector row, secret value or None) -> ObjectStore
StoreFactory = Callable[[dict, "str | None"], ObjectStore]
