"""
The GCS adapter (app/connectors/gcs.py) against a local FAKE of the GCS JSON API.

The real google-cloud-storage library talks HTTP to this in-process server (through
STORAGE_EMULATOR_HOST, the same switch fake-gcs-server uses), so listing, pagination, generation-exact
downloads and the error translation are exercised for real. fake-gcs-server itself was not
available where this was written; CI/compose can point STORAGE_EMULATOR_HOST at it instead.
"""

from __future__ import annotations

import io
import json
import threading
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("google.cloud.storage")

from app.connectors import gcs  # noqa: E402
from tests.test_connectors import admin, stack, wengine  # noqa: E402,F401  (fixtures)
from app.connectors.base import (  # noqa: E402
    ConnectorAccessError,
    ConnectorAuthError,
    ConnectorTransientError,
    ObjectGoneError,
    ObjectTooLargeError,
)
from app.connectors.validation import build_gcs_config, matches, next_run, validate_schedule  # noqa: E402

PAGE = 2


class FakeGcs:
    def __init__(self) -> None:
        self.objects: dict[str, dict[str, tuple[int, bytes, str]]] = {}   # bucket -> name -> (generation, data, updated)
        self.status: dict[str, int] = {}                                  # bucket -> forced HTTP status
        self.requests: list[str] = []


@pytest.fixture()
def fake(monkeypatch):
    state = FakeGcs()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, code: int) -> None:
            self._send(code, json.dumps({"error": {"code": code, "message": "SECRET-LOOKING-DETAIL"}}).encode())

        def do_GET(self):  # noqa: N802
            state.requests.append(self.path)
            url = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(url.query)
            parts = [urllib.parse.unquote(p) for p in url.path.split("/")]
            # /storage/v1/b/{bucket}/o            list
            # /download/storage/v1/b/{bucket}/o/{name}?alt=media   or /storage/v1/b/{bucket}/o/{name}?alt=media
            if "b" not in parts:
                return self._error(404)
            bucket = parts[parts.index("b") + 1]
            if state.status.get(bucket):
                return self._error(state.status[bucket])
            if bucket not in state.objects:
                return self._error(404)
            rest = parts[parts.index("b") + 3:]
            if not rest:                                        # list
                prefix = qs.get("prefix", [""])[0]
                names = sorted(n for n in state.objects[bucket] if n.startswith(prefix))
                start = int(qs.get("pageToken", ["0"])[0])
                limit = min(PAGE, int(qs.get("maxResults", [str(PAGE)])[0]))
                page = names[start:start + limit]
                body = {"kind": "storage#objects", "items": [
                    {"kind": "storage#object", "name": n, "bucket": bucket,
                     "generation": str(state.objects[bucket][n][0]), "metageneration": "1",
                     "size": str(len(state.objects[bucket][n][1])),
                     "updated": state.objects[bucket][n][2], "timeCreated": state.objects[bucket][n][2]}
                    for n in page]}
                if start + limit < len(names):
                    body["nextPageToken"] = str(start + limit)
                return self._send(200, json.dumps(body).encode())
            name = "/".join(rest)
            obj = state.objects[bucket].get(name)
            gen = qs.get("generation", [None])[0]
            if obj is None or (gen is not None and int(gen) != obj[0]):
                return self._error(404)
            if qs.get("alt") == ["media"]:
                return self._send(200, obj[1], "application/octet-stream")
            return self._send(200, json.dumps({"name": name, "bucket": bucket, "generation": str(obj[0]),
                                               "size": str(len(obj[1])), "updated": obj[2]}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("STORAGE_EMULATOR_HOST", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setattr(gcs, "RETRY_DEADLINE_SECONDS", 2)
    yield state
    server.shutdown()


def _store(bucket: str = "b1"):
    return gcs.GcsObjectStore.from_secret(bucket, None)        # emulator: anonymous credentials


def test_list_paginates_filters_by_prefix_and_reports_generation_size_and_utc_time(fake):
    fake.objects["b1"] = {f"in/f{i}.csv": (100 + i, b"x" * i, f"2026-03-0{i + 1}T10:30:00.250Z") for i in range(5)}
    fake.objects["b1"]["other/x.csv"] = (7, b"zz", "2026-03-01T00:00:00Z")
    got = list(_store().list_objects("in/"))
    assert [o.key for o in got] == [f"in/f{i}.csv" for i in range(5)]       # 5 objects over 3 pages
    assert [o.version for o in got] == ["100", "101", "102", "103", "104"]
    assert [o.size for o in got] == [0, 1, 2, 3, 4]
    assert got[2].updated == datetime(2026, 3, 3, 10, 30, 0, 250000) and got[2].updated.tzinfo is None
    assert len(list(_store().list_objects(""))) == 6
    _store().check_access()
    assert sum("/o?" in r for r in fake.requests) >= 5, "the library really talked to the fake over HTTP"


def test_download_is_generation_exact_and_size_limited(fake):
    fake.objects["b1"] = {"in/a.csv": (5, b"hello", "2026-03-01T00:00:00Z")}
    store = _store()
    [info] = list(store.list_objects("in/"))
    buf = io.BytesIO()
    store.download(info, buf, 1000)
    assert buf.getvalue() == b"hello"
    assert any("alt=media" in r and "generation=5" in r for r in fake.requests), fake.requests
    fake.objects["b1"]["in/a.csv"] = (6, b"newer!", "2026-03-02T00:00:00Z")   # overwritten after listing
    with pytest.raises(ObjectGoneError):
        store.download(info, io.BytesIO(), 1000)
    with pytest.raises(ObjectTooLargeError):
        store.download(type(info)(info.key, info.version, 5000, info.updated), io.BytesIO(), 1000)


def test_http_errors_become_fixed_secret_free_connector_errors(fake):
    fake.objects["b1"] = {"in/a.csv": (1, b"x", "2026-03-01T00:00:00Z")}
    for code, exc in ((401, ConnectorAuthError), (403, ConnectorAccessError), (503, ConnectorTransientError)):
        fake.status["b1"] = code
        with pytest.raises(exc) as ei:
            _store().check_access()
        with pytest.raises(exc):
            list(_store().list_objects(""))
        assert "SECRET-LOOKING-DETAIL" not in str(ei.value) and str(ei.value) == exc.message
    fake.status.clear()
    with pytest.raises(ConnectorAccessError):                    # a bucket that does not exist
        _store("nope").check_access()


def test_credentials_handling(monkeypatch):
    monkeypatch.delenv("STORAGE_EMULATOR_HOST", raising=False)
    with pytest.raises(ConnectorAuthError):                      # no secret and no emulator: never ADC
        gcs.GcsObjectStore.from_secret("b1", None)
    for junk in ("not json", "{}", json.dumps({"type": "service_account", "private_key": "NOT-A-KEY-SENTINEL"})):
        with pytest.raises(ConnectorAuthError) as ei:
            gcs.GcsObjectStore.from_secret("b1", junk)
        assert "SENTINEL" not in str(ei.value) and str(ei.value) == ConnectorAuthError.message
    crypto = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.rsa")
    from cryptography.hazmat.primitives import serialization
    key = crypto.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()
    sa = json.dumps({"type": "service_account", "project_id": "demo-proj", "private_key_id": "k1",
                     "private_key": key, "client_email": "sa@demo-proj.iam.gserviceaccount.com",
                     "token_uri": "https://oauth2.googleapis.com/token"})
    store = gcs.GcsObjectStore.from_secret("b1", sa)             # builds a client; makes no network call
    assert store.client.project == "demo-proj"


def test_validation_helpers():
    assert build_gcs_config("my-bucket.01", "in/2026 q1/", "*.csv") == {
        "bucket": "my-bucket.01", "prefix": "in/2026 q1/", "pattern": "*.csv"}
    for bad in (("UPPER", "", "*.csv"), ("ab", "", "*.csv"), ("my-bucket", "/x", "*.csv"),
                ("my-bucket", "a/../b", "*.csv"), ("my-bucket", "", "a;b"), ("my-bucket", "", "")):
        with pytest.raises(ValueError):
            build_gcs_config(*bad)
    cfg = {"prefix": "in/", "pattern": "*.csv"}
    assert matches(cfg, "in/a.csv") and matches(cfg, "in/sub/a.csv")
    assert not matches(cfg, "in/a.txt") and not matches(cfg, "out/a.csv") and not matches(cfg, "in/dir/")
    validate_schedule("*/15 * * * *", 300)
    for bad in ("* * * * *", "*/2 * * * *", "61 * * * *", "a b c d e", "@hourly", ""):
        with pytest.raises(ValueError):
            validate_schedule(bad, 300)
    assert next_run("0 * * * *", datetime(2026, 1, 1, 10, 30)) == datetime(2026, 1, 1, 11, 0)
    assert next_run("0 * * * *", datetime(2026, 1, 1, 11, 0)) == datetime(2026, 1, 1, 12, 0)


def test_secrets_provider_env_file_and_traversal(tmp_path, monkeypatch):
    from app.core.secrets_provider import EnvFileSecretsProvider, SecretError
    (tmp_path / "from-file").write_text("file-secret\n")
    p = EnvFileSecretsProvider(env={"CONNECTOR_SECRET__MY_REF": " env-secret "}, directory=tmp_path)
    assert p.get("my-ref") == "env-secret" and p.get("from-file") == "file-secret"
    for bad in ("missing", "../etc/passwd", "a/b", "UPPER", "", "x" * 101):
        with pytest.raises(SecretError):
            p.get(bad)


# ---------------------------------------------------------------------
# The real adapter + the real puller + Postgres: no FakeStore anywhere
# ---------------------------------------------------------------------

def test_puller_with_the_real_gcs_adapter_against_the_fake_api(fake, admin, wengine, tmp_path):
    import uuid

    from sqlalchemy import text

    from app.connectors.puller import run_pull
    from app.storage import get_storage
    from tests.test_company_b import B_MAPPING
    from tests.test_connectors import SMALL, TA

    fake.objects["acme-bucket"] = {
        "in/a.csv": (11, SMALL, "2026-03-01T10:00:00Z"),
        "in/b.csv": (12, SMALL + b"PUMP-A102,2026-03-01T00:05:00,Pump house 1,401,25001,151\n", "2026-03-01T11:00:00Z"),
        "in/c.txt": (13, b"ignored", "2026-03-01T12:00:00Z")}
    cid = str(uuid.uuid4())
    with admin.begin() as conn:
        conn.execute(text("INSERT INTO connectors (id, tenant_id, name, type, config, schedule, mapping, secret_ref, next_run_at) "
                          "VALUES (:i, :t, 'real-adapter', 'gcs', CAST(:c AS JSONB), '0 * * * *', CAST(:m AS JSONB), NULL, NOW())"),
                     {"i": cid, "t": TA, "m": json.dumps(B_MAPPING),
                      "c": json.dumps({"bucket": "acme-bucket", "prefix": "in/", "pattern": "*.csv"})})
    try:
        result = run_pull(wengine, cid, TA, storage=get_storage())          # default factory = GCS
        assert (result.status, result.seen, result.queued) == ("ok", 2, 2), result
        with admin.connect() as conn:
            rows = conn.execute(text("SELECT o.object_key, o.object_version, o.status, u.storage_key, u.size_bytes "
                                     "FROM connector_objects o JOIN uploads u ON u.upload_id = o.upload_id "
                                     "WHERE o.connector_id = :c ORDER BY 1"), {"c": cid}).all()
        assert [(r[0], r[1], r[2]) for r in rows] == [("in/a.csv", "11", "queued"), ("in/b.csv", "12", "queued")]
        for r in rows:                                                       # bytes arrived intact
            with get_storage().open(r[3], tenant_id=TA) as f:
                assert len(f.read()) == r[4]
        assert run_pull(wengine, cid, TA, storage=get_storage()).queued == 0   # idempotent over real listing
        fake.status["acme-bucket"] = 403
        r = run_pull(wengine, cid, TA, storage=get_storage())
        assert (r.status, r.message) == ("failed", "the bucket was not found or access to it is denied")
    finally:
        with admin.begin() as conn:
            for t in ("connector_objects", "connector_runs"):
                conn.execute(text(f"DELETE FROM {t} WHERE connector_id = :c"), {"c": cid})
            conn.execute(text("DELETE FROM connectors WHERE id = :c"), {"c": cid})
            conn.execute(text("DELETE FROM jobs WHERE tenant_id = :t"), {"t": TA})
            conn.execute(text("DELETE FROM uploads WHERE tenant_id = :t"), {"t": TA})
