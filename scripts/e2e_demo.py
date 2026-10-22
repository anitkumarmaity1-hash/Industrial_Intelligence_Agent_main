"""End-to-end demo flow, runnable against a live stack or in-process (tests/test_e2e_demo.py).

  1. create two fresh tenants (A = "Company B" sample data, B = bystander) with admins and a viewer
  2. A's admin logs in -> uploads the Company B CSV -> previews -> dry-runs and confirms the mapping
  3. the job succeeds (worker) -> fleet health lists the machines, with anomalies and a series
  4. an investigation returns and appears in the history
  5. tenant B sees none of it; A's viewer cannot write

    DATABASE_URL=<owner url> python scripts/e2e_demo.py --api http://localhost:8000
        (the worker must be running: `docker compose up` starts it)

Every step asserts; the script exits non-zero on the first failure. It creates `e2e-*` tenants and,
unless --keep, removes them at the end (owner connection).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Callable

import requests
from sqlalchemy import create_engine, text

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import seed_demo  # noqa: E402

SAMPLE = HERE.parent / "sample_data" / "company_b"
PASSWORD = "e2e-demo-password-" + uuid.uuid4().hex[:8]
CLEAN_TABLES = ("investigations", "audit_log", "rejected_rows", "jobs", "uploads", "ingestion_checkpoints",
                "sensor_readings", "machine_anomalies", "sensor_summary", "sensor_registry",
                "tenant_settings", "users", "machines")

Request = Callable[..., "requests.Response"]


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def run_flow(request: Request, engine, wait_for_job: Callable[[str, dict], dict],
             created: list[str] | None = None) -> dict:
    """`request(method, path, **kw)` -> response; `wait_for_job(job_id, headers)` -> final job dict."""
    suffix = uuid.uuid4().hex[:6]
    ta, tb = f"e2e-a-{suffix}", f"e2e-b-{suffix}"
    if created is not None:
        created.extend([ta, tb])        # so the caller can clean up even if a step below fails
    with engine.begin() as c:
        for t in (ta, tb):
            seed_demo.ensure_tenant(c, t, f"E2E {t}")
            seed_demo.ensure_user(c, t, "admin@e2e.example", PASSWORD, "admin")
        seed_demo.ensure_user(c, ta, "viewer@e2e.example", PASSWORD, "viewer")

    def login(tenant: str, who: str) -> dict:
        r = request("POST", "/auth/login", json={"tenant": tenant, "email": f"{who}@e2e.example", "password": PASSWORD})
        check(r.status_code == 200, f"{tenant}/{who} logs in")
        return {"Authorization": f"Bearer {r.json()['access_token']}"}

    admin, viewer, other = login(ta, "admin"), login(ta, "viewer"), login(tb, "admin")
    csv_bytes = (SAMPLE / "readings.csv").read_bytes()
    mapping = json.loads((SAMPLE / "mapping.json").read_text())
    n_machines = len({ln.split(b",")[0] for ln in csv_bytes.splitlines()[1:]})

    check(request("GET", "/fleet/summary", headers=admin).json()["total"] == 0, "a new tenant starts with an empty fleet")
    up = request("POST", "/uploads", json={"filename": "readings.csv"}, headers=admin).json()
    uid = up["upload_id"]
    check(request("POST", f"/uploads/{uid}/content", data=csv_bytes,
                  headers={**admin, "Content-Type": "text/csv"}).status_code == 200, "the CSV uploads")
    pv = request("POST", f"/uploads/{uid}/preview", json={"rows": 5}, headers=admin).json()
    check({c["name"] for c in pv["columns"]} >= {"AssetTag", "ReadTime", "PressKPa"}, "the preview lists the file's columns")
    dry = request("POST", f"/uploads/{uid}/mapping", json={"mapping": mapping, "confirm": False}, headers=admin).json()
    check(dry["valid"] and dry["report"]["total_errors"] == 0, "the mapping validates on a sample")
    job = request("POST", f"/uploads/{uid}/mapping", json={"mapping": mapping, "confirm": True}, headers=admin).json()["job"]
    done = wait_for_job(job["job_id"], admin)
    check(done["status"] == "succeeded", f"the job succeeds (rows loaded: {done['rows_loaded']})")

    fleet = request("GET", "/fleet/summary", headers=admin).json()
    check(fleet["total"] == n_machines == len(fleet["items"]), f"fleet health lists all {n_machines} machines")
    check(all(i["risk_level"] != "UNKNOWN" for i in fleet["items"]), "every machine has a risk level")
    anomalies = {i["machine_id"]: request("GET", f"/machines/{i['machine_id']}/anomalies", headers=admin).json()
                 for i in fleet["items"]}
    check(any(anomalies.values()), f"fleet health shows anomalies ({sum(map(len, anomalies.values()))} total)")
    flagged = next(m for m, a in anomalies.items() if a)
    sensors = request("GET", "/sensors", headers=admin).json()
    check({s["sensor_name"] for s in sensors} == set(mapping["sensors"]), "the sensor registry matches the mapping")
    series = request("GET", f"/machines/{flagged}/series", params={"sensor": sensors[0]["sensor_name"]}, headers=admin).json()
    check(len(series["points"]) > 0 and series["normal_max"] is not None or series["normal_min"] is not None,
          "a sensor series with the tenant's normal range is returned")

    inv = request("POST", "/investigate", json={"question": f"Why is {flagged} abnormal?", "machine_id": flagged}, headers=admin)
    check(inv.status_code == 200 and inv.json()["machine_id"] == flagged, f"an investigation of {flagged} returns")
    hist = request("GET", "/investigations", headers=viewer).json()
    check(len(hist) == 1 and hist[0]["machine_id"] == flagged, "it appears in the history (and a viewer can read it)")

    check(request("GET", "/fleet/summary", headers=other).json()["total"] == 0, "tenant B's fleet is empty")
    check(request("GET", "/investigations", headers=other).json() == [], "tenant B has no investigations")
    check(request("GET", "/jobs", headers=other).json() == [], "tenant B has no jobs")
    check(request("GET", f"/machines/{flagged}/health", headers=other).status_code == 404, "tenant B cannot read A's machine")
    check(request("GET", f"/jobs/{job['job_id']}", headers=other).status_code == 404, "tenant B cannot read A's job")
    check(request("POST", "/uploads", json={"filename": "x.csv"}, headers=viewer).status_code == 403, "a viewer cannot upload")
    return {"tenants": [ta, tb], "machines": n_machines, "flagged": flagged}


def cleanup(engine, tenants: list[str]) -> None:
    with engine.begin() as c:
        for table in CLEAN_TABLES:
            c.execute(text(f"DELETE FROM {table} WHERE tenant_id = ANY(:t)"), {"t": tenants})
        c.execute(text("DELETE FROM tenants WHERE tenant_id = ANY(:t)"), {"t": tenants})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default=os.environ.get("API_BASE_URL", "http://localhost:8000"))
    ap.add_argument("--timeout", type=int, default=300, help="seconds to wait for the job")
    ap.add_argument("--keep", action="store_true", help="do not delete the e2e tenants afterwards")
    args = ap.parse_args(argv)
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("DATABASE_URL (owner) is required to create the e2e tenants.")
        return 1
    base = args.api.rstrip("/")

    def request(method, path, **kw):
        return requests.request(method, base + path, timeout=120, **kw)

    def wait_for_job(job_id: str, headers: dict) -> dict:
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            j = request("GET", f"/jobs/{job_id}", headers=headers).json()
            if j["status"] in ("succeeded", "failed"):
                return j
            time.sleep(3)
        raise AssertionError(f"job {job_id} did not finish within {args.timeout}s (is the worker running?)")

    engine = create_engine(url)
    seed_demo.wait_for_api(base)
    created: list[str] = []
    try:
        result = run_flow(request, engine, wait_for_job, created)
        print("E2E OK:", json.dumps({k: result[k] for k in ("machines", "flagged")}))
        return 0
    except AssertionError as exc:
        print("E2E FAILED:", exc)
        return 1
    finally:
        if created and not args.keep:
            cleanup(engine, created)


if __name__ == "__main__":
    raise SystemExit(main())
