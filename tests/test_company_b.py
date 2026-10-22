"""
Phase 1 acceptance test: a SECOND fleet with a different sensor vocabulary
is onboarded end to end and detected with no code change, and tenant A's
credential can never reach tenant B's data.

Company B ("company-b-test") runs pumps and reports discharge pressure
(kPa, stored in bar), motor current (mA, stored in A) and bearing
temperature (degF, stored in degC) - none of the AI4I columns. Its raw
export is synthesized below with KNOWN injected faults:

    PUMP-A101  discharge pressure jumps ~+120 kPa for 20 readings  -> z-score
    PUMP-A102  motor current 45 A (limit 40 A) for 20 readings     -> range
    PUMP-A103  healthy control                                     -> nothing

Noise is bounded (uniform), so a healthy machine's z-score cannot reach 3
once it has a baseline; the first 24 h of each machine are excluded from the
"healthy stays quiet" assertion because the original engine has no minimum
baseline (documented in README Limitations).

Company A ("company-a-test") is a deliberately different tenant that owns a
machine with the SAME id as one of B's (PUMP-A101), so isolation is tested
against the nastiest case: a shared machine id.

Needs DATABASE_URL and Java/Spark, like tests/test_pipeline.py.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from app.core.anomaly_rules import load_anomaly_rules
from app.core.tenancy import generate_api_key, hash_api_key
from app.database import queries
from app.main import app
from app.onboarding.load import apply_registry, load_bundle, read_bundle
from app.onboarding.sensors import (
    SensorMapping,
    normalize_sensor_readings,
    write_bundle,
)
from app.rag import retriever as retriever_module
from app.rag.chunking import Chunk, save_chunks
from spark_jobs.ingestion import get_spark_session
from spark_jobs.long_format import detect, load_long_readings, load_machines
from tests.helpers import env

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

A, B = "company-a-test", "company-b-test"
A_KEY, B_KEY = generate_api_key(), generate_api_key()
START = datetime(2026, 3, 1)
N = 864                      # 3 days of 5-minute readings per machine
FAULT = slice(700, 720)      # day 3, i.e. long after the 24 h warm-up
POISON = "ZXQ-9 centrifugal pump seal flush procedure"

B_MAPPING = {
    "dataset_kind": "sensor_readings",
    "format": "wide",
    "machine_id": "AssetTag",
    "timestamp": "ReadTime",
    "production_line": "Area",
    "sensors": {
        "discharge_pressure": {"source": "PressKPa", "unit": "kPa", "target_unit": "bar",
                               "normal_min": 1.5, "normal_max": 6.0},
        "motor_current": {"source": "AmpsMilli", "unit": "mA", "target_unit": "A", "normal_max": 40.0},
        "bearing_temp": {"source": "BrgTempF", "unit": "F", "target_unit": "C", "normal_max": 85.0},
    },
    "anomaly": {"z_score_threshold": 3.0, "rolling_window_readings": 288},
}


def _b_export() -> pd.DataFrame:
    rng = np.random.RandomState(7)
    frames = []
    for tag in ("PUMP-A101", "PUMP-A102", "PUMP-A103"):
        pressure = 400.0 + rng.uniform(-10, 10, N)            # kPa  (= 4.0 bar)
        current = 25000.0 + rng.uniform(-800, 800, N)         # mA   (= 25 A)
        temp = 150.0 + rng.uniform(-1.5, 1.5, N)              # degF (~ 65.6 C)
        if tag == "PUMP-A101":
            pressure[FAULT] += 120.0
        if tag == "PUMP-A102":
            current[FAULT] = 45000.0
        frames.append(pd.DataFrame({
            "AssetTag": tag,
            "ReadTime": [(START + timedelta(minutes=5 * i)).isoformat() for i in range(N)],
            "Area": "Pump house 1",
            "PressKPa": pressure.round(3), "AmpsMilli": current.round(1), "BrgTempF": temp.round(3),
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def engine():
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    return create_engine(url)


def _cleanup(conn) -> None:
    for table in ("investigations", "investigation_audit_log", "sensor_readings", "maintenance_records",
                  "machine_anomalies", "sensor_summary", "machines", "sensor_registry",
                  "tenant_settings"):
        conn.execute(text(f"DELETE FROM {table} WHERE tenant_id IN (:a, :b)"), {"a": A, "b": B})
    conn.execute(text("DELETE FROM tenants WHERE tenant_id IN (:a, :b)"), {"a": A, "b": B})


@pytest.fixture(scope="module")
def spark():
    s = get_spark_session("pytest-company-b")
    yield s
    s.stop()


@pytest.fixture(scope="module")
def fleet_b(engine, spark, tmp_path_factory):
    """Onboard B through the real code path, run detection, load results."""
    import load_postgres  # scripts/load_postgres.py

    with engine.begin() as conn:
        _cleanup(conn)
        conn.execute(text("""
            INSERT INTO tenants (tenant_id, name, api_key_hash) VALUES
              (:a, 'Company A (test)', :ah), (:b, 'Company B (test)', :bh)
        """), {"a": A, "b": B, "ah": hash_api_key(A_KEY), "bh": hash_api_key(B_KEY)})

    work = tmp_path_factory.mktemp("company_b")
    mapping = SensorMapping.from_dict(B_MAPPING)
    data = normalize_sensor_readings(_b_export(), mapping)
    assert data.report.ok, data.report.as_text()
    bundle = work / "bundle"
    write_bundle(bundle, data, mapping)

    readings, machines, registry = read_bundle(bundle)
    with engine.begin() as conn:
        loaded = load_bundle(conn, B, readings, machines, registry)
        rules = load_anomaly_rules(conn, B)

    long_df = load_long_readings(spark, str(bundle / "sensor_readings.csv"))
    machines_df = load_machines(spark, str(bundle / "machines.csv"))
    summary, anomalies = detect(long_df, machines_df, rules)
    out = work / "processed"
    summary.write.mode("overwrite").partitionBy("machine_id").parquet(str(out / "sensor_summary.parquet"))
    anomalies.write.mode("overwrite").partitionBy("machine_id").parquet(str(out / "machine_anomalies.parquet"))
    load_postgres.load_machines(engine, B, out)
    load_postgres.load_sensor_summary(engine, B, out)
    load_postgres.load_machine_anomalies(engine, B, out)

    # Company A: a different vocabulary, and the SAME machine id as B's PUMP-A101.
    with engine.begin() as conn:
        queries.upsert_sensor_registry(conn, [
            {"sensor_name": "spindle_vibration", "unit": "mm_s", "normal_max": 7.0}], tenant_id=A)
        queries.upsert_machines(conn, [
            {"machine_id": "PUMP-A101", "production_line": "A-LINE", "type": None},
            {"machine_id": "FAN-7", "production_line": "A-LINE", "type": None}], tenant_id=A)
        queries.insert_sensor_readings(conn, [
            {"machine_id": "PUMP-A101", "sensor_name": "spindle_vibration",
             "ts": START + timedelta(hours=i), "value": 2.0 + i / 100} for i in range(5)], tenant_id=A)
        queries.insert_sensor_readings(conn, [
            {"machine_id": "FAN-7", "sensor_name": "spindle_vibration",
             "ts": START, "value": 9.9}], tenant_id=A)
    yield {"loaded": loaded, "bundle": bundle, "data": data}
    with engine.begin() as conn:
        _cleanup(conn)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _h(key: str) -> dict[str, str]:
    return {"X-API-Key": key}


def _anoms(engine, machine: str) -> pd.DataFrame:
    with engine.connect() as conn:
        rows = queries.get_machine_anomalies(conn, machine, limit=1000, tenant_id=B)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Onboarding -> storage
# ---------------------------------------------------------------------

def test_b_was_onboarded_in_converted_units(engine, fleet_b):
    assert fleet_b["loaded"]["readings_inserted"] == 3 * 3 * N
    with engine.connect() as conn:
        reg = {r["sensor_name"]: r for r in queries.list_sensor_registry(conn, tenant_id=B)}
        pressure = queries.get_sensor_readings(
            conn, "PUMP-A103", "discharge_pressure", limit=1, tenant_id=B)[0]["value"]
        temp = queries.get_sensor_readings(
            conn, "PUMP-A103", "bearing_temp", limit=1, tenant_id=B)[0]["value"]
    assert set(reg) == {"discharge_pressure", "motor_current", "bearing_temp"}
    assert reg["discharge_pressure"]["unit"] == "bar" and reg["motor_current"]["unit"] == "A"
    assert 3.89 < pressure < 4.11          # 400 kPa +-10  ->  bar
    assert 65.0 < temp < 66.3              # 150 degF +-1.5 -> degC


def test_reloading_the_same_bundle_is_idempotent(engine, fleet_b):
    readings, machines, registry = read_bundle(fleet_b["bundle"])
    with engine.begin() as conn:
        before = conn.execute(text(
            "SELECT COUNT(*) FROM sensor_readings WHERE tenant_id = :t"), {"t": B}).scalar()
        again = load_bundle(conn, B, readings, machines, registry)
        after = conn.execute(text(
            "SELECT COUNT(*) FROM sensor_readings WHERE tenant_id = :t"), {"t": B}).scalar()
    assert again["readings_inserted"] == 0 and before == after == 3 * 3 * N


# ---------------------------------------------------------------------
# Detection: (a) correct anomalies found
# ---------------------------------------------------------------------

def test_pressure_fault_is_found_by_zscore_on_a_sensor_the_code_never_heard_of(engine, fleet_b):
    a = _anoms(engine, "PUMP-A101")
    fault_start = START + timedelta(minutes=5 * FAULT.start)
    fault_end = START + timedelta(minutes=5 * FAULT.stop)
    in_fault = a[(a["detected_at"] >= fault_start) & (a["detected_at"] < fault_end)]
    assert len(in_fault) >= 1
    assert in_fault["triggered_reasons"].str.contains(
        "Discharge pressure statistically unlike").any()


def test_current_fault_is_found_by_the_range_rule(engine, fleet_b):
    a = _anoms(engine, "PUMP-A102")
    fault_start = START + timedelta(minutes=5 * FAULT.start)
    fault_end = START + timedelta(minutes=5 * FAULT.stop)
    in_fault = a[(a["detected_at"] >= fault_start) & (a["detected_at"] < fault_end)]
    ranged = in_fault[in_fault["triggered_reasons"].str.contains("Motor current outside its normal range")]
    assert len(ranged) == FAULT.stop - FAULT.start      # every faulty reading, none missed
    assert ranged["triggered_reasons"].str.contains(r"\[None, 40.0\] A").all()


def test_healthy_pump_is_quiet_once_it_has_a_baseline(engine, fleet_b):
    a = _anoms(engine, "PUMP-A103")
    warm = a[a["detected_at"] >= START + timedelta(days=1)] if len(a) else a
    assert len(warm) == 0


def test_no_anomaly_reasons_mention_ai4i_rules_for_b(engine, fleet_b):
    for machine in ("PUMP-A101", "PUMP-A102"):
        text_all = " ".join(_anoms(engine, machine)["triggered_reasons"].fillna(""))
        for ai4i_phrase in ("AI4I", "Heat dissipation", "Overstrain", "Power outside"):
            assert ai4i_phrase not in text_all


def test_investigation_flags_the_faulty_pump_and_not_the_healthy_one(client, fleet_b):
    sick = client.post("/investigate", headers=_h(B_KEY),
                       json={"machine_id": "PUMP-A101", "question": "Why is this pump at risk?"})
    assert sick.status_code == 200, sick.text
    assert sick.json()["risk_level"] in ("MEDIUM", "HIGH")
    # The fault is 20 of 864 readings: B's last-24h trend has 20/288 = 7% anomalous?
    # No - the fault is on day 3, inside the last 24 h, so it must register.
    fleet = client.post("/investigate", headers=_h(B_KEY),
                        json={"question": "Which machines currently show abnormal behavior?"})
    assert fleet.status_code == 200, fleet.text
    ranked = {r["machine_id"]: r for r in fleet.json()["fleet_ranking"]}
    assert set(ranked) <= {"PUMP-A101", "PUMP-A102", "PUMP-A103"}
    assert "PUMP-A103" not in ranked or ranked["PUMP-A103"]["risk_level"] == "LOW"


def test_a_tenant_can_change_its_rules_with_no_code_change(engine, fleet_b):
    """Tighten B's current limit and the same engine flags more: config only."""
    with engine.begin() as conn:
        queries.upsert_sensor_registry(conn, [
            {"sensor_name": "motor_current", "unit": "A", "normal_max": 20.0}], tenant_id=B)
        rules = load_anomaly_rules(conn, B)
        queries.upsert_sensor_registry(conn, [
            {"sensor_name": "motor_current", "unit": "A", "normal_max": 40.0}], tenant_id=B)
    current = next(s for s in rules.sensors if s.sensor_name == "motor_current")
    assert current.normal_max == 20.0
    assert rules.ai4i is None                      # B never gets the AI4I pack


# ---------------------------------------------------------------------
# (b) Isolation: A's key can never see B's readings, machines, registry, documents
# ---------------------------------------------------------------------

def test_registries_are_disjoint(client, fleet_b):
    a = {s["sensor_name"] for s in client.get("/sensors", headers=_h(A_KEY)).json()}
    b = {s["sensor_name"] for s in client.get("/sensors", headers=_h(B_KEY)).json()}
    assert a == {"spindle_vibration"}
    assert b == {"discharge_pressure", "motor_current", "bearing_temp"}


def test_machines_are_disjoint_and_a_shared_id_resolves_to_each_own(client, fleet_b):
    a = {m["machine_id"]: m for m in client.get("/machines", headers=_h(A_KEY)).json()}
    b = {m["machine_id"]: m for m in client.get("/machines", headers=_h(B_KEY)).json()}
    assert set(a) == {"PUMP-A101", "FAN-7"}
    assert set(b) == {"PUMP-A101", "PUMP-A102", "PUMP-A103"}
    assert a["PUMP-A101"]["production_line"] == "A-LINE"
    assert b["PUMP-A101"]["production_line"] == "Pump house 1"
    assert a["PUMP-A101"]["type"] is None


def test_readings_never_cross_tenants(client, fleet_b):
    a = client.get("/machines/PUMP-A101/readings", headers=_h(A_KEY), params={"limit": 5000}).json()
    b = client.get("/machines/PUMP-A101/readings", headers=_h(B_KEY), params={"limit": 5000}).json()
    assert {r["sensor_name"] for r in a} == {"spindle_vibration"} and len(a) == 5
    assert {r["sensor_name"] for r in b} == {"discharge_pressure", "motor_current", "bearing_temp"}
    assert not ({r["sensor_name"] for r in a} & {r["sensor_name"] for r in b})


def test_other_tenants_machines_are_404_on_every_route(client, fleet_b):
    for path in ("/machines/PUMP-A102", "/machines/PUMP-A102/health", "/machines/PUMP-A102/anomalies",
                 "/machines/PUMP-A102/sensors", "/machines/PUMP-A102/readings"):
        assert client.get(path, headers=_h(A_KEY)).status_code == 404, path
    for path in ("/machines/FAN-7", "/machines/FAN-7/readings"):
        assert client.get(path, headers=_h(B_KEY)).status_code == 404, path
    r = client.post("/investigate", headers=_h(A_KEY),
                    json={"machine_id": "PUMP-A102", "question": "Why is this pump at risk?"})
    assert r.status_code == 404


def test_b_machine_data_is_invisible_to_a_even_for_the_shared_id(client, fleet_b):
    assert client.get("/machines/PUMP-A101/anomalies", headers=_h(A_KEY)).json() == []
    assert len(client.get("/machines/PUMP-A101/anomalies", headers=_h(B_KEY)).json()) > 0


def test_a_cannot_write_a_reading_with_one_of_bs_sensors(engine, fleet_b):
    """Database-level guard (FK to sensor_registry): A never declared bearing_temp."""
    with pytest.raises(Exception) as exc:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO sensor_readings (tenant_id, machine_id, sensor_name, ts, value)
                VALUES (:a, 'PUMP-A101', 'bearing_temp', '2026-04-01', 1.0)"""), {"a": A})
    assert "foreign key" in str(exc.value).lower()


def test_documents_are_not_shared_between_the_two_fleets(client, fleet_b, tmp_path):
    base = tmp_path / "document_chunks.jsonl"
    chunk = Chunk(chunk_id="ZXQ-0", doc_id="ZXQ-9", title="ZXQ-9 seal manual", doc_type="manual",
                  section="1", text=POISON, chunk_index=0, source_path="zxq.md",
                  effective_date="2026-01-01", data_class="SYNTHETIC", failure_modes=[])
    save_chunks([chunk], tmp_path / "tenants" / B / "document_chunks.jsonl")
    save_chunks([chunk.__class__(**{**chunk.__dict__, "doc_id": "DEMO-1", "chunk_id": "DEMO-1-0",
                                    "text": "unrelated demo manual"})], base)
    retriever_module._retrievers.clear()
    try:
        with env(CHUNKS_PATH=str(base), RAG_BACKEND="local", PINECONE_API_KEY=None,
                 GOOGLE_CLOUD_PROJECT=None):
            assert {h.doc_id for h in retriever_module.get_retriever(tenant_id=B).search(
                "centrifugal pump seal flush procedure")} == {"ZXQ-9"}
            assert retriever_module.get_retriever(tenant_id=A).search(
                "centrifugal pump seal flush procedure") == []
            r = client.post("/investigate", headers=_h(A_KEY), json={
                "machine_id": "PUMP-A101", "question": "centrifugal pump seal flush procedure"})
            assert r.status_code == 200, r.text
            assert all("ZXQ" not in str(c) for c in r.json()["supporting_citations"])
    finally:
        retriever_module._retrievers.clear()


def test_b_tenant_settings_do_not_leak_into_the_demo_tenant(engine, fleet_b):
    with engine.connect() as conn:
        demo = load_anomaly_rules(conn, "default")
        b = load_anomaly_rules(conn, B)
    assert {s.sensor_name for s in demo.active_sensors} == {
        "torque_nm", "tool_wear_min", "production_rate", "defect_rate", "energy_consumption_kwh"}
    assert demo.ai4i is not None and b.ai4i is None
    assert {s.sensor_name for s in b.active_sensors} == {"discharge_pressure", "motor_current", "bearing_temp"}
