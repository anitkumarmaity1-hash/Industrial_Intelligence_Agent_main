"""
Onboard a new tenant's own sensor/maintenance export.

Usage:
    python scripts/onboard_tenant.py \\
        --tenant acme --name "Acme Manufacturing" \\
        --operational-csv their_export.csv --operational-mapping acme_op_mapping.json \\
        --maintenance-csv their_maint.csv --maintenance-mapping acme_maint_mapping.json

What it does, in order (each step's output is what the next stage reads —
nothing here is a parallel path, it's the front of the existing pipeline):
  1. Creates the tenant (scripts/manage_tenants.py create), if it doesn't
     already exist, and prints its API key.
  2. Validates the mapping config(s) against app.onboarding.mapping, and
     normalizes the raw CSV(s) into the exact schema
     spark_jobs/ingestion.py already validates (app.onboarding.normalize),
     writing them to spark_jobs.ingestion.operational_path_for(tenant) /
     maintenance_path_for(tenant).
  3. Prints the exact two follow-up commands (they are NOT run
     automatically — each is a separate, inspectable step with its own
     failure mode):
       python scripts/run_pipeline.py --tenant <tenant>
       python scripts/load_postgres.py --tenant <tenant> \\
           --processed-dir data/processed/tenants/<tenant>

A mapping config is JSON; see app/onboarding/mapping.py for the schema, or
--print-example-mapping operational|maintenance for a starting point.

Two operational paths exist:
  * legacy (--operational-csv): AI4I-style readings (air/process temperature,
    rotational speed, torque, tool wear) under the tenant's own names/units -
    see app/onboarding/mapping.py.
  * arbitrary sensors (--sensor-csv + --sensor-mapping, Phase 1): any set of
    sensors with declared units, per-sensor normal ranges and z-score
    thresholds - see app/onboarding/sensors.py. Writes a long-format bundle
    (sensor_readings.csv, machines.csv, sensor_registry.json); then
        python scripts/load_sensor_readings.py --tenant <t>
        python scripts/run_generic_pipeline.py --tenant <t>
        python scripts/load_postgres.py --tenant <t> --processed-dir ...
    Bad rows are reported per line; nothing is written unless the file is
    clean, or --allow-partial is given (then bad readings are dropped).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.tenancy import validate_tenant_id  # noqa: E402
from app.onboarding.mapping import MappingConfig  # noqa: E402
from app.onboarding.normalize import (  # noqa: E402
    NormalizationError,
    normalize_maintenance,
    normalize_operational,
)
from app.onboarding.paths import maintenance_path_for, operational_path_for, raw_dir_for  # noqa: E402
from app.onboarding.sensors import (  # noqa: E402
    SensorMapping,
    SensorMappingError,
    ai4i_preset_dict,
    normalize_sensor_readings,
    write_bundle,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

EXAMPLE_OPERATIONAL_MAPPING = {
    "dataset_kind": "operational",
    "columns": {
        "machine_id": "AssetID",
        "timestamp": "ReadingTime",
        "production_line": "Line",
        "type": "QualityGrade",
        "air_temperature_k": {"source_column": "AirTempC", "unit": "C"},
        "process_temperature_k": {"source_column": "ProcessTempC", "unit": "C"},
        "rotational_speed_rpm": {"source_column": "SpindleRPM", "unit": "rpm"},
        "torque_nm": {"source_column": "TorqueLbFt", "unit": "lbf_ft"},
        "tool_wear_min": {"source_column": "ToolWearHours", "unit": "hours"},
    },
    "machine_type_map": {"Standard": "L", "Advanced": "M", "Premium": "H"},
}
EXAMPLE_MAINTENANCE_MAPPING = {
    "dataset_kind": "maintenance",
    "columns": {
        "machine_id": "AssetID",
        "event_date": "ServiceDate",
        "event_type": "WorkType",
        "technician_notes": "Notes",
        "resolved": "Status",
    },
}


EXAMPLE_SENSOR_MAPPING = {
    "dataset_kind": "sensor_readings",
    "format": "wide",
    "machine_id": "AssetTag",
    "timestamp": "ReadTime",
    "production_line": "Area",
    "sensors": {
        "discharge_pressure": {"source": "PressKPa", "unit": "kPa", "target_unit": "bar",
                               "normal_min": 1.5, "normal_max": 6.0},
        "motor_current": {"source": "AmpsRMS", "unit": "A", "normal_max": 40.0,
                          "z_score_threshold": 3.5},
        "bearing_temp": {"source": "BrgTempF", "unit": "F", "target_unit": "C",
                         "normal_max": 85.0},
    },
    "anomaly": {"z_score_threshold": 3.0, "rolling_window_readings": 288},
}


def _ensure_tenant(tenant_id: str, name: str | None) -> None:
    check = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "manage_tenants.py"), "list"],
        capture_output=True, text=True)
    if any(line.split()[0] == tenant_id for line in check.stdout.splitlines() if line.split()):
        logger.info("Tenant %s already exists; reusing it.", tenant_id)
        return
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "manage_tenants.py"),
         "create", tenant_id, "--name", name or tenant_id],
        capture_output=True, text=True)
    print(result.stderr or result.stdout)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tenant")
    parser.add_argument("--name", help="Display name (used only if the tenant doesn't exist yet)")
    parser.add_argument("--operational-csv")
    parser.add_argument("--operational-mapping")
    parser.add_argument("--maintenance-csv")
    parser.add_argument("--maintenance-mapping")
    parser.add_argument("--sensor-csv", help="Raw export with the tenant's OWN sensors (Phase 1)")
    parser.add_argument("--sensor-mapping", help="JSON mapping for --sensor-csv ('preset': 'ai4i' is allowed)")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Write the clean readings even if some rows were rejected")
    parser.add_argument("--print-example-mapping", choices=["operational", "maintenance", "sensors", "ai4i"])
    parser.add_argument("--skip-create-tenant", action="store_true",
                        help="Assume the tenant already exists (e.g. re-onboarding new data)")
    args = parser.parse_args()

    if args.print_example_mapping:
        example = {"operational": EXAMPLE_OPERATIONAL_MAPPING, "maintenance": EXAMPLE_MAINTENANCE_MAPPING,
                   "sensors": EXAMPLE_SENSOR_MAPPING, "ai4i": ai4i_preset_dict()}[args.print_example_mapping]
        print(json.dumps(example, indent=2))
        return 0

    if not args.tenant:
        parser.error("--tenant is required (unless using --print-example-mapping)")
    if not (args.operational_csv or args.maintenance_csv or args.sensor_csv):
        parser.error("provide at least one of --operational-csv, --maintenance-csv or --sensor-csv "
                     "(each paired with its own --*-mapping)")
    if args.sensor_csv and not args.sensor_mapping:
        parser.error("--sensor-csv requires --sensor-mapping")
    if args.operational_csv and not args.operational_mapping:
        parser.error("--operational-csv requires --operational-mapping")
    if args.maintenance_csv and not args.maintenance_mapping:
        parser.error("--maintenance-csv requires --maintenance-mapping")

    try:
        tenant_id = validate_tenant_id(args.tenant)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    # Validate the sensor data BEFORE creating anything, so a bad export
    # leaves no half-onboarded tenant behind.
    sensor_result = None
    if args.sensor_csv:
        try:
            sensor_mapping = SensorMapping.from_json_file(args.sensor_mapping)
            sensor_result = normalize_sensor_readings(args.sensor_csv, sensor_mapping)
        except (SensorMappingError, FileNotFoundError) as exc:
            logger.error("%s", exc)
            return 1
        logger.info("%s", sensor_result.report.as_text())
        if not sensor_result.report.ok and not args.allow_partial:
            logger.error("Sensor file has problems (above); nothing written. Fix them, or "
                         "re-run with --allow-partial to keep only the clean readings.")
            return 1
        if sensor_result.report.readings_accepted == 0:
            logger.error("No usable readings; nothing written.")
            return 1

    if not args.skip_create_tenant:
        _ensure_tenant(tenant_id, args.name)

    try:
        if sensor_result is not None:
            paths = write_bundle(raw_dir_for(tenant_id), sensor_result, sensor_mapping)
            logger.info("Wrote %d readings for %d machine(s) -> %s",
                        len(sensor_result.readings), len(sensor_result.machines), paths["readings"].parent)

        if args.operational_csv:
            mapping = MappingConfig.from_json_file(args.operational_mapping)
            df = normalize_operational(args.operational_csv, mapping)
            out_path = Path(operational_path_for(tenant_id))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            logger.info("Wrote %d operational rows -> %s", len(df), out_path)

        if args.maintenance_csv:
            mapping = MappingConfig.from_json_file(args.maintenance_mapping)
            df = normalize_maintenance(args.maintenance_csv, mapping)
            out_path = Path(maintenance_path_for(tenant_id))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            logger.info("Wrote %d maintenance rows -> %s", len(df), out_path)
    except (ValueError, NormalizationError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 1

    processed_dir = os.path.join("data", "processed", "tenants", tenant_id)
    if sensor_result is not None:
        logger.info(
            "\nNext steps (sensor data):\n"
            "  python scripts/load_sensor_readings.py --tenant %s\n"
            "  python scripts/run_generic_pipeline.py --tenant %s\n"
            "  python scripts/load_postgres.py --tenant %s --processed-dir %s",
            tenant_id, tenant_id, tenant_id, processed_dir)
        if not (args.operational_csv or args.maintenance_csv):
            return 0
    logger.info(
        "\nNext steps:\n"
        "  python scripts/run_pipeline.py --tenant %s\n"
        "  python scripts/load_postgres.py --tenant %s --processed-dir %s\n"
        "  python scripts/ingest_documents.py --tenant %s   # if they have maintenance manuals",
        tenant_id, tenant_id, processed_dir, tenant_id,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
