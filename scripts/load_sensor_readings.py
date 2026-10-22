"""
Load a tenant's onboarded sensor bundle into Postgres (sensor_registry,
tenant_settings.anomaly, machines, sensor_readings). Idempotent - safe to
re-run; already-present readings are left untouched.

    python scripts/load_sensor_readings.py --tenant acme
    python scripts/load_sensor_readings.py --tenant default --ai4i-demo   # demo tenant, wide -> long

--ai4i-demo reads data/raw/synthetic_operational.csv and loads its eight
AI4I sensors into sensor_readings in long format (about 2.5M rows; opt-in,
not part of the committed seed). The AI4I detection itself still runs on the
wide file via scripts/run_pipeline.py - see CHANGES_PHASE1.md.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402

from app.core.tenancy import DEFAULT_TENANT_ID, validate_tenant_id  # noqa: E402
from app.onboarding.load import load_bundle, read_bundle  # noqa: E402
from app.onboarding.paths import operational_path_for, raw_dir_for  # noqa: E402
from app.onboarding.sensors import ai4i_wide_to_long  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tenant", default=DEFAULT_TENANT_ID)
    parser.add_argument("--dir", type=Path, help="Bundle directory (default: the tenant's raw dir)")
    parser.add_argument("--ai4i-demo", action="store_true",
                        help="Unpivot the demo tenant's wide AI4I operational CSV into sensor_readings")
    args = parser.parse_args()
    tenant_id = validate_tenant_id(args.tenant)

    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        logger.error("DATABASE_URL not set.")
        return 1
    engine = create_engine(url)

    if args.ai4i_demo:
        if tenant_id != DEFAULT_TENANT_ID:
            logger.error("--ai4i-demo is only for the demo tenant.")
            return 1
        wide = pd.read_csv(operational_path_for(tenant_id))
        readings = ai4i_wide_to_long(wide)
        machines = wide[["machine_id", "production_line", "type"]].drop_duplicates()
        registry = None   # migration 0005 already seeded the demo registry
    else:
        readings, machines, registry = read_bundle(args.dir or raw_dir_for(tenant_id))

    with engine.begin() as conn:
        result = load_bundle(conn, tenant_id, readings, machines, registry)
    logger.info("tenant=%s: %s", tenant_id, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
