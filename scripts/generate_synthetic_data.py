"""
Generate synthetic industrial operational data.

Produces two CSV files under data/raw/:
  - synthetic_operational.csv : 5-minute sensor + business-metric readings
                                 for 18 machines over SIMULATION_DAYS days
  - synthetic_maintenance.csv : maintenance log entries tied to the
                                 injected failure scenarios

This data is 100% synthetic. It is generated to be *statistically
consistent* with the real AI4I 2020 dataset's documented generation
process and failure thresholds (see pyspark/config.py), but it is NOT
real sensor data and must never be represented as such.

Key modeling choice: tool wear is bounded by routine preventive
replacement (as happens in real factories) for every machine. Only a
machine inside an active, not-yet-repaired TWF/OSF-type degradation
window is allowed to keep accruing wear past the routine-replacement
point — that persistence is exactly what makes it anomalous and
detectable.

Run:
    python scripts/generate_synthetic_data.py
"""

from __future__ import annotations

import csv
import math
import os
import random
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from spark_jobs import config as cfg  # noqa: E402

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OPERATIONAL_PATH = os.path.join(OUTPUT_DIR, "synthetic_operational.csv")
MAINTENANCE_PATH = os.path.join(OUTPUT_DIR, "synthetic_maintenance.csv")

FIELDNAMES = [
    "machine_id",
    "timestamp",
    "production_line",
    "type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "production_rate",
    "energy_consumption_kwh",
    "defect_rate",
    "injected_scenario_active",
]


def build_fleet() -> list[dict]:
    """Assign each machine an id, production line, and quality type."""
    rng = random.Random(cfg.RANDOM_SEED)
    scenario_by_machine = {s.machine_id: s for s in cfg.INJECTED_SCENARIOS}

    types = list(cfg.TYPE_WEIGHTS.keys())
    weights = list(cfg.TYPE_WEIGHTS.values())

    fleet = []
    for i in range(1, cfg.N_MACHINES + 1):
        machine_id = f"M-{i:02d}"
        line = f"LINE-{((i - 1) // cfg.MACHINES_PER_LINE) + 1}"
        machine_type = rng.choices(types, weights=weights, k=1)[0]
        scenario = scenario_by_machine.get(machine_id)
        fleet.append(
            {
                "machine_id": machine_id,
                "production_line": line,
                "type": machine_type,
                "scenario": scenario.scenario if scenario else "HEALTHY",
                "cfg": scenario,
            }
        )
    return fleet


def ramp(day: float, onset: float, span: float) -> float:
    """0.0 before onset, linear ramp to 1.0 over `span` days, then holds at 1.0."""
    if day < onset:
        return 0.0
    return min(1.0, (day - onset) / span)


def wear_window_active(machine: dict, day: float) -> tuple[bool, str | None]:
    """
    Is this machine currently inside an active, un-repaired tool-wear-type
    degradation window (TWF / TWF_RECURRING / OSF)? Returns (active, note).
    """
    scenario = machine["scenario"]
    c = machine["cfg"]
    if scenario == "TWF":
        active = day >= c.onset_day and (c.repair_day is None or day < c.repair_day)
        return active, "tool_wear_drift" if active else None
    if scenario == "TWF_RECURRING":
        first = c.onset_day <= day < c.repair_day
        second = c.second_onset_day is not None and day >= c.second_onset_day and (
            c.second_repair_day is None or day < c.second_repair_day
        )
        return (first or second), "tool_wear_drift_recurring" if (first or second) else None
    if scenario == "OSF":
        active = day >= c.onset_day and (c.repair_day is None or day < c.repair_day)
        return active, "overstrain_drift" if active else None
    return False, None


def wear_ramp_fraction(machine: dict, day: float) -> float:
    """Ramp fraction (0..1) for the currently-active wear scenario window."""
    scenario = machine["scenario"]
    c = machine["cfg"]
    if scenario in ("TWF", "OSF"):
        return ramp(day, c.onset_day, 10.0)
    if scenario == "TWF_RECURRING":
        if day < c.repair_day:
            return ramp(day, c.onset_day, 6.0)
        return ramp(day, c.second_onset_day, 6.0)
    return 0.0


def simulate_machine(machine: dict, writer: csv.DictWriter, start: datetime) -> None:
    rng = random.Random(f"{cfg.RANDOM_SEED}-{machine['machine_id']}")
    machine_type = machine["type"]
    scenario = machine["scenario"]
    c = machine["cfg"]

    n_steps = int((cfg.SIMULATION_DAYS * 24 * 60) / cfg.READING_INTERVAL_MINUTES)
    interval_hours = cfg.READING_INTERVAL_MINUTES / 60.0

    air_temp = cfg.AIR_TEMP_BASE_K
    process_temp = cfg.AIR_TEMP_BASE_K + cfg.PROCESS_TEMP_OFFSET_K
    tool_wear = rng.uniform(0, 8)

    for step in range(n_steps):
        ts = start + timedelta(minutes=step * cfg.READING_INTERVAL_MINUTES)
        day = step * cfg.READING_INTERVAL_MINUTES / (24 * 60)
        note = None

        # --- temperatures: bounded random walk (matches AI4I's own process) ---
        air_temp = 0.98 * (air_temp + rng.gauss(0, cfg.AIR_TEMP_WALK_SD)) + 0.02 * cfg.AIR_TEMP_BASE_K
        process_temp_target = air_temp + cfg.PROCESS_TEMP_OFFSET_K
        process_temp = 0.95 * (process_temp + rng.gauss(0, cfg.PROCESS_TEMP_WALK_SD)) + 0.05 * process_temp_target

        # --- HDF perturbation: narrow the temp differential + suppress speed ---
        hdf_f = 0.0
        if scenario == "HDF":
            active = c.onset_day <= day and (c.repair_day is None or day < c.repair_day)
            if active:
                hdf_f = ramp(day, c.onset_day, 10.0)
                note = "heat_dissipation_drift"
        process_temp_eff = process_temp - hdf_f * 4.5

        # --- torque: stationary distribution + scenario-specific drift ---
        torque = max(0.5, rng.gauss(cfg.TORQUE_MEAN_NM, cfg.TORQUE_SD_NM))

        wear_active, wear_note = wear_window_active(machine, day)
        wear_f = 0.0
        if wear_active:
            note = wear_note
            wear_f = wear_ramp_fraction(machine, day)
            torque += wear_f * 5.0 * (1.4 if scenario == "OSF" else 1.0)
        torque = max(0.5, torque)

        # --- power: sampled directly (calibrated to real AI4I stats), then
        # rotational speed is DERIVED from power/torque — this preserves the
        # real physical coupling between torque and speed instead of
        # sampling them independently (see config.py for why that matters).
        pwf_f = 0.0
        if scenario == "PWF":
            active = c.onset_day <= day and (c.repair_day is None or day < c.repair_day)
            if active:
                pwf_f = ramp(day, c.onset_day, 10.0)
                note = "power_drift"
        power_target = rng.gauss(cfg.POWER_MEAN_W, cfg.POWER_SD_W) + pwf_f * cfg.PWF_SCENARIO_POWER_SHIFT_W

        rotational_speed = (power_target / torque) * (60 / (2 * math.pi))
        rotational_speed -= hdf_f * 300  # HDF scenario suppresses speed directly
        rotational_speed = min(
            cfg.ROTATIONAL_SPEED_MAX_RPM, max(cfg.ROTATIONAL_SPEED_MIN_RPM, rotational_speed)
        )

        # --- tool wear: baseline rate + bounded routine replacement ---
        base_rate = cfg.TOOL_WEAR_RATE_PER_HOUR[machine_type]
        accel_rate = 0.0
        if wear_active:
            accel_rate = wear_f * (
                cfg.ACCEL_OSF_WEAR_RATE_PER_HOUR if scenario == "OSF" else cfg.ACCEL_TWF_WEAR_RATE_PER_HOUR
            )
        tool_wear += (base_rate + accel_rate) * interval_hours

        if not wear_active and tool_wear >= cfg.ROUTINE_TOOL_REPLACEMENT_THRESHOLD_MIN:
            tool_wear = rng.uniform(0, 8)  # routine preventive replacement

        # --- RNF: sharp, unexplained blip, no precursor ---
        if scenario == "RNF" and abs(day - c.onset_day) < (cfg.READING_INTERVAL_MINUTES / (24 * 60)):
            torque += 48.0
            process_temp_eff += 6.5
            note = "random_unexplained_blip"

        # --- business-layer metrics, correlated with active degradation ---
        strain = 0.0
        if note:
            strain = min(
                1.0,
                max(
                    (torque - cfg.TORQUE_MEAN_NM) / 35.0,
                    max(0.0, tool_wear - 150) / 150.0,
                    0.0,
                ),
            )
        production_rate = cfg.BASE_PRODUCTION_RATE * (1 - 0.35 * strain) + rng.gauss(0, 1.5)
        energy_consumption = cfg.BASE_ENERGY_KWH * (1 + 0.25 * strain) + rng.gauss(0, 0.3)
        defect_rate = cfg.BASE_DEFECT_RATE + 0.15 * strain + max(0, rng.gauss(0, 0.003))

        writer.writerow(
            {
                "machine_id": machine["machine_id"],
                "timestamp": ts.isoformat(),
                "production_line": machine["production_line"],
                "type": machine_type,
                "air_temperature_k": round(air_temp, 3),
                "process_temperature_k": round(process_temp_eff, 3),
                "rotational_speed_rpm": round(rotational_speed, 2),
                "torque_nm": round(torque, 3),
                "tool_wear_min": round(tool_wear, 2),
                "production_rate": round(max(0.0, production_rate), 2),
                "energy_consumption_kwh": round(max(0.0, energy_consumption), 3),
                "defect_rate": round(min(1.0, max(0.0, defect_rate)), 4),
                "injected_scenario_active": note or "",
            }
        )


def generate_operational_data(fleet: list[dict]) -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OPERATIONAL_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for machine in fleet:
            simulate_machine(machine, writer, start)


def generate_maintenance_records(fleet: list[dict]) -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    fieldnames = ["machine_id", "event_date", "event_type", "technician_notes", "resolved"]

    notes_map = {
        "TWF": "Inspected and replaced tool after wear-related torque drift.",
        "HDF": "Cleaned cooling fins and airflow path; verified rotational speed restored to spec.",
        "PWF": "Checked drive motor and power supply; power draw back within normal band.",
        "OSF": "Overstrain condition flagged; parts on order, machine still running under load.",
        "RNF": "Investigated isolated anomaly; no recurring pattern found, logged as one-off.",
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(MAINTENANCE_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for machine in fleet:
            c = machine["cfg"]
            if c is None:
                continue

            if c.scenario == "TWF_RECURRING":
                writer.writerow(
                    {
                        "machine_id": machine["machine_id"],
                        "event_date": (start + timedelta(days=c.repair_day)).date().isoformat(),
                        "event_type": "corrective_maintenance",
                        "technician_notes": "Replaced worn cutting tool; torque and wear reset to baseline.",
                        "resolved": "true",
                    }
                )
                writer.writerow(
                    {
                        "machine_id": machine["machine_id"],
                        "event_date": (start + timedelta(days=c.second_repair_day)).date().isoformat(),
                        "event_type": "corrective_maintenance",
                        "technician_notes": "Second tool-wear episode — same failure mode as prior repair. Recommend reviewing feed-rate settings, not just replacing the tool.",
                        "resolved": "true",
                    }
                )
            else:
                resolved = c.repair_day is not None
                event_day = c.repair_day if c.repair_day is not None else c.onset_day + 12
                writer.writerow(
                    {
                        "machine_id": machine["machine_id"],
                        "event_date": (start + timedelta(days=event_day)).date().isoformat(),
                        "event_type": "corrective_maintenance",
                        "technician_notes": notes_map[c.scenario],
                        "resolved": "true" if resolved else "false",
                    }
                )


def main():
    fleet = build_fleet()
    print(
        f"Generating synthetic data for {len(fleet)} machines "
        f"over {cfg.SIMULATION_DAYS} days at {cfg.READING_INTERVAL_MINUTES}-min intervals..."
    )
    for m in fleet:
        print(f"  {m['machine_id']} | {m['production_line']} | type={m['type']} | scenario={m['scenario']}")

    generate_operational_data(fleet)
    generate_maintenance_records(fleet)

    with open(OPERATIONAL_PATH) as f:
        n_rows = sum(1 for _ in f) - 1
    print(f"\nWrote {n_rows:,} rows to {OPERATIONAL_PATH}")
    print(f"Wrote maintenance log to {MAINTENANCE_PATH}")


if __name__ == "__main__":
    main()
