---
doc_id: MAN-200
title: CNC Milling Station Equipment Manual (Extract)
doc_type: manual
failure_modes: TWF, HDF, PWF, OSF
equipment_class: CNC milling station
version: 5.0
effective_date: 2025-06-30
data_class: SYNTHETIC
---

# MAN-200 — CNC Milling Station Equipment Manual (Extract)

> **Data class: SYNTHETIC.** Written for the Industrial Intelligence Agent
> demonstration project. This is not a real manufacturer manual and the
> equipment described does not exist.

## 1. Machine variants

Stations are supplied in three quality variants. The variant determines
tolerance class, nominal tool wear rate and the overstrain limit.

| Variant | Tolerance class | Nominal wear accrual | Overstrain limit (wear x torque) |
|---|---|---|---|
| L | standard | approx. 2 min/operating hour | 11,000 min·Nm |
| M | precision | approx. 3 min/operating hour | 12,000 min·Nm |
| H | high precision | approx. 5 min/operating hour | 13,000 min·Nm |

Variant is a fixed property of the station and is recorded in the machine
register. It is not configurable at the controller.

## 2. Nominal operating envelope

| Parameter | Nominal | Notes |
|---|---|---|
| Air temperature | approx. 300 K | ambient at enclosure inlet |
| Process temperature | approx. air temperature + 10 K | at the cutting zone |
| Rotational speed | 1100–2950 rpm | below 1380 rpm is a low-speed condition |
| Torque | approx. 40 Nm nominal | varies with cutting load |
| Delivered power | 3500–9000 W | derived, see SOP-103 |
| Tool wear | reset at replacement | 200–240 min is the failure band |

These are envelope values for a healthy station under normal load. They are not
alarm setpoints in themselves; the condition definitions in SOP-101 through
SOP-103 are what constitute a recorded fault condition.

## 3. Documented failure modes

The station has five recorded failure modes. Four are deterministic conditions
on measured values; the fifth is not.

- **TWF — tool wear failure.** Accumulated wear in the 200–240 minute band.
- **HDF — heat dissipation failure.** Process-to-air temperature difference
  below 8.6 K while rotational speed is below 1380 rpm.
- **PWF — power failure.** Derived power outside the 3500–9000 W band.
- **OSF — overstrain failure.** Wear x torque above the variant limit in
  section 1.
- **RNF — random failure.** A low-probability event not explained by any of the
  measured conditions above. RNF exists precisely because some stoppages have no
  detectable precursor in the recorded signals. When a machine shows a single
  isolated event with no supporting trend in any monitored parameter, an
  unexplained random event is a legitimate conclusion, and inventing a cause for
  it is worse than reporting that none was found.

## 4. Tool holder torque specification

Tool holders are seated to 25 Nm ±2 Nm using a calibrated torque wrench. Do not
use the spindle brake to resist tightening torque. Re-verify seating torque after
the first thermal cycle following a replacement.

## 5. Coolant pump reference values

| Parameter | Healthy range |
|---|---|
| Pump outlet pressure | 2.8–3.6 bar |
| Pump current draw | 3.1–4.0 A |
| Coolant concentration | 6–9% |

Readings outside these ranges support a cooling-system hypothesis under SOP-102
section 4 step 5.

## 6. Maintenance intervals

| Task | Interval |
|---|---|
| Tool wear check | every shift |
| Coolant level and concentration | every shift |
| Filter and strainer inspection | weekly |
| Heat exchanger cleaning | monthly |
| Belt tension and coupling check | monthly |
| Spindle runout measurement | quarterly |
| Lubrication system service | quarterly |

## 7. Data recording

Sensor values are sampled at 5-minute intervals and rolled up into hourly
summary windows for reporting. A summary window reports averages for continuous
values and the maximum observed severity for condition flags, so a short-lived
excursion inside an hour is visible in the window's flag counts even though it is
smoothed in the averages. Investigations that need excursion-level detail must
look at the flagged-event record, not the hourly averages.
