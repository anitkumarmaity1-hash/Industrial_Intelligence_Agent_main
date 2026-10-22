---
doc_id: SOP-101
title: Tool Wear Monitoring and Replacement Procedure
doc_type: sop
failure_modes: TWF, OSF
equipment_class: CNC milling station
version: 3.1
effective_date: 2025-11-04
data_class: SYNTHETIC
---

# SOP-101 — Tool Wear Monitoring and Replacement Procedure

> **Data class: SYNTHETIC.** This document was written for the Industrial
> Intelligence Agent demonstration project. It is not a real manufacturer
> procedure and must not be used to operate real equipment. The numeric
> thresholds referenced here mirror the published thresholds of the public
> UCI AI4I 2020 Predictive Maintenance dataset so that the knowledge base is
> consistent with the sensor data the system analyses.

## 1. Purpose and scope

This procedure covers scheduled and condition-based replacement of cutting
tools on CNC milling stations across LINE-1, LINE-2 and LINE-3. It applies to
all machine quality variants (type L, M and H). It does not cover spindle
bearing replacement (see MAN-200) or coolant system faults (see SOP-102).

## 2. Wear accrual behaviour

Tool wear is reported in accumulated minutes of cutting time and increases
monotonically until the tool is replaced, at which point the counter resets
to zero. Expected accrual under normal load:

| Machine type | Nominal wear accrual | Typical time to routine replacement |
|---|---|---|
| H (high quality variant) | approx. 5 min wear per operating hour | approx. 30 operating hours |
| M (medium quality variant) | approx. 3 min wear per operating hour | approx. 50 operating hours |
| L (low quality variant) | approx. 2 min wear per operating hour | approx. 75 operating hours |

Accrual that is materially faster than the nominal rate for the machine type
is itself a signal. It usually indicates increased cutting resistance —
workpiece hardness variation, incorrect feed rate, or a tool that was already
chipped at installation — rather than a fault in the wear sensor.

## 3. Replacement thresholds

### 3.1 Routine preventive replacement

Replace the cutting tool during the next planned stoppage once accumulated
wear reaches **150 minutes**. Routine replacement at this point is deliberate:
it keeps the machine well clear of the degraded-performance band described in
3.2 and means a routine change is never mistaken for a fault condition.

### 3.2 Tool wear failure (TWF) band

Accumulated tool wear between **200 and 240 minutes** is the documented
tool-wear-failure band. A tool operated inside this band is past its service
life and is associated with dimensional drift on the finished part, increased
surface roughness, and a raised defect rate.

A machine whose reported wear has entered this band while still in production
means routine replacement at 150 minutes was missed or deferred. Treat it as
an open maintenance item, not a routine one.

### 3.3 Overstrain (OSF) interaction

Tool wear alone is not sufficient to assess overstrain risk. Overstrain is
evaluated as the product of accumulated tool wear (minutes) and applied torque
(Nm), against a type-specific limit:

| Machine type | Overstrain limit (wear x torque) |
|---|---|
| L | 11,000 min·Nm |
| M | 12,000 min·Nm |
| H | 13,000 min·Nm |

A worn tool operated at high torque can cross the overstrain limit well before
wear alone reaches the TWF band. When the overstrain product is rising, reduce
feed rate or replace the tool early; do not wait for the 150-minute routine
point.

## 4. Replacement procedure

1. Raise a maintenance work order referencing the machine ID and current
   reported tool wear.
2. Isolate the machine following SAF-300 (lockout/tagout). Do not begin tool
   removal until isolation is verified.
3. Allow the spindle to come to a complete stop and the tool holder to cool.
4. Remove the tool. Inspect the cutting edge for chipping, built-up edge, and
   flank wear before discarding it — the failure mode of the removed tool is
   the cheapest diagnostic evidence available.
5. Install the replacement tool and verify the holder is seated to the torque
   specified in MAN-200 section 4.
6. Reset the tool wear counter at the machine controller. A missed reset is a
   common cause of an apparently healthy machine reporting implausibly high
   wear immediately after service.
7. Run a first-article check on the first three parts produced.
8. Record the replacement in the maintenance log, including observed wear
   pattern and whether the change was routine or condition-based.

## 5. Post-replacement verification

After replacement, defect rate and production rate should return to their
baseline within two operating hours. If the defect rate stays elevated after a
tool change, the tool was probably not the dominant cause — re-open the
investigation and check spindle runout and workpiece fixturing rather than
fitting a second new tool.

## 6. Escalation

Escalate to the maintenance engineering lead when any of the following holds:

- Wear repeatedly re-enters the TWF band on the same machine within 30 days.
- Wear accrual rate exceeds twice the nominal rate for the machine type.
- The overstrain product crosses the type limit more than once per week.

Recurring wear-related events on a single machine usually point to an
underlying mechanical cause (spindle alignment, fixturing, or worn collet)
that repeated tool replacement will not resolve.
