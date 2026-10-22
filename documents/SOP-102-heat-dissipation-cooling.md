---
doc_id: SOP-102
title: Heat Dissipation and Cooling System Procedure
doc_type: sop
failure_modes: HDF
equipment_class: CNC milling station
version: 2.4
effective_date: 2025-09-18
data_class: SYNTHETIC
---

# SOP-102 — Heat Dissipation and Cooling System Procedure

> **Data class: SYNTHETIC.** Written for the Industrial Intelligence Agent
> demonstration project. Not a real manufacturer procedure. Thresholds mirror
> the published thresholds of the public UCI AI4I 2020 dataset.

## 1. Purpose and scope

Covers the coolant loop, process-air handling and heat-dissipation checks for
CNC milling stations. Applies to all machine types on LINE-1, LINE-2 and
LINE-3. Electrical drive faults are out of scope (see SOP-103).

## 2. Thermal measurements

Two temperatures are recorded per machine:

- **Air temperature** — ambient air at the machine enclosure inlet, in Kelvin.
  Nominal operation is around 300 K with slow drift across a shift.
- **Process temperature** — measured at the cutting zone, in Kelvin. Under
  healthy operation it sits approximately 10 K above air temperature.

The diagnostically meaningful quantity is the **difference** between process
and air temperature, not either absolute value. A machine running in a warm
corner of the plant will show a high absolute process temperature and still be
dissipating heat perfectly well.

## 3. Heat dissipation failure (HDF) condition

Heat dissipation is considered failed when **both** of the following hold at
the same time:

1. The difference between process temperature and air temperature falls below
   **8.6 K**, and
2. Rotational speed is below **1380 rpm**.

The two conditions together matter. A narrowing temperature difference on its
own can simply mean the machine is idling or lightly loaded. A narrowing
difference *while the spindle is also turning slowly* indicates the machine is
no longer moving heat away from the cutting zone at the rate the process
requires — typically restricted coolant flow, a fouled heat exchanger, or a
failing coolant pump.

A single window that crosses this condition is worth noting. A sustained trend
across several consecutive hourly windows, with the temperature difference
narrowing progressively, is the pattern that warrants inspection.

## 4. Inspection procedure

Perform in this order; each step is cheaper than the one after it.

1. **Coolant level and concentration.** Check the reservoir level and refractometer
   reading. Low level or diluted coolant is the most common cause and the
   fastest to correct.
2. **Coolant flow at the nozzle.** Confirm flow reaches the cutting zone and is
   aimed correctly. A displaced nozzle produces exactly the same signature as a
   pump fault.
3. **Filter and strainer.** Inspect for swarf loading. Replace if restricted.
4. **Heat exchanger fins.** Check for dust blanketing and clean if fouled.
5. **Coolant pump.** Check pump current draw and outlet pressure against the
   values in MAN-200 section 5. Replace the pump only after steps 1–4 have been
   excluded.
6. **Temperature sensor integrity.** If all of the above are healthy and the
   temperature difference is still reported as narrow, verify the process
   temperature probe. A drifting or detached probe reports a false narrow
   difference and is frequently mistaken for a cooling fault.

## 5. Interaction with production metrics

A genuine heat dissipation problem usually shows up in business metrics before
it shows up as a hard failure: defect rate rises as thermal growth changes part
dimensions, and operators often reduce feed rate to compensate, which lowers
production rate. Energy consumption may rise as the cooling system works
harder.

Observing all three together — narrowing temperature difference, falling
production rate, rising defect rate — strengthens the cooling hypothesis. It
does not by itself prove causation; each of those metrics has other possible
causes, and the confirmation is the physical inspection in section 4.

## 6. Safety

Coolant lines may be pressurised and the heat exchanger may be hot. Isolate and
allow the machine to cool per SAF-300 before opening any part of the loop.
Used coolant is a controlled waste stream and must not be poured to drain.

## 7. Escalation

Escalate to facilities engineering if multiple machines on the same production
line show a narrowing temperature difference within the same shift. A
line-wide pattern points at shared plant services — chilled water supply,
compressed air, or ambient ventilation — rather than at any individual machine.
