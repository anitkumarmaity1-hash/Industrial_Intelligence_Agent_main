---
doc_id: SOP-103
title: Power Delivery and Drivetrain Procedure
doc_type: sop
failure_modes: PWF
equipment_class: CNC milling station
version: 1.9
effective_date: 2025-12-02
data_class: SYNTHETIC
---

# SOP-103 — Power Delivery and Drivetrain Procedure

> **Data class: SYNTHETIC.** Written for the Industrial Intelligence Agent
> demonstration project. Not a real manufacturer procedure. Thresholds mirror
> the published thresholds of the public UCI AI4I 2020 dataset.

## 1. Purpose and scope

Covers the spindle drive, motor, coupling and power-monitoring checks for CNC
milling stations. Cooling faults are out of scope (see SOP-102); tool wear is
out of scope (see SOP-101).

## 2. How delivered power is derived

Delivered mechanical power is not measured directly. It is computed from two
recorded values:

    power (W) = torque (Nm) x rotational speed (rad/s)

where rotational speed in rad/s is the reported rpm multiplied by 2*pi and
divided by 60.

Torque and rotational speed are physically coupled: under a fixed drive
setting, an increase in cutting load raises torque and pulls speed down. This
is why the two values must be interpreted together. Reading either one in
isolation produces misleading conclusions.

## 3. Safe operating band

Delivered power is expected to stay between **3500 W and 9000 W**. Crossing
either limit is recorded as a power failure (PWF) condition.

### 3.1 Power below 3500 W

Low delivered power means the drivetrain is not transmitting the commanded
load. Candidate causes, in rough order of frequency:

- Slipping coupling or loose drive belt.
- Worn or glazed belt failing to transmit torque.
- Drive parameter drift or incorrect program feed/speed override.
- Torque sensor under-reporting.

### 3.2 Power above 9000 W

High delivered power means the drivetrain is working harder than the process
should require. Candidate causes:

- Increased cutting resistance from a worn tool (cross-check SOP-101 — a
  high-power reading with high accumulated wear is more likely a tool problem
  than a drive problem).
- Bearing degradation adding parasitic friction load.
- Inadequate lubrication on ways or ball screws.
- Material or fixturing change increasing the actual load.

## 4. Inspection procedure

1. Record torque, rotational speed and derived power for the affected period
   before touching anything. Values after a restart are not comparable.
2. Check for a mechanical cause first: belt tension and condition, coupling
   security, spindle runout, and audible bearing noise.
3. Check lubrication delivery to ways and screws.
4. Compare against the machine's own recent baseline rather than against the
   fleet. Machines differ, and a fleet-average comparison produces false
   positives on legitimately heavier-duty stations.
5. Verify the torque sensor and drive feedback only after mechanical causes
   have been excluded. Sensor faults are real but less common than mechanical
   ones, and replacing a sensor to chase a mechanical fault wastes a shift.

## 5. Distinguishing a transient from a trend

Isolated power excursions occur during normal operation — tool entry, interrupted
cuts and material inconsistency all produce brief spikes. A single flagged
reading is not evidence of a drivetrain fault.

What warrants inspection is a **sustained shift in the distribution**: the mean
delivered power moving away from its own established baseline across many
consecutive windows, or excursions becoming steadily more frequent. Report the
trend and the window it covers, not the single worst reading.

## 6. Safety

The spindle drive holds stored energy. Isolate and verify zero energy state per
SAF-300 before removing guards or working on the belt, coupling or motor
terminals. Never defeat the enclosure interlock to observe the drive under load.

## 7. Escalation

Escalate to electrical maintenance if power excursions persist after mechanical
causes have been excluded, or if the drive reports any fault code. Escalate to
production engineering if the excursions align with a specific part program
rather than with a specific machine — that pattern points to the program, not
the equipment.
