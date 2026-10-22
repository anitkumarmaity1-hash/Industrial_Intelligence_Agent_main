---
doc_id: SAF-300
title: Machine Isolation, Lockout/Tagout and Safe Access
doc_type: safety
failure_modes: 
equipment_class: CNC milling station
version: 4.2
effective_date: 2025-10-10
data_class: SYNTHETIC
---

# SAF-300 — Machine Isolation, Lockout/Tagout and Safe Access

> **Data class: SYNTHETIC.** Written for the Industrial Intelligence Agent
> demonstration project. It is not a real safety procedure and must never be
> relied on for work on real equipment. Real isolation procedures are
> site-specific and legally mandated.

## 1. When isolation is required

Isolate the machine before any of the following:

- Tool replacement or tool holder work.
- Opening the coolant loop, filter housing or heat exchanger.
- Any work on the spindle drive, belt, coupling or motor terminals.
- Removing any fixed guard or defeating any interlock.
- Clearing a jam or swarf blockage inside the enclosure.

Diagnostic observation from outside the enclosure with all guards in place does
not require isolation. Reading sensor data or reviewing history never requires
isolation.

## 2. Lockout/tagout sequence

1. Notify the line supervisor and affected operators before isolating.
2. Bring the machine to a normal controlled stop at the controller.
3. Isolate the electrical supply at the local disconnect and apply a personal
   lock and tag. Each person working on the machine applies their own lock.
4. Isolate stored energy: pneumatic supply vented, coolant loop
   depressurised, and any suspended axis mechanically supported.
5. Verify zero energy state by attempting a normal start and by testing at the
   designated test point.
6. Only after verification may guards be removed.

Restoration is the reverse order. The person who applied a lock is the only
person who may remove it.

## 3. Thermal and pressure hazards

The cutting zone, tool holder, heat exchanger and coolant lines may be hot and
pressurised after operation. Allow the machine to cool before opening the
coolant loop. Do not loosen a fitting to check pressure.

## 4. Hazards specific to investigated conditions

- **Suspected overstrain or high delivered power.** The drivetrain may be under
  load and may release stored energy when a coupling or belt is disturbed.
  Isolate fully before touching the drive.
- **Suspected cooling fault.** Assume the coolant is hot. Assume the loop holds
  pressure even if the pump is off.
- **Tool in the wear failure band.** A worn tool may be chipped. Handle the
  removed tool as sharp waste.

## 5. Waste and materials handling

Used coolant, swarf and worn cutting tools are controlled waste streams.
Coolant must not be discharged to drain. Worn tools go to the sharps stream,
not to general waste.

## 6. Decision-support limits

Any condition flagged by a monitoring or analytics system is an indication that
inspection may be warranted. It is not authorisation to work on the machine, it
does not replace the isolation sequence in section 2, and it does not constitute
a determination that the machine is safe or unsafe. A competent person performs
the physical assessment.
