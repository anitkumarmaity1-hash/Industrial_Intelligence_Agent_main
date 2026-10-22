---
doc_id: INC-500
title: Historical Incident Reports (2025)
doc_type: incident_report
failure_modes: TWF, HDF, PWF, OSF, RNF
equipment_class: CNC milling station
version: 1.0
effective_date: 2025-12-31
data_class: SYNTHETIC
---

# INC-500 — Historical Incident Reports (2025)

> **Data class: SYNTHETIC.** These incidents are fabricated for the Industrial
> Intelligence Agent demonstration project. No real factory, customer, machine
> or downtime event is described here. Dates precede the simulated operating
> window of the demonstration dataset, so these are prior history relative to
> the sensor data the system analyses.

## INC-2025-014 — M-07, LINE-2, tool wear escalated to overstrain

**Date:** 2025-03-12 · **Variant:** M · **Outcome:** resolved, 4.5 h downtime

Accumulated tool wear reached the 200–240 minute failure band after a routine
replacement was deferred across two consecutive shifts during a delivery push.
Torque was also running above its usual level for the part program, so the
overstrain product crossed the 12,000 min·Nm limit for the M variant before wear
alone would have been treated as urgent.

Inspection found heavy flank wear and a chipped cutting edge. The part fixture
was found slightly out of position, which had raised cutting load and therefore
both torque and wear rate.

**What the evidence actually supported:** wear and torque rose together and the
fixture was found displaced. The fixture is the plausible common cause; it was
not independently proven to be the origin, because it was corrected in the same
intervention as the tool change.

**Lesson recorded:** deferring a routine 150-minute replacement under high torque
is materially riskier than deferring it under normal torque, because the
overstrain limit is reached first.

## INC-2025-027 — M-11, LINE-1, coolant restriction misdiagnosed as pump failure

**Date:** 2025-05-28 · **Variant:** L · **Outcome:** resolved, 6 h downtime

Process-to-air temperature difference narrowed progressively over about nine
hours while rotational speed drifted below 1380 rpm. Defect rate roughly doubled
over the same period.

The coolant pump was replaced first, on the assumption that a pump fault was the
cause. The symptom did not clear. A second inspection found the coolant filter
heavily loaded with swarf and the nozzle displaced away from the cutting zone.
Cleaning the filter and re-aiming the nozzle restored the temperature difference
within one hour.

**Lesson recorded:** the cheap checks in SOP-102 section 4 exist in that order
for a reason. Replacing the most expensive suspect component first cost roughly
four hours and did not address the cause.

## INC-2025-039 — M-02, LINE-1, power excursions traced to part program

**Date:** 2025-07-09 · **Variant:** M · **Outcome:** resolved, no downtime

Delivered power exceeded 9000 W intermittently across two shifts. Mechanical
inspection found belt tension, coupling and spindle runout all within
specification, and tool wear was low.

The excursions were found to align with a specific part program introduced the
previous week, which commanded a deeper cut than the previous revision. The
program was corrected by production engineering. No machine fault existed.

**Lesson recorded:** when excursions follow a program rather than a machine, the
machine is not the problem. Checking whether the same program runs on other
stations is faster than dismantling a drivetrain.

## INC-2025-051 — M-14, LINE-3, isolated anomaly with no cause found

**Date:** 2025-09-02 · **Variant:** H · **Outcome:** closed, no action, no downtime

A single flagged reading was recorded with no supporting movement in tool wear,
temperature difference, torque, rotational speed or defect rate, either before or
after the event. A full inspection found nothing abnormal. The machine continued
to run normally for the following eight weeks.

**Outcome recorded as unexplained.** The event is consistent with the random
failure mode described in MAN-200 section 3 or with a momentary sensor glitch.
No cause was assigned, deliberately: assigning one would have created a false
maintenance history for this station.

## INC-2025-063 — M-06, LINE-2, recurring wear events on one station

**Date:** 2025-11-19 · **Variant:** L · **Outcome:** resolved, 8 h downtime

Four wear-driven tool replacements were logged on M-06 within 26 days, against a
line average of about one per month. Tool grade and supplier were unchanged, and
comparable stations on LINE-2 were unaffected.

Spindle runout was measured at roughly twice the quarterly-check limit. The
spindle bearing assembly was replaced, after which wear accrual returned to the
nominal rate for the L variant.

**Lesson recorded:** a repeating symptom on a single station, where comparable
stations are unaffected, points at that station rather than at consumables.
Three of the four replacements before the runout measurement treated the symptom
and not the cause.
