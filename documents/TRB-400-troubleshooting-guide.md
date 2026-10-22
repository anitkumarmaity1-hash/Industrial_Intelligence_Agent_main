---
doc_id: TRB-400
title: Symptom-Based Troubleshooting Guide
doc_type: troubleshooting
failure_modes: TWF, HDF, PWF, OSF, RNF
equipment_class: CNC milling station
version: 2.0
effective_date: 2026-01-15
data_class: SYNTHETIC
---

# TRB-400 — Symptom-Based Troubleshooting Guide

> **Data class: SYNTHETIC.** Written for the Industrial Intelligence Agent
> demonstration project. Not a real manufacturer troubleshooting guide.

## 1. How to use this guide

Entries are organised by observed symptom, because that is what an operations
team has before a diagnosis exists. Each entry lists candidate causes in rough
order of likelihood, the evidence that discriminates between them, and the
procedure to follow.

Candidate causes are hypotheses. Two conditions appearing together is a
correlation and a reason to inspect; it is not proof that one caused the other.
Record which candidate the physical inspection actually confirmed, so the next
investigation of the same symptom starts from evidence rather than from this
list.

## 2. Falling production rate

**Candidate causes**

1. Worn tool causing reduced feed rate or rework (SOP-101).
2. Operator-applied feed override in response to quality problems.
3. Cooling restriction causing thermal growth and dimensional drift (SOP-102).
4. Drivetrain slip reducing delivered power (SOP-103).
5. Upstream material or scheduling constraint — not a machine fault at all.

**Discriminating evidence**

Check accumulated tool wear first; it is the cheapest signal. If wear is low and
the temperature difference is narrowing, look at cooling. If both are normal but
delivered power sits below the safe band, look at the drivetrain. If every
monitored parameter is normal, the cause is likely upstream of the machine and
no amount of machine inspection will find it.

## 3. Rising defect rate

**Candidate causes**

1. Tool wear past service life (SOP-101).
2. Thermal growth from a cooling restriction (SOP-102).
3. Spindle runout or fixturing wear.
4. Workpiece material variation.

**Discriminating evidence**

Defect rate rising together with accumulated wear points at the tool. Defect
rate rising with a narrowing temperature difference points at cooling. Defect
rate rising with no movement in any sensor parameter points at material or
fixturing, neither of which is instrumented.

## 4. Rising energy consumption

**Candidate causes**

1. Increased cutting resistance from a worn tool.
2. Bearing or lubrication degradation adding friction load.
3. Cooling system working harder against a fouled exchanger.

**Discriminating evidence**

Energy rising together with delivered power above the safe band points at
mechanical load. Energy rising while delivered power stays normal points at
auxiliary systems, most often the cooling loop.

## 5. Narrowing process-to-air temperature difference

Follow SOP-102 section 4. Before starting, confirm rotational speed: a narrow
difference at normal speed is usually light loading, while a narrow difference at
low speed is the documented heat-dissipation condition.

## 6. Delivered power outside the safe band

Follow SOP-103. Establish first whether the excursion is a transient or a
sustained shift in the distribution. Check accumulated tool wear before
suspecting the drive — a worn tool raising cutting resistance is a more frequent
cause of high delivered power than a drivetrain fault.

## 7. Repeated tool changes on the same machine

Repeated wear-driven replacements on one station, when comparable stations on the
same line are not affected, suggest a persistent mechanical cause rather than
tool quality. Measure spindle runout, check collet condition and check fixture
repeatability before ordering a different tool grade.

## 8. Single isolated anomaly with no supporting trend

One flagged reading with no movement in any other parameter, before or after, is
consistent with a random event (MAN-200 section 3, RNF) or with a momentary
sensor glitch.

Recommended action is to note it and continue monitoring, not to schedule an
intervention. Constructing a mechanical explanation for an isolated reading
generates unnecessary maintenance work and erodes confidence in genuine alerts.
Escalate only if the event repeats or if other parameters begin to move.

## 9. Multiple machines affected simultaneously

When several machines on the same line show the same symptom in the same shift,
investigate shared services before individual machines: electrical supply,
compressed air, chilled water, ambient ventilation, a common material batch, or a
recently changed part program. A fault that appears on many machines at once is
rarely a fault in many machines at once.
