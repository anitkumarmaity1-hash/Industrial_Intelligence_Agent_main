"""
Machine and intent extraction from the user's question — audit finding F1.

Before this module, nothing ever read the question text. The agent's
target came only from the request's optional `machine_id` field, so:

  * "Why is M-17 underperforming?" with no machine_id silently ran a
    FLEET scan (the master prompt's headline example question).
  * machine_id="M-04" plus a question about M-17 silently investigated
    M-04 and labelled the report with the wrong machine.
  * "Why is M-99 underperforming?" returned a fleet report instead of
    telling the user M-99 doesn't exist.

This module is deliberately deterministic (regex + set membership, no
LLM call), for the same reason route_intent is: which machine a question
names is a *fact about the text*, not a judgement call, and a wrong
guess here means investigating the wrong machine. It is also pure — no
database access. The caller passes in the set of known machine IDs, so
this is unit-testable offline and the graph's own signature is unchanged.

Resolution rules (see `resolve_target`):

  1. The text names exactly one machine        -> that machine.
     - If an explicit machine_id disagrees    -> 422, never guess.
  2. The text names several machines           -> 422 (the graph
     investigates one machine at a time; picking one silently would be
     a wrong answer to the other half of the question).
  3. The text names a machine that isn't in the fleet -> 404, not a
     silent fallback to a fleet scan.
  4. The text names nothing:
     - explicit machine_id                     -> that machine (the
       Streamlit "scope to selected machine" flow relies on this).
     - no explicit machine_id                  -> fleet scan.
     In both sub-cases, a note is attached when the question's wording
     points the other way, so the fallback is visible to the user
     instead of silent.

`Intent` currently has two values, so intent is a function of the
resolved machine: a machine -> machine_investigation, none ->
fleet_scan. If more intents are added later (e.g. a docs-only
"what procedure applies" question), the detection belongs here.

Production-readiness fix 12. The `M-\d{2}` pattern below was, until now,
hardcoded — it is exactly the demo fleet's own convention, and the audit's
own Company B ("asset_id", e.g. A-118) and Company C ("equipment_code",
e.g. EQ-042) examples don't match it at all, so a question naming their
machines would silently fail to resolve. `MachineIdScheme` pulls the
prefix, digit-group regex, optional zero-padding and the "machine N"
wording pattern out into a small, tenant-overridable value object; every
function below takes an optional `scheme` and falls back to
`DEFAULT_MACHINE_ID_SCHEME` (this module's original patterns, unchanged)
when it's omitted. A tenant's own scheme is stored in `tenant_settings`
and loaded by `app.core.tenant_settings.load_tenant_settings` — this
module still does no I/O and stays pure/offline-testable. This covers any
tenant whose asset IDs are "<prefix><separator><digits>" under a scheme
they declare; a genuinely free-form ID format (no fixed prefix, no digit
run) would need a small custom parser, which is out of scope for this fix.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Collection
from dataclasses import dataclass, field

from app.agents.state import Intent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MachineIdScheme:
    """How to recognize and canonicalize this tenant's machine IDs in free text.

    Args:
        id_prefix: The literal prefix a canonical ID starts with (e.g. "M-",
            "A-", "EQ-"). Always included verbatim in the canonical form.
        code_separators: Character class of separators allowed between the
            prefix's letters and the digits in free text (e.g. "-_" also
            matches "M17"/"M_17"); does not need to include the en-dash
            variants, those are added automatically.
        digit_width: Zero-pad the digits to this many places in the
            canonical form (2 -> "M-07"); 0 disables padding (the digits
            are used exactly as matched, still without leading zeros
            collapsed, e.g. a literal "007" stays "007").
        word_pattern: Regex (one capture group, the digits) matching the
            "machine 17" / "asset #17" spoken-word form; None disables it
            for tenants whose vocabulary doesn't use one.
    """

    name: str
    id_prefix: str
    code_separators: str = "-_"
    digit_width: int = 2
    word_pattern: str | None = r"\bmachine\s*(?:no\.?|number|#)?\s*(\d{1,3})\b"

    def canonicalize(self, digits: str) -> str:
        if self.digit_width:
            return f"{self.id_prefix}{int(digits):0{self.digit_width}d}"
        return f"{self.id_prefix}{digits}"

    def compiled_code_pattern(self) -> re.Pattern[str]:
        """The prefix's own letters, then an optional separator (including
        copy/paste en-dash variants), then 1-3 digits. Lookarounds stop it
        matching inside longer tokens ("ABM-5", "M-17A") the same way the
        original hardcoded pattern did."""
        letters = re.escape(self.id_prefix.rstrip("".join(set(self.code_separators))))
        sep_class = re.escape(self.code_separators) + "\u2010-\u2015"
        return re.compile(
            rf"(?<![A-Za-z0-9]){letters}[{sep_class}]?(\d{{1,3}})(?![A-Za-z0-9])",
            re.IGNORECASE,
        )

    def compiled_word_pattern(self) -> re.Pattern[str] | None:
        return re.compile(self.word_pattern, re.IGNORECASE) if self.word_pattern else None


# The original demo-fleet scheme, unchanged: "M-17" / "m17" / "M_17" /
# "M–17" (en dash) / "M-7" (zero-padded to "M-07"), plus "machine 17" /
# "machine #17" / "machine no. 17". Every caller that omits `scheme` gets
# exactly this, byte-for-byte what the original hardcoded patterns matched.
DEFAULT_MACHINE_ID_SCHEME = MachineIdScheme(
    name="default", id_prefix="M-", code_separators="-_", digit_width=2,
)

# A bare space between the prefix and the digits is deliberately NOT
# accepted by any scheme: "5 M 3" style fragments are far likelier to be
# noise than machine IDs, and a false positive here becomes a wrong 404 or
# a wrong machine.

# Wording that asks about the whole fleet rather than one machine.
# Used only to decide whether to attach an explanatory note — it never
# overrides an explicit machine_id or a machine named in the text. This is
# generic phrasing (not tied to any one tenant's ID format), so it stays a
# module-level pattern rather than part of MachineIdScheme.
_FLEET_WIDE = re.compile(
    r"\b(?:machines|equipment|assets|fleet|lines)\b"
    r"|\b(?:factory|plant|shop[- ]?floor)[- ]?wide\b"
    r"|\b(?:every|each|any|all)\s+(?:the\s+)?(?:machine|line|asset)\b"
    r"|\binspect\s+first\b"
    r"|\bacross\s+(?:the\s+)?(?:plant|factory|fleet|lines)\b",
    re.IGNORECASE,
)


def canonical_machine_id(number: int) -> str:
    """Fleet IDs are zero-padded to two digits (M-07, M-17).

    Kept for backward compatibility (tests, and any caller that only ever
    dealt with the demo fleet's own convention) — equivalent to
    `DEFAULT_MACHINE_ID_SCHEME.canonicalize(str(number))`.
    """
    return DEFAULT_MACHINE_ID_SCHEME.canonicalize(str(number))


@dataclass(frozen=True)
class ParsedQuery:
    """What the question text itself says, before any validation."""

    machine_ids: tuple[str, ...]  # canonical form, de-duplicated, in order
    fleet_wide: bool


@dataclass(frozen=True)
class ResolvedTarget:
    """Final decision handed to the agent."""

    machine_id: str | None
    intent: Intent
    notes: tuple[str, ...] = field(default_factory=tuple)


class QueryResolutionError(Exception):
    """The request can't be mapped to exactly one sensible target.

    `status_code` is the HTTP status the API layer should return; this
    module stays framework-free so it doesn't import FastAPI.
    """

    status_code = 422


class UnknownMachineError(QueryResolutionError):
    status_code = 404


class AmbiguousMachineError(QueryResolutionError):
    status_code = 422


def parse_question(
    question: str, scheme: MachineIdScheme | None = None
) -> ParsedQuery:
    """Extract machine references and fleet-wide wording from free text.

    `scheme` selects the tenant's own machine-ID format (production-
    readiness fix 12); omitting it uses `DEFAULT_MACHINE_ID_SCHEME`, the
    demo fleet's "M-NN" convention, unchanged.
    """
    scheme = scheme or DEFAULT_MACHINE_ID_SCHEME
    patterns = [scheme.compiled_code_pattern()]
    word_pattern = scheme.compiled_word_pattern()
    if word_pattern is not None:
        patterns.append(word_pattern)

    found: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(question):
            machine_id = scheme.canonicalize(match.group(1))
            if machine_id not in found:
                found.append(machine_id)
    return ParsedQuery(
        machine_ids=tuple(found),
        fleet_wide=bool(_FLEET_WIDE.search(question)),
    )


def resolve_target(
    question: str,
    explicit_machine_id: str | None,
    known_machine_ids: Collection[str],
    scheme: MachineIdScheme | None = None,
) -> ResolvedTarget:
    """Decide which machine (if any) the agent should investigate.

    Precondition: `explicit_machine_id`, when given, has already been
    checked against the fleet by the caller (the API layer 404s on it
    first, with the same message as GET /machines/{id}).

    `scheme`: see `parse_question` — this tenant's machine-ID format.
    """
    parsed = parse_question(question, scheme)

    unknown = [m for m in parsed.machine_ids if m not in known_machine_ids]
    if unknown:
        raise UnknownMachineError(
            f"Machine {unknown[0]!r} mentioned in the question was not found in the fleet."
        )

    if len(parsed.machine_ids) > 1:
        raise AmbiguousMachineError(
            "The question names several machines "
            f"({', '.join(parsed.machine_ids)}). Ask about one machine at a "
            "time, or omit machine IDs for a fleet-wide scan."
        )

    if len(parsed.machine_ids) == 1:
        mentioned = parsed.machine_ids[0]
        if explicit_machine_id is not None and explicit_machine_id != mentioned:
            raise AmbiguousMachineError(
                f"The question is about {mentioned} but machine_id={explicit_machine_id} "
                "was supplied. Make them match, or drop machine_id to use the one in the question."
            )
        logger.info(
            "target resolved: machine=%s source=%s",
            mentioned,
            "request+question" if explicit_machine_id else "question",
        )
        return ResolvedTarget(machine_id=mentioned, intent="machine_investigation")

    # The question names no machine.
    if explicit_machine_id is not None:
        notes: tuple[str, ...] = ()
        if parsed.fleet_wide:
            notes = (
                f"The question reads as fleet-wide but machine_id={explicit_machine_id} "
                f"was supplied, so only {explicit_machine_id} was investigated. "
                "Omit machine_id for a fleet scan.",
            )
        logger.info("target resolved: machine=%s source=request",
                    explicit_machine_id)
        return ResolvedTarget(
            machine_id=explicit_machine_id, intent="machine_investigation", notes=notes
        )

    notes = ()
    if not parsed.fleet_wide:
        notes = (
            "No machine was identified in the question or request, so a "
            "fleet-wide scan was run. Name a machine (e.g. M-17) for a "
            "single-machine investigation.",
        )
    logger.info(
        "target resolved: fleet scan (fleet_wide_wording=%s)", parsed.fleet_wide)
    return ResolvedTarget(machine_id=None, intent="fleet_scan", notes=notes)
