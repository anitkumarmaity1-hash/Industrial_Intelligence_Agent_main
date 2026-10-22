"""
Gemini synthesis via Vertex AI — Phase 6.

Swaps the internals of the Phase 5 template (app/agents/synthesis.py) for
a real LLM call, exactly as that module's docstring said Phase 6 would:
same evidence in, same RESPONSE FORMAT out, no change to the graph shape
around it (app/agents/graph.py, app/agents/nodes.py structure is
untouched — only build_recommendation's body gains a try-then-fallback).

What Gemini is and is NOT allowed to do here, deliberately narrow:
  - IS allowed: turn already-gathered, already-risk-scored, already
    hypothesis-labeled evidence into fluent prose for the RESPONSE
    FORMAT's prose sections (What changed?, Investigation, Possible
    causes, Recommended action, Confidence & limitations).
  - IS NOT allowed: decide risk_level (app.agents.risk.assess_risk,
    Phase 5, remains the only source), invent root-cause candidates
    beyond root_cause_candidates (app.agents.nodes.identify_root_causes,
    Phase 5), call any tool, or see anything the graph didn't already
    put in AgentState. The prompt below is explicit that
    possible_causes_prose must cover exactly the candidates already
    identified — rephrased, not expanded.
This is EVIDENCE BEFORE EXPLANATION and DETERMINISTIC COMPUTATION OUTSIDE
THE LLM applied literally: Gemini explains a verdict, it does not render one.

Audit F7 fix. The above was previously aspirational, not enforced —
nothing checked that Gemini's prose actually stayed inside those rules.
GeminiSynthesizer.synthesize now runs the response through
_validate_report_against_evidence before returning it: every number in
the prose must trace back to the evidence JSON, and possible_causes_prose
must cover root_cause_candidates_already_identified one-for-one. Either
failure raises, and nodes.build_recommendation's existing try/except
falls back to the deterministic template exactly as it already does for
a schema mismatch or a network error — this is a new way to fail the
same way, not a new failure mode the caller has to special-case.

Same SDK/auth as app/rag/embeddings.py (google-genai, vertexai=True,
Application Default Credentials) — imported lazily so the rest of the
agent package, and all of Phase 5's tests, run with no Google SDK
installed and no credentials present.

Citations are NOT part of what Gemini writes. app/agents/synthesis.py
still builds "Supporting evidence" and the citations list returned to
the API directly from maintenance_evidence/historical_evidence — an LLM
is not a trustworthy source for which doc_ids were actually retrieved,
so it is never asked to produce that list.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.agents import synthesis
from app.agents.state import AgentState
from app.core.genai_client import build_genai_client

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"

_SYSTEM_INSTRUCTION = (
    "You are an industrial operations decision-support assistant. You write ONLY from "
    "the evidence JSON given to you in the prompt. Never invent a machine statistic, "
    "sensor reading, failure cause, or maintenance fact that is not present in that JSON. "
    "Any number, percentage, or rate you state must be copied from the evidence JSON "
    "(e.g. use the 'recent_vs_baseline' figures verbatim for the What changed? section) — "
    "never compute, round, or restate a number in a form that doesn't appear in the JSON. "
    "State causes as possibilities the evidence suggests, never as confirmed diagnoses — "
    "always defer to a qualified technician for final judgment. possible_causes_prose must "
    "have exactly one entry per item in root_cause_candidates_already_identified, in the "
    "same order — rephrase each one, never merge, drop, or add to them. If the evidence is "
    "thin, say so plainly in confidence_limitations rather than filling the gap with a guess."
)


class GeminiReportSections(BaseModel):
    """Schema Gemini must fill, returned via Vertex structured output.

    Deliberately excludes machine_id, risk_level, and root-cause identity
    — those are already decided upstream (see module docstring) and are
    not the model's to restate, and possibly get wrong.
    """

    what_changed: list[str] = Field(min_length=1, max_length=6)
    investigation_summary: str = Field(min_length=1)
    possible_causes_prose: list[str] = Field(min_length=1, max_length=6)
    recommended_action: str = Field(min_length=1)
    confidence_limitations: str = Field(min_length=1)


def _build_evidence_payload(state: AgentState) -> dict[str, Any]:
    """The evidence dict Gemini sees, factored out of _build_prompt so
    app.agents.llm._validate_report_against_evidence (audit F7) can
    re-derive the exact same payload to check Gemini's prose against,
    rather than re-parsing the rendered prompt string.
    """
    # Audit F8 (finding 1): the same deterministic last-24h-vs-prior-
    # baseline comparison app.agents.synthesis renders into the
    # deterministic template's "What changed?" section (see
    # synthesis._what_changed_lines) — handed to Gemini as structured,
    # already-computed numbers so it narrates them instead of computing
    # (and potentially getting wrong, or wording ambiguously) its own
    # percentage from the raw trend. Also gives the F7 guardrail below a
    # literal string to match a "17.2%" (etc.) claim in Gemini's prose
    # against, instead of having to re-derive percentages from floats.
    baseline_comparison = synthesis.compute_recent_vs_baseline(
        state.get("sensor_trend_baseline") or [])

    return {
        "machine_id": state.get("machine_id"),
        "user_question": state.get("user_query"),
        "risk_level_already_determined": state.get("risk_level"),
        "sensor_snapshot": state.get("sensor_evidence"),
        "sensor_trend_windows_most_recent_first": state.get("sensor_trend") or [],
        "recent_vs_baseline": baseline_comparison,
        "anomaly_events_most_recent_first": state.get("anomaly_metrics") or [],
        "maintenance_history": state.get("historical_evidence") or [],
        "maintenance_documentation_hits": [
            {"doc_id": d["doc_id"], "title": d["title"],
                "section": d["section"], "text": d["text"]}
            for d in (state.get("maintenance_evidence") or [])
        ],
        "root_cause_candidates_already_identified": state.get("root_cause_candidates") or [],
    }


def _build_prompt(state: AgentState) -> str:
    """Serialize exactly the evidence Gemini is allowed to see. Internal
    routing/error-tracking fields (intent, errors) are deliberately left
    out — they're not evidence about the machine, they're bookkeeping.
    """
    payload = _build_evidence_payload(state)
    return (
        "Evidence (JSON, already gathered and verified — treat as ground truth; "
        "do not add facts beyond it):\n"
        f"{json.dumps(payload, indent=2, default=str)}\n\n"
        "Write the investigation report sections. `possible_causes_prose` must cover "
        "exactly the items in root_cause_candidates_already_identified, rephrased for "
        "readability — do not add candidates beyond that list, and do not drop any. Use "
        "`recent_vs_baseline`'s own figures (verbatim) for any 'what changed' percentage "
        "or rate you state."
    )


# --- Audit F7: hallucination guardrail --------------------------------
#
# Nothing here checks whether Gemini's prose is a *good* explanation —
# that's unautomatable and out of scope. It checks the narrower, testable
# claim the audit finding was actually about: every number Gemini states
# (a percentage, a rate, a reading count, an anomaly score, ...) must be
# traceable to something the evidence JSON actually said, and the model
# must not silently drop or invent root-cause candidates. Either failure
# raises, which (see GeminiSynthesizer.synthesize's docstring and
# app.agents.nodes.build_recommendation) makes the caller fall back to
# the deterministic template rather than show an unverified number.
#
# Numbers embedded inside a date/time string (e.g. "2026-01-01T10:00:00")
# are deliberately excluded — matched only when NOT adjacent to another
# digit, '.', '-', ':' or '/' — so this doesn't flag timestamp components
# as "invented statistics". Bare single-digit whole numbers (0-9) are
# also excluded: "one possible cause" / a lone anomaly count is common,
# expected prose that would otherwise produce constant false positives;
# this checks substantive figures (rates, scores, multi-digit counts,
# decimals, percentages) where a hallucinated statistic actually matters.
_NUMBER_TOKEN_PATTERN = re.compile(
    r"(?<![\d.\-:/])(?:\d{2,}(?:\.\d+)?|\d\.\d+)%?(?![\d.\-:/])"
)


def _numbers_in(text: str) -> set[str]:
    return {tok.rstrip("%") for tok in _NUMBER_TOKEN_PATTERN.findall(text)}


def _evidence_number_pool(payload: dict[str, Any]) -> set[str]:
    """Every substantive number appearing anywhere in the evidence JSON,
    as the same normalized tokens _numbers_in extracts from Gemini's
    prose — so a number is "supported" if it appears anywhere in the
    evidence, not only in the specific field a human might expect.
    """
    return _numbers_in(json.dumps(payload, default=str))


def _validate_report_against_evidence(
    sections: "GeminiReportSections", payload: dict[str, Any]
) -> None:
    """Raises ValueError (caught by GeminiSynthesizer.synthesize's caller
    the same way a schema mismatch already is) when either check fails:

      1. possible_causes_prose doesn't have exactly one entry per
         root_cause_candidates_already_identified — the audit's "'cover
         exactly these candidates' isn't enforced" finding.
      2. A substantive number in any prose section doesn't appear
         anywhere in the evidence JSON — the audit's "nothing verifies
         that numbers or dates in what_changed exist in the payload"
         finding. what_changed is the section the finding named, but the
         same guardrail applies to every prose section Gemini writes,
         since a hallucinated number in Possible causes or Confidence &
         limitations is exactly as unverified.
    """
    expected_causes = payload.get(
        "root_cause_candidates_already_identified") or []
    if len(sections.possible_causes_prose) != len(expected_causes):
        raise ValueError(
            f"Gemini returned {len(sections.possible_causes_prose)} possible_causes_prose "
            f"entries but root_cause_candidates_already_identified has {len(expected_causes)} "
            "— it must cover exactly the identified candidates, one-for-one."
        )

    evidence_numbers = _evidence_number_pool(payload)
    prose_sections = {
        "what_changed": "\n".join(sections.what_changed),
        "investigation_summary": sections.investigation_summary,
        "possible_causes_prose": "\n".join(sections.possible_causes_prose),
        "recommended_action": sections.recommended_action,
        "confidence_limitations": sections.confidence_limitations,
    }
    unsupported: list[str] = []
    for section_name, text in prose_sections.items():
        stated = _numbers_in(text)
        missing = stated - evidence_numbers
        if missing:
            unsupported.append(f"{section_name}: {sorted(missing)}")
    if unsupported:
        raise ValueError(
            "Gemini's prose contains numbers that don't appear anywhere in the evidence "
            "JSON (possible hallucinated statistic): " + "; ".join(unsupported)
        )


class GeminiSynthesizer:
    """Calls Gemini via Vertex AI to write the prose sections of an
    investigation report from evidence already in AgentState."""

    def __init__(
        self,
        project: str,
        location: str = "global",
        model_name: str = DEFAULT_GEMINI_MODEL,
        temperature: float | None = None,
    ) -> None:
        """Create the Vertex client.

        Raises:
            ImportError: if `google-genai` is not installed.
            ValueError: if no project is configured.
        """
        if not project:
            raise ValueError(
                "A Google Cloud project is required for Gemini synthesis.")

        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError(
                "google-genai is required for Gemini synthesis "
                "(pip install -r requirements-cloud.txt). Without it, /investigate "
                "falls back to the deterministic template (app.agents.synthesis)."
            ) from exc

        # Shared timeout + bounded-retry policy (app/core/genai_client.py).
        self._client = build_genai_client(project, location)
        self.model_name = model_name
        # None -> omit from the request and use the model's default. See
        # Settings.gemini_temperature for why 0.2 is no longer hardcoded.
        self._temperature = temperature
        logger.info("Gemini synthesizer ready: model=%s region=%s",
                    model_name, location)

    def synthesize(self, state: AgentState) -> GeminiReportSections:
        """Call Gemini and return validated report sections.

        Raises:
            RuntimeError: empty response, the response didn't match
                GeminiReportSections, or it failed the audit-F7
                hallucination guardrail (a stated number not traceable to
                the evidence JSON, or possible_causes_prose not covering
                root_cause_candidates_already_identified one-for-one —
                see _validate_report_against_evidence). Any Vertex/network
                exception also propagates as-is. This method never
                returns a partial, guessed, or unverified report — the
                caller (nodes.py) decides the fallback, and it always
                falls back to the deterministic template rather than show
                the person a number this method couldn't verify.
        """
        from google.genai import types

        payload = _build_evidence_payload(state)
        config = types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=GeminiReportSections,
            temperature=self._temperature,
        )
        response: Any = self._client.models.generate_content(
            model=self.model_name,
            contents=_build_prompt(state),
            config=config,
        )
        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")
        try:
            sections = GeminiReportSections.model_validate_json(response.text)
        except ValidationError as exc:
            raise RuntimeError(
                f"Gemini response did not match the expected schema: {exc}") from exc

        try:
            _validate_report_against_evidence(sections, payload)
        except ValueError as exc:
            raise RuntimeError(
                f"Gemini response failed the evidence-verification guardrail: {exc}") from exc

        return sections


_synthesizer: GeminiSynthesizer | None = None
_synthesizer_unavailable: bool = False


def get_synthesizer(refresh: bool = False) -> GeminiSynthesizer | None:
    """Process-wide GeminiSynthesizer singleton, built lazily.

    Returns None (not raise) when Gemini isn't configured or usable —
    unlike the RAG retriever, which /investigate cannot run without,
    Gemini synthesis is optional: the deterministic template is a
    complete, working fallback (see app/agents/synthesis.py), so a
    missing/broken Gemini setup should degrade the report quality, not
    break the endpoint. Failure is cached for the process lifetime
    (`_synthesizer_unavailable`) so a misconfigured deployment doesn't
    retry the same doomed construction on every single request.
    """
    global _synthesizer, _synthesizer_unavailable
    if refresh:
        _synthesizer, _synthesizer_unavailable = None, False
    if _synthesizer is not None:
        return _synthesizer
    if _synthesizer_unavailable:
        return None

    from app.core.config import get_settings

    settings = get_settings()
    if not settings.gemini_configured:
        logger.info(
            "Gemini synthesis not configured (no GOOGLE_CLOUD_PROJECT or disabled) — using deterministic template.")
        _synthesizer_unavailable = True
        return None

    try:
        _synthesizer = GeminiSynthesizer(
            # type: ignore[arg-type]  - gemini_configured guarantees non-None
            project=settings.gcp_project,
            location=settings.gemini_location,
            model_name=settings.gemini_model,
            temperature=settings.gemini_temperature,
        )
    except (ImportError, ValueError) as exc:
        logger.warning(
            "Gemini synthesizer unavailable, falling back to deterministic template: %s", exc)
        _synthesizer_unavailable = True
        return None
    return _synthesizer
