"""
LLM-driven supplementary evidence-gathering planner — audit F6 revision.

Audit findings this module addresses:

  F1 (earlier phase): "LangGraph routing is deterministic, not LLM-driven
  tool planning." route_intent still deterministically decides
  fleet_scan vs. machine_investigation (that's a fact about the request —
  was a machine_id given or not — not a judgment call worth delegating
  to an LLM).

  F6 (this revision): the original version of this module let the LLM
  decide whether to call ALL FOUR evidence tools, including
  get_sensor_evidence and get_anomaly_metrics — the two risk.assess_risk
  depends on directly. The audit's verdict: "The default 'agent' is a
  fixed pipeline... the LLM-planned path is opt-in, unverified live, has
  no tests." Worse, a scripted "lazy" planner that skipped those two
  tools could make a 100%-anomalous machine report LOW risk (see
  tests/test_agent.py's test_lazy_planner_skipping_sensor_and_anomaly_
  tools_is_unknown_not_low and this module's own tests for the fake-LLM
  version of that attack).

  The fix, per the audit's "best design": sensor and anomaly evidence
  are no longer tools this module offers at all — app/agents/graph.py
  gathers both deterministically, unconditionally, before this planner
  (or its deterministic fallback) ever runs; see that module's docstring
  for the current graph shape. What THIS module still does is real: an
  actual LLM (Gemini, via a native google-genai tool-calling loop — see
  the implementation note below) decides whether pulling maintenance
  history and/or searching the documentation knowledge base would add
  value to a specific investigation, given a summary of the evidence
  already gathered, and picks what to search for. That is a genuine,
  bounded LLM planning decision — not risk-critical, so the audit's
  concern about a bad decision here doesn't reach risk_level or the
  citations' factual grounding, but a real decision nonetheless (see
  app/agents/nodes.plan_supplementary_evidence for the evidence summary
  the model is given to decide from).

Implementation note (read this if you're wondering why there's no
ChatVertexAI here despite the paragraph above): the obvious choice,
LangChain's `langchain-google-vertexai` package, pins `pyarrow<24.0.0`
and pulls in google-cloud-aiplatform/storage/vectorsearch — which
directly conflicts with this project's own `pyarrow==25.0.1` pin
(requirements.txt, used by the Spark/Parquet pipeline) and adds a
second, mostly-unused Google SDK alongside the `google-genai` one
`app/agents/llm.py` already uses for synthesis. `google-genai` has no
pyarrow dependency at all and already talks to Gemini via Vertex
(`vertexai=True`), so `_GenAIToolCallingChat` below is a small adapter
that drives google-genai's own native function-calling API but exposes
exactly the two methods (`bind_tools`, `invoke`) the rest of this
module and app/agents/nodes.py need — LangChain's `StructuredTool` and
message classes (`SystemMessage`, `HumanMessage`, `AIMessage`,
`ToolMessage`) are still doing the real work of describing tools and
representing the conversation; only the network call underneath is
google-genai instead of langchain-google-vertexai. This keeps a single
Google SDK in the whole project and avoids the dependency conflict
entirely rather than papering over it with a version pin fight.

Gemini 3 thought signatures (audit F5): Gemini 3 models return an
encrypted `thought_signature` on function-call parts and return HTTP 400
if a later request replays that function call without it. LangChain's
AIMessage has no field for it, so an earlier version of this adapter —
which rebuilt function calls from `AIMessage.tool_calls` — would have
worked on gemini-2.5-flash and then failed on the first tool round after
the move to Gemini 3. The fix is to keep the model's own response:
`_genai_response_to_ai_message` stores the raw Content (JSON-serialised,
so it survives a checkpointer) in `AIMessage.additional_kwargs`, and
`_messages_to_genai_contents` replays that Content byte-for-byte instead
of reconstructing it. Rebuilding from `tool_calls` remains only as a
fallback for messages that never came from the model (with a warning).
Related: all function responses for one model turn must go back together
in a single user Content, so consecutive ToolMessages are now merged.

Feature-detected, not opt-in-off-by-default (audit F6): app/core/config.py's
agentic_routing_enabled now defaults to true, mirroring
gemini_synthesis_enabled's shape — this runs automatically whenever
GOOGLE_CLOUD_PROJECT is configured, with AGENTIC_ROUTING_ENABLED kept as
an explicit kill-switch for a reproducible, fully-deterministic demo run.
This is a smaller behavioural change to default-on than it would have
been before the redesign above: this module can no longer affect
risk_level, only whether historical records and doc search happen (and
in what order/with what query) versus the deterministic fallback
(app/agents/nodes.gather_historical_evidence,
app/agents/nodes.gather_maintenance_evidence) doing the same two calls
unconditionally. When it's off, or Gemini/google-genai isn't available,
app/agents/graph.py falls back to that deterministic pair — the existing
test suite (and any credential-less run) is unaffected.

What the planner is and is NOT allowed to do (same discipline as
app/agents/llm.py's Gemini-synthesis boundary):
  - IS allowed: decide whether to call get_historical_evidence and/or
    search_maintenance_documents, and what query string to search the
    maintenance docs with.
  - IS NOT allowed: fetch sensor or anomaly evidence (not offered as
    tools at all — see above), compute risk_level (app.agents.risk.
    assess_risk is still the only source, run after this module's work
    is done), invent root causes, or write anything into AgentState's
    evidence fields itself — every tool call here still goes through the
    exact same app.agents.tools functions the deterministic fallback
    uses, so the data reaching Postgres/RAG is byte-for-byte identical
    either way; the only thing that changes is *who decides whether/how*
    the two supplementary tools run.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy.engine import Connection

from app.agents import tools
from app.core.genai_client import build_genai_client
from app.core.tenancy import DEFAULT_TENANT_ID
from app.rag.retriever import DocumentRetriever

logger = logging.getLogger(__name__)

# AIMessage.additional_kwargs key holding the model's own response Content
# (JSON-serialised) so its thought_signature parts can be replayed verbatim.
RAW_CONTENT_KEY = "genai_content"

DEFAULT_PLANNER_MODEL = "gemini-3.6-flash"

# Hard cap on planner <-> tool-execution round trips. Not a suggestion to
# the model — enforced in code (app/agents/nodes.py.plan_supplementary_evidence
# checks state['planner_rounds'] before making another LLM call) so a
# confused or looping model can never hang a request. Only two tools now
# (audit F6 removed get_sensor_evidence/get_anomaly_metrics — see module
# docstring), so the longest sane plan is "call both in parallel, then
# stop" (2 rounds) or "call one, look at it, call the other, then stop"
# (3 rounds); 3 is that ceiling with no margin to spare, so the cap stays
# slightly above it rather than exactly at it.
MAX_PLANNER_TOOL_ROUNDS = 4

PLANNER_SYSTEM_INSTRUCTION = (
    "You are the SUPPLEMENTARY evidence-gathering planner for an industrial "
    "operations investigation agent. The machine's current sensor health and "
    "anomaly history have ALREADY been gathered deterministically before you "
    "were called, are NOT tools you can call, and are summarized for you in "
    "the operator's message below — a separate, deterministic step computes "
    "the machine's risk level from that data regardless of anything you decide "
    "here. Your only job is to decide, given that summary and the operator's "
    "question, whether pulling this machine's maintenance/repair history "
    "and/or searching the maintenance documentation knowledge base would add "
    "value — and if so, what to search for. You do not compute risk or "
    "diagnose the machine yourself.\n\n"
    "Call get_historical_evidence when the question concerns past repairs or "
    "recurring problems, or when the anomaly summary suggests this may not be "
    "a first occurrence. Call search_maintenance_documents (with a specific "
    "query describing the symptom, e.g. 'tool wear excessive heat', not just "
    "the machine ID) when the question could be answered by documented "
    "procedures or known failure modes — the anomaly summary's detail is "
    "usually a better query than the operator's raw question. It is a valid, "
    "reportable outcome to call neither tool if the sensor/anomaly summary "
    "alone already answers the question (e.g. a machine with no anomalies and "
    "a healthy status). Call a tool only once each; do not repeat a call you "
    "already made. When you are done (including if you decided to call "
    "nothing), reply with a short plain-text confirmation such as 'Evidence "
    "gathering complete.' — do not summarize findings yourself; a separate "
    "step handles that."
)


def build_planner_tools(
    conn: Connection,
    retriever: DocumentRetriever | None,
    machine_id: str,
    errors: list[str],
    collected: dict[str, Any],
    tenant_id: str = DEFAULT_TENANT_ID,
) -> list[Any]:
    """Wrap the two SUPPLEMENTARY tool functions (app.agents.tools)
    as LangChain StructuredTools the planner LLM can call — audit F6
    deliberately excludes get_machine_health/get_sensor_summary and
    get_machine_anomalies here; those are gathered unconditionally by
    app/agents/nodes.py before this planner ever runs (see this module's
    docstring), so they are not offered as tools at all: there is no tool
    name a confused or adversarial planner could fail to call, or call
    incorrectly, that would leave risk.assess_risk without evidence.

    Each wrapper calls the exact same deterministic function the
    deterministic fallback (app.agents.nodes.gather_historical_evidence /
    gather_maintenance_evidence) uses (identical query logic, identical
    error handling via the shared `errors` list), then does two things
    with the result: writes it into `collected` under the AgentState key
    it belongs to (read back into state once the loop ends by
    app.agents.nodes.finalize_supplementary_evidence), and returns a
    compact JSON summary as the tool's return value so the LLM can see
    what it got back — the LLM never sees the raw evidence at planning
    time, only counts, which keeps this call cheap and keeps the model
    from trying to reason about the evidence itself (that's Phase 6's
    synthesizer's job, later, once risk_level and root causes are already
    decided).

    machine_id and tenant_id are closed over, not LLM-supplied tool
    arguments: they are facts about the request the planner should never
    be trusted to type correctly, so a hallucinated/malformed machine_id
    can never reach a query — and a prompt-injected planner can never
    steer a query at another tenant's data. `conn`, `retriever`, `errors` and `collected` are all
    request-scoped objects built fresh per investigation in
    app.agents.graph.build_graph, exactly like the closures nodes.py
    already uses for the deterministic fallback.
    """
    # Deferred import: langchain_core is only required when the
    # supplementary planner is actually configured (see get_planner_llm
    # below and app/core/config.py's agentic_routing_configured) — a
    # credential-less run, and the existing test suite, never import it.
    from langchain_core.tools import StructuredTool

    def _get_historical_evidence() -> str:
        """Fetch this machine's maintenance/repair event history. Call when the question concerns past repairs or recurring issues."""
        records = tools.query_machine_history(
            conn, machine_id, errors, tenant_id=tenant_id)
        collected["historical_evidence"] = records
        return json.dumps({"maintenance_records_returned": len(records)}, default=str)

    def _search_maintenance_documents(query: str) -> str:
        """Search the maintenance-documentation knowledge base for procedures or known failure modes relevant to this machine. Give a specific query describing the symptom, e.g. 'tool wear excessive heat'."""
        docs = tools.search_maintenance_documents(
            f"{machine_id} {query}".strip(), errors, retriever=retriever
        )
        collected["maintenance_evidence"] = docs
        return json.dumps(
            {"documents_returned": len(docs), "doc_ids": [
                d["doc_id"] for d in docs]},
            default=str,
        )

    return [
        StructuredTool.from_function(
            func=_get_historical_evidence, name="get_historical_evidence"),
        StructuredTool.from_function(
            func=_search_maintenance_documents, name="search_maintenance_documents"
        ),
    ]


_planner_llm: Any | None = None
_planner_unavailable: bool = False


def _tool_to_function_declaration(tool: Any) -> Any:
    """Convert a LangChain StructuredTool (see build_planner_tools) into a
    google.genai.types.FunctionDeclaration. Deferred import for the same
    reason as build_planner_tools' langchain_core import."""
    from google.genai import types

    schema: dict[str, Any] = {"type": "object", "properties": {}}
    if tool.args_schema is not None:
        json_schema = tool.args_schema.model_json_schema()
        schema["properties"] = json_schema.get("properties", {})
        if json_schema.get("required"):
            schema["required"] = json_schema["required"]

    return types.FunctionDeclaration(
        name=tool.name,
        description=tool.description or "",
        parameters_json_schema=schema,
    )


def _messages_to_genai_contents(messages: list[Any]) -> tuple[str | None, list[Any]]:
    """Convert LangChain BaseMessages (SystemMessage, HumanMessage,
    AIMessage with tool_calls, ToolMessage) into (system_instruction,
    contents) for google.genai's generate_content. Gemini's own contract:
    at most one system instruction (pulled out separately, not sent as a
    turn), and a function's result goes back as a function_response Part
    inside a role="user" turn — Gemini has no separate "tool" role.

    Two Gemini rules this function exists to honour (audit F5):
      * A model turn is replayed from the raw Content stored in
        AIMessage.additional_kwargs[RAW_CONTENT_KEY] when present, so
        Gemini 3 thought signatures come back exactly as returned.
      * The function responses answering one model turn go in ONE user
        Content, one part per call — consecutive ToolMessages are merged.
    """
    from google.genai import types
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    system_instruction: str | None = None
    contents: list[Any] = []
    pending_responses: list[Any] = []

    def _flush_responses() -> None:
        if pending_responses:
            contents.append(types.Content(
                role="user", parts=list(pending_responses)))
            pending_responses.clear()

    for msg in messages:
        if isinstance(msg, ToolMessage):
            pending_responses.append(types.Part.from_function_response(
                name=msg.name, response={"result": msg.content}
            ))
            continue
        _flush_responses()

        if isinstance(msg, SystemMessage):
            system_instruction = str(msg.content)
        elif isinstance(msg, HumanMessage):
            contents.append(types.Content(role="user", parts=[
                            types.Part.from_text(text=str(msg.content))]))
        elif isinstance(msg, AIMessage):
            raw = (msg.additional_kwargs or {}).get(RAW_CONTENT_KEY)
            if raw is not None:
                contents.append(types.Content.model_validate(raw))
                continue
            if msg.tool_calls:
                logger.warning(
                    "Replaying a function call rebuilt from tool_calls, not from the "
                    "model's own response — Gemini 3 will reject this with a 400 "
                    "(missing thought_signature). Only expected for messages not "
                    "produced by _GenAIToolCallingChat."
                )
            parts = []
            if msg.content:
                parts.append(types.Part.from_text(text=str(msg.content)))
            for call in (msg.tool_calls or []):
                parts.append(types.Part.from_function_call(
                    name=call["name"], args=call.get("args") or {}))
            if parts:
                contents.append(types.Content(role="model", parts=parts))

    _flush_responses()
    return system_instruction, contents


def _genai_response_to_ai_message(response: Any) -> Any:
    """Convert a google.genai GenerateContentResponse back into a
    LangChain AIMessage — the shape app.agents.nodes.plan_supplementary_evidence
    and LangGraph's ToolNode both expect (`.content`, `.tool_calls`).

    The model's raw Content is kept in additional_kwargs (JSON-safe) so
    thought signatures survive the round trip; see this module's docstring.
    """
    from langchain_core.messages import AIMessage

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    candidates = getattr(response, "candidates", None) or []
    content = candidates[0].content if candidates and candidates[0].content else None
    parts = content.parts if content else []
    for i, part in enumerate(parts or []):
        if getattr(part, "text", None) and not getattr(part, "thought", False):
            text_parts.append(part.text)
        fc = getattr(part, "function_call", None)
        if fc is not None:
            tool_calls.append({
                "name": fc.name,
                "args": dict(fc.args or {}),
                "id": f"call_{i}",
            })

    additional_kwargs: dict[str, Any] = {}
    if content is not None:
        additional_kwargs[RAW_CONTENT_KEY] = content.model_dump(
            mode="json", exclude_none=True)

    return AIMessage(
        content="".join(text_parts),
        tool_calls=tool_calls,
        additional_kwargs=additional_kwargs,
    )


class _GenAIToolCallingChat:
    """Minimal LangChain-ChatModel-shaped adapter around google.genai's
    native Gemini function calling — see this module's docstring for why
    this exists instead of langchain-google-vertexai. Deliberately narrow:
    implements only bind_tools() and invoke(), the two calls
    app.agents.nodes.plan_supplementary_evidence makes; it is not a
    general-purpose LangChain ChatModel.
    """

    def __init__(
        self,
        client: Any,
        model_name: str,
        genai_tools: Any | None = None,
        temperature: float | None = None,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._genai_tools = genai_tools
        # None -> omit and use the model default (Google's Gemini 3
        # guidance: don't lower it). See Settings.planner_temperature.
        self._temperature = temperature

    def bind_tools(self, tools: list[Any]) -> "_GenAIToolCallingChat":
        from google.genai import types

        declarations = [_tool_to_function_declaration(t) for t in tools]
        genai_tools = [types.Tool(function_declarations=declarations)]
        return _GenAIToolCallingChat(
            self._client, self._model_name, genai_tools, self._temperature)

    def invoke(self, messages: list[Any]) -> Any:
        from google.genai import types

        system_instruction, contents = _messages_to_genai_contents(messages)
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=self._genai_tools,
            temperature=self._temperature,
        )
        response = self._client.models.generate_content(
            model=self._model_name, contents=contents, config=config
        )
        return _genai_response_to_ai_message(response)


def get_planner_llm(refresh: bool = False) -> Any | None:
    """Process-wide planner LLM, built lazily — unbound; tools are bound
    per-request in app.agents.nodes.plan_supplementary_evidence, since the
    tools themselves close over request-scoped conn/retriever/machine_id
    (see build_planner_tools).

    Returns None (not raise) when the supplementary planner isn't
    configured, isn't enabled, or google-genai isn't installed —
    app.agents.graph.build_graph falls back to the deterministic
    historical+maintenance-doc pair in that case, exactly the same
    degrade-not-crash contract app.agents.llm.get_synthesizer uses for
    Gemini synthesis. Uses the
    same google-genai SDK and Application Default Credentials as
    app.agents.llm.GeminiSynthesizer — see this module's docstring for
    why, unlike an early draft of this module, it does NOT depend on
    langchain-google-vertexai.

    Disclosure, same as app/agents/llm.py: this was verified against the
    google-genai 2.24.0 and langchain-core 1.6.3 import surfaces
    (FunctionDeclaration/Tool/Content/Part construction, StructuredTool's
    JSON schema output), not against a live Vertex AI call — this
    development sandbox has no egress to Google. Confirm one real
    planner call against your own GOOGLE_CLOUD_PROJECT before relying on
    this for a live demo.
    """
    global _planner_llm, _planner_unavailable
    if refresh:
        _planner_llm, _planner_unavailable = None, False
    if _planner_llm is not None:
        return _planner_llm
    if _planner_unavailable:
        return None

    from app.core.config import get_settings

    settings = get_settings()
    if not settings.agentic_routing_configured:
        logger.info(
            "Supplementary evidence-gathering planner not configured "
            "(AGENTIC_ROUTING_ENABLED=false and/or GOOGLE_CLOUD_PROJECT unset) — "
            "using the deterministic historical+maintenance-doc fallback. "
            "Sensor/anomaly evidence is gathered deterministically either way."
        )
        _planner_unavailable = True
        return None

    try:
        from google import genai
    except ImportError as exc:
        logger.warning(
            "google-genai is not installed (pip install -r requirements-cloud.txt) "
            "— supplementary evidence-gathering planner unavailable, falling back to the "
            "deterministic historical+maintenance-doc pair: %s",
            exc,
        )
        _planner_unavailable = True
        return None

    try:
        # Shared timeout + bounded-retry policy (app/core/genai_client.py).
        client = build_genai_client(
            settings.gcp_project, settings.gemini_location)
        _planner_llm = _GenAIToolCallingChat(
            client, settings.planner_model, temperature=settings.planner_temperature
        )
        logger.info(
            "Evidence-gathering planner ready: model=%s region=%s",
            settings.planner_model,
            settings.gemini_location,
        )
    except Exception as exc:  # noqa: BLE001 - any construction failure degrades, never crashes
        logger.warning(
            "Planner client construction failed, agentic routing unavailable, "
            "falling back to the deterministic historical+maintenance-doc pair: %s",
            exc,
        )
        _planner_unavailable = True
        return None
    return _planner_llm
