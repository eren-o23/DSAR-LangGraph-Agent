from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple, TypeVar

from pydantic import BaseModel, ValidationError

from dsar_langgraph_agent.ollama_client import OllamaLLMClient, OllamaLLMConfig, OllamaLLMError
from dsar_langgraph_agent.triage_schemas import (
    ClassificationResult,
    DSARRequestType,
    FallbackCause,
    LLMFallbackWarning,
    ReasoningEntry,
    RiskFlag,
    RiskResult,
    ScopingResult,
)
from dsar_langgraph_agent.triage_agents import (
    _append_reasoning,
    classification_agent,
    risk_flag_agent,
    scoping_agent,
)
from dsar_langgraph_agent.triage_state import TriageState

try:
    from langgraph.config import get_store as _get_store
except ImportError:  # pragma: no cover — older LG versions
    _get_store = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMNodeConfig:
    base_url: Optional[str] = "http://localhost:11434/v1"  # None → use OpenAI's default endpoint
    api_key: str = "ollama"
    model: str = "llama3.1"
    timeout_s: float = 60.0

    @classmethod
    def from_env(cls) -> "LLMNodeConfig":
        """Auto-detect provider from environment variables.

        - If ``OPENAI_API_KEY`` is set, return a config that targets the real
          OpenAI API using ``gpt-4o-mini`` (cheap, fast — ideal for quick testing).
        - Otherwise return the default Ollama config.
        """
        import os
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            return cls(
                base_url=None,       # OpenAI SDK will use https://api.openai.com/v1
                api_key=openai_key,
                model="gpt-4o-mini",  # cheap + fast for testing
                timeout_s=30.0,
            )
        return cls()  # default: Ollama


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _client(cfg: LLMNodeConfig) -> OllamaLLMClient:
    return OllamaLLMClient(
        OllamaLLMConfig(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            model=cfg.model,
            timeout_s=cfg.timeout_s,
        )
    )


def _as_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str)


def _classify_exception(exc: Exception) -> Tuple[FallbackCause, str]:
    """Map a raw exception to a structured (cause, detail) pair."""
    detail = str(exc)
    lower = detail.lower()

    if isinstance(exc, OllamaLLMError):
        # OllamaLLMError wraps both network/timeout failures and JSON parse failures.
        if any(kw in lower for kw in ("timed out", "timeout", "connect", "connection")):
            return "timeout", detail
        if any(kw in lower for kw in ("json", "parse", "locate", "empty model")):
            return "bad_json", detail
        return "unknown", detail

    if isinstance(exc, ValidationError):
        return "schema_error", detail

    if isinstance(exc, (json.JSONDecodeError, ValueError)) and "json" in lower:
        return "bad_json", detail

    return "unknown", detail


def _append_warning(state: TriageState, warning: LLMFallbackWarning) -> List[LLMFallbackWarning]:
    """Return a new list with *warning* appended; never mutates state in place."""
    existing: List[LLMFallbackWarning] = list(state.get("llm_warnings") or [])
    existing.append(warning)
    return existing


def _run_with_fallback(
    *,
    agent_name: str,
    state: TriageState,
    llm_call: Callable[[], T],
    post_validate: Callable[[T], T],
    fallback_result: T,
    state_key: str,
    extra_state: dict | None = None,
    llm_entry_factory: Callable[[T], ReasoningEntry],
    fallback_entry_factory: Callable[[], ReasoningEntry],
) -> TriageState:
    """
    Execute *llm_call*, validate the result via *post_validate*, and write
    it into ``state[state_key]``.

    **Reasoning history** is always updated:
    - **Success** → one ``source="llm"`` entry built by *llm_entry_factory*.
    - **Failure** → two consecutive entries:
        1. ``source="llm_fallback"`` — records the attempt and failure detail.
        2. ``source="deterministic"`` — built by *fallback_entry_factory* from the
           deterministic result, timestamped at the moment it is used.

    On **any** failure the existing ``llm_warnings`` list is also updated via
    :class:`LLMFallbackWarning`.

    *extra_state* is merged into the returned state dict (used by the
    classification agent to also clear ``human_feedback``).
    """
    new_warnings: List[LLMFallbackWarning] | None = None
    result: T
    existing_history: List[ReasoningEntry] = list(state.get("reasoning_history") or [])

    try:
        raw = llm_call()
        result = post_validate(raw)
        new_history = existing_history + [llm_entry_factory(result)]
    except Exception as exc:
        cause, detail = _classify_exception(exc)
        log.warning(
            "[%s] LLM fallback triggered (cause=%s): %s",
            agent_name, cause, detail,
        )
        warning = LLMFallbackWarning(agent=agent_name, cause=cause, detail=detail)
        new_warnings = _append_warning(state, warning)
        result = fallback_result

        # Two-entry representation: the failed LLM attempt, then the deterministic fallback.
        failed_entry = ReasoningEntry(
            agent=agent_name,
            source="llm_fallback",
            decision_made="llm_failed",
            rationale=f"[{cause}] {detail}",
        )
        new_history = existing_history + [failed_entry, fallback_entry_factory()]

    out: TriageState = {
        **state,
        state_key: result,
        "reasoning_history": new_history,
    }
    if new_warnings is not None:
        out["llm_warnings"] = new_warnings
    if extra_state:
        out.update(extra_state)  # type: ignore[typeddict-item]
    return out


# ---------------------------------------------------------------------------
# LLM node implementations
# ---------------------------------------------------------------------------

def classification_agent_llm(state: TriageState, *, cfg: LLMNodeConfig) -> TriageState:
    """
    LLM-powered classification with deterministic fallback and episodic retrieval.

    **Episodic retrieval** — At the start of every call the node queries the
    LangGraph Store for up to 2 previously approved triage episodes whose
    ``request_text`` is similar to the current request.  These are surfaced to
    the LLM as few-shot examples so it can anchor its decision on real
    precedents rather than reasoning from scratch.

    ``store.search()`` is called with ``query=<request_text>`` and ``limit=2``.
    On plain ``InMemoryStore`` (no embedding backend) the query string is
    accepted but silently ignored; results are returned by insertion order.
    Switching to a vector-backed store (e.g. ``AsyncPostgresStore`` with an
    embedder) makes the semantic search active without any code changes here.

    If the store contains no episodes yet the function handles an empty list
    gracefully and the prompt is unchanged.

    Failure modes handled:
    - Ollama timeout / connection error  → cause=``timeout``
    - Non-JSON or unparseable response   → cause=``bad_json``
    - Valid JSON but fails Pydantic schema → cause=``schema_error``
    - Any other exception                → cause=``unknown``

    On fallback the deterministic :func:`classification_agent` result is used.
    The reasoning history will contain two consecutive entries for this agent:
    one ``llm_fallback`` entry and one ``deterministic`` entry.
    """
    from dsar_langgraph_agent.triage_graph import _EPISODE_NAMESPACE

    req = state.get("request_text", "")
    human_feedback = state.get("human_feedback")

    # ------------------------------------------------------------------
    # Episodic retrieval — fetch up to 2 similar precedents from store
    # ------------------------------------------------------------------
    matched_precedents: List[dict] = []
    try:
        ep_store = _get_store() if _get_store is not None else None
        if ep_store is not None:
            hits = ep_store.search(
                _EPISODE_NAMESPACE,
                query=req,   # semantic search when backed by a vector store;
                limit=2,     # silently ignored as ranking on plain InMemoryStore
            )
            for hit in hits:
                v = hit.value or {}
                classification_dict = v.get("classification") or {}
                final_decision = classification_dict.get("request_type", "unknown")
                matched_precedents.append({
                    "request_text":  v.get("request_text", ""),
                    "final_decision": final_decision,
                })
    except Exception as retrieval_exc:  # never crash the node due to store issues
        log.warning(
            "[classification_agent_llm] Episodic retrieval failed, continuing without precedents: %s",
            retrieval_exc,
        )
        matched_precedents = []

    # Run deterministic agent first — it handles human_feedback correction and
    # clears the field, giving us a fully-prepared fallback result + rationale.
    # NOTE: we only extract the result; the deterministic history entry is
    # discarded here and replaced by _run_with_fallback's own entries.
    deterministic_state = classification_agent(state)
    deterministic_result: ClassificationResult = deterministic_state["classification"]

    system_prompt = (
        "You are a DSAR triage classifier.\n"
        "Return ONLY valid JSON matching this schema:\n"
        '{ "request_type": "access|deletion|portability|unknown", "confidence": 0.0-1.0, "rationale": "..." }\n'
        "No markdown, no code fences, no extra keys."
    )
    user_prompt = f"DSAR request text:\n{req}\n"

    # Inject precedents as few-shot context when available.
    if matched_precedents:
        examples = "\n".join(
            f'  - Request: "{p["request_text"]}" → Decision: {p["final_decision"]}'
            for p in matched_precedents
        )
        user_prompt += (
            "\n--- Relevant precedents from approved past decisions ---\n"
            "Use these as guidance, but make an independent decision based on the current request:\n"
            f"{examples}\n"
            "--- End of precedents ---\n"
        )

    if human_feedback:
        user_prompt += (
            "\n--- Human reviewer correction ---\n"
            "The previous classification was rejected. The reviewer provided the following feedback. "
            "You MUST incorporate it and produce a corrected classification:\n"
            f"{human_feedback.strip()}\n"
            "--- End of correction ---\n"
        )

    def _call() -> ClassificationResult:
        data = _client(cfg).chat_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        result = ClassificationResult.model_validate(data, strict=False)
        if result.request_type not in (
            DSARRequestType.access,
            DSARRequestType.deletion,
            DSARRequestType.portability,
            DSARRequestType.unknown,
        ):
            raise ValidationError.from_exception_data(  # type: ignore[attr-defined]
                title="ClassificationResult",
                input_type="python",
                line_errors=[{
                    "type": "enum",
                    "loc": ("request_type",),
                    "msg": f"Invalid request_type value: {result.request_type!r}",
                    "input": result.request_type,
                    "ctx": {},
                }],
            )
        return result

    def _post_validate(r: ClassificationResult) -> ClassificationResult:
        return r  # validation is already done inside _call()

    def _llm_entry(r: ClassificationResult) -> ReasoningEntry:
        return ReasoningEntry(
            agent="classification_agent_llm",
            source="llm",
            decision_made=f"request_type={r.request_type.value}, confidence={r.confidence}",
            rationale=r.rationale or "LLM provided no rationale.",
        )

    def _fallback_entry() -> ReasoningEntry:
        return ReasoningEntry(
            agent="classification_agent_llm",
            source="deterministic",
            decision_made=(
                f"request_type={deterministic_result.request_type.value}, "
                f"confidence={deterministic_result.confidence}"
            ),
            rationale=deterministic_result.rationale or "Deterministic fallback applied.",
        )

    return _run_with_fallback(
        agent_name="classification_agent_llm",
        state=state,
        llm_call=_call,
        post_validate=_post_validate,
        fallback_result=deterministic_result,
        state_key="classification",
        # Always clear human_feedback whether we used the LLM or fell back.
        extra_state={"human_feedback": None},
        llm_entry_factory=_llm_entry,
        fallback_entry_factory=_fallback_entry,
    )


def scoping_agent_llm(state: TriageState, *, cfg: LLMNodeConfig) -> TriageState:
    """
    LLM-powered scoping with deterministic fallback.

    Failure modes and history semantics are the same as
    :func:`classification_agent_llm`.
    """
    req = state.get("request_text", "")
    # Only extract the result; discard the deterministic history entry.
    deterministic_result: ScopingResult = scoping_agent(state)["scope"]
    classification = state["classification"].model_dump()

    # ------------------------------------------------------------------
    # Semantic Memory — fetch authorized data inventory from store
    # ------------------------------------------------------------------
    system_catalog_lines: List[str] = []
    try:
        ep_store = _get_store() if _get_store is not None else None
        if ep_store is not None:
            from dsar_langgraph_agent.triage_graph import _INVENTORY_NAMESPACE
            hits = ep_store.search(_INVENTORY_NAMESPACE)
            for hit in hits:
                v = hit.value or {}
                categories = ", ".join(v.get("data_categories", []))
                system_catalog_lines.append(
                    f"System ID: {v.get('id')}\n"
                    f"Name: {v.get('name')}\n"
                    f"Description: {v.get('description')}\n"
                    f"Data Categories: {categories}\n"
                    f"Retention Period: {v.get('retention_period')}"
                )
    except Exception as retrieval_exc:  # never crash the node due to store issues
        log.warning(
            "[scoping_agent_llm] Inventory retrieval failed, continuing without catalog: %s",
            retrieval_exc,
        )

    system_catalog = "\n\n".join(system_catalog_lines)

    system_prompt = (
        "You are a Data Discovery Agent. You must ONLY select systems from the AUTHORIZED DATA INVENTORY.\n"
        "Return ONLY valid JSON matching this schema:\n"
        '{ "systems": ["..."], "rationale": "..." }\n'
        "Rules:\n"
        "- systems must be a list of short system identifiers (snake_case).\n"
        '- For every system you select, you must cite the specific "data_categories" from the inventory that justify its inclusion.\n'
        "- If no systems match the request categories, return an empty list.\n"
        "- no markdown, no code fences, no extra keys."
    )

    if system_catalog:
        system_prompt += f"\n\n### AUTHORIZED DATA INVENTORY\n{system_catalog}\n"
    user_prompt = (
        "Given the DSAR request text and classification, propose which internal systems to query.\n\n"
        f"Classification:\n{_as_json(classification)}\n\n"
        f"Request text:\n{req}\n"
    )

    def _call() -> ScopingResult:
        data = _client(cfg).chat_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        return ScopingResult.model_validate(data, strict=False)

    def _post_validate(result: ScopingResult) -> ScopingResult:
        seen: set[str] = set()
        normalised: List[str] = []
        for s in result.systems:
            s2 = (s or "").strip()
            if not s2 or s2 in seen:
                continue
            normalised.append(s2)
            seen.add(s2)
        result.systems = normalised
        return result

    def _llm_entry(r: ScopingResult) -> ReasoningEntry:
        num_systems = len(r.systems)
        base_rationale = f"Consulted the Semantic Knowledge Base (Data Inventory) to identify {num_systems} relevant systems.\n"
        return ReasoningEntry(
            agent="scoping_agent_llm",
            source="llm",
            decision_made=f"systems={r.systems}",
            rationale=base_rationale + (r.rationale or "LLM provided no rationale."),
        )

    def _fallback_entry() -> ReasoningEntry:
        return ReasoningEntry(
            agent="scoping_agent_llm",
            source="deterministic",
            decision_made=f"systems={deterministic_result.systems}",
            rationale=deterministic_result.rationale or "Deterministic fallback applied.",
        )

    return _run_with_fallback(
        agent_name="scoping_agent_llm",
        state=state,
        llm_call=_call,
        post_validate=_post_validate,
        fallback_result=deterministic_result,
        state_key="scope",
        llm_entry_factory=_llm_entry,
        fallback_entry_factory=_fallback_entry,
    )


def risk_flag_agent_llm(state: TriageState, *, cfg: LLMNodeConfig) -> TriageState:
    """
    LLM-powered risk flagging with deterministic fallback.

    Failure modes and history semantics are the same as
    :func:`classification_agent_llm`.

    Post-validation enforces that every flag is a known :class:`RiskFlag`
    enum value; unrecognised values raise :class:`pydantic.ValidationError`
    instead of being silently dropped.
    """
    req = state.get("request_text", "")
    # Only extract the result; discard the deterministic history entry.
    deterministic_result: RiskResult = risk_flag_agent(state)["risk"]
    classification = state["classification"].model_dump()
    scope = state["scope"].model_dump()

    system_prompt = (
        "You are a DSAR triage risk flagger.\n"
        "Return ONLY valid JSON matching this schema:\n"
        '{ "flags": ["third_party_data_involved|request_from_minor|conflicting_legal_basis|identity_unclear|ambiguous_scope"],\n'
        '  "notes": ["..."] }\n'
        "No markdown, no code fences, no extra keys.\n"
        "flags must contain only the exact string values listed above."
    )
    user_prompt = (
        "Given the DSAR request text, classification, and scope, flag complicating factors.\n\n"
        f"Classification:\n{_as_json(classification)}\n\n"
        f"Scope:\n{_as_json(scope)}\n\n"
        f"Request text:\n{req}\n"
    )

    def _call() -> RiskResult:
        data = _client(cfg).chat_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        return RiskResult.model_validate(data, strict=False)

    def _post_validate(result: RiskResult) -> RiskResult:
        allowed = {f.value for f in RiskFlag}
        seen: set[str] = set()
        normalised: List[RiskFlag] = []
        unknown: List[str] = []

        for f in result.flags:
            f_val = f.value if isinstance(f, RiskFlag) else str(f)
            if f_val not in allowed:
                unknown.append(f_val)
                continue
            if f_val in seen:
                continue
            normalised.append(RiskFlag(f_val))
            seen.add(f_val)

        if unknown:
            raise ValidationError.from_exception_data(  # type: ignore[attr-defined]
                title="RiskResult",
                input_type="python",
                line_errors=[{
                    "type": "enum",
                    "loc": ("flags",),
                    "msg": f"Unknown RiskFlag value(s) returned by LLM: {unknown!r}",
                    "input": unknown,
                    "ctx": {},
                }],
            )

        result.flags = normalised
        result.notes = [n for n in (result.notes or []) if (n or "").strip()]
        return result

    def _llm_entry(r: RiskResult) -> ReasoningEntry:
        flag_values = [f.value for f in r.flags]
        return ReasoningEntry(
            agent="risk_flag_agent_llm",
            source="llm",
            decision_made=f"flags={flag_values}",
            rationale="; ".join(r.notes) if r.notes else "LLM reported no risk flags.",
        )

    def _fallback_entry() -> ReasoningEntry:
        flag_values = [f.value for f in deterministic_result.flags]
        return ReasoningEntry(
            agent="risk_flag_agent_llm",
            source="deterministic",
            decision_made=f"flags={flag_values}",
            rationale=(
                "; ".join(deterministic_result.notes)
                if deterministic_result.notes
                else "Deterministic fallback: no risk flags detected."
            ),
        )

    return _run_with_fallback(
        agent_name="risk_flag_agent_llm",
        state=state,
        llm_call=_call,
        post_validate=_post_validate,
        fallback_result=deterministic_result,
        state_key="risk",
        llm_entry_factory=_llm_entry,
        fallback_entry_factory=_fallback_entry,
    )
