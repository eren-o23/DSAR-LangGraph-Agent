from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List

from dsar_langgraph_agent.ollama_client import OllamaLLMClient, OllamaLLMConfig, OllamaLLMError
from dsar_langgraph_agent.triage_schemas import (
    ClassificationResult,
    DSARRequestType,
    RiskFlag,
    RiskResult,
    ScopingResult,
)
from dsar_langgraph_agent.triage_agents import classification_agent, risk_flag_agent, scoping_agent
from dsar_langgraph_agent.triage_state import TriageState


@dataclass(frozen=True)
class LLMNodeConfig:
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "llama3.1"
    timeout_s: float = 30.0


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


def classification_agent_llm(state: TriageState, *, cfg: LLMNodeConfig) -> TriageState:
    req = state.get("request_text", "")
    deterministic = classification_agent(state)["classification"]

    system_prompt = (
        "You are a DSAR triage classifier.\n"
        "Return ONLY valid JSON matching this schema:\n"
        '{ "request_type": "access|deletion|portability|unknown", "confidence": 0.0-1.0, "rationale": "..." }\n'
        "No markdown, no code fences, no extra keys."
    )
    user_prompt = f"DSAR request text:\n{req}\n"

    try:
        data = _client(cfg).chat_json(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0)
        result = ClassificationResult.model_validate(data)
        if result.request_type not in (
            DSARRequestType.access,
            DSARRequestType.deletion,
            DSARRequestType.portability,
            DSARRequestType.unknown,
        ):
            raise ValueError("Invalid request_type")
        return {**state, "classification": result}
    except Exception as e:
        fallback = deterministic.model_copy()
        fallback.confidence = min(fallback.confidence, 0.4)
        fallback.rationale = (fallback.rationale + " " if fallback.rationale else "") + f"(LLM unavailable/invalid: {e})"
        return {**state, "classification": fallback}


def scoping_agent_llm(state: TriageState, *, cfg: LLMNodeConfig) -> TriageState:
    req = state.get("request_text", "")
    deterministic = scoping_agent(state)["scope"]

    classification = state["classification"].model_dump()

    system_prompt = (
        "You are a DSAR triage scoping agent.\n"
        "Return ONLY valid JSON matching this schema:\n"
        '{ "systems": ["..."], "rationale": "..." }\n'
        "Rules:\n"
        "- systems must be a list of short system identifiers (snake_case).\n"
        "- no markdown, no code fences, no extra keys."
    )
    user_prompt = (
        "Given the DSAR request text and classification, propose which internal systems to query.\n\n"
        f"Classification:\n{_as_json(classification)}\n\n"
        f"Request text:\n{req}\n"
    )

    try:
        data = _client(cfg).chat_json(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0)
        result = ScopingResult.model_validate(data)
        # Normalize: keep stable order, remove empties/dupes
        seen = set()
        normalized: List[str] = []
        for s in result.systems:
            s2 = (s or "").strip()
            if not s2 or s2 in seen:
                continue
            normalized.append(s2)
            seen.add(s2)
        result.systems = normalized
        return {**state, "scope": result}
    except Exception as e:
        fallback = deterministic.model_copy()
        fallback.rationale = (fallback.rationale + " " if fallback.rationale else "") + f"(LLM unavailable/invalid: {e})"
        return {**state, "scope": fallback}


def risk_flag_agent_llm(state: TriageState, *, cfg: LLMNodeConfig) -> TriageState:
    req = state.get("request_text", "")
    deterministic = risk_flag_agent(state)["risk"]

    classification = state["classification"].model_dump()
    scope = state["scope"].model_dump()

    system_prompt = (
        "You are a DSAR triage risk flagger.\n"
        "Return ONLY valid JSON matching this schema:\n"
        '{ "flags": ["third_party_data_involved|request_from_minor|conflicting_legal_basis|identity_unclear|ambiguous_scope"],\n'
        '  "notes": ["..."] }\n'
        "No markdown, no code fences, no extra keys."
    )
    user_prompt = (
        "Given the DSAR request text, classification, and scope, flag complicating factors.\n\n"
        f"Classification:\n{_as_json(classification)}\n\n"
        f"Scope:\n{_as_json(scope)}\n\n"
        f"Request text:\n{req}\n"
    )

    try:
        data = _client(cfg).chat_json(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0)
        result = RiskResult.model_validate(data)

        # Validate/normalize flags to known enum values; drop unknowns instead of failing.
        allowed = {f.value for f in RiskFlag}
        normalized_flags: List[RiskFlag] = []
        seen = set()
        for f in result.flags:
            f_val = f.value if isinstance(f, RiskFlag) else str(f)
            if f_val not in allowed or f_val in seen:
                continue
            normalized_flags.append(RiskFlag(f_val))
            seen.add(f_val)

        result.flags = normalized_flags
        result.notes = [n for n in (result.notes or []) if (n or "").strip()]
        return {**state, "risk": result}
    except Exception as e:
        fallback = deterministic.model_copy()
        fallback.notes = list(fallback.notes) + [f"(LLM unavailable/invalid: {e})"]
        return {**state, "risk": fallback}

