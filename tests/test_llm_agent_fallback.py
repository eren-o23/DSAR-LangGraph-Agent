"""
Tests for LLM agent fallback logic.

These tests run without a live Ollama instance by monkey-patching the client's
chat_json method to simulate the three main failure modes:
  - OllamaLLMError wrapping a timeout / connection failure
  - OllamaLLMError wrapping a bad-JSON / unparseable response
  - Valid JSON that fails Pydantic schema validation

After each failure the deterministic agent output must be present in state and
a structured LLMFallbackWarning must be appended to state["llm_warnings"].
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from dsar_langgraph_agent.ollama_client import OllamaLLMError
from dsar_langgraph_agent.triage_llm_agents import (
    LLMNodeConfig,
    classification_agent_llm,
    risk_flag_agent_llm,
    scoping_agent_llm,
)
from dsar_langgraph_agent.triage_schemas import DSARRequestType, LLMFallbackWarning
from dsar_langgraph_agent.triage_agents import classification_agent, scoping_agent, risk_flag_agent

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_CFG = LLMNodeConfig()

_CLASSIFY_TEXT = "Please delete my account and all my personal data."
_SCOPE_TEXT = _CLASSIFY_TEXT
_RISK_TEXT = "My child wants to access their account. Delete everything."


def _base_state(text: str) -> dict:
    """Return a minimal TriageState-compatible dict seeded through deterministic agents."""
    s: dict = {"request_text": text}
    s = {**s, **classification_agent(s)}  # type: ignore[arg-type]
    s = {**s, **scoping_agent(s)}  # type: ignore[arg-type]
    s = {**s, **risk_flag_agent(s)}  # type: ignore[arg-type]
    return s


# ---------------------------------------------------------------------------
# classification_agent_llm
# ---------------------------------------------------------------------------

class TestClassificationAgentLLMFallback:

    def _patch_path(self):
        return "dsar_langgraph_agent.triage_llm_agents.OllamaLLMClient.chat_json"

    def test_timeout_triggers_fallback(self):
        state = _base_state(_CLASSIFY_TEXT)
        exc = OllamaLLMError("LLM call failed: timed out after 30s")
        with patch(self._patch_path(), side_effect=exc):
            result = classification_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

        assert result["classification"].request_type == DSARRequestType.deletion
        warnings = result.get("llm_warnings", [])
        assert len(warnings) == 1
        w: LLMFallbackWarning = warnings[0]
        assert w.agent == "classification_agent_llm"
        assert w.cause == "timeout"
        assert "timed out" in w.detail.lower()

    def test_bad_json_triggers_fallback(self):
        state = _base_state(_CLASSIFY_TEXT)
        exc = OllamaLLMError("Failed to parse JSON from model response")
        with patch(self._patch_path(), side_effect=exc):
            result = classification_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

        warnings = result.get("llm_warnings", [])
        assert len(warnings) == 1
        assert warnings[0].cause == "bad_json"

    def test_schema_error_triggers_fallback(self):
        """LLM returns valid JSON but wrong schema (e.g., missing required field)."""
        state = _base_state(_CLASSIFY_TEXT)
        bad_payload = {"confidence": 0.9, "rationale": "ok"}  # missing request_type
        with patch(self._patch_path(), return_value=bad_payload):
            result = classification_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

        warnings = result.get("llm_warnings", [])
        assert len(warnings) == 1
        assert warnings[0].cause == "schema_error"
        # Fallback result must still be the deterministic answer
        assert result["classification"].request_type == DSARRequestType.deletion

    def test_invalid_request_type_enum_triggers_fallback(self):
        """LLM returns a request_type value not in the allowed enum."""
        state = _base_state(_CLASSIFY_TEXT)
        bad_payload = {"request_type": "refund", "confidence": 0.9, "rationale": "ok"}
        with patch(self._patch_path(), return_value=bad_payload):
            result = classification_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

        warnings = result.get("llm_warnings", [])
        assert len(warnings) == 1
        # Pydantic enum coercion will fail → schema_error
        assert warnings[0].cause == "schema_error"

    def test_success_no_warnings(self):
        state = _base_state(_CLASSIFY_TEXT)
        good_payload = {"request_type": "deletion", "confidence": 0.95, "rationale": "LLM says so"}
        with patch(self._patch_path(), return_value=good_payload):
            result = classification_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

        assert result.get("llm_warnings", []) == []
        assert result["classification"].request_type == DSARRequestType.deletion
        assert result["classification"].confidence == 0.95
        # human_feedback must always be cleared
        assert result.get("human_feedback") is None

    def test_warnings_accumulate_across_agents(self):
        """Simulate classification failing then scoping failing; both warnings kept."""
        state = _base_state(_CLASSIFY_TEXT)
        timeout_exc = OllamaLLMError("LLM call failed: connection refused")
        bad_json_exc = OllamaLLMError("Could not locate JSON object in model response")

        with patch(self._patch_path(), side_effect=timeout_exc):
            state = classification_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

        with patch(self._patch_path(), side_effect=bad_json_exc):
            state = scoping_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

        warnings = state.get("llm_warnings", [])
        assert len(warnings) == 2
        assert warnings[0].agent == "classification_agent_llm"
        assert warnings[0].cause == "timeout"
        assert warnings[1].agent == "scoping_agent_llm"
        assert warnings[1].cause == "bad_json"


# ---------------------------------------------------------------------------
# scoping_agent_llm
# ---------------------------------------------------------------------------

class TestScopingAgentLLMFallback:

    def _patch_path(self):
        return "dsar_langgraph_agent.triage_llm_agents.OllamaLLMClient.chat_json"

    def test_timeout_triggers_fallback(self):
        state = _base_state(_SCOPE_TEXT)
        exc = OllamaLLMError("LLM call failed: timed out")
        with patch(self._patch_path(), side_effect=exc):
            result = scoping_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

        warnings = result.get("llm_warnings", [])
        assert len(warnings) == 1
        assert warnings[0].agent == "scoping_agent_llm"
        assert warnings[0].cause == "timeout"
        # Fallback scope must include standard systems
        assert "identity_store" in result["scope"].systems

    def test_schema_error_triggers_fallback(self):
        state = _base_state(_SCOPE_TEXT)
        bad_payload = {"systems": 42}  # wrong type
        with patch(self._patch_path(), return_value=bad_payload):
            result = scoping_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

        warnings = result.get("llm_warnings", [])
        assert len(warnings) == 1
        assert warnings[0].cause == "schema_error"

    def test_success_no_warnings(self):
        state = _base_state(_SCOPE_TEXT)
        good_payload = {"systems": ["crm", "app_db"], "rationale": "Standard deletion scope."}
        with patch(self._patch_path(), return_value=good_payload):
            result = scoping_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

        assert result.get("llm_warnings", []) == []
        assert result["scope"].systems == ["crm", "app_db"]


# ---------------------------------------------------------------------------
# risk_flag_agent_llm
# ---------------------------------------------------------------------------

class TestRiskFlagAgentLLMFallback:

    def _patch_path(self):
        return "dsar_langgraph_agent.triage_llm_agents.OllamaLLMClient.chat_json"

    def test_timeout_triggers_fallback(self):
        state = _base_state(_RISK_TEXT)
        exc = OllamaLLMError("LLM call failed: timed out")
        with patch(self._patch_path(), side_effect=exc):
            result = risk_flag_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

        warnings = result.get("llm_warnings", [])
        assert len(warnings) == 1
        assert warnings[0].agent == "risk_flag_agent_llm"
        assert warnings[0].cause == "timeout"

    def test_unknown_flag_triggers_schema_error_fallback(self):
        """Unknown flag value from LLM must raise a ValidationError → schema_error."""
        state = _base_state(_RISK_TEXT)
        bad_payload = {
            "flags": ["request_from_minor", "made_up_flag"],
            "notes": ["Some note"],
        }
        with patch(self._patch_path(), return_value=bad_payload):
            result = risk_flag_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

        warnings = result.get("llm_warnings", [])
        assert len(warnings) == 1
        assert warnings[0].cause == "schema_error"
        assert "made_up_flag" in warnings[0].detail

    def test_success_no_warnings(self):
        state = _base_state(_RISK_TEXT)
        good_payload = {
            "flags": ["request_from_minor"],
            "notes": ["Minor detected."],
        }
        with patch(self._patch_path(), return_value=good_payload):
            result = risk_flag_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

        assert result.get("llm_warnings", []) == []
        from dsar_langgraph_agent.triage_schemas import RiskFlag
        assert RiskFlag.request_from_minor in result["risk"].flags

    def test_duplicate_flags_deduplicated_without_fallback(self):
        """Duplicate flags are silently removed — no fallback triggered."""
        state = _base_state(_RISK_TEXT)
        payload = {
            "flags": ["ambiguous_scope", "ambiguous_scope"],
            "notes": [],
        }
        with patch(self._patch_path(), return_value=payload):
            result = risk_flag_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

        assert result.get("llm_warnings", []) == []
        from dsar_langgraph_agent.triage_schemas import RiskFlag
        assert result["risk"].flags.count(RiskFlag.ambiguous_scope) == 1
