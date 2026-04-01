"""
Tests for the Reasoning History feature.

Covers:
  - Deterministic path: 3 entries (classify → scope → risk), all source="deterministic",
    with exact keyword-level rationale.
  - LLM success path: source="llm" entry with the LLM's own rationale string.
  - LLM fallback path (two-entry): source="llm_fallback" immediately followed by
    source="deterministic" for the same agent.
  - Human checkpoint approve: final source="human" entry recorded.
  - Human checkpoint revise + second pass: history shows revise entry then the
    re-classification entries, confirming history accumulates across the cycle.
  - Warning accumulation does not corrupt history, and vice-versa.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dsar_langgraph_agent.ollama_client import OllamaLLMError
from dsar_langgraph_agent.triage_agents import (
    classification_agent,
    risk_flag_agent,
    scoping_agent,
)
from dsar_langgraph_agent.triage_graph import build_triage_graph
from dsar_langgraph_agent.triage_llm_agents import (
    LLMNodeConfig,
    classification_agent_llm,
    risk_flag_agent_llm,
    scoping_agent_llm,
)
from dsar_langgraph_agent.triage_schemas import DSARRequestType, HumanReviewDecision, RiskFlag

_CFG = LLMNodeConfig()
_PATCH = "dsar_langgraph_agent.triage_llm_agents.OllamaLLMClient.chat_json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deterministic_state(text: str) -> dict:
    """Build a fully populated TriageState via the three deterministic agents."""
    s: dict = {"request_text": text}
    s = {**s, **classification_agent(s)}  # type: ignore[arg-type]
    s = {**s, **scoping_agent(s)}   # type: ignore[arg-type]
    s = {**s, **risk_flag_agent(s)} # type: ignore[arg-type]
    return s


# ---------------------------------------------------------------------------
# 1. Deterministic path
# ---------------------------------------------------------------------------

class TestDeterministicHistory:

    def test_three_entries_in_order(self):
        state = _deterministic_state("Please delete all my data.")
        history = state.get("reasoning_history", [])
        assert len(history) == 3
        assert history[0].agent == "classification_agent"
        assert history[1].agent == "scoping_agent"
        assert history[2].agent == "risk_flag_agent"

    def test_all_entries_source_deterministic(self):
        state = _deterministic_state("Please delete all my data.")
        for entry in state["reasoning_history"]:
            assert entry.source == "deterministic"

    def test_classification_rationale_contains_matched_keyword(self):
        state = _deterministic_state("Please delete all my personal data.")
        h = state["reasoning_history"][0]
        assert h.decision_made.startswith("request_type=deletion")
        assert '"delete"' in h.rationale
        assert "deletion rule" in h.rationale

    def test_portability_keyword_reported(self):
        state = _deterministic_state("I want to download my data in machine readable format.")
        h = state["reasoning_history"][0]
        assert "portability" in h.decision_made
        # first match is "download my data"
        assert '"download my data"' in h.rationale or '"machine readable"' in h.rationale

    def test_access_keyword_reported(self):
        state = _deterministic_state("Please send me a copy of my data.")
        h = state["reasoning_history"][0]
        assert "access" in h.decision_made
        assert "matched access rule" in h.rationale

    def test_unknown_has_no_keyword_match_note(self):
        state = _deterministic_state("Hello, I have a question.")
        h = state["reasoning_history"][0]
        assert "unknown" in h.decision_made
        assert "No DSAR keyword matched" in h.rationale

    def test_scoping_decision_made_contains_systems(self):
        state = _deterministic_state("Please delete all my data.")
        h = state["reasoning_history"][1]
        assert "identity_store" in h.decision_made

    def test_scoping_rationale_mentions_rule(self):
        state = _deterministic_state("Please delete all my support chat history.")
        h = state["reasoning_history"][1]
        assert "standard identity+product rule" in h.rationale
        # support keyword should also appear
        assert "comms systems included" in h.rationale

    def test_risk_decision_made_contains_flags(self):
        state = _deterministic_state(
            "Please delete my child's data and all information you hold."
        )
        h = state["reasoning_history"][2]
        assert "request_from_minor" in h.decision_made

    def test_risk_rationale_contains_triggering_keyword(self):
        state = _deterministic_state("My child wants their data removed.")
        h = state["reasoning_history"][2]
        assert '"my child"' in h.rationale or '"minor"' in h.rationale
        assert "request_from_minor" in h.rationale

    def test_risk_no_flags_rationale(self):
        state = _deterministic_state("I just want a copy of my data.")
        h = state["reasoning_history"][2]
        assert "No risk keywords matched" in h.rationale
        assert "[]" in h.decision_made

    def test_timestamps_are_set(self):
        state = _deterministic_state("Please delete my account.")
        for entry in state["reasoning_history"]:
            assert entry.timestamp  # non-empty ISO string


# ---------------------------------------------------------------------------
# 2. LLM success path
# ---------------------------------------------------------------------------

class TestLLMSuccessHistory:

    def test_classification_llm_success_adds_llm_entry(self):
        state: dict = {"request_text": "Delete my account."}
        payload = {"request_type": "deletion", "confidence": 0.97, "rationale": "Clear deletion intent."}
        with patch(_PATCH, return_value=payload):
            result = classification_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

        history = result.get("reasoning_history", [])
        assert len(history) == 1
        entry = history[0]
        assert entry.agent == "classification_agent_llm"
        assert entry.source == "llm"
        assert entry.decision_made == "request_type=deletion, confidence=0.97"
        assert entry.rationale == "Clear deletion intent."

    def test_scoping_llm_success_adds_llm_entry(self):
        base: dict = {"request_text": "Delete my account."}
        base = {**base, **classification_agent(base)}  # type: ignore[arg-type]
        # Clean history from the deterministic classify step
        base["reasoning_history"] = []
        payload = {"systems": ["crm", "app_db"], "rationale": "Standard deletion scope."}
        with patch(_PATCH, return_value=payload):
            result = scoping_agent_llm(base, cfg=_CFG)  # type: ignore[arg-type]

        history = result.get("reasoning_history", [])
        assert len(history) == 1
        entry = history[0]
        assert entry.agent == "scoping_agent_llm"
        assert entry.source == "llm"
        assert "crm" in entry.decision_made
        assert entry.rationale == "Standard deletion scope."

    def test_risk_llm_success_adds_llm_entry(self):
        state = _deterministic_state("My child's data please.")
        state["reasoning_history"] = []  # isolate
        payload = {"flags": ["request_from_minor"], "notes": ["Minor detected."]}
        with patch(_PATCH, return_value=payload):
            result = risk_flag_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

        history = result.get("reasoning_history", [])
        assert len(history) == 1
        entry = history[0]
        assert entry.agent == "risk_flag_agent_llm"
        assert entry.source == "llm"
        assert "request_from_minor" in entry.decision_made
        assert "Minor detected." in entry.rationale

    def test_llm_success_no_warnings(self):
        state: dict = {"request_text": "Delete my account."}
        payload = {"request_type": "deletion", "confidence": 0.9, "rationale": "Deletion."}
        with patch(_PATCH, return_value=payload):
            result = classification_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]
        assert result.get("llm_warnings", []) == []


# ---------------------------------------------------------------------------
# 3. LLM fallback path — two-entry representation
# ---------------------------------------------------------------------------

class TestLLMFallbackHistory:

    def _timeout_exc(self):
        return OllamaLLMError("LLM call failed: timed out after 30s")

    def _bad_json_exc(self):
        return OllamaLLMError("Could not locate JSON object in model response")

    # --- classification ---

    def test_timeout_produces_two_history_entries(self):
        state: dict = {"request_text": "Delete my account."}
        with patch(_PATCH, side_effect=self._timeout_exc()):
            result = classification_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

        history = result.get("reasoning_history", [])
        assert len(history) == 2
        assert history[0].source == "llm_fallback"
        assert history[1].source == "deterministic"

    def test_fallback_entry_agents_match(self):
        state: dict = {"request_text": "Delete my account."}
        with patch(_PATCH, side_effect=self._timeout_exc()):
            result = classification_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

        history = result["reasoning_history"]
        assert history[0].agent == "classification_agent_llm"
        assert history[1].agent == "classification_agent_llm"

    def test_llm_fallback_entry_records_failure_cause(self):
        state: dict = {"request_text": "Delete my account."}
        with patch(_PATCH, side_effect=self._timeout_exc()):
            result = classification_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

        failed = result["reasoning_history"][0]
        assert failed.decision_made == "llm_failed"
        assert "[timeout]" in failed.rationale
        assert "timed out" in failed.rationale.lower()

    def test_deterministic_fallback_entry_reflects_correct_decision(self):
        state: dict = {"request_text": "Delete my account."}
        with patch(_PATCH, side_effect=self._timeout_exc()):
            result = classification_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

        det_entry = result["reasoning_history"][1]
        assert "deletion" in det_entry.decision_made
        assert det_entry.rationale  # non-empty

    def test_bad_json_fallback_cause_in_rationale(self):
        state: dict = {"request_text": "Delete my account."}
        with patch(_PATCH, side_effect=self._bad_json_exc()):
            result = classification_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

        failed = result["reasoning_history"][0]
        assert "[bad_json]" in failed.rationale

    def test_schema_error_fallback(self):
        state: dict = {"request_text": "Delete my account."}
        bad_payload = {"confidence": 0.9, "rationale": "oops"}  # missing request_type
        with patch(_PATCH, return_value=bad_payload):
            result = classification_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

        history = result["reasoning_history"]
        assert len(history) == 2
        assert history[0].source == "llm_fallback"
        assert "[schema_error]" in history[0].rationale
        assert history[1].source == "deterministic"

    # --- scoping ---

    def test_scoping_fallback_two_entries(self):
        state = _deterministic_state("Delete my account.")
        state["reasoning_history"] = []
        with patch(_PATCH, side_effect=self._timeout_exc()):
            result = scoping_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

        history = result["reasoning_history"]
        assert len(history) == 2
        assert history[0].source == "llm_fallback"
        assert history[0].agent == "scoping_agent_llm"
        assert history[1].source == "deterministic"
        assert history[1].agent == "scoping_agent_llm"
        assert "identity_store" in history[1].decision_made

    # --- risk ---

    def test_risk_fallback_two_entries(self):
        state = _deterministic_state("My child's data, delete everything.")
        state["reasoning_history"] = []
        with patch(_PATCH, side_effect=self._timeout_exc()):
            result = risk_flag_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

        history = result["reasoning_history"]
        assert len(history) == 2
        assert history[0].source == "llm_fallback"
        assert history[1].source == "deterministic"
        assert "request_from_minor" in history[1].decision_made

    # --- accumulation across agents ---

    def test_history_accumulates_across_two_fallbacks(self):
        """Both classify and scope fail → history has 4 entries total (2+2)."""
        state: dict = {"request_text": "Delete my account."}
        with patch(_PATCH, side_effect=self._timeout_exc()):
            state = classification_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]
        with patch(_PATCH, side_effect=self._bad_json_exc()):
            state = scoping_agent_llm(state, cfg=_CFG)          # type: ignore[arg-type]

        history = state["reasoning_history"]
        assert len(history) == 4
        assert history[0].source == "llm_fallback"
        assert history[0].agent == "classification_agent_llm"
        assert history[1].source == "deterministic"
        assert history[1].agent == "classification_agent_llm"
        assert history[2].source == "llm_fallback"
        assert history[2].agent == "scoping_agent_llm"
        assert history[3].source == "deterministic"
        assert history[3].agent == "scoping_agent_llm"

    def test_fallback_history_and_llm_warnings_both_populated(self):
        state: dict = {"request_text": "Delete my account."}
        with patch(_PATCH, side_effect=self._timeout_exc()):
            result = classification_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

        assert len(result["reasoning_history"]) == 2
        assert len(result["llm_warnings"]) == 1
        assert result["llm_warnings"][0].cause == "timeout"


# ---------------------------------------------------------------------------
# 4. Human checkpoint history (integration via build_triage_graph)
# ---------------------------------------------------------------------------

class TestHumanCheckpointHistory:

    def _run_to_checkpoint(self, text: str, thread_id: str):
        app = build_triage_graph()
        config = {"configurable": {"thread_id": thread_id}}
        app.invoke({"request_text": text}, config=config)
        return app, config

    def test_approve_adds_human_entry(self):
        from langgraph.types import Command

        app, config = self._run_to_checkpoint(
            "Please delete my account.", "history_approve_1"
        )
        decision = HumanReviewDecision(approved=True, reviewer="tester", notes="Looks good.")
        resumed = app.invoke(Command(resume=decision.model_dump()), config=config)

        history = resumed["output"].reasoning_history
        human_entries = [e for e in history if e.source == "human"]
        assert len(human_entries) == 1
        h = human_entries[0]
        assert h.agent == "human_checkpoint"
        assert "approved" in h.decision_made
        assert "tester" in h.decision_made
        assert "Looks good." in h.rationale

    def test_approve_history_has_classify_scope_risk_then_human(self):
        from langgraph.types import Command

        app, config = self._run_to_checkpoint(
            "Please delete my account.", "history_approve_2"
        )
        decision = HumanReviewDecision(approved=True, reviewer="tester", notes="ok")
        resumed = app.invoke(Command(resume=decision.model_dump()), config=config)

        history = resumed["output"].reasoning_history
        sources = [e.source for e in history]
        agents = [e.agent for e in history]

        # First three are deterministic agent entries
        assert agents[0] == "classification_agent"
        assert agents[1] == "scoping_agent"
        assert agents[2] == "risk_flag_agent"
        # Last entry is the human checkpoint
        assert agents[-1] == "human_checkpoint"
        assert sources[-1] == "human"

    def test_revise_adds_human_revise_entry_then_reclassification(self):
        from langgraph.types import Command

        app, config = self._run_to_checkpoint(
            "I want all my data.", "history_revise_1"
        )

        # Revise — provide feedback that changes classification
        revise = HumanReviewDecision(
            approved=False,
            reviewer="tester",
            human_feedback="This should be portability, not access.",
        )
        # This will pause again at the second human_checkpoint
        app.invoke(Command(resume=revise.model_dump()), config=config)

        snap = app.get_state(config)
        history = snap.values.get("reasoning_history", [])

        # Must contain a "revise" human entry
        revise_entries = [e for e in history if e.source == "human" and "revise" in e.decision_made]
        assert len(revise_entries) == 1
        assert "portability" in revise_entries[0].rationale

        # Must contain a second classification entry (the corrected one)
        classify_entries = [e for e in history if e.agent == "classification_agent"]
        assert len(classify_entries) == 2  # original + revised pass

        # The second classification decision should reflect the correction
        second_classify = classify_entries[1]
        assert "portability" in second_classify.decision_made

    def test_full_output_reasoning_history_forwarded(self):
        """TriageOutput.reasoning_history is populated (build_output passes it through)."""
        from langgraph.types import Command

        app, config = self._run_to_checkpoint(
            "Please delete my support chat history.", "history_forward_1"
        )
        decision = HumanReviewDecision(approved=True, reviewer="qa")
        resumed = app.invoke(Command(resume=decision.model_dump()), config=config)

        output = resumed["output"]
        assert len(output.reasoning_history) >= 4  # classify + scope + risk + human
        # Scoping should mention support keyword
        scope_entry = next(e for e in output.reasoning_history if e.agent == "scoping_agent")
        assert "comms systems included" in scope_entry.rationale
