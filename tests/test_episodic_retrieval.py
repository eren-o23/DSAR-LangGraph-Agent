"""
Tests for episodic retrieval in classification_agent_llm.

Covers:
  - Empty store: no crash, matched_precedents defaults to [] and prompt is unchanged.
  - Non-empty store: search() is called with query=request_text and limit=2.
  - Precedents are injected into the user_prompt as few-shot examples.
  - Retrieval cap: only up to 2 results are included even if the store has more.
  - Bad store / get_store() failure: node continues without crashing.
  - Stored value missing expected fields: handled gracefully without KeyError.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, call, patch

import pytest
from langgraph.store.memory import InMemoryStore

from dsar_langgraph_agent.triage_graph import _EPISODE_NAMESPACE
from dsar_langgraph_agent.triage_llm_agents import LLMNodeConfig, classification_agent_llm

_CFG  = LLMNodeConfig()
_PATCH_LLM   = "dsar_langgraph_agent.triage_llm_agents.OllamaLLMClient.chat_json"
# We patch _get_store at the name it's bound to *inside* the module under test.
_PATCH_STORE = "dsar_langgraph_agent.triage_llm_agents._get_store"

_GOOD_RESPONSE = {
    "request_type": "deletion",
    "confidence":   0.95,
    "rationale":    "Clear deletion intent.",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(text: str = "Delete my account.") -> dict:
    return {"request_text": text}


def _make_ep_store(*records: tuple[str, str]) -> InMemoryStore:
    """Populate a fresh InMemoryStore with (request_text, request_type) pairs."""
    s = InMemoryStore()
    for i, (text, req_type) in enumerate(records):
        s.put(
            _EPISODE_NAMESPACE,
            f"ep-{i}",
            {
                "episode_id":        f"ep-{i}",
                "request_text":      text,
                "classification":    {"request_type": req_type, "confidence": 0.9},
                "scope":             {"systems": []},
                "risk":              {"flags": []},
                "human_review":      {"approved": True, "reviewer": "tester"},
                "reasoning_history": [],
                "llm_warnings":      [],
            },
        )
    return s


def _run(state: dict, ep_store) -> tuple[dict, List[str]]:
    """
    Run classification_agent_llm with a fake store and capture the user_prompt.

    ``ep_store`` may be:
    - an InMemoryStore  → used as-is
    - None              → _get_store returns None
    - an Exception type → _get_store raises that exception
    - a MagicMock       → used as-is (for call-counting)
    """
    captured: List[str] = []

    def fake_chat_json(*, system_prompt, user_prompt, temperature=0.0):
        captured.append(user_prompt)
        return _GOOD_RESPONSE

    def fake_get_store():
        if isinstance(ep_store, type) and issubclass(ep_store, Exception):
            raise ep_store("store error")
        return ep_store

    with patch(_PATCH_LLM, side_effect=fake_chat_json):
        with patch(_PATCH_STORE, fake_get_store):
            result = classification_agent_llm(state, cfg=_CFG)  # type: ignore[arg-type]

    return result, captured


# ---------------------------------------------------------------------------
# 1. Empty store — no crash, no precedents section
# ---------------------------------------------------------------------------

class TestEmptyStore:

    def test_no_crash_when_store_empty(self):
        result, _ = _run(_state(), _make_ep_store())
        assert result["classification"].request_type.value == "deletion"

    def test_no_precedents_section_when_store_empty(self):
        _, prompts = _run(_state(), _make_ep_store())
        assert "Relevant precedents" not in prompts[0]

    def test_no_crash_when_get_store_returns_none(self):
        result, _ = _run(_state(), None)
        assert result["classification"] is not None

    def test_no_crash_when_get_store_raises(self):
        result, _ = _run(_state(), RuntimeError)
        assert result["classification"] is not None


# ---------------------------------------------------------------------------
# 2. Store with episodes — search called with correct args + prompt injected
# ---------------------------------------------------------------------------

class TestStoreWithEpisodes:

    def test_search_called_with_query_and_limit(self):
        """search() must receive query=<request_text> and limit=2."""
        ep_store = _make_ep_store(
            ("I want to be forgotten.", "deletion"),
            ("Send me my data.",        "access"),
        )
        # Wrap the real search in a MagicMock so we can assert on call args,
        # without replacing the method attribute (which is read-only on the slot).
        search_mock = MagicMock(wraps=ep_store.search)

        class WrappedStore:
            """Thin wrapper that delegates to ep_store but lets us spy on search."""
            def search(self, *args, **kwargs):
                return search_mock(*args, **kwargs)

        result, _ = _run(_state("Delete everything."), WrappedStore())
        search_mock.assert_called_once()
        pos_args, kw_args = search_mock.call_args
        assert pos_args[0] == _EPISODE_NAMESPACE
        assert kw_args.get("query") == "Delete everything."
        assert kw_args.get("limit") == 2

    def test_precedents_section_injected_into_prompt(self):
        ep_store = _make_ep_store(("Please erase my account.", "deletion"))
        _, prompts = _run(_state("Delete my account."), ep_store)
        assert "Relevant precedents from approved past decisions" in prompts[0]
        assert "Please erase my account." in prompts[0]
        assert "deletion" in prompts[0]

    def test_both_request_text_and_final_decision_present(self):
        ep_store = _make_ep_store(
            ("I want portability.", "portability"),
            ("Send me a copy.",     "access"),
        )
        _, prompts = _run(_state("Give me all my data."), ep_store)
        prompt = prompts[0]
        # At least one of the two stored episodes should appear
        has_ep1 = "I want portability." in prompt and "portability" in prompt
        has_ep2 = "Send me a copy." in prompt and "access" in prompt
        assert has_ep1 or has_ep2

    def test_at_most_two_precedents_in_prompt(self):
        """The limit=2 cap means at most 2 bullet points end up in the prompt."""
        ep_store = _make_ep_store(
            ("Delete account.",   "deletion"),
            ("Remove my data.",   "deletion"),
            ("Erase everything.", "deletion"),
            ("Forget me.",        "deletion"),
            ("Wipe my record.",   "deletion"),
        )
        _, prompts = _run(_state("Delete everything."), ep_store)
        count = prompts[0].count("→ Decision:")
        assert count <= 2, f"Expected ≤ 2 precedents in prompt, got {count}"

    def test_no_precedents_section_when_store_empty(self):
        _, prompts = _run(_state(), _make_ep_store())
        assert "Relevant precedents" not in prompts[0]


# ---------------------------------------------------------------------------
# 3. Resilience — store errors never crash the node
# ---------------------------------------------------------------------------

class TestStoreErrorResilience:

    def test_search_exception_caught_node_continues(self):
        """If search() raises, the node must still return a valid classification."""
        broken = MagicMock()
        broken.search.side_effect = ConnectionError("store down")
        result, _ = _run(_state(), broken)
        assert result["classification"] is not None
        assert result["classification"].request_type.value == "deletion"

    def test_missing_classification_field_in_stored_value_no_key_error(self):
        """Stored episode without 'classification' key → final_decision defaults to 'unknown'."""
        s = InMemoryStore()
        s.put(_EPISODE_NAMESPACE, "bad-ep", {"request_text": "something"})  # no classification

        _, prompts = _run(_state(), s)
        # Either the precedent was skipped entirely, or it was included with 'unknown'
        prompt = prompts[0]
        if "Relevant precedents" in prompt:
            assert "unknown" in prompt

    def test_none_value_in_stored_item_no_error(self):
        """Store returning a SearchItem with value=None must not crash."""
        fake_item = MagicMock()
        fake_item.value = None
        broken = MagicMock()
        broken.search.return_value = [fake_item]
        result, _ = _run(_state(), broken)
        assert result["classification"] is not None
