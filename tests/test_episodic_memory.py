"""
Tests for Episodic Memory via LangGraph Store.

Each test class creates a fresh InMemoryStore and passes it via the
``episode_store`` parameter of ``build_triage_graph`` to avoid cross-test
pollution from the module-level singleton.

Covers:
  - An approved triage is persisted to the ('precedents', 'triage') namespace.
  - The episode_id is surfaced on the TriageOutput returned to the caller.
  - The stored value contains the expected fields (request_text, classification,
    scope, risk, human_review, reasoning_history, llm_warnings).
  - A rejected / revise decision does NOT create an episode entry.
  - Multiple approved runs create distinct episodes (different UUIDs).
"""

from __future__ import annotations

import re

import pytest
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from dsar_langgraph_agent.triage_graph import build_triage_graph, _EPISODE_NAMESPACE
from dsar_langgraph_agent.triage_schemas import HumanReviewDecision


# ---------------------------------------------------------------------------
# Fixtures / constants
# ---------------------------------------------------------------------------

_APPROVE = HumanReviewDecision(approved=True, reviewer="tester", notes="ok")
_REVISE  = HumanReviewDecision(
    approved=False,
    reviewer="tester",
    human_feedback="This is portability, not deletion.",
)
_TEXT = "Please delete all my personal data from your systems."
_UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def _fresh_store() -> InMemoryStore:
    """Return a clean, isolated store for one test class."""
    return InMemoryStore()


def _run_approved(text: str, thread_id: str, ep_store: InMemoryStore) -> dict:
    """Run graph → pause → approve → return final state dict."""
    app    = build_triage_graph(episode_store=ep_store)
    config = {"configurable": {"thread_id": thread_id}}
    app.invoke({"request_text": text}, config=config)
    return app.invoke(Command(resume=_APPROVE.model_dump()), config=config)


# ---------------------------------------------------------------------------
# 1. Episode is persisted on approval
# ---------------------------------------------------------------------------

class TestEpisodePersisted:

    def setup_method(self):
        self.ep_store = _fresh_store()

    def test_episode_stored_after_approval(self):
        _run_approved(_TEXT, "ep_1a", self.ep_store)
        assert len(self.ep_store.search(_EPISODE_NAMESPACE)) == 1

    def test_episode_id_on_output(self):
        result = _run_approved(_TEXT, "ep_1b", self.ep_store)
        eid = result["output"].episode_id
        assert eid is not None
        assert _UUID4_RE.match(eid), f"episode_id {eid!r} is not a valid UUID4"

    def test_episode_id_in_state(self):
        result = _run_approved(_TEXT, "ep_1c", self.ep_store)
        assert result.get("episode_id") is not None

    def test_stored_value_has_required_fields(self):
        _run_approved(_TEXT, "ep_1d", self.ep_store)
        item = self.ep_store.search(_EPISODE_NAMESPACE)[0]
        for field in (
            "episode_id", "request_text", "classification", "scope",
            "risk", "human_review", "reasoning_history", "llm_warnings",
        ):
            assert field in item.value, f"Missing field: {field}"

    def test_stored_request_text_matches(self):
        _run_approved(_TEXT, "ep_1e", self.ep_store)
        item = self.ep_store.search(_EPISODE_NAMESPACE)[0]
        assert item.value["request_text"] == _TEXT

    def test_stored_classification_correct(self):
        _run_approved(_TEXT, "ep_1f", self.ep_store)
        item = self.ep_store.search(_EPISODE_NAMESPACE)[0]
        assert item.value["classification"]["request_type"] == "deletion"

    def test_stored_reasoning_history_nonempty(self):
        _run_approved(_TEXT, "ep_1g", self.ep_store)
        item = self.ep_store.search(_EPISODE_NAMESPACE)[0]
        assert len(item.value["reasoning_history"]) >= 4  # classify+scope+risk+human

    def test_stored_human_review_approved(self):
        _run_approved(_TEXT, "ep_1h", self.ep_store)
        item = self.ep_store.search(_EPISODE_NAMESPACE)[0]
        assert item.value["human_review"]["approved"] is True
        assert item.value["human_review"]["reviewer"] == "tester"

    def test_key_matches_output_episode_id(self):
        result = _run_approved(_TEXT, "ep_1i", self.ep_store)
        eid    = result["output"].episode_id
        keys   = {i.key for i in self.ep_store.search(_EPISODE_NAMESPACE)}
        assert eid in keys

    def test_stored_value_episode_id_matches_key(self):
        result = _run_approved(_TEXT, "ep_1j", self.ep_store)
        eid    = result["output"].episode_id
        item   = self.ep_store.get(_EPISODE_NAMESPACE, eid)
        assert item is not None
        assert item.value["episode_id"] == eid


# ---------------------------------------------------------------------------
# 2. Multiple approvals → distinct episodes
# ---------------------------------------------------------------------------

class TestMultipleEpisodes:

    def setup_method(self):
        self.ep_store = _fresh_store()

    def test_two_approvals_create_two_distinct_keys(self):
        r1 = _run_approved("Delete my account.",      "ep_2a", self.ep_store)
        r2 = _run_approved("Please remove all data.", "ep_2b", self.ep_store)
        assert len(self.ep_store.search(_EPISODE_NAMESPACE)) == 2
        assert r1["output"].episode_id != r2["output"].episode_id

    def test_episode_ids_look_like_uuids(self):
        r = _run_approved("Delete everything.", "ep_2c", self.ep_store)
        assert _UUID4_RE.match(r["output"].episode_id)


# ---------------------------------------------------------------------------
# 3. Revise decision does NOT store an episode prematurely
# ---------------------------------------------------------------------------

class TestNoEpisodeOnRevise:

    def setup_method(self):
        self.ep_store = _fresh_store()

    def test_revise_does_not_create_episode_prematurely(self):
        """Episodes are only written after final approval, not on a revise."""
        app    = build_triage_graph(episode_store=self.ep_store)
        config = {"configurable": {"thread_id": "ep_3a"}}

        # First pass → pause at human_checkpoint
        app.invoke({"request_text": _TEXT}, config=config)
        assert len(self.ep_store.search(_EPISODE_NAMESPACE)) == 0

        # Revise → loops back to classify, pauses again
        app.invoke(Command(resume=_REVISE.model_dump()), config=config)
        assert len(self.ep_store.search(_EPISODE_NAMESPACE)) == 0

        # Final approval → exactly one episode written
        app.invoke(Command(resume=_APPROVE.model_dump()), config=config)
        assert len(self.ep_store.search(_EPISODE_NAMESPACE)) == 1
