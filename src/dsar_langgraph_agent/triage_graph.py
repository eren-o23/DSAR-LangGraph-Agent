from __future__ import annotations

import importlib.resources as _ir
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dsar_langgraph_agent.triage_agents import (
    _append_reasoning,
    classification_agent,
    risk_flag_agent,
    scoping_agent,
)
from dsar_langgraph_agent.triage_schemas import (
    HumanReviewDecision,
    ReasoningEntry,
    TriageOutput,
)
from dsar_langgraph_agent.triage_state import TriageState

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Episodic memory store — single shared instance for the process lifetime.
# Passed to graph.compile(store=store) so every node can access it via
# langgraph.config.get_store() without any additional state threading.
# ---------------------------------------------------------------------------
from langgraph.store.memory import InMemoryStore  # noqa: E402

store = InMemoryStore()

# Namespace under which approved triage episodes are stored.
_EPISODE_NAMESPACE: tuple[str, str] = ("precedents", "triage")

# Namespace under which the data-system inventory is stored.
_INVENTORY_NAMESPACE: tuple[str, str] = ("inventory", "systems")


# ---------------------------------------------------------------------------
# Semantic memory — data inventory loader
# ---------------------------------------------------------------------------

def load_data_inventory(target_store: InMemoryStore | None = None) -> List[str]:
    """
    Load ``data_inventory.json`` into the LangGraph Store under the
    ``('inventory', 'systems')`` namespace.

    Each system is stored with its ``id`` as the key and the full JSON object
    as the value.  The function is **idempotent**: calling it multiple times
    with the same data will overwrite existing entries (same key) rather than
    create duplicates.

    Parameters
    ----------
    target_store:
        The :class:`~langgraph.store.memory.InMemoryStore` to write into.
        Defaults to the module-level ``store`` singleton.

    Returns
    -------
    List[str]
        The list of system ids that were loaded.
    """
    _store = target_store if target_store is not None else store
    pkg = _ir.files("dsar_langgraph_agent.data")
    raw = (pkg / "data_inventory.json").read_text(encoding="utf-8")
    systems: List[Dict[str, Any]] = json.loads(raw)

    loaded: List[str] = []
    for system in systems:
        system_id = system["id"]
        _store.put(_INVENTORY_NAMESPACE, system_id, system)
        loaded.append(system_id)
        log.debug("Loaded inventory system: %s", system_id)

    log.info(
        "Data inventory loaded into store (namespace=%s, systems=%s)",
        _INVENTORY_NAMESPACE, loaded,
    )
    return loaded


# Pre-load the inventory into the module-level store at import time so that
# every graph instance compiled with the default store has inventory available
# immediately, without requiring an explicit startup call from the caller.
try:
    load_data_inventory()
except Exception as _e:  # pragma: no cover
    log.warning("Could not pre-load data inventory: %s", _e)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def build_output(state: TriageState) -> TriageState:
    output = TriageOutput(
        classification=state["classification"],
        scope=state["scope"],
        risk=state["risk"],
        human_review=state.get("human_review"),
        llm_warnings=list(state.get("llm_warnings") or []),
        reasoning_history=list(state.get("reasoning_history") or []),
        # episode_id is written by record_episode_node *after* this node,
        # so it will be None here and set on a second state update.
        episode_id=state.get("episode_id"),
    )
    return {**state, "output": output}


@dataclass(frozen=True)
class HumanCheckpointPayload:
    classification: Dict[str, Any]
    scope: Dict[str, Any]
    risk: Dict[str, Any]


def human_checkpoint(state: TriageState) -> TriageState:
    """
    Human-in-the-loop checkpoint using LangGraph's interrupt mechanism.

    The reviewer must resume with a dict that includes at minimum:
      { "approved": true/false, ... }

    To trigger a revision loop, set ``approved`` to ``false`` and supply
    a ``human_feedback`` string explaining what needs to be corrected:
      { "approved": false, "human_feedback": "This looks like portability, not access." }
    """
    from langgraph.types import interrupt

    payload = HumanCheckpointPayload(
        classification=state["classification"].model_dump(),
        scope=state["scope"].model_dump(),
        risk=state["risk"].model_dump(),
    )

    decision_dict = interrupt(payload.__dict__)
    decision = HumanReviewDecision.model_validate(decision_dict)

    next_state: TriageState = {**state, "human_review": decision}

    # Propagate human_feedback into the top-level state key so that
    # the ClassificationAgent can read it on the next iteration.
    next_state["human_feedback"] = decision.human_feedback or None

    overrides = decision.overrides or {}
    for k, v in overrides.items():
        next_state[k] = v

    # Record the human decision in the reasoning history.
    action = "approved" if decision.approved else "revise"
    reviewer_note = decision.human_feedback or decision.notes or ""
    human_entry = ReasoningEntry(
        agent="human_checkpoint",
        source="human",
        decision_made=f"{action}, reviewer={decision.reviewer}",
        rationale=reviewer_note if reviewer_note else f"Reviewer marked as {action}.",
    )
    next_state["reasoning_history"] = _append_reasoning(next_state, human_entry)

    return next_state


def record_episode_node(state: TriageState) -> TriageState:
    """
    Episodic memory node — persists the approved triage episode to the
    LangGraph Store under the ``('precedents', 'triage')`` namespace.

    The store is injected at runtime via ``langgraph.config.get_store()``;
    the same ``InMemoryStore`` instance that was passed to ``compile()`` is
    returned, so no state threading is needed.

    Stored value schema
    -------------------
    {
        "request_text":      str,
        "classification":    dict,   # ClassificationResult.model_dump()
        "scope":             dict,   # ScopingResult.model_dump()
        "risk":              dict,   # RiskResult.model_dump()
        "human_review":      dict | None,
        "reasoning_history": list[dict],
        "llm_warnings":      list[dict],
        "episode_id":        str,    # UUID key used for this entry
    }
    """
    from langgraph.config import get_store

    episode_id = str(uuid.uuid4())

    value: Dict[str, Any] = {
        "episode_id":        episode_id,
        "request_text":      state.get("request_text", ""),
        "classification":    state["classification"].model_dump(),
        "scope":             state["scope"].model_dump(),
        "risk":              state["risk"].model_dump(),
        "human_review":      (
            state["human_review"].model_dump() if state.get("human_review") else None
        ),
        "reasoning_history": [
            e.model_dump() for e in (state.get("reasoning_history") or [])
        ],
        "llm_warnings": [
            w.model_dump() for w in (state.get("llm_warnings") or [])
        ],
    }

    ep_store = get_store()
    ep_store.put(_EPISODE_NAMESPACE, episode_id, value)

    log.info(
        "Episode persisted to store (namespace=%s, key=%s)",
        _EPISODE_NAMESPACE, episode_id,
    )

    # Propagate the episode_id back into state so build_output can surface it.
    updated_output = state["output"].model_copy(update={"episode_id": episode_id})
    return {**state, "episode_id": episode_id, "output": updated_output}


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def route_after_human(state: TriageState) -> str:
    """
    Conditional edge: decide whether to finalise or loop back to classify.

    Returns
    -------
    "classify"  – reviewer chose 'revise' (approved is False and feedback given)
    "finalize"  – reviewer approved the triage result
    """
    review = state.get("human_review")
    if review is not None and not review.approved and state.get("human_feedback"):
        return "classify"
    return "finalize"


def route_after_finalize(state: TriageState) -> str:
    """
    Conditional edge: after ``build_output``, decide whether to record the
    episode or exit immediately.

    Returns
    -------
    "record_episode"  – the reviewer approved; persist the episode
    "__end__"         – no human review recorded (shouldn't happen in normal
                        flow, but safe default)
    """
    review = state.get("human_review")
    if review is not None and review.approved:
        return "record_episode"
    return "__end__"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_triage_graph(
    *,
    use_llm: bool = False,
    llm_config: Optional[Dict[str, Any]] = None,
    episode_store: Optional[Any] = None,
):
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import StateGraph

    if llm_config is None:
        llm_config = {}

    # Use the provided store (for isolated testing) or the module-level singleton.
    _store = episode_store if episode_store is not None else store

    graph = StateGraph(TriageState)

    if use_llm:
        from dsar_langgraph_agent.triage_llm_agents import (
            LLMNodeConfig,
            classification_agent_llm,
            risk_flag_agent_llm,
            scoping_agent_llm,
        )

        cfg = LLMNodeConfig(**llm_config)
        graph.add_node("classify", lambda s: classification_agent_llm(s, cfg=cfg))
        graph.add_node("scope",    lambda s: scoping_agent_llm(s, cfg=cfg))
        graph.add_node("risk",     lambda s: risk_flag_agent_llm(s, cfg=cfg))
    else:
        graph.add_node("classify", classification_agent)
        graph.add_node("scope",    scoping_agent)
        graph.add_node("risk",     risk_flag_agent)

    graph.add_node("human_checkpoint",  human_checkpoint)
    graph.add_node("finalize",          build_output)
    graph.add_node("record_episode",    record_episode_node)

    # Edges
    graph.set_entry_point("classify")
    graph.add_edge("classify", "scope")
    graph.add_edge("scope",    "risk")
    graph.add_edge("risk",     "human_checkpoint")

    # human_checkpoint → classify (revise) or finalize (approve)
    graph.add_conditional_edges(
        "human_checkpoint",
        route_after_human,
        {"classify": "classify", "finalize": "finalize"},
    )

    # finalize → record_episode (approved) or __end__ (safety fallback)
    graph.add_conditional_edges(
        "finalize",
        route_after_finalize,
        {"record_episode": "record_episode", "__end__": "__end__"},
    )

    graph.set_finish_point("record_episode")

    # compile with both the checkpointer (for HITL interrupts) and the store
    # (for episodic memory).
    return graph.compile(checkpointer=MemorySaver(), store=_store)


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

def run_triage(
    request_text: str,
    *,
    human_review: Optional[HumanReviewDecision] = None,
) -> TriageOutput:
    """
    Convenience helper for non-interactive usage (e.g., tests).
    """
    app = build_triage_graph()
    config = {"configurable": {"thread_id": "triage"}}

    try:
        result = app.invoke({"request_text": request_text}, config=config)
        return result["output"]
    except Exception as e:
        if human_review is None:
            raise

        try:
            from langgraph.types import Command  # type: ignore
        except Exception:
            raise e

        resumed = app.invoke(Command(resume=human_review.model_dump()), config=config)
        return resumed["output"]
