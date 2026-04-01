from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

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


def build_output(state: TriageState) -> TriageState:
    output = TriageOutput(
        classification=state["classification"],
        scope=state["scope"],
        risk=state["risk"],
        human_review=state.get("human_review"),
        llm_warnings=list(state.get("llm_warnings") or []),
        reasoning_history=list(state.get("reasoning_history") or []),
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


def build_triage_graph(
    *,
    use_llm: bool = False,
    llm_config: Optional[Dict[str, Any]] = None,
):
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import StateGraph

    if llm_config is None:
        llm_config = {}

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
        graph.add_node("scope", lambda s: scoping_agent_llm(s, cfg=cfg))
        graph.add_node("risk", lambda s: risk_flag_agent_llm(s, cfg=cfg))
    else:
        graph.add_node("classify", classification_agent)
        graph.add_node("scope", scoping_agent)
        graph.add_node("risk", risk_flag_agent)

    graph.add_node("human_checkpoint", human_checkpoint)
    graph.add_node("finalize", build_output)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "scope")
    graph.add_edge("scope", "risk")
    graph.add_edge("risk", "human_checkpoint")
    # Conditional routing: 'revise' → back to classify, 'approve' → finalize
    graph.add_conditional_edges(
        "human_checkpoint",
        route_after_human,
        {"classify": "classify", "finalize": "finalize"},
    )

    return graph.compile(checkpointer=MemorySaver())


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

