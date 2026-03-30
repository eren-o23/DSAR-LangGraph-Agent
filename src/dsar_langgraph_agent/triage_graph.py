from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from dsar_langgraph_agent.triage_agents import classification_agent, risk_flag_agent, scoping_agent
from dsar_langgraph_agent.triage_schemas import (
    HumanReviewDecision,
    TriageOutput,
)
from dsar_langgraph_agent.triage_state import TriageState


def build_output(state: TriageState) -> TriageState:
    output = TriageOutput(
        classification=state["classification"],
        scope=state["scope"],
        risk=state["risk"],
        human_review=state.get("human_review"),
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
    overrides = decision.overrides or {}
    for k, v in overrides.items():
        next_state[k] = v
    return next_state


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
    graph.add_edge("human_checkpoint", "finalize")

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

