from __future__ import annotations

from typing import List, Optional, TypedDict

from dsar_langgraph_agent.triage_schemas import (
    ClassificationResult,
    HumanReviewDecision,
    LLMFallbackWarning,
    ReasoningEntry,
    RiskResult,
    ScopingResult,
    TriageOutput,
)


class TriageState(TypedDict, total=False):
    request_text: str
    classification: ClassificationResult
    scope: ScopingResult
    risk: RiskResult
    human_review: HumanReviewDecision
    # Carries the reviewer's correction text when they choose 'revise'.
    # Cleared (set to None) once the ClassificationAgent has consumed it.
    human_feedback: Optional[str]
    # Accumulates structured warnings from LLM agent fallback events.
    llm_warnings: List[LLMFallbackWarning]
    # Ordered audit trail: one entry appended per node per execution.
    reasoning_history: List[ReasoningEntry]
    # Set by record_episode_node after the episode is persisted.
    episode_id: Optional[str]
    output: TriageOutput

