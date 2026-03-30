from __future__ import annotations

from typing import TypedDict

from dsar_langgraph_agent.triage_schemas import (
    ClassificationResult,
    HumanReviewDecision,
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
    output: TriageOutput

