from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DSARRequestType(str, Enum):
    access = "access"
    deletion = "deletion"
    portability = "portability"
    unknown = "unknown"


class ClassificationResult(BaseModel):
    request_type: DSARRequestType
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    rationale: str = Field(default="")


class ScopingResult(BaseModel):
    systems: List[str] = Field(default_factory=list)
    rationale: str = Field(default="")


class RiskFlag(str, Enum):
    third_party_data_involved = "third_party_data_involved"
    request_from_minor = "request_from_minor"
    conflicting_legal_basis = "conflicting_legal_basis"
    identity_unclear = "identity_unclear"
    ambiguous_scope = "ambiguous_scope"


class RiskResult(BaseModel):
    flags: List[RiskFlag] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class HumanReviewDecision(BaseModel):
    approved: bool
    reviewer: str = Field(default="human")
    notes: str = Field(default="")
    overrides: Dict[str, Any] = Field(default_factory=dict)
    # Populated when the reviewer chooses 'revise' — forwarded to the
    # ClassificationAgent so it can correct its previous decision.
    human_feedback: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# LLM fallback telemetry
# ---------------------------------------------------------------------------

FallbackCause = Literal["timeout", "bad_json", "schema_error", "unknown"]
ReasoningSource = Literal["deterministic", "llm", "llm_fallback", "human"]


class LLMFallbackWarning(BaseModel):
    """Recorded in state["llm_warnings"] whenever an LLM agent falls back to
    its deterministic counterpart.  Keeps error metadata out of domain fields."""

    agent: str = Field(description="Node name that triggered the fallback.")
    cause: FallbackCause = Field(description="High-level reason for the fallback.")
    detail: str = Field(default="", description="str(exception) from the failure.")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp of the fallback event.",
    )


# ---------------------------------------------------------------------------
# Reasoning history
# ---------------------------------------------------------------------------


class ReasoningEntry(BaseModel):
    """One step in the audit trail — appended by every node on every execution."""

    agent: str = Field(description="Node / agent name that produced this entry.")
    source: ReasoningSource = Field(description="How the decision was reached.")
    decision_made: str = Field(description="Short, human-readable summary of the outcome.")
    rationale: str = Field(
        default="",
        description="The specific rule triggered, LLM thought extracted, or reviewer note.",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp of when this entry was recorded.",
    )


class TriageOutput(BaseModel):
    classification: ClassificationResult
    scope: ScopingResult
    risk: RiskResult
    human_review: Optional[HumanReviewDecision] = None
    llm_warnings: List[LLMFallbackWarning] = Field(default_factory=list)
    reasoning_history: List[ReasoningEntry] = Field(default_factory=list)
    # Set when the episode was persisted to the LangGraph Store.
    episode_id: Optional[str] = Field(default=None)

