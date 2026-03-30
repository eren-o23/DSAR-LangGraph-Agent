from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

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


class TriageOutput(BaseModel):
    classification: ClassificationResult
    scope: ScopingResult
    risk: RiskResult
    human_review: Optional[HumanReviewDecision] = None

