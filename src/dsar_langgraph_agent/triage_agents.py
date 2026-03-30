from __future__ import annotations

from typing import List

from dsar_langgraph_agent.triage_schemas import (
    ClassificationResult,
    DSARRequestType,
    RiskFlag,
    RiskResult,
    ScopingResult,
)
from dsar_langgraph_agent.triage_state import TriageState


def _contains_any(text: str, needles: List[str]) -> bool:
    t = (text or "").lower()
    return any(n.lower() in t for n in needles)


def classification_agent(state: TriageState) -> TriageState:
    text = state.get("request_text", "")
    t = text.lower()

    if _contains_any(t, ["delete", "erasure", "remove my data", "right to be forgotten"]):
        req_type = DSARRequestType.deletion
        conf = 0.85
        rationale = "Deletion/erasure language detected."
    elif _contains_any(t, ["port", "portability", "machine readable", "transfer my data", "download my data"]):
        req_type = DSARRequestType.portability
        conf = 0.8
        rationale = "Portability/transfer language detected."
    elif _contains_any(t, ["access", "copy of my data", "what data", "subject access", "dsar"]):
        req_type = DSARRequestType.access
        conf = 0.8
        rationale = "Access/copy language detected."
    else:
        req_type = DSARRequestType.unknown
        conf = 0.35
        rationale = "No clear DSAR request type keywords detected."

    return {
        **state,
        "classification": ClassificationResult(
            request_type=req_type,
            confidence=conf,
            rationale=rationale,
        ),
    }


def scoping_agent(state: TriageState) -> TriageState:
    classification = state["classification"]
    text = state.get("request_text", "").lower()

    systems: List[str] = []
    rationale_bits: List[str] = []

    core_identity = ["identity_store", "crm"]
    comms = ["support_ticketing", "email_system"]
    product = ["app_db", "analytics"]
    storage = ["object_storage"]
    billing = ["billing"]

    if classification.request_type in (
        DSARRequestType.access,
        DSARRequestType.portability,
        DSARRequestType.deletion,
    ):
        systems.extend(core_identity)
        systems.extend(product)
        rationale_bits.append("Standard identity + product systems included for DSAR processing.")

    if _contains_any(text, ["support", "ticket", "complaint", "chat"]):
        systems.extend(comms)
        rationale_bits.append("Support/contact language detected; included support + email systems.")

    if _contains_any(text, ["invoice", "payment", "card", "billing", "subscription"]):
        systems.extend(billing)
        rationale_bits.append("Billing language detected; included billing systems.")

    if classification.request_type == DSARRequestType.portability:
        systems.extend(storage)
        rationale_bits.append("Portability typically requires export packaging; included object storage.")

    if classification.request_type == DSARRequestType.unknown:
        rationale_bits.append("Request type unclear; scope is conservative and may need clarification.")

    seen = set()
    systems_deduped = []
    for s in systems:
        if s not in seen:
            systems_deduped.append(s)
            seen.add(s)

    return {
        **state,
        "scope": ScopingResult(systems=systems_deduped, rationale=" ".join(rationale_bits).strip()),
    }


def risk_flag_agent(state: TriageState) -> TriageState:
    text = state.get("request_text", "").lower()
    flags: List[RiskFlag] = []
    notes: List[str] = []

    if _contains_any(text, ["my child", "minor", "under 13", "under 16", "under 18", "guardian", "parent"]):
        flags.append(RiskFlag.request_from_minor)
        notes.append("Potential minor/guardian context; additional checks may be required.")

    if _contains_any(text, ["third party", "someone else", "other person", "include my family", "include my friend"]):
        flags.append(RiskFlag.third_party_data_involved)
        notes.append("Potential third-party data; may require redaction or consent checks.")

    if _contains_any(text, ["legal basis", "contract", "legitimate interest", "retain", "required by law", "litigation hold"]):
        flags.append(RiskFlag.conflicting_legal_basis)
        notes.append("Retention/legal basis language present; deletion may be restricted.")

    if _contains_any(text, ["i think", "maybe", "not sure", "different email", "old account"]):
        flags.append(RiskFlag.identity_unclear)
        notes.append("Identity hints are uncertain; ensure ID verification and identifier reconciliation.")

    if _contains_any(text, ["everything", "all data", "any and all", "all information"]):
        flags.append(RiskFlag.ambiguous_scope)
        notes.append("Broad scope request; consider clarification and proportionality controls.")

    return {**state, "risk": RiskResult(flags=flags, notes=notes)}

