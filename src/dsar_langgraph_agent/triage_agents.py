from __future__ import annotations

from typing import List, Optional, Tuple

from dsar_langgraph_agent.triage_schemas import (
    ClassificationResult,
    DSARRequestType,
    ReasoningEntry,
    RiskFlag,
    RiskResult,
    ScopingResult,
)
from dsar_langgraph_agent.triage_state import TriageState


# ---------------------------------------------------------------------------
# Low-level text helpers
# ---------------------------------------------------------------------------

def _contains_any(text: str, needles: List[str]) -> bool:
    t = (text or "").lower()
    return any(n.lower() in t for n in needles)


def _first_match(text: str, needles: List[str]) -> Optional[str]:
    """Return the first needle found in *text* (case-insensitive), or None."""
    t = (text or "").lower()
    for n in needles:
        if n.lower() in t:
            return n
    return None


# ---------------------------------------------------------------------------
# Reasoning history helper  (exported — also used by triage_llm_agents)
# ---------------------------------------------------------------------------

def _append_reasoning(state: TriageState, entry: ReasoningEntry) -> List[ReasoningEntry]:
    """Return *state*'s reasoning_history with *entry* appended (immutable pattern)."""
    existing: List[ReasoningEntry] = list(state.get("reasoning_history") or [])
    existing.append(entry)
    return existing


# ---------------------------------------------------------------------------
# Deterministic agents
# ---------------------------------------------------------------------------

_DELETION_KW = ["delete", "erasure", "remove my data", "right to be forgotten"]
_PORTABILITY_KW = ["port", "portability", "machine readable", "transfer my data", "download my data"]
_ACCESS_KW = [
    "access", "subject access", "dsar", "copy of my data", "send me", "show me",
    "provide me with", "see my data", "view my data", "what data", "what information",
    "what information do you have about me", "what data do you have about me",
    "data you hold", "all data you hold", "information you hold",
    "all information you hold", "shared with your", "passed to", "marketing partners",
]

_SUPPORT_KW = ["support", "ticket", "complaint", "chat"]
_BILLING_KW = ["invoice", "payment", "card", "billing", "subscription"]

# (keywords, flag, note) — defines risk flag detection in a single table so
# both the flagging logic and the keyword-level history rationale share one source.
_RISK_FLAG_RULES: List[Tuple[List[str], RiskFlag, str]] = [
    (
        ["my child", "minor", "under 13", "under 16", "under 18", "guardian", "parent"],
        RiskFlag.request_from_minor,
        "Potential minor/guardian context; additional checks may be required.",
    ),
    (
        ["third party", "someone else", "other person", "include my family", "include my friend"],
        RiskFlag.third_party_data_involved,
        "Potential third-party data; may require redaction or consent checks.",
    ),
    (
        ["legal basis", "contract", "legitimate interest", "retain", "required by law", "litigation hold"],
        RiskFlag.conflicting_legal_basis,
        "Retention/legal basis language present; deletion may be restricted.",
    ),
    (
        ["i think", "maybe", "not sure", "different email", "old account"],
        RiskFlag.identity_unclear,
        "Identity hints are uncertain; ensure ID verification and identifier reconciliation.",
    ),
    (
        ["everything", "all data", "any and all", "all information"],
        RiskFlag.ambiguous_scope,
        "Broad scope request; consider clarification and proportionality controls.",
    ),
]


def classification_agent(state: TriageState) -> TriageState:
    text = state.get("request_text", "")
    t = text.lower()

    # --- Baseline keyword classification ---
    if (kw := _first_match(t, _DELETION_KW)):
        req_type = DSARRequestType.deletion
        conf = 0.85
        rationale = "Deletion/erasure language detected."
        rule_rationale = f'Keyword "{kw}" matched deletion rule.'
    elif (kw := _first_match(t, _PORTABILITY_KW)):
        req_type = DSARRequestType.portability
        conf = 0.8
        rationale = "Portability/transfer language detected."
        rule_rationale = f'Keyword "{kw}" matched portability rule.'
    elif (kw := _first_match(t, _ACCESS_KW)):
        req_type = DSARRequestType.access
        conf = 0.8
        rationale = "Access/copy language detected."
        rule_rationale = f'Keyword "{kw}" matched access rule.'
    else:
        req_type = DSARRequestType.unknown
        conf = 0.35
        rationale = "No clear DSAR request type keywords detected."
        rule_rationale = "No DSAR keyword matched any classification rule."

    # --- Human-feedback correction pass ---
    # When the reviewer chose 'revise', their guidance is stored in
    # state["human_feedback"].  Parse it for explicit type mentions and
    # override the baseline decision accordingly.
    human_feedback: Optional[str] = state.get("human_feedback")  # type: ignore[assignment]
    if human_feedback:
        fb = human_feedback.lower()
        correction_map = {
            DSARRequestType.deletion: ["delet", "erasure", "erase", "remov", "right to be forgotten"],
            DSARRequestType.portability: ["portab", "transfer", "machine readable", "download"],
            DSARRequestType.access: ["access", "subject access", "dsar", "copy", "view"],
        }
        corrected = False
        for candidate_type, hints in correction_map.items():
            if _contains_any(fb, hints):
                req_type = candidate_type
                conf = min(1.0, conf + 0.1)
                rationale = (
                    f"{rationale} [Revised per human feedback: \"{human_feedback.strip()}\"]"
                )
                rule_rationale = (
                    f"{rule_rationale} \u2192 corrected to {candidate_type.value} "
                    f'via human feedback: "{human_feedback.strip()}"'
                )
                corrected = True
                break

        if not corrected:
            rationale = (
                f"{rationale} [Human feedback received but no type override applied: "
                f'"{human_feedback.strip()}"]'
            )
            rule_rationale = (
                f"{rule_rationale} \u2192 human feedback received but no keyword match: "
                f'"{human_feedback.strip()}"'
            )

    entry = ReasoningEntry(
        agent="classification_agent",
        source="deterministic",
        decision_made=f"request_type={req_type.value}, confidence={conf}",
        rationale=rule_rationale,
    )

    return {
        **state,
        # Clear human_feedback once consumed so it doesn't affect a second cycle.
        "human_feedback": None,
        "classification": ClassificationResult(
            request_type=req_type,
            confidence=conf,
            rationale=rationale,
        ),
        "reasoning_history": _append_reasoning(state, entry),
    }


def scoping_agent(state: TriageState) -> TriageState:
    classification = state["classification"]
    text = state.get("request_text", "").lower()

    systems: List[str] = []
    rationale_bits: List[str] = []
    history_bits: List[str] = []

    if classification.request_type in (
        DSARRequestType.access,
        DSARRequestType.portability,
        DSARRequestType.deletion,
    ):
        systems.extend(["identity_store", "crm"])
        systems.extend(["app_db", "analytics"])
        rationale_bits.append("Standard identity + product systems included for DSAR processing.")
        history_bits.append(
            f"request_type={classification.request_type.value} → standard identity+product rule applied"
        )

    if (kw := _first_match(text, _SUPPORT_KW)):
        systems.extend(["support_ticketing", "email_system"])
        rationale_bits.append("Support/contact language detected; included support + email systems.")
        history_bits.append(f'Keyword "{kw}" → comms systems included')

    if (kw := _first_match(text, _BILLING_KW)):
        systems.extend(["billing"])
        rationale_bits.append("Billing language detected; included billing systems.")
        history_bits.append(f'Keyword "{kw}" → billing system included')

    if classification.request_type == DSARRequestType.portability:
        systems.extend(["object_storage"])
        rationale_bits.append("Portability typically requires export packaging; included object storage.")
        history_bits.append("portability rule → object_storage included")

    if classification.request_type == DSARRequestType.unknown:
        rationale_bits.append("Request type unclear; scope is conservative and may need clarification.")
        history_bits.append("unknown request_type → conservative scope, no systems added")

    seen: set = set()
    systems_deduped: List[str] = []
    for s in systems:
        if s not in seen:
            systems_deduped.append(s)
            seen.add(s)

    history_rationale = "; ".join(history_bits) if history_bits else "No scoping rules matched."

    entry = ReasoningEntry(
        agent="scoping_agent",
        source="deterministic",
        decision_made=f"systems={systems_deduped}",
        rationale=history_rationale,
    )

    return {
        **state,
        "scope": ScopingResult(
            systems=systems_deduped,
            rationale=" ".join(rationale_bits).strip(),
        ),
        "reasoning_history": _append_reasoning(state, entry),
    }


def risk_flag_agent(state: TriageState) -> TriageState:
    text = state.get("request_text", "").lower()
    flags: List[RiskFlag] = []
    notes: List[str] = []
    history_bits: List[str] = []

    for keywords, flag, note in _RISK_FLAG_RULES:
        if (kw := _first_match(text, keywords)):
            flags.append(flag)
            notes.append(note)
            history_bits.append(f'Keyword "{kw}" → {flag.value}')

    history_rationale = (
        "; ".join(history_bits) if history_bits else "No risk keywords matched any rule."
    )

    entry = ReasoningEntry(
        agent="risk_flag_agent",
        source="deterministic",
        decision_made=f"flags={[f.value for f in flags]}",
        rationale=history_rationale,
    )

    return {
        **state,
        "risk": RiskResult(flags=flags, notes=notes),
        "reasoning_history": _append_reasoning(state, entry),
    }
