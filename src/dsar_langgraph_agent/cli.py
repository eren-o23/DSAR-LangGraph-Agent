from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from dsar_langgraph_agent.triage_graph import build_triage_graph
from dsar_langgraph_agent.triage_schemas import HumanReviewDecision


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DSAR triage LangGraph workflow (with human checkpoint).")
    parser.add_argument("--text", required=True, help="Incoming DSAR request text.")
    parser.add_argument("--thread-id", default="triage_cli", help="LangGraph thread id for checkpointing/resume.")
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve at human checkpoint (no prompt).")
    args = parser.parse_args()

    app = build_triage_graph()
    config = {"configurable": {"thread_id": args.thread_id}}

    try:
        result = app.invoke({"request_text": args.text}, config=config)
        print(_safe_json(result["output"].model_dump()))
        return 0
    except Exception as e:
        payload: Dict[str, Any] | None = None
        for attr in ("value", "payload", "data"):
            if hasattr(e, attr):
                candidate = getattr(e, attr)
                if isinstance(candidate, dict):
                    payload = candidate
                    break

        if payload is None and "interrupt" not in e.__class__.__name__.lower():
            raise

        if payload is None:
            payload = {"note": "Interrupted; payload unavailable on this LangGraph version."}

        print("Human review required. Proposed triage output (pre-approval):")
        print(_safe_json(payload))

        if args.auto_approve:
            decision = HumanReviewDecision(approved=True, reviewer="auto", notes="auto-approved by CLI")
        else:
            approved = input("Approve? [y/N]: ").strip().lower() in ("y", "yes")
            reviewer = input("Reviewer name (optional): ").strip() or "human"
            notes = input("Notes (optional): ").strip()
            decision = HumanReviewDecision(approved=approved, reviewer=reviewer, notes=notes, overrides={})

        from langgraph.types import Command

        resumed = app.invoke(Command(resume=decision.model_dump()), config=config)
        print(_safe_json(resumed["output"].model_dump()))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

