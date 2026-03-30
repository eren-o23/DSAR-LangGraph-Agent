import pytest

from dsar_langgraph_agent.triage_graph import build_triage_graph
from dsar_langgraph_agent.triage_schemas import DSARRequestType, HumanReviewDecision


def test_triage_graph_interrupt_and_resume():
    app = build_triage_graph()
    config = {"configurable": {"thread_id": "test_triage_1"}}

    request_text = "Please delete my account and remove my personal data. I might have an old account too."

    try:
        app.invoke({"request_text": request_text}, config=config)
        assert False, "Expected human checkpoint interrupt"
    except Exception as e:
        assert "interrupt" in e.__class__.__name__.lower() or hasattr(e, "value") or hasattr(e, "payload")

    from langgraph.types import Command

    decision = HumanReviewDecision(approved=True, reviewer="tester", notes="ok")
    resumed = app.invoke(Command(resume=decision.model_dump()), config=config)

    output = resumed["output"]
    assert output.human_review is not None
    assert output.human_review.approved is True
    assert output.classification.request_type == DSARRequestType.deletion
    assert "identity_store" in output.scope.systems
    assert len(output.risk.flags) >= 1

