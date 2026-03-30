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


@pytest.mark.skipif(
    __import__("os").getenv("RUN_OLLAMA_TESTS", "0") != "1",
    reason="Set RUN_OLLAMA_TESTS=1 and run Ollama locally to enable this test.",
)
def test_triage_graph_llm_mode_interrupt_and_resume():
    # This test expects a local Ollama server with the requested model available.
    app = build_triage_graph(
        use_llm=True,
        llm_config={
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "model": "llama3.1",
            "timeout_s": 30.0,
        },
    )
    config = {"configurable": {"thread_id": "test_triage_llm_1"}}

    request_text = "I want a copy of all data you have on me, including support chats. Also delete anything you can."

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
    assert output.classification.request_type in (
        DSARRequestType.access,
        DSARRequestType.deletion,
        DSARRequestType.portability,
        DSARRequestType.unknown,
    )

