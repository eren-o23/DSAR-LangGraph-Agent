## DSAR LangGraph Agent

A small **LangGraph** workflow that triages an incoming DSAR request through:

- **ClassificationAgent**: categorises the request as access / deletion / portability
- **ScopingAgent**: proposes which internal systems to query based on request type
- **RiskFlagAgent**: flags complicating factors (third party data, minor, conflicting legal basis, etc.)
- **Human-in-the-loop checkpoint**: a required review step implemented via LangGraph’s `interrupt`

### Project layout

- `src/dsar_langgraph_agent/triage_schemas.py`: Pydantic schemas for triage output
- `src/dsar_langgraph_agent/triage_graph.py`: LangGraph state machine + nodes + interrupt checkpoint
- `src/dsar_langgraph_agent/cli.py`: CLI runner that pauses at the interrupt, collects approval, then resumes
- `tests/test_triage_graph.py`: unit test that validates interrupt + resume works

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/requirements.txt
```

### Run (CLI)

```bash
python -m dsar_langgraph_agent.cli --text "Please delete my account and remove my personal data."
```

### Run tests

```bash
pytest -q
```

