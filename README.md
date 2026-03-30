## DSAR LangGraph Agent

A small **LangGraph** workflow that triages an incoming DSAR request through:

- **ClassificationAgent**: categorises the request as access / deletion / portability
- **ScopingAgent**: proposes which internal systems to query based on request type
- **RiskFlagAgent**: flags complicating factors (third party data, minor, conflicting legal basis, etc.)
- **Human-in-the-loop checkpoint**: a required review step implemented via LangGraph’s `interrupt`

### Project layout

- `src/dsar_langgraph_agent/triage_schemas.py`: Pydantic schemas for triage output
- `src/dsar_langgraph_agent/triage_agents.py`: deterministic (non-LLM) agent implementations
- `src/dsar_langgraph_agent/triage_llm_agents.py`: Ollama-powered (LLM) agent implementations
- `src/dsar_langgraph_agent/ollama_client.py`: Ollama OpenAI-compatible client wrapper
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

Deterministic mode (default):

```bash
python -m dsar_langgraph_agent.cli --text "Please delete my account and remove my personal data."
```

LLM mode (Ollama):

1) Start Ollama and pull a model:

```bash
ollama serve
ollama pull llama3.1
```

2) Run with `--use-llm`:

```bash
python -m dsar_langgraph_agent.cli --use-llm --model llama3.1 --text "Please delete my account and remove my personal data."
```

### Run tests

```bash
pytest -q
```

Optional Ollama integration test:

```bash
RUN_OLLAMA_TESTS=1 pytest -q
```

