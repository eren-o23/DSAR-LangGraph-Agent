## DSAR LangGraph Agent

A **governed AI workflow** for triaging Data Subject Access Requests (DSARs), built with **LangGraph** and powered by **open-source LLMs via Ollama**.

This project demonstrates how AI agents can be safely integrated into compliance workflows using:
- structured outputs
- deterministic fallbacks
- human-in-the-loop checkpoints
- full interrupt + resume capability

---

## Why this matters

Handling DSARs is:
- time-consuming
- error-prone
- legally sensitive

This system shows how to:
- automate **triage decisions with AI**
- maintain **auditability and control**
- ensure **humans stay in the loop for critical decisions**

---

## How it works

The workflow is implemented as a **LangGraph state machine**:

1. **ClassificationAgent**
   - Determines request type (access / deletion / portability)

2. **ScopingAgent**
   - Identifies relevant systems and required data

3. **RiskFlagAgent**
   - Flags legal/compliance risks (e.g. third-party data, minors)

4. **Human Review (interrupt checkpoint)**
   - Execution pauses for approval
   - Operator can review outputs before continuing

5. **Resume execution**
   - Continues from the exact same state after approval

---

## LLM + Deterministic Hybrid Design

The system supports two modes:

### Deterministic (default)
- rule-based logic
- fully predictable
- used as a fallback

### LLM-powered (Ollama)
- uses open models (e.g. `llama3.1`, `qwen`)
- improves classification, scoping, and risk detection

### Safety mechanism
If the LLM:
- fails
- returns invalid JSON
- produces low-quality output

the system **automatically falls back to deterministic logic**

This ensures:
- reliability
- auditability
- production-safe behaviour

---

## Project structure

- `src/dsar_langgraph_agent/triage_schemas.py`: Pydantic schemas for triage state
- `src/dsar_langgraph_agent/triage_agents.py`: deterministic agent implementations
- `src/dsar_langgraph_agent/triage_llm_agents.py`: LLM-powered agent implementations
- `src/dsar_langgraph_agent/ollama_client.py`: OpenAI-compatible Ollama client
- `src/dsar_langgraph_agent/triage_graph.py`: LangGraph workflow + interrupt checkpoint
- `src/dsar_langgraph_agent/cli.py`: CLI runner with resume capability
- `tests/test_triage_graph.py`: tests (including interrupt/resume validation)

---

## Setup

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements/requirements.txt

---

## Run

### Deterministic mode (default)

    python -m dsar_langgraph_agent.cli --text "Please delete my account and remove my personal data."

---

### LLM mode (Ollama)

1) Start Ollama and pull a model:

    ollama serve
    ollama pull llama3.1

2) Run with LLM enabled:

    python -m dsar_langgraph_agent.cli --use-llm --model llama3.1 --text "Please delete my account and remove my personal data."

---

## Run tests

    pytest -q

Optional Ollama integration test:

    RUN_OLLAMA_TESTS=1 pytest -q

---

## Key features

- LangGraph-based stateful agent workflow
- Interrupt + resume (human-in-the-loop)
- Structured JSON outputs (no hallucinated actions)
- LLM + deterministic fallback architecture
- Open-model support via Ollama
- CLI interface for quick testing

---

## Hackathon focus

This project demonstrates:
- practical AI agent orchestration (not just chatbots)
- safe deployment patterns for AI in regulated domains
- integration of open models into real workflows

---

## Future improvements

- Web dashboard for DSAR case management
- Integration with real data sources (CRM, support tools)
- Audit log persistence (PostgreSQL / event sourcing)
- Confidence scoring + escalation policies
- Multi-language request handling
