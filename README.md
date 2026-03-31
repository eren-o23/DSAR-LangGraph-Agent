# DSAR LangGraph Agent

A governed AI workflow for triaging Data Subject Access Requests (DSARs), built with LangGraph and powered by open-source LLMs via Ollama.

---

## Why this exists

Under GDPR and similar regulations, organisations must respond to DSARs within strict deadlines. In practice, triage is slow, inconsistent, and handled manually: a person reads the request, works out what type it is, figures out which systems are involved, and flags anything legally complicated. That process is time-consuming, error-prone, and hard to audit.

This project automates that triage step while keeping a human in the loop before anything is finalised. It is designed for regulated environments: every decision is structured, every fallback is deterministic, and no action is taken without explicit approval.

---

## How it works

The workflow runs as a LangGraph state machine. Each stage reads from and writes to a shared state object, which is persisted across the human checkpoint.

```
DSAR request text
       │
       ▼
ClassificationAgent   →  access / deletion / portability
       │
       ▼
ScopingAgent          →  systems that need to be queried
       │
       ▼
RiskFlagAgent         →  third-party data, minors, conflicting legal basis
       │
       ▼
── interrupt() ──     →  execution pauses; human reviews full state
       │
  approve / revise
       │
       ▼
Finalised triage output
```

**ClassificationAgent** reads the request and categorises it into one of three DSAR types.

**ScopingAgent** maps the request type to the systems that would need to be queried — CRM, data warehouse, email platform, and so on.

**RiskFlagAgent** checks for anything that would complicate the response: third-party data involved, request from a minor, conflicting legal basis, or ambiguous identity.

**Human review checkpoint** pauses execution using LangGraph's `interrupt()` mechanism. The full triage state is surfaced to an operator, who can approve or return it with feedback. The graph resumes from exactly the same state.

---

## Demo

```bash
$ python -m dsar_langgraph_agent.cli \
    --text "Please delete my account and all personal data you hold about me."

[ClassificationAgent]  type=deletion  confidence=high
[ScopingAgent]         systems=[CRM, email_platform, data_warehouse]
[RiskFlagAgent]        flags=[]  risk_level=low

── CHECKPOINT: human review required ──
Triage output ready for approval. Press [a] to approve or [r] to revise.

> a

[APPROVED] Triage finalised. Output written to state.
```

---

## LLM + deterministic hybrid design

Most AI workflow tools treat the LLM as the source of truth. This system treats it as an enhancement to deterministic logic — not a replacement for it.

In **deterministic mode** (the default), triage decisions are made by rule-based logic: fast, fully predictable, and auditable. In **LLM mode**, an open model running locally via Ollama improves classification and risk detection on ambiguous requests.

If the LLM fails, returns invalid JSON, or produces output that fails schema validation, the system falls back to deterministic logic automatically. No silent failures, no hallucinated decisions.

This makes the system safe to run in a regulated environment whether or not a local model is available.

---

## Project structure

```
src/dsar_langgraph_agent/
├── triage_schemas.py      Pydantic schemas for triage state
├── triage_agents.py       Deterministic agent implementations
├── triage_llm_agents.py   LLM-powered agent implementations
├── ollama_client.py       OpenAI-compatible Ollama client
├── triage_graph.py        LangGraph workflow + interrupt checkpoint
└── cli.py                 CLI runner with resume capability

tests/
└── test_triage_graph.py   Tests including interrupt/resume validation
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/requirements.txt
```

---

## Running the agent

**Deterministic mode**

```bash
python -m dsar_langgraph_agent.cli \
  --text "Please delete my account and remove my personal data."
```

**LLM mode (requires Ollama)**

```bash
ollama serve
ollama pull llama3.1

python -m dsar_langgraph_agent.cli \
  --use-llm --model llama3.1 \
  --text "Please delete my account and remove my personal data."
```

---

## Tests

```bash
pytest -q
```

With Ollama integration tests:

```bash
RUN_OLLAMA_TESTS=1 pytest -q
```

---

## What's next

The next phase extends the project with two capabilities developed in parallel.

**Long-term memory** will allow the agents to learn from past DSARs. If a particular data controller consistently involves third-party processors, the RiskFlagAgent will surface that pattern rather than treating every request from scratch.

**RAG over policy documents** will ground the ScopingAgent in your actual data inventory and records of processing activities, rather than having system knowledge baked into the prompt. This is more maintainable and far more auditable for a real compliance use case.

Both are being built as part of preparation for the AMD × lablab.ai hackathon (AI Agents & Agentic Workflows track).

---

## Built with

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Ollama](https://ollama.com)
- [Pydantic](https://docs.pydantic.dev)
