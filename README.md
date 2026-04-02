# DSAR LangGraph Agent

A governed AI workflow for triaging Data Subject Access Requests (DSARs), built with LangGraph. Runs on a local LLM via Ollama or the OpenAI API.

---

## Why this exists

Under GDPR and similar regulations, organisations must respond to DSARs within strict deadlines. In practice, triage is slow, inconsistent, and handled manually: a person reads the request, works out what type it is, figures out which systems are involved, and flags anything legally complicated. That process is time-consuming, error-prone, and hard to audit.

This project automates that triage step while keeping a human in the loop before anything is finalised. Every decision is structured, every fallback is deterministic, and no action is taken without explicit approval.

---

## How it works

The workflow runs as a LangGraph state machine. Each stage reads from and writes to a shared state object, persisted across the human checkpoint.

```
DSAR request text
       │
       ▼
ClassificationAgent   →  access / deletion / portability / unknown
  ↑ episodic memory        (retrieves similar past approved decisions as precedents)
       │
       ▼
ScopingAgent          →  systems that need to be queried
  ↑ semantic memory        (authorised data inventory — must cite data categories)
       │
       ▼
RiskFlagAgent         →  third-party data, minors, conflicting legal basis
       │
       ▼
── interrupt() ──     →  execution pauses; human reviews full state + audit trail
       │
  approve / revise
       │
       ▼
Finalised output + episode persisted to store
```

**ClassificationAgent** categorises the request into one of three DSAR types. In LLM mode, it first retrieves up to 2 similar past approved decisions from the episode store and injects them as few-shot precedents.

**ScopingAgent** maps the request to the internal systems that need to be queried. In LLM mode, it consults the authorised data inventory and must cite the specific data category from the inventory that justifies each system's inclusion. It cannot reference systems outside the inventory.

**RiskFlagAgent** checks for complicating factors: third-party data, requests from minors, conflicting legal basis, unclear identity, or ambiguous scope.

**Human review checkpoint** pauses execution using LangGraph's `interrupt()` mechanism. The operator sees the full proposed decision, rationale, and reasoning history, then approves or returns it with correction feedback. If revised, the LLM agents re-run with the feedback injected into the prompt.

**Episode store** — on approval, the full triage episode (request, decisions, reasoning history) is persisted to the LangGraph store and becomes available as a precedent for future requests.

---

## Memory architecture

The system uses two types of long-term agent memory:

**Semantic memory** is the verified data inventory: the set of internal systems the agent is authorised to reference, with their data categories, descriptions, and retention periods. The ScopingAgent is grounded in this inventory and cannot fabricate systems outside it.

**Episodic memory** is a case history of past approved triage decisions. Before classifying a new request, the ClassificationAgent retrieves the most similar past episodes and uses them as examples — like a compliance team consulting their own precedents. Switching to a vector-backed store activates true semantic similarity search with no code changes.

---

## LLM + deterministic hybrid design

In **deterministic mode** (the default), all triage decisions are made by rule-based logic: fast, fully predictable, and auditable without any model dependency.

In **LLM mode**, an LLM improves classification and risk detection on ambiguous requests. If the LLM fails — timeout, invalid JSON, or schema validation failure — the system falls back to deterministic logic automatically and records the failure in a structured `LLMFallbackWarning`. No silent failures, no hallucinated decisions.

---

## Demo

```
$ python -m dsar_langgraph_agent --text "Delete my account" --openai --auto-approve

────────────────────  DSAR Triage Report — Review pass #1  ────────────────────

  Request Type    │ DELETION  (confidence: 100%)
  Rationale       │ The request explicitly states to delete the account.
  Scoped Systems  │ crm, production_sql
  Scope Rationale │ CRM holds the master identity record and account status.
                  │ Production SQL holds purchase history and order IDs.
  Risk Flags      │ ✓ No risk flags

  🤖 [1/3] classification_agent_llm → request_type=deletion, confidence=1.0
  🤖 [2/3] scoping_agent_llm        → systems=['crm', 'production_sql']
  🤖 [3/3] risk_flag_agent_llm      → flags=[]

Auto-approving…

  ✅  Final DSAR Triage Summary
  Request Type    │ DELETION  (confidence: 100%)
  Scoped Systems  │ crm, production_sql
  Risk Flags      │ ✓ None
  Audit Trail     │ 4 reasoning step(s) recorded
  Episode ID      │ b459b449-…  (persisted to store)
```

---

## Project structure

```
src/dsar_langgraph_agent/
├── triage_schemas.py      Pydantic schemas for all triage state
├── triage_agents.py       Deterministic agent implementations
├── triage_llm_agents.py   LLM-powered agents with deterministic fallback
├── ollama_client.py       OpenAI-compatible LLM client (Ollama or OpenAI API)
├── triage_graph.py        LangGraph workflow, memory stores, interrupt checkpoint
├── cli.py                 CLI runner with human review loop
└── data/
    └── data_inventory.json  Authorised system inventory (semantic memory)

tests/
├── test_triage_graph.py        Graph flow and interrupt/resume
├── test_llm_agent_fallback.py  Fallback logic for all three LLM agents
├── test_episodic_memory.py     Episode persistence and retrieval
├── test_episodic_retrieval.py  Precedent injection into classification
├── test_data_inventory.py      Inventory loading and scoping constraints
└── test_reasoning_history.py   Audit trail correctness
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
python -m dsar_langgraph_agent \
  --text "Please delete my account and remove my personal data."
```

**LLM mode — local (requires Ollama)**

```bash
ollama serve
ollama pull llama3.1

python -m dsar_langgraph_agent \
  --use-llm --model llama3.1 \
  --text "Please delete my account and remove my personal data."
```

**LLM mode — OpenAI API**

```bash
export OPENAI_API_KEY=your_key_here

python -m dsar_langgraph_agent \
  --openai \
  --text "Please delete my account and remove my personal data."
```

Use `--openai-model` to override the model (default: `gpt-4o-mini`). Use `--auto-approve` to skip the interactive human checkpoint.

---

## Tests

```bash
pytest -q
```

---

## What's next

- **Vector store backend** — swap `InMemoryStore` for a vector-backed store (e.g. `AsyncPostgresStore` with an embedder) to enable true semantic similarity search over past episodes. The retrieval code requires no changes.
- **RAG over policy documents** — ground the agents in your actual Records of Processing Activities and retention schedules, rather than a static inventory file.
- **API endpoint** — expose the workflow as a REST API for integration into a broader compliance platform.

Being built as part of preparation for the AMD × lablab.ai hackathon (AI Agents & Agentic Workflows track).

---

## Built with

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Ollama](https://ollama.com)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [Pydantic](https://docs.pydantic.dev)
