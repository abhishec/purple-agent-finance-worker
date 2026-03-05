# Finance AI Worker

> One of four **mini AI workers built on BrainOS** — the Reflexive Agent Architecture framework that achieved **3/3 (100%)** on τ²-Bench. Each worker is a lightweight, self-contained cognitive unit that runs the same PRIME → EXECUTE → REFLECT loop tuned to its domain.

**AgentX Phase 2 — Finance Agent Track**

---

An autonomous finance AI worker built on **BrainOS** that executes SEC filing analysis, options pricing, crypto trading decisions, Treasury QA, and portfolio optimization via a dynamic finite state machine, Autonomous Capability Engine (ACE), and a format-compliance post-processor tuned to the AgentX-AgentBeats Finance benchmark.

## Overview

Finance Mini AI Worker connects to an MCP tool server and synthesizes a task-specific FSM on every invocation — zero hardcoded process definitions. The Autonomous Capability Engine pre-warms domain capabilities in the background during the PRIME phase, so they are available before the primary solve begins. A format normalization pass runs last to guarantee benchmark-required JSON and XML output shapes survive through the MoA and self-reflection quality gates.

## Architecture

```
POST /  (A2A JSON-RPC 2.0)
        │
        ▼
    PRIME
    ├── Privacy guard              (zero API cost)
    ├── RL primer                  (past-task patterns)
    ├── Session context            (Haiku-compressed history)
    ├── Process type classification (Haiku + keyword fallback)
    ├── Dynamic FSM synthesis      (Haiku, cached after first encounter)
    ├── ACE pre-warming            (start_promises_for_task, background)
    ├── Format directive injection (finance_output_adapter, zero API cost)
    ├── Tool discovery             (MCP + local registry)
    ├── Tool gap detection         (LLM phase, ACE synthesizes missing tools)
    ├── Tool deduplication         (MCP + seeded + ACE overlap prevention)
    └── Knowledge + entities       (domain facts, cross-task memory)
        │
        ▼
    EXECUTE
    ├── UCB1 bandit selects strategy: fsm | five_phase | moa
    ├── await_promises()           (ACE pre-warmed capabilities available)
    ├── Finance FSM: RETRIEVE → PARSE → ANALYZE → VALIDATE → COMPLETE
    ├── Post-execution: compute verifier · numeric MoA · output validation
    │                   self-reflection · ACE gap observation (background)
    └── Format normalization pass  (Haiku re-wrap, LAST — after all quality gates)
        │
        ▼
    REFLECT
    ├── FSM checkpoint saved
    ├── Session memory compressed  (async, Haiku)
    ├── RL outcome recorded        → case_log.json
    ├── UCB1 bandit updated
    ├── ACE capability graph updated
    └── Knowledge extracted        (quality ≥ 0.5)
```

## Finance FSM States

| State | Purpose |
|---|---|
| `RETRIEVE` | Fetch external data — filings, prices, trade records, API responses |
| `PARSE` | Extract structured data from raw documents and API payloads |
| `ANALYZE` | Multi-step reasoning — trends, comparisons, classification, cross-reference |
| `VALIDATE` | Cross-check results against source, verify numerical consistency |
| `COMPUTE` | Arithmetic from collected data — Black-Scholes, amortization, portfolio math |
| `COMPLETE` | Final structured answer |

Complexity-adaptive depth: tasks scored 1–3 get 3 states; 4–6 get 5; 7–10 get 8. Parallel state groups (`[ANALYZE || COMPUTE]`) combine independent work into a single prompt.

## Key Innovations

### 1. Finance Output Adapter
Zero-API-cost format compliance layer. Detects the required output shape from the task text itself (every green agent embeds its format spec in the task) and injects a format directive into the system context before execution begins. Covers 10 formats:

| Format Key | Used By | Shape |
|---|---|---|
| `json_risk_classification` | Alpha-Cortex | `{"risk_classification": [...]}` |
| `json_business_summary` | Alpha-Cortex | `{"business_summary": {industry, products, geography}}` |
| `json_consistency_check` | Alpha-Cortex | `{"consistency_check": [...]}` |
| `json_trading_decision` | FAB++ / AgentBusters | `{"action", "size", "stop_loss", "take_profit", "reasoning", "confidence"}` |
| `json_cot_answer` | FAB++ / AgentBusters | `{"cot": "...", "answer": "..."}` |
| `json_options` | FAB++ | `{"result": {"price", "greeks", "assessment"}}` |
| `xml_final_answer` | OfficeQA | `<FINAL_ANSWER>\nvalue\n</FINAL_ANSWER>` |
| `csv_data_integration` | TPC-DI tasks | CSV with header row |
| `portfolio_allocation` | Portfolio tasks | `TICKER: XX% — reason` per line |
| `json_generic` | General JSON tasks | Template extracted from task text |

### 2. Format Normalization Pass (Fix 2)
Runs **after** all quality gates (MoA / self-reflection / compute verifier). Those passes can rewrite a correctly-formatted answer back into prose. The normalization pass:
- `_quick_format_check()` — marker-based fast path, zero API if already correct
- `xml_final_answer` — regex number extraction, zero API cost
- All JSON formats — Haiku call to re-extract into the exact required shape
- `json_generic` — template extracted dynamically from `"Return JSON: {...}"` in the task text

### 3. Autonomous Capability Engine (ACE)
Replaces static regex gap patterns entirely. The ACE:
- **CapabilityStore** — persistent registry with metadata (`capability_store.json`)
- **CapabilityGraph** — `task_type → capability` learned map (`capability_graph.json`)
- **CapabilityObserver** — scans LLM output for gap signals after each execution
- **CapabilityAcquirer** — full pipeline: formalize → library search → compose → synthesize → test
- **CapabilityPromise** — async non-blocking: `start()` fires in background, `await_promises()` collects before primary solve
- Library-aware sandbox: scipy / numpy / pandas / sympy detected at runtime

### 4. Dynamic FSM v2
`PROCESS_DEFINITIONS = {}` — all types synthesized via Haiku. Never blocks execution: synthesis failure falls back to `RETRIEVE → ANALYZE → COMPLETE`. FSM mutation rules handle runtime recovery:
- `COMPUTE_failed` → insert `RECOMPUTE`
- `RETRIEVE_timeout` → insert `USE_CACHED`
- `VALIDATE_low_confidence` → insert `CROSS_CHECK`

### 5. Tool Deduplication
Two dedup passes prevent `400 Tool names must be unique` from the Anthropic API:
- After PRIME assembles MCP + seeded + ACE-synthesized tools
- After `await_promises()` adds newly synthesized capabilities

## Component Reference

| Module | Role |
|---|---|
| `server.py` | FastAPI application; A2A JSON-RPC 2.0 handler |
| `worker_brain.py` | Core cognitive loop: PRIME / EXECUTE / REFLECT |
| `fsm_runner.py` | Finance FSM engine; parallel state groups; branching hints |
| `dynamic_fsm.py` | Haiku-based FSM synthesizer; complexity-adaptive depth; mutation rules |
| `finance_output_adapter.py` | Zero-API format detection and directive injection |
| `autonomous_capability_engine.py` | ACE: CapabilityStore, CapabilityGraph, CapabilityPromise |
| `smart_classifier.py` | 10 finance task types; Haiku classifier + keyword fallback |
| `dynamic_tools.py` | LLM gap detection; ACE-aware sandboxed synthesis; seeded finance tools |
| `claude_executor.py` | Agentic Claude execution loop (max 20 iterations, 25 tool calls) |
| `five_phase_executor.py` | PLAN → GATHER → SYNTHESIZE → ARTIFACT → INSIGHT executor |
| `self_moa.py` | Dual top-p mixture-of-agents with word-overlap consensus |
| `strategy_bandit.py` | UCB1 multi-armed bandit; learns win rate per strategy per process type |
| `token_budget.py` | 10K token budget; state-aware model selection |
| `compute_verifier.py` | COMPUTE-state arithmetic audit; correction pass before COMPLETE |
| `self_reflection.py` | Answer completeness scoring; improvement pass if below threshold |
| `output_validator.py` | Required-field checking per process type |
| `schema_adapter.py` | Schema drift resilience; 5-tier fuzzy column matching |
| `rl_loop.py` | Case log persistence; quality scoring; RL primer construction |
| `knowledge_extractor.py` | Post-task domain fact extraction; keyword-keyed retrieval |
| `entity_extractor.py` | Zero-cost regex entity tracking across tasks |
| `mcp_bridge.py` | MCP tool bridge; pre-flight parameter validation; schema patching |
| `session_context.py` | Multi-turn session state; FSM checkpoints; Haiku memory compression |
| `recovery_agent.py` | Tool failure recovery: synonym → decompose → Haiku advice → degrade |
| `privacy_guard.py` | PII and credential detection before any API call |
| `paginated_tools.py` | Cursor-loop bulk data fetching across all pagination styles |

## Requirements

Python 3.12+

```
fastapi>=0.115
uvicorn[standard]>=0.30
anthropic>=0.34
httpx>=0.27
pydantic>=2.0
boto3>=1.34
scipy>=1.11
numpy>=1.26
pandas>=2.0
sympy>=1.12
```

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Claude API key |
| `GREEN_AGENT_MCP_URL` | Yes | — | MCP tool server base URL |
| `FALLBACK_MODEL` | No | `claude-sonnet-4-6` | Default execution model |
| `TOOL_TIMEOUT` | No | `10` | Seconds per tool call |
| `TASK_TIMEOUT` | No | `120` | Seconds per task |

## Running

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
export GREEN_AGENT_MCP_URL=http://localhost:9009
python main.py --host 0.0.0.0 --port 9010
```

## Docker

```bash
docker pull public.ecr.aws/d9m7h3k5/agentbench-finance:latest
docker run -e ANTHROPIC_API_KEY=sk-ant-... \
           -e GREEN_AGENT_MCP_URL=http://localhost:9009 \
           -p 9010:9010 \
           public.ecr.aws/d9m7h3k5/agentbench-finance:latest
```

## API

All requests use A2A JSON-RPC 2.0.

**Endpoints**

| Endpoint | Method | Description |
|---|---|---|
| `/` | POST | `tasks/send` — submit a finance task |
| `/.well-known/agent-card.json` | GET | Agent capability declaration |
| `/health` | GET | Health check |
| `/rl/status` | GET | Case log, bandit state, ACE stats, FSM cache |
| `/training/status` | GET | Training data seeding status |
| `/training/sync` | POST | Trigger training data re-sync |

**Request format**

```json
{
  "jsonrpc": "2.0",
  "method": "tasks/send",
  "id": "task-001",
  "params": {
    "id": "task-001",
    "message": {
      "role": "user",
      "parts": [{ "text": "Calculate the Black-Scholes price of a European call option..." }]
    },
    "metadata": {
      "policy_doc": "",
      "tools_endpoint": "https://mcp.example.com",
      "session_id": "worker-abc"
    }
  }
}
```

## Simulation Results

Pre-benchmark simulation against all three green agents (10 tasks):

```
Tasks completed:   10/10
Format compliant:  10/10  ← judge sees this score
Avg time/task:     53s
```

| Agent | Task Type | Format | Result |
|---|---|---|---|
| OfficeQA | Treasury `<FINAL_ANSWER>` XML | xml_final_answer | ✓ |
| FAB++ | Black-Scholes options pricing | json_options | ✓ |
| FAB++ | Crypto trading decision | json_trading_decision | ✓ |
| FAB++ | Chain-of-thought financial reasoning | json_cot_answer | ✓ |
| Alpha-Cortex | SEC 10-K risk classification | json_risk_classification | ✓ |
| Alpha-Cortex | SEC 10-K business summary | json_business_summary | ✓ |
| Alpha-Cortex | SEC 10-K consistency check | json_consistency_check | ✓ |
| Portfolio | ETF retirement allocation | portfolio_allocation | ✓ |
| Calculator | Mortgage amortization JSON | json_generic | ✓ |

## Tech Stack

- **Runtime:** Python 3.12, FastAPI, uvicorn
- **LLM:** Anthropic Claude — Haiku for classification, synthesis, normalization; Sonnet for ANALYZE and COMPUTE
- **FSM:** Dynamic finance FSM with ACE capability synthesis
- **Format compliance:** Zero-API regex detection + Haiku normalization post-processor
- **Tool bridge:** MCP HTTP with pre-flight validation and schema drift correction
- **Numerics:** scipy / numpy / sympy in ACE-synthesized tool sandbox
- **RL:** UCB1 bandit + case log + quality scoring + knowledge extraction
- **Storage:** Local JSON (ACE store/graph, bandit state, entity memory, knowledge base, case log)

## License

Apache 2.0
