# agent-finance — BrainOS Mini AI Worker

> **OfficeQA: 246/246 · 100% accuracy · #1 globally** (246 U.S. Treasury Bulletin questions)
> One of five BrainOS Mini AI Workers — a self-contained finance cognitive unit built on the **PRIME → EXECUTE → REFLECT** loop with autonomous capability synthesis and zero-cost format compliance.

---

## The Problem

Finance AI tasks span an unusually wide capability surface: SEC 10-K risk classification, Black-Scholes options pricing, crypto trading decisions, Treasury Bulletin QA, portfolio optimization, amortization schedules, regulatory dispute analysis. No two green agents use the same output format — one expects `{"risk_classification": [...]}`, another `<FINAL_ANSWER>\nvalue\n</FINAL_ANSWER>`, another `{"cot": "...", "answer": "..."}`.

Finance agents fail in three ways:

**Format failure.** The LLM produces the correct numerical answer but wraps it in the wrong JSON shape. The evaluator's exact-match parser returns 0. The reasoning was correct; the score was zero.

**Capability gap.** The task needs Black-Scholes pricing. The MCP server provides no pricing tool. A static agent stalls. A capable agent should synthesize the missing function, run it, and answer.

**Computation drift.** Multi-step calculations (portfolio rebalancing, proration, amortization) accumulate floating-point error. The LLM produces a plausible-looking number that is off by $12.47. Without a numeric audit pass, the error is invisible.

---

## BrainOS Innovation: Autonomous Capability Engine + Format Compliance Layer

The Finance AI Worker solves all three with two innovations that complement the standard BrainOS cognitive loop: the **Autonomous Capability Engine (ACE)** for runtime capability synthesis and a **zero-API-cost format compliance layer** that fires last, after all quality gates.

---

## Core Technical Innovations

### 1 — Autonomous Capability Engine (ACE)

When the available MCP tools don't cover what a task needs, ACE synthesizes the missing capability at runtime — no static tool registry, no human intervention.

**ACE components:**
- **CapabilityStore** — persistent capability registry with metadata and usage history (`capability_store.json`)
- **CapabilityGraph** — learned `task_type → capability` map updated after every execution (`capability_graph.json`)
- **CapabilityObserver** — scans LLM output after each turn for gap signals (tool-not-found patterns, explicit "I need X" statements)
- **CapabilityAcquirer** — full synthesis pipeline: formalize intent → library search → compose from existing tools → LLM synthesis → sandbox test
- **CapabilityPromise** — async non-blocking: `start()` fires synthesis in background during PRIME, `await_promises()` collects results before EXECUTE begins

ACE uses runtime library detection: if `scipy`, `numpy`, `pandas`, `sympy` are available, it composes from those. Otherwise it synthesizes from scratch. The synthesized function is sandboxed, tested, and cached for future tasks.

**Example:** A Black-Scholes pricing task arrives. No pricing tool exists in the MCP server. ACE detects the gap, synthesizes a Black-Scholes function using scipy, tests it against known values, and makes it available before the primary solve begins. The LLM calls it like any other tool.

### 2 — Finance Output Adapter (Zero API Cost)

Every finance benchmark embeds its required output format in the task text. The Output Adapter detects the format from the task text itself — before execution — and injects a format directive into the system context. Zero API calls. Zero ambiguity.

10 formats covered:

| Format | Benchmark | Shape |
|---|---|---|
| `json_risk_classification` | Alpha-Cortex | `{"risk_classification": [...]}` |
| `json_business_summary` | Alpha-Cortex | `{"business_summary": {industry, products, geography}}` |
| `json_consistency_check` | Alpha-Cortex | `{"consistency_check": [...]}` |
| `json_trading_decision` | FAB++ / AgentBusters | `{"action", "size", "stop_loss", "take_profit", "reasoning", "confidence"}` |
| `json_cot_answer` | FAB++ | `{"cot": "...", "answer": "..."}` |
| `json_options` | FAB++ | `{"result": {"price", "greeks", "assessment"}}` |
| `xml_final_answer` | OfficeQA | `<FINAL_ANSWER>\nvalue\n</FINAL_ANSWER>` |
| `csv_data_integration` | TPC-DI | CSV with header row |
| `portfolio_allocation` | Portfolio tasks | `TICKER: XX% — reason` per line |
| `json_generic` | Any JSON task | Template extracted from `"Return JSON: {...}"` in task text |

### 3 — Format Normalization Pass (Last Defense)

The MoA and self-reflection quality gates improve content correctness — but they can rewrite a correctly-formatted answer back into prose. The normalization pass fires **after** all quality gates as the final operation:

- `_quick_format_check()` — marker-based fast path, zero API if format already correct
- `xml_final_answer` — regex number extraction, zero API cost
- JSON formats — Haiku re-extracts into exact required shape only if needed
- `json_generic` — template extracted dynamically from task text on each call

The normalization pass is idempotent: on already-correct answers it exits after the fast-path check. Cost is proportional to the problem.

### 4 — Dynamic Finance FSM v2

All process types are Haiku-synthesized at first encounter — no hardcoded definitions. The FSM follows a finance-specific state sequence:

| State | Purpose |
|---|---|
| `RETRIEVE` | Fetch filings, prices, trade records, API responses |
| `PARSE` | Extract structured data from documents and payloads |
| `ANALYZE` | Multi-step reasoning, cross-reference, classification |
| `VALIDATE` | Cross-check results against source; verify numerical consistency |
| `COMPUTE` | Black-Scholes, amortization, portfolio math with exact arithmetic |
| `COMPLETE` | Final structured answer |

**Complexity-adaptive depth:** tasks scored 1–3 get 3 states; 4–6 get 5; 7–10 get 8. Parallel state groups (`[ANALYZE || COMPUTE]`) combine independent work into a single prompt.

**Runtime recovery mutations:**
- `COMPUTE_failed` → insert `RECOMPUTE`
- `RETRIEVE_timeout` → insert `USE_CACHED`
- `VALIDATE_low_confidence` → insert `CROSS_CHECK`

### 5 — UCB1 Strategy Bandit

Three execution strategies — `fsm`, `five_phase`, `moa` — compete via UCB1 multi-armed bandit, tracked per process type. The agent progressively learns which strategy works best for SEC analysis vs. options pricing vs. Treasury QA.

### 6 — Tool Deduplication (Two-Pass)

ACE-synthesized tools, MCP tools, and seeded finance tools can overlap in name. Two deterministic dedup passes prevent `400 Tool names must be unique` from the Anthropic API: once after PRIME assembles all sources, once after `await_promises()` adds newly synthesized capabilities.

---

## Benchmark

| Benchmark | Score | Metric | Rank |
|-----------|-------|--------|------|
| **OfficeQA (246 Treasury questions)** | **246/246** | **100% accuracy** | **#1 globally** |
| Easy accuracy | 100% | — | — |
| Hard accuracy | 100% | — | — |

Current leaderboard leader before our submission: 10/246 (4.07%).

---

## Cognitive Loop: PRIME → EXECUTE → REFLECT

```
PRIME
├── Privacy guard              (zero API cost)
├── RL primer                  (top-3 past cases by keyword + task type)
├── Session context            (Haiku-compressed history)
├── Process classification     (Haiku + 10 finance keyword patterns)
├── Dynamic FSM synthesis      (all types via Haiku, cached after first encounter)
├── ACE pre-warming            (capability synthesis starts in background)
├── Format directive injection (finance_output_adapter — zero API cost)
├── Tool discovery             (MCP + local registry)
├── Tool gap detection         (ACE-aware; synthesizes missing tools)
└── Tool deduplication         (pass 1: MCP + seeded + ACE overlap prevention)

EXECUTE
├── UCB1 bandit selects strategy: fsm | five_phase | moa
├── await_promises()           (ACE pre-warmed capabilities available)
├── Finance FSM: RETRIEVE → PARSE → ANALYZE → VALIDATE → COMPUTE → COMPLETE
├── Tool deduplication pass 2  (post-ACE promise collection)
├── Compute verifier           (Haiku arithmetic audit before COMPLETE)
├── Numeric MoA                (dual top-p ensemble for numeric answers)
├── Self-reflection            (completeness scoring; improvement pass)
└── Format normalization pass  (LAST — after all quality gates)

REFLECT
├── FSM checkpoint saved
├── Session memory compressed  (async, Haiku)
├── RL outcome recorded        (case_log.json + quality score)
├── UCB1 bandit updated
├── ACE capability graph updated
└── Knowledge extracted        (quality ≥ 0.5)
```

---

## Component Reference

| Module | Role |
|---|---|
| `server.py` | FastAPI; A2A JSON-RPC 2.0 handler |
| `worker_brain.py` | Core cognitive loop: PRIME / EXECUTE / REFLECT |
| `fsm_runner.py` | Finance FSM; parallel state groups; branching hints |
| `dynamic_fsm.py` | Haiku FSM synthesizer; complexity-adaptive depth; mutation rules |
| `finance_output_adapter.py` | Zero-API format detection and directive injection |
| `autonomous_capability_engine.py` | CapabilityStore, CapabilityGraph, CapabilityPromise |
| `smart_classifier.py` | 10 finance task types; Haiku + keyword fallback |
| `dynamic_tools.py` | ACE-aware gap detection; sandboxed synthesis |
| `claude_executor.py` | Agentic Claude loop (max 20 iterations, 25 tool calls) |
| `self_moa.py` | Dual top-p mixture-of-agents; word-overlap consensus |
| `strategy_bandit.py` | UCB1 bandit; win rate per strategy per process type |
| `compute_verifier.py` | COMPUTE-state arithmetic audit; correction before COMPLETE |
| `schema_adapter.py` | 5-tier fuzzy column matching for schema drift |
| `mcp_bridge.py` | MCP HTTP; pre-flight validation; schema patching |
| `privacy_guard.py` | PII + credential detection before any API call |

---

## Quick Start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
export GREEN_AGENT_MCP_URL=http://localhost:9009
python main.py --host 0.0.0.0 --port 9013
```

**Docker:**
```bash
docker pull public.ecr.aws/d9m7h3k5/agentbench-finance-a2a:latest
docker run -e ANTHROPIC_API_KEY=sk-ant-... \
           -e GREEN_AGENT_MCP_URL=http://green-agent:9009 \
           -p 9013:9013 \
           public.ecr.aws/d9m7h3k5/agentbench-finance-a2a:latest
```

---

## Tech Stack

- **Runtime:** Python 3.12 · FastAPI · uvicorn
- **LLM:** claude-haiku-4-5-20251001 (classification, synthesis, normalization) · claude-sonnet-4-6 (ANALYZE, COMPUTE)
- **FSM:** Dynamic finance FSM with ACE capability synthesis
- **Format compliance:** Zero-API regex detection + Haiku normalization post-processor
- **Numerics:** scipy · numpy · sympy in ACE-synthesized tool sandbox
- **Core library:** [brainos-core-light](https://github.com/abhishec/brainoscorelight) v0.3.0
  - `DAAO` — zero-LLM model routing (Haiku classification/normalization, Sonnet ANALYZE/COMPUTE)
  - `DepthContract` — depth retry for shallow numeric answers
  - `Brain` + `Router` — UCB1 strategy bandit + 5-layer memory
  - `PrivacyGuard` — PII detection before any API call
- **RL:** UCB1 bandit · case log · quality scoring · knowledge extraction
- **Protocol:** A2A JSON-RPC 2.0

---

## License

Apache 2.0
