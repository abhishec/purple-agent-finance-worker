from __future__ import annotations
import json
import os
import time
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.worker_brain import run_worker
from src.training_loader import seed_from_training_data, is_stale
from src.dynamic_fsm import get_synthesis_stats
from src.dynamic_tools import seed_amortization_tool, seed_http_fetch_tool, get_tool_registry_stats
from src.autonomous_capability_engine import get_ace_stats
from src.strategy_bandit import get_stats as get_bandit_stats
from src.report_analyzer import analyze_and_save, load_intelligence
from src.context_rl import get_context_stats

app = FastAPI(title="Finance Mini AI Worker", version="1.0.0")

AGENT_CARD = {
    "name": "Finance Mini AI Worker",
    "description": (
        "Finance-domain AI Worker for AgentX-AgentBeats Phase 2 Finance Track. "
        "Zero hardcoded tools — Autonomous Capability Engine (ACE) synthesizes math "
        "capabilities on demand using scipy/numpy/pandas. Fully dynamic FSM synthesized "
        "per task type via Haiku. Handles: SEC 10-K filings, Black-Scholes options "
        "pricing, crypto trading decisions, US Treasury Q&A, portfolio optimization, "
        "risk classification, and data integration."
    ),
    "version": "1.0.0",
    "url": os.getenv("PURPLE_AGENT_CARD_URL", "https://finance.agentbench.usebrainos.com"),
    "capabilities": {"streaming": False, "tools": True},
    "skills": [{
        "id": "finance-analysis",
        "name": "Finance AI Worker",
        "description": (
            "End-to-end finance task execution: SEC 10-K filing analysis (risk/summary/consistency), "
            "options pricing (Black-Scholes, Greeks), crypto trading decisions (BUY/SELL/HOLD), "
            "US Treasury Bulletin Q&A, portfolio optimization (Sharpe/MPT), risk classification, "
            "data integration (TPC-DI/CSV), and general financial calculations."
        ),
    }],
}


@app.on_event("startup")
async def on_startup():
    """
    Startup: seed tools + training data.
    - Amortization + HTTP fetch seeded into ACE store (guaranteed available)
    - Training data seeded from S3/HTTP (non-blocking background thread)
    """
    seed_amortization_tool()
    seed_http_fetch_tool()

    import threading
    def _seed():
        try:
            seed_from_training_data(force=False)
        except Exception:
            pass  # Never crash startup
    threading.Thread(target=_seed, daemon=True).start()


@app.get("/.well-known/agent-card.json")
async def agent_card():
    return JSONResponse(AGENT_CARD)


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "finance-mini-ai-worker", "version": "1.0.0"}


@app.post("/")
async def a2a_handler(request: Request):
    body = await request.json()

    # JSON-RPC 2.0
    jsonrpc_id = body.get("id")

    if body.get("method") != "tasks/send":
        return JSONResponse({
            "jsonrpc": "2.0", "id": jsonrpc_id,
            "error": {"code": -32601, "message": "Method not found"},
        })

    params = body.get("params", {})
    task_id = params.get("id", str(uuid.uuid4()))
    message = params.get("message", {})
    metadata = params.get("metadata", {})

    task_text = "".join(p.get("text", "") for p in message.get("parts", []))
    policy_doc = metadata.get("policy_doc", "")
    tools_endpoint = metadata.get("tools_endpoint", "")
    session_id = metadata.get("session_id", task_id)

    try:
        answer = await run_worker(
            task_text=task_text,
            policy_doc=policy_doc,
            tools_endpoint=tools_endpoint,
            task_id=task_id,
            session_id=session_id,
        )
    except Exception as exc:
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": jsonrpc_id,
            "error": {"code": -32603, "message": f"Internal error: {exc}", "data": None},
        })

    return JSONResponse({
        "jsonrpc": "2.0",
        "id": jsonrpc_id,
        "result": {
            "id": task_id,
            "status": {"state": "completed"},
            "artifacts": [{"parts": [{"text": answer}]}],
        },
    })


@app.get("/rl/status")
async def rl_status():
    """
    Comprehensive status endpoint: RL loop, ACE, FSM synthesis, bandit, KB.
    """
    base_dir = os.path.join(os.path.dirname(__file__), "..")

    # Case log stats
    case_log_path = os.path.join(base_dir, "case_log.json")
    case_stats: dict = {"total": 0, "successes": 0, "failures": 0, "avg_quality": 0.0}
    try:
        if os.path.exists(case_log_path):
            with open(case_log_path) as f:
                cases = json.load(f)
            if cases:
                qualities = [c.get("quality", 0) for c in cases]
                case_stats = {
                    "total": len(cases),
                    "successes": sum(1 for c in cases if c.get("outcome") == "success"),
                    "failures": sum(1 for c in cases if c.get("outcome") == "failure"),
                    "avg_quality": round(sum(qualities) / len(qualities), 3),
                }
    except Exception:
        pass

    # Knowledge base stats
    kb_path = os.path.join(base_dir, "knowledge_base.json")
    kb_stats: dict = {"total_entries": 0, "domains_covered": []}
    try:
        if os.path.exists(kb_path):
            with open(kb_path) as f:
                kb_entries = json.load(f)
            domains = list({e.get("domain", "unknown") for e in kb_entries})
            kb_stats = {
                "total_entries": len(kb_entries),
                "domains_covered": sorted(domains),
            }
    except Exception:
        pass

    # Context RL stats
    ctx_stats: dict = {}
    try:
        ctx_stats = get_context_stats()
    except Exception:
        pass

    return {
        "status": "ok",
        "total_cases": case_stats.get("total", 0),
        "avg_quality": case_stats.get("avg_quality", 0.0),
        "case_log": case_stats,
        "knowledge_base": kb_stats,
        "context_rl": ctx_stats,
        "dynamic_fsm": get_synthesis_stats(),
        "dynamic_tools": get_tool_registry_stats(),
        "autonomous_capability_engine": get_ace_stats(),
        "strategy_bandit": get_bandit_stats(),
    }


@app.get("/training/status")
async def training_status():
    """S3 training seed status."""
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    seeded_marker = os.path.join(base_dir, ".training_seeded")
    stale = True
    try:
        stale = is_stale()
    except Exception:
        pass
    intel: dict = {}
    try:
        intel = load_intelligence()
    except Exception:
        pass
    return {
        "status": "ok",
        "seeded": os.path.exists(seeded_marker),
        "stale": stale,
        "benchmark_intelligence": intel,
    }


@app.post("/training/sync")
async def training_sync():
    """Force refresh from S3 / HTTP benchmark endpoint."""
    results: dict = {}
    try:
        results["seed"] = seed_from_training_data(force=True)
    except Exception as e:
        results["seed_error"] = str(e)
    try:
        results["analyze"] = analyze_and_save(force=True)
    except Exception as e:
        results["analyze_error"] = str(e)
    results["status"] = "ok" if not any("error" in k for k in results) else "partial"
    return results
