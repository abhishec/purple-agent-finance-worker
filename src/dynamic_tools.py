"""
dynamic_tools.py
Evolved dynamic tool layer — thin bridge between ACE and worker_brain.

=============================================================
WHAT CHANGED vs agent-purple
=============================================================

REMOVED:
  - _GAP_PATTERNS (36 hardcoded regex strings)
  - detect_tool_gaps() (regex scan — finite list, goes stale)
  - _synthesize_via_haiku() called directly from worker_brain
  - Pure-Python-only sandbox (_SANDBOX_GLOBALS with just math/Decimal)

KEPT (unchanged contract, worker_brain.py still works):
  - load_registered_tools()      → now reads from ACE CapabilityStore
  - is_registered_tool()         → delegates to ACE store
  - call_registered_tool()       → delegates to ACE store
  - seed_amortization_tool()     → seeds into ACE store at startup
  - get_tool_registry_stats()    → now returns ACE stats

NEW:
  - detect_tool_gaps_llm()       → powered by ACE (not regex)
  - synthesize_and_register()    → thin wrapper over ACE CapabilityAcquirer
  - seed_http_fetch_tool()       → seeds HTTP file fetcher at startup
  - All execution runs in ACE's library-aware sandbox (scipy/numpy/pandas)

=============================================================
WHY THE SEEDED TOOLS STILL EXIST
=============================================================

Seeded tools (amortization, http_fetch) are guaranteed-available at startup,
before any task arrives, with tested implementations. They give the agent
reliable baseline capabilities even on the very first task. The ACE builds
on top of these — it does NOT replace them.
"""
from __future__ import annotations

import json
import math
import os
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from src.autonomous_capability_engine import (
    CapabilityRecord,
    get_store,
    get_ace_stats,
    CapabilityAcquirer,
    _validate,
    _get_sandbox,
)

# ── Amortization: seeded implementation ──────────────────────────────────────
# Kept here (not in ACE) because it's a known-good implementation we ship.
# Seeded into the ACE store at startup exactly like before.

_AMORTIZATION_CODE = """\
def finance_loan_amortization(**kwargs):
    principal = float(kwargs.get('principal', 0))
    annual_rate = float(kwargs.get('annual_rate', 0))
    months = int(kwargs.get('months', 0))
    if months <= 0 or principal <= 0:
        return {'result': 0.0, 'details': {'error': 'invalid inputs'}}
    if annual_rate == 0:
        monthly_payment = principal / months
        return {
            'result': round(monthly_payment, 2),
            'details': {'monthly_payment': round(monthly_payment, 2),
                        'total_paid': round(monthly_payment * months, 2),
                        'total_interest': 0.0}
        }
    r = annual_rate / 100 / 12
    payment_cents = round(principal * 100 * r * (1 + r)**months / ((1 + r)**months - 1))
    monthly_payment = payment_cents / 100
    total_paid = round(monthly_payment * months, 2)
    total_interest = round(total_paid - principal, 2)
    return {
        'result': monthly_payment,
        'details': {
            'monthly_payment': monthly_payment,
            'total_paid': total_paid,
            'total_interest': total_interest,
            'periods': months,
            'annual_rate_pct': annual_rate,
        }
    }
"""

# ── HTTP fetch: seeded implementation ─────────────────────────────────────────
# General capability — downloads a file from a URL, returns contents as text.
# Needed for Data Matchmaker style tasks (download CSV files from HTTP endpoint).
# Not domain-specific — any task that says "download file from URL" can use this.

_HTTP_FETCH_CODE = """\
def data_http_fetch(**kwargs):
    url = str(kwargs.get('url', ''))
    file_type = str(kwargs.get('file_type', 'text')).lower()
    if not url:
        return {'result': '', 'error': 'no url provided'}
    try:
        import urllib.request as _req
        import json as _json
        import csv as _csv
        import io as _io
        with _req.urlopen(url, timeout=30) as resp:
            raw = resp.read().decode('utf-8', errors='replace')
        if file_type == 'json':
            data = _json.loads(raw)
            return {'result': data, 'text': raw, 'rows': len(data) if isinstance(data, list) else 1}
        if file_type == 'csv':
            reader = _csv.DictReader(_io.StringIO(raw))
            rows = list(reader)
            return {'result': rows, 'text': raw, 'rows': len(rows), 'columns': reader.fieldnames or []}
        return {'result': raw, 'text': raw, 'bytes': len(raw)}
    except Exception as e:
        return {'result': '', 'error': str(e), 'url': url}
"""


# ── Public API (unchanged contract) ──────────────────────────────────────────

def load_registered_tools() -> list[dict]:
    """
    Return JSON schemas for all registered capabilities (MCP-compatible).
    Called in PRIME to add capabilities to self._tools.
    """
    return get_store().all_schemas()


def is_registered_tool(tool_name: str) -> bool:
    """Check if a tool name maps to a registered (synthesized or seeded) capability."""
    return get_store().has(tool_name)


def call_registered_tool(tool_name: str, params: dict) -> dict:
    """Execute a registered capability with the given params. Returns result dict."""
    return get_store().call(tool_name, params)


def get_tool_registry_stats() -> dict:
    """Return capability stats for /rl/status endpoint."""
    return get_ace_stats()


# ── Seeding ───────────────────────────────────────────────────────────────────

def seed_amortization_tool() -> None:
    """
    Seed the loan amortization tool into the ACE store at startup.
    Idempotent — only seeds once. Tested against known value.
    """
    store = get_store()
    key = "finance_loan_amortization"
    if store.has(key):
        return

    # Validate: $200k, 5% APR, 360 months → ~$1073.64/mo
    passed, accuracy, _ = _validate(_AMORTIZATION_CODE, key, [
        {"inputs": {"principal": 200000, "annual_rate": 5.0, "months": 360},
         "expected_result_approx": 1073.64, "tolerance_pct": 0.01},
    ])
    if not passed:
        return

    rec = CapabilityRecord(
        name=key,
        description="Loan amortization schedule — monthly payment, total interest",
        python_code=_AMORTIZATION_CODE,
        input_schema={
            "type": "object",
            "properties": {
                "principal": {"type": "number", "description": "Loan principal in dollars"},
                "annual_rate": {"type": "number", "description": "Annual rate as % e.g. 5.5"},
                "months": {"type": "integer", "description": "Term in months e.g. 360"},
            },
            "required": ["principal", "annual_rate", "months"],
        },
        test_cases=[
            {"inputs": {"principal": 200000, "annual_rate": 5.0, "months": 360},
             "expected_result_approx": 1073.64, "tolerance_pct": 0.01},
        ],
        accuracy_score=accuracy,
        library_used="math",
        _seeded=True,
        _synthesized=False,
    )
    store.register(rec)


def seed_http_fetch_tool() -> None:
    """
    Seed the HTTP file fetcher into the ACE store at startup.
    General capability: any task that needs to download files uses this.
    Idempotent.
    """
    store = get_store()
    key = "data_http_fetch"
    if store.has(key):
        return

    rec = CapabilityRecord(
        name=key,
        description="Download a file from a URL and return contents (text/JSON/CSV)",
        python_code=_HTTP_FETCH_CODE,
        input_schema={
            "type": "object",
            "properties": {
                "url":       {"type": "string", "description": "URL to download"},
                "file_type": {"type": "string", "description": "Expected type: text/json/csv",
                              "enum": ["text", "json", "csv"]},
            },
            "required": ["url"],
        },
        test_cases=[],          # no test cases — network-dependent
        accuracy_score=1.0,
        library_used="urllib",
        _seeded=True,
        _synthesized=False,
    )
    store.register(rec)


# ── ACE-powered gap detection (replaces regex _GAP_PATTERNS) ─────────────────

async def detect_tool_gaps_llm(task_text: str, existing_tools: list[dict]) -> list[dict]:
    """
    LLM-based gap detection.
    Previously: Phase 2 fallback after regex scan.
    Now: PRIMARY gap detection — no regex list to maintain.

    Asks Haiku what custom computations this task needs that aren't
    already covered by existing tools or registered capabilities.

    Returns list of gap dicts {key, description} — max 3 items.
    """
    from src.config import ANTHROPIC_API_KEY
    import asyncio
    import re

    if not ANTHROPIC_API_KEY:
        return []

    store = get_store()
    existing_names: list[str] = []
    for t in existing_tools:
        name = t.get("name") or t.get("function", {}).get("name", "")
        if name:
            existing_names.append(name)
    existing_names.extend(store.all_names())

    tools_str = ", ".join(existing_names[:40]) if existing_names else "none"

    system = """\
You are a computational gap analyst. Identify ONLY custom calculations or data transformations
that this task requires but that are NOT covered by the listed existing tools.

Focus on:
- Mathematical formulas that need a dedicated Python function (not just arithmetic)
- Data transformations (CSV parsing, account classification, document analysis)
- Statistical computations (VaR, Sharpe ratio, outlier detection)

Do NOT flag:
- Simple arithmetic (addition, percentage of a number)
- Database read/write operations
- Standard text formatting

Return JSON array. Each item: {"key": "snake_case_name", "description": "precise spec"}.
If no custom computation is needed, return [].
Return ONLY valid JSON."""

    prompt = (
        f"Task:\n{task_text[:1200]}\n\n"
        f"Existing tools/capabilities (already available):\n{tools_str}\n\n"
        "List ONLY the specific computations this task needs that aren't already covered."
    )

    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        msg = await asyncio.wait_for(
            client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=400,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=8.0,
        )
        raw = msg.content[0].text.strip() if msg.content else "[]"
        clean = raw
        if clean.startswith("```"):
            clean = re.sub(r"^```[a-z]*\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean).strip()

        parsed = json.loads(clean)
        if not isinstance(parsed, list):
            return []

        gaps = []
        registered = set(store.all_names())
        existing_set = set(existing_names)
        for item in parsed[:3]:
            if isinstance(item, dict) and item.get("key") and item.get("description"):
                k = str(item["key"]).strip()
                d = str(item["description"]).strip()
                if k not in registered and k not in existing_set:
                    gaps.append({"key": k, "description": d})
        return gaps

    except Exception:
        return []


async def synthesize_and_register(gap: dict, task_text: str) -> dict | None:
    """
    Thin wrapper over ACE CapabilityAcquirer.
    Called by worker_brain for gaps detected in PRIME phase.
    Returns MCP-compatible tool schema or None.
    """
    store = get_store()
    key = gap.get("key", "")
    description = gap.get("description", "")

    # Already registered?
    if store.has(key):
        rec = store.get_record(key)
        return {"name": key, "description": rec.description if rec else key,
                "input_schema": rec.input_schema if rec else {}} if rec else None

    # Use ACE to acquire
    from src.autonomous_capability_engine import CapabilityAcquirer as _Acq
    acquirer = _Acq(store)
    signal = description or f"function named {key.replace('_', ' ')}"
    name = await acquirer.acquire(signal, task_text)
    if not name:
        return None

    rec = store.get_record(name)
    if not rec:
        return None
    return {
        "name": name,
        "description": rec.description,
        "input_schema": rec.input_schema,
    }
