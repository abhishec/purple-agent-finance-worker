"""
run_tests.py — Comprehensive test suite for Finance Mini AI Worker
Covers: format detection, FSM synthesis, classifiers, tools, worker brain helpers
Usage: python3 run_tests.py
"""
from __future__ import annotations
import asyncio
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(__file__))

# ── colour helpers ─────────────────────────────────────────────────────────────
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; C = "\033[96m"; RESET = "\033[0m"; BOLD = "\033[1m"
def ok(msg):   print(f"  {G}✓{RESET} {msg}")
def fail(msg): print(f"  {R}✗ FAIL: {msg}{RESET}")
def hr():      print(f"{Y}{'─'*70}{RESET}")

passed = 0
failed = 0
sections: list[tuple[str, int, int]] = []

def section(name: str):
    print(f"\n{BOLD}{C}{name}{RESET}")
    hr()

def expect(condition: bool, label: str, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1; ok(label)
    else:
        failed += 1; fail(f"{label}  →  {detail}" if detail else label)

# ══════════════════════════════════════════════════════════════════════════════
# 1. FINANCE OUTPUT ADAPTER
# ══════════════════════════════════════════════════════════════════════════════
section("1. Finance Output Adapter — format detection")

from src.finance_output_adapter import detect_output_format, build_format_directive, get_format_name

# risk classification
k = detect_output_format('Return JSON: {"risk_classification": ["Market Risk"]}')
expect(k == "json_risk_classification", "detects risk_classification JSON", k)

# business summary
k = detect_output_format('Return JSON: {"business_summary": {"industry": "...", "products": "...", "geography": "..."}}')
expect(k == "json_business_summary", "detects business_summary JSON", k)

# consistency check
k = detect_output_format('Return JSON: {"consistency_check": [...]} Which risks ARE discussed in Section 7')
expect(k == "json_consistency_check", "detects consistency_check JSON", k)

# crypto trading
k = detect_output_format('BTCUSDT ohlcv data: {"action": "BUY", "size": 0.1}')
expect(k == "json_trading_decision", "detects trading_decision JSON", k)

# cot + answer
k = detect_output_format('Solve this problem. Return JSON with {"cot": "...", "answer": "..."}')
expect(k == "json_cot_answer", "detects cot_answer JSON", k)

# options pricing
k = detect_output_format('Calculate the Black-Scholes price for this call option with strike 150')
expect(k == "json_options", "detects options pricing", k)

# treasury / XML final answer
k = detect_output_format('Using the Treasury Bulletin, what was total public debt outstanding?')
expect(k == "xml_final_answer", "detects xml_final_answer (Treasury)", k)

# FINAL_ANSWER tag explicit
k = detect_output_format('Answer the question. <FINAL_ANSWER> format required')
expect(k == "xml_final_answer", "detects FINAL_ANSWER tag", k)

# CSV / data integration
k = detect_output_format('TPC-DI data integration task. Return CSV format with customer_id columns')
expect(k == "csv_data_integration", "detects csv_data_integration", k)

# portfolio allocation
k = detect_output_format('Allocate this retirement portfolio among VTI, VXUS and BND tickers in percent')
expect(k == "portfolio_allocation", "detects portfolio_allocation", k)

# generic JSON fallback
k = detect_output_format('Calculate earnings per share and Return JSON with the result')
expect(k == "json_generic", "detects json_generic fallback", k)

# no format
k = detect_output_format('What is the capital of France?')
expect(k is None, "returns None for non-finance text", str(k))

# directives not empty
d = build_format_directive("json_risk_classification")
expect("risk_classification" in d, "directive contains key field", d[:60])
expect(len(d) > 50, "directive is non-trivial length")

# human name lookup
expect(get_format_name("json_cot_answer") == "CoT + Answer JSON", "get_format_name works")
expect(get_format_name("unknown_key") == "unknown_key", "get_format_name fallback")

# ══════════════════════════════════════════════════════════════════════════════
# 2. SMART CLASSIFIER — keyword fallback (finance-specific types)
# ══════════════════════════════════════════════════════════════════════════════
section("2. Smart Classifier — keyword fallback (finance types)")

from src.smart_classifier import _keyword_fallback

# Finance agent has finance-domain process types, not BP types
cases = [
    ("Analyze the 10-K SEC filing risk classification categories", "sec_filing_analysis"),
    ("Calculate Black-Scholes option price delta gamma strike expir", "options_trading"),
    ("BTCUSDT trading decision stop_loss take_profit action BUY", "crypto_trading"),
    ("Treasury Bulletin fiscal year public debt total expenditure", "treasury_qa"),
    ("Allocate retirement portfolio VTI VXUS BND tickers percent", "portfolio_optimization"),
    ("market risk operational risk cybersecurity risk VAR stress test", "risk_classification"),
    ("TPC-DI CSV data integration customer_id columns join", "data_integration"),
    ("Calculate amortization NPV IRR compound interest present value", "financial_calculation"),
    ("compliance regulatory KYC AML SOX GDPR audit report", "compliance_reporting"),
    ("What is the weather like today in Paris?", "general"),
]

for text, expected in cases:
    result = _keyword_fallback(text)
    expect(result == expected, f"finance keyword '{expected}'", f"got '{result}'")

# ══════════════════════════════════════════════════════════════════════════════
# 3. DYNAMIC FSM — valid states and synthesis helpers
# ══════════════════════════════════════════════════════════════════════════════
section("3. Dynamic FSM — structure and caching")

from src.dynamic_fsm import _VALID_STATES, _cache as _FSM_CACHE, FSM_MUTATION_RULES, synthesize_if_needed, get_synthesis_stats

expect(isinstance(_VALID_STATES, frozenset), "_VALID_STATES is a frozenset")
expect(len(_VALID_STATES) >= 8, f"at least 8 valid states, got {len(_VALID_STATES)}")
expect("RETRIEVE" in _VALID_STATES, "RETRIEVE in valid states")
expect("ANALYZE" in _VALID_STATES, "ANALYZE in valid states")
expect("VALIDATE" in _VALID_STATES, "VALIDATE in valid states")
expect("COMPLETE" in _VALID_STATES, "COMPLETE in valid states")
expect("PARSE" in _VALID_STATES, "PARSE in valid states (finance domain)")
expect("COMPUTE" in _VALID_STATES, "COMPUTE in valid states")

expect(isinstance(FSM_MUTATION_RULES, dict), "FSM_MUTATION_RULES is a dict")
expect(len(FSM_MUTATION_RULES) >= 2, f"at least 2 mutation rules, got {len(FSM_MUTATION_RULES)}")

expect(isinstance(_FSM_CACHE, dict), "_FSM_CACHE is a dict (possibly empty at start)")

# FSM cache is serializable
try:
    json.dumps(_FSM_CACHE)
    expect(True, "_FSM_CACHE is JSON-serializable")
except Exception as e:
    expect(False, "_FSM_CACHE JSON-serializable", str(e))

# ══════════════════════════════════════════════════════════════════════════════
# 4. DYNAMIC TOOLS — built-in finance tools
# ══════════════════════════════════════════════════════════════════════════════
section("4. Dynamic Tools — built-in finance calculators")

from src.dynamic_tools import (
    load_registered_tools, is_registered_tool, call_registered_tool,
    get_tool_registry_stats, seed_amortization_tool, seed_http_fetch_tool
)

# Seed tools first
seed_amortization_tool()
seed_http_fetch_tool()

tools_list = load_registered_tools()
expect(isinstance(tools_list, list), "load_registered_tools returns list")
expect(len(tools_list) >= 1, f"at least 1 seeded tool, got {len(tools_list)}")
expect(is_registered_tool("finance_loan_amortization"), "finance_loan_amortization is registered")
expect(is_registered_tool("data_http_fetch"), "data_http_fetch is registered")

# Test amortization via call_registered_tool (uses months not years)
try:
    # 30 years = 360 months, 6% annual_rate (passed as %), $200k → ~$1199/mo
    r2 = call_registered_tool("finance_loan_amortization",
                               {"principal": 200000, "annual_rate": 6.0, "months": 360})
    mp = r2.get("details", {}).get("monthly_payment", 0)
    expect(1100 < mp < 1300, f"amortization monthly_payment ≈ $1199, got {mp:.2f}")
    expect("details" in r2, "amortization result has 'details' key")
except Exception as e:
    expect(False, "call_registered_tool amortization works", str(e))

# Test smaller loan
try:
    r3 = call_registered_tool("finance_loan_amortization",
                               {"principal": 50000, "annual_rate": 5.0, "months": 60})
    mp3 = r3.get("details", {}).get("monthly_payment", 0)
    expect(900 < mp3 < 960, f"50k/5yr monthly ≈ $943, got {mp3:.2f}")
except Exception as e:
    expect(False, "smaller amortization calc", str(e))

# Stats
stats = get_tool_registry_stats()
expect(isinstance(stats, dict), "get_tool_registry_stats returns dict")
expect(stats.get("seeded", 0) >= 1, f"stats shows ≥1 seeded tools, got {stats}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. TOKEN BUDGET
# ══════════════════════════════════════════════════════════════════════════════
section("5. Token Budget — context management")

from src.token_budget import TokenBudget

tb = TokenBudget(budget=4000)
expect(tb.remaining > 0, "remaining > 0 on init")
expect(not tb.should_skip_llm, "should_skip_llm False initially")

# consume takes text (counts tokens by len//4 approx), so pass large string
tb.consume("x" * 16000, "test")   # ~4000 tokens → exhausts budget
expect(tb.should_skip_llm, "should_skip_llm True after heavy consumption")

tb2 = TokenBudget(budget=8000)
tb2.consume("x" * 400, "test")    # ~100 tokens → not exhausted
expect(not tb2.should_skip_llm, "budget not exhausted at small consumption")

# ══════════════════════════════════════════════════════════════════════════════
# 6. STRATEGY BANDIT — UCB1 arm selection (module-level functions)
# ══════════════════════════════════════════════════════════════════════════════
section("6. Strategy Bandit — UCB1 arm selection")

from src.strategy_bandit import select_strategy, record_outcome, get_stats

for ptype in ["risk_classification", "options_trading", "crypto_trading", "treasury_qa"]:
    try:
        arm = select_strategy(ptype)
        expect(isinstance(arm, str) and len(arm) > 0, f"select_strategy({ptype}) → '{arm}'")
    except Exception as e:
        expect(False, f"select_strategy({ptype}) doesn't crash", str(e))

try:
    arm = select_strategy("risk_classification")
    record_outcome("risk_classification", arm, 0.85)
    expect(True, "record_outcome doesn't crash")
except Exception as e:
    expect(False, "record_outcome doesn't crash", str(e))

stats = get_stats()
expect(isinstance(stats, dict), f"get_stats() returns dict")

# ══════════════════════════════════════════════════════════════════════════════
# 7. ENTITY EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════
section("7. Entity Extractor — extraction")

from src.entity_extractor import extract_entities, get_entity_context

text = "Apple Inc reported revenue of $394 billion in FY2023. CEO Tim Cook announced record iPhone sales."
entities = extract_entities(text, domain="finance")
expect(isinstance(entities, list), f"extract_entities returns list, got {type(entities)}")

ctx = get_entity_context(text, top_k=3)
expect(isinstance(ctx, str), "get_entity_context returns string")

# ══════════════════════════════════════════════════════════════════════════════
# 8. CONTEXT RL — confidence and drift
# ══════════════════════════════════════════════════════════════════════════════
section("8. Context RL — confidence and drift")

from src.context_rl import get_confidence, should_inject, is_drift_detected, get_context_stats

try:
    conf = get_confidence("risk_classification", "financial")
    expect(0.0 <= conf <= 1.0, f"get_confidence in [0,1]: {conf:.2f}")
    inj = should_inject("crypto_trading", "technical")
    expect(isinstance(inj, bool), f"should_inject returns bool: {inj}")
    drift = is_drift_detected("treasury_qa", "numerical")
    expect(isinstance(drift, bool), f"is_drift_detected returns bool: {drift}")
    cstats = get_context_stats()
    expect(isinstance(cstats, dict), "get_context_stats returns dict")
except Exception as e:
    expect(False, "context_rl functions don't crash", str(e)[:100])

# ══════════════════════════════════════════════════════════════════════════════
# 9. OUTPUT VALIDATOR — answer quality checks
# ══════════════════════════════════════════════════════════════════════════════
section("9. Output Validator — answer quality checks")

from src.output_validator import validate_output

r = validate_output('{"risk_classification": ["Market Risk", "Operational Risk"]}', "sec_filing_analysis")
expect(isinstance(r, dict), f"validate_output returns dict")

r2 = validate_output("The quarterly revenue was $42.5 billion.", "financial_calculation")
expect(isinstance(r2, dict), "validate_output handles plain text")

r3 = validate_output("", "general")
expect(isinstance(r3, dict), "validate_output handles empty string gracefully")

# ══════════════════════════════════════════════════════════════════════════════
# 10. WORKER BRAIN — format normalizer helpers (no API)
# ══════════════════════════════════════════════════════════════════════════════
section("10. Worker Brain — format normalizer (no-API path)")

# We test the quick_format_check and xml extraction path (zero API cost)
# Import the module-level helpers by running a minimal import
try:
    import importlib
    wb = importlib.import_module("src.worker_brain")

    # Test _quick_format_check
    qfc = getattr(wb, "_quick_format_check", None)
    if qfc:
        expect(qfc('{"risk_classification": ["Market Risk"]}', "json_risk_classification"),
               "_quick_format_check: risk JSON already OK")
        expect(not qfc("Here is my analysis of the risks...", "json_risk_classification"),
               "_quick_format_check: prose not OK for risk JSON")
        expect(qfc("<FINAL_ANSWER>\n46.2\n</FINAL_ANSWER>", "xml_final_answer"),
               "_quick_format_check: FINAL_ANSWER already OK")
        expect(qfc("VTI: 60%\nVXUS: 40%", "portfolio_allocation"),
               "_quick_format_check: portfolio skips (None marker)")
    else:
        expect(False, "_quick_format_check exists in worker_brain")

    # Test xml path of _normalize_to_format (zero API)
    norm = getattr(wb, "_normalize_to_format", None)
    if norm:
        # XML path: regex number extraction, no API call
        result = asyncio.get_event_loop().run_until_complete(
            norm("The total public debt outstanding was 22,500 billion dollars", "xml_final_answer", "")
        )
        expect(result and "<FINAL_ANSWER>" in result, "xml normalization extracts number", str(result))
        expect(result and "22500" in result.replace(",", ""), "xml normalization correct value", str(result))

        # Already-correct: should return None (fast path)
        result2 = asyncio.get_event_loop().run_until_complete(
            norm('<FINAL_ANSWER>\n46.2\n</FINAL_ANSWER>', "xml_final_answer", "")
        )
        expect(result2 is None, "xml normalization skips already-correct answer (returns None)")
    else:
        expect(False, "_normalize_to_format exists in worker_brain")

except Exception as e:
    expect(False, "worker_brain imports cleanly", traceback.format_exc()[-200:])

# ══════════════════════════════════════════════════════════════════════════════
# 11. FSM RUNNER — FSMRunner class interface
# ══════════════════════════════════════════════════════════════════════════════
section("11. FSM Runner — class interface")

import inspect
from src.fsm_runner import FSMRunner, FSMState, FSMContext

expect(issubclass(FSMRunner, object), "FSMRunner class importable")
expect(issubclass(FSMState, str), "FSMState is a string enum")

# Check FSMRunner has expected methods (uses 'advance' not 'run')
runner_methods = [m for m in dir(FSMRunner) if not m.startswith("__")]
expect("advance" in runner_methods, f"FSMRunner has advance method")
expect("is_terminal" in runner_methods, "FSMRunner has is_terminal method")
expect("get_summary" in runner_methods, "FSMRunner has get_summary method")

# FSMRunner instantiation (requires session_id)
try:
    runner = FSMRunner(task_text="calculate bond yield", session_id="test-001",
                      process_type="financial_calculation")
    expect(True, "FSMRunner instantiates with session_id")
    expect(hasattr(runner, "ctx"), "FSMRunner has ctx attribute")
    expect(isinstance(runner.is_terminal, bool), "FSMRunner.is_terminal property returns bool")
except Exception as e:
    expect(False, "FSMRunner instantiates", str(e)[:80])

# ══════════════════════════════════════════════════════════════════════════════
# 12. SERVER — routes (no live server needed)
# ══════════════════════════════════════════════════════════════════════════════
section("12. Server — app factory")

try:
    from src.server import app
    expect(app is not None, "FastAPI app imports")
    routes = [r.path for r in app.routes]
    expect("/health" in routes or any("health" in r for r in routes),
           f"health route exists: {routes}")
    # A2A handler is at "/" (POST), not "/process"
    expect("/" in routes or any("/process" in r for r in routes),
           f"A2A handler route exists: {routes}")
except Exception as e:
    expect(False, "server app imports cleanly", str(e)[:120])

# ══════════════════════════════════════════════════════════════════════════════
# 13. KNOWLEDGE EXTRACTOR — module-level API
# ══════════════════════════════════════════════════════════════════════════════
section("13. Knowledge Extractor — persistence")

from src.knowledge_extractor import _load, extract_and_store

try:
    entries = _load()
    expect(isinstance(entries, list), f"knowledge_extractor._load() returns list ({len(entries)} entries)")
    expect(callable(extract_and_store), "extract_and_store is callable (async)")
except Exception as e:
    expect(False, "knowledge_extractor loads cleanly", str(e)[:100])

# ══════════════════════════════════════════════════════════════════════════════
# 14. RL LOOP — case log round-trip
# ══════════════════════════════════════════════════════════════════════════════
section("14. RL Loop — case log")

from src.rl_loop import record_outcome as rl_record, build_rl_primer, score_quality

try:
    # record_outcome(task_text, answer, tool_count, policy_passed, error, domain)
    rl_record(
        task_text="calculate bond yield to maturity",
        answer="yield = 4.2%",
        tool_count=1,
        policy_passed=True,
        domain="financial_calculation",
    )
    expect(True, "rl_loop.record_outcome doesn't crash")
except Exception as e:
    expect(False, "rl_loop.record_outcome doesn't crash", str(e)[:120])

try:
    primer = build_rl_primer("calculate bond yield to maturity")
    expect(isinstance(primer, str), f"build_rl_primer returns str (len={len(primer)})")
except Exception as e:
    expect(False, "build_rl_primer doesn't crash", str(e)[:80])

score = score_quality("The yield is 4.2%", tool_count=1, policy_passed=True)
expect(0.0 <= score <= 1.0, f"score_quality in [0,1]: {score:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}{'═'*70}{RESET}")
total = passed + failed
pct = 100 * passed // total if total else 0
color = G if pct == 100 else (Y if pct >= 80 else R)
print(f"{BOLD}{color}  RESULT: {passed}/{total} tests passed ({pct}%){RESET}")
print(f"{'═'*70}")
if failed > 0:
    print(f"{R}  {failed} test(s) failed — see ✗ above{RESET}\n")
else:
    print(f"{G}  All tests green! ✓{RESET}\n")

sys.exit(0 if failed == 0 else 1)
