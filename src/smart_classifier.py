"""
smart_classifier.py
LLM-based finance task type classification.

A single Haiku call (~200ms) semantically classifies the task.
The classified type becomes the key for dynamic_fsm synthesis — right type
signal = right FSM workflow = right tool pre-warming = better benchmark scores.

Finance track task types map to known benchmark green agents:
  sec_filing_analysis  → Alpha-Cortex (900 SEC 10-K filings)
  options_trading      → FAB++ (Black-Scholes, Greeks)
  crypto_trading       → FAB++ (BTCUSDT trading decisions)
  treasury_qa          → OfficeQA (697 Treasury PDF questions)
  portfolio_*          → Various green agents
  data_integration     → Data Matchmaker / TPC-DI

Falls back to keyword matching if Haiku unavailable or times out.
"""
from __future__ import annotations
import asyncio
import json
import re

from src.config import ANTHROPIC_API_KEY

CLASSIFIER_MODEL = "claude-haiku-4-5-20251001"
CLASSIFIER_TIMEOUT = 5.0

# Finance track process types.
# The dynamic FSM synthesizes workflows for ALL types including any novel ones
# Haiku returns — nothing is filtered out.
VALID_PROCESS_TYPES = {
    "sec_filing_analysis",      # SEC 10-K filings, annual reports, risk/summary/MD&A
    "options_trading",          # Black-Scholes, Greeks, call/put strategy
    "crypto_trading",           # BTCUSDT, OHLCV, trading decisions (BUY/SELL/HOLD)
    "treasury_qa",              # US Treasury Bulletin PDF questions
    "portfolio_optimization",   # Portfolio allocation, Sharpe ratio, MPT
    "risk_classification",      # Risk categorization, VaR, stress testing
    "data_integration",         # TPC-DI, CSV joins, data cleaning
    "financial_calculation",    # Amortization, NPV, IRR, general math
    "market_analysis",          # Price trends, technical indicators
    "compliance_reporting",     # Regulatory reports, audit, KYC/AML
    "general",
}

_CLASSIFIER_PROMPT = """\
You are a finance task classifier for AI benchmark evaluation.
Given a task description, output the single best process type.

Finance task types:
- sec_filing_analysis: SEC 10-K/10-Q filings, annual reports, risk factors, MD&A sections, business summaries, consistency checks
- options_trading: Black-Scholes option pricing, option Greeks (Delta/Gamma/Theta/Vega/Rho), call/put strategies, iron condor, covered call, implied volatility, option assessment
- crypto_trading: Bitcoin/cryptocurrency OHLCV candlestick data, trading decisions (BUY/SELL/HOLD/CLOSE), position sizing, stop-loss, take-profit, portfolio balance
- treasury_qa: US Treasury Bulletin, federal expenditures, public debt, fiscal year tables, Treasury PDF questions, FINAL_ANSWER format
- portfolio_optimization: Portfolio allocation (% per ticker), Sharpe ratio, Modern Portfolio Theory, diversification, retirement/college savings
- risk_classification: Risk categorization (Market Risk, Operational Risk, etc.), VaR, stress testing, risk scoring, risk matrix
- data_integration: Data joining, CSV processing, TPC-DI benchmark, data cleaning, customer-account matching
- financial_calculation: Loan amortization, NPV, IRR, compound interest, depreciation, time value of money, general finance math
- market_analysis: Price trends, technical indicators, volume analysis, market data interpretation, moving averages
- compliance_reporting: Regulatory reports, compliance checks, KYC/AML, audit findings
- general: anything that doesn't clearly fit the above

Respond with JSON only: {"process_type": "<type>", "confidence": 0.0-1.0, "reasoning": "<one sentence>"}"""


async def classify_process_type(task_text: str) -> tuple[str, float]:
    """
    Classify task into a finance process type using Haiku.
    Returns (process_type, confidence).
    Falls back to keyword matching on timeout or error.
    """
    if not ANTHROPIC_API_KEY:
        return _keyword_fallback(task_text), 0.5

    try:
        return await asyncio.wait_for(
            _call_classifier(task_text),
            timeout=CLASSIFIER_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return _keyword_fallback(task_text), 0.5
    except Exception:
        return _keyword_fallback(task_text), 0.5


async def _call_classifier(task_text: str) -> tuple[str, float]:
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    resp = await client.messages.create(
        model=CLASSIFIER_MODEL,
        max_tokens=120,
        system=_CLASSIFIER_PROMPT,
        messages=[{"role": "user", "content": task_text[:500]}],
    )
    text = resp.content[0].text if resp.content else ""
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```[a-z]*\n?", "", clean)
        clean = re.sub(r"\n?```$", "", clean).strip()
    parsed = json.loads(clean)
    ptype = parsed.get("process_type", "general")
    conf = float(parsed.get("confidence", 0.7))
    # Trust whatever Haiku returns — novel types are valid (dynamic_fsm handles them)
    if not ptype:
        return _keyword_fallback(task_text), 0.4
    return ptype, conf


def _keyword_fallback(task_text: str) -> str:
    """Keyword-based classification used as fallback and test utility."""
    KEYWORDS: dict[str, list[str]] = {
        "sec_filing_analysis":    ["10-k", "10-q", "sec filing", "annual report", "risk factor",
                                   "md&a", "management discussion", "business summary", "form 10-k",
                                   "consistency check", "risk classification"],
        "options_trading":        ["black-scholes", "black scholes", "option price", "option pricing",
                                   "greeks", "delta", "gamma", "theta", "vega", "call", "put",
                                   "strike", "expir", "implied volatility", "iron condor",
                                   "covered call", "protective put"],
        "crypto_trading":         ["btcusdt", "bitcoin", "crypto", "ohlcv", "open_positions",
                                   "trading decision", "stop_loss", "take_profit", "action",
                                   "btc", "eth", "usdt", "balance"],
        "treasury_qa":            ["treasury bulletin", "treasury", "federal expenditure",
                                   "public debt", "fiscal year", "final_answer", "total expenditure",
                                   "department of", "u.s. treasury"],
        "portfolio_optimization": ["portfolio", "allocation", "sharpe", "diversif",
                                   "vti", "vxus", "bnd", "retirement", "college savings",
                                   "ticker", "percent"],
        "risk_classification":    ["risk classification", "market risk", "operational risk",
                                   "cybersecurity risk", "geopolitical risk", "supply chain risk",
                                   "var ", "value at risk", "stress test", "risk score"],
        "data_integration":       ["tpc-di", "csv", "data integration", "customer_id",
                                   "join", "data match", "columns", "data clean"],
        "financial_calculation":  ["amortization", "npv", "irr", "compound interest",
                                   "depreciation", "principal", "annual rate", "monthly payment",
                                   "time value", "present value", "future value"],
        "market_analysis":        ["price trend", "volume", "moving average", "technical analysis",
                                   "market data", "candlestick", "support", "resistance"],
        "compliance_reporting":   ["compliance", "regulatory", "kyc", "aml", "audit report",
                                   "sox", "gdpr", "pci"],
    }
    text = task_text.lower()
    best, best_score = "general", 0
    for ptype, kws in KEYWORDS.items():
        score = sum(1 for kw in kws if kw in text)
        if score > best_score:
            best_score, best = score, ptype
    return best
