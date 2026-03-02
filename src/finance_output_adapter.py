"""
finance_output_adapter.py
Zero-API-cost output format compliance for finance benchmarks.

=============================================================
THE PROBLEM THIS SOLVES
=============================================================

Claude Opus 4.5 scores 39.8 on AgentBusters vs GPT-4o at 69.5.
30 points of that gap is NOT model capability — it's FORMAT COMPLIANCE.

Different green agents require completely different output shapes:
  Alpha-Cortex:   {"risk_classification": ["Market Risk", ...]}
  AgentBusters:   {"cot": "reasoning...", "answer": "42"}
  Crypto trading: {"action": "BUY", "size": 0.1, "stop_loss": 0.0, ...}
  OfficeQA:       <FINAL_ANSWER>\n46.2\n</FINAL_ANSWER>
  Data tasks:     CSV with exact column order

Without knowing which green agent is sending the task, we detect the
required format FROM THE TASK TEXT ITSELF — which always contains
format instructions from the green agent.

This is NOT business logic. It's compliance enforcement.
The dynamic FSM still handles all reasoning and computation.
This module just tells Claude "put your answer in THIS shape."

=============================================================
DETECTION APPROACH
=============================================================

Every green agent embeds its format requirements in the task text:
  Alpha-Cortex: "Return JSON: {"risk_classification": [...]}"
  AgentBusters: ends questions with format spec
  OfficeQA: no explicit format — but pattern match on Treasury question style

We detect these patterns with regex (zero API cost) and inject a
format directive into the system context BEFORE execution starts.
Claude then knows from the start what shape to produce.
"""
from __future__ import annotations

import re

# ── Format definitions ────────────────────────────────────────────────────────
# Each entry: (regex_pattern, format_key, directive_text)
# Patterns match task text. First match wins.
# Directives are injected into system context verbatim.

_FORMAT_RULES: list[tuple[str, str, str]] = [

    # ── Alpha-Cortex: Risk Classification ─────────────────────────────────────
    (
        r'Return JSON.*?risk_classification|risk_classification.*?Return JSON|'
        r'Categories:.*?Market Risk.*?Operational Risk',
        "json_risk_classification",
        (
            "OUTPUT FORMAT — return ONLY this exact JSON structure, nothing else:\n"
            '{"risk_classification": ["Category1", "Category2"]}\n'
            "Use ONLY these category names exactly:\n"
            "Market Risk, Operational Risk, Financial Risk, Legal/Regulatory Risk,\n"
            "Technology Risk, Cybersecurity Risk, Competition Risk, Supply Chain Risk,\n"
            "Human Capital/Talent Risk, Environmental/Climate Risk, "
            "COVID-19/Pandemic Risk, Geopolitical Risk\n"
            "Include only categories that are clearly present. Typically 3-6 categories."
        ),
    ),

    # ── Alpha-Cortex: Business Summary ────────────────────────────────────────
    (
        r'Return JSON.*?business_summary|business_summary.*?industry.*?products.*?geography',
        "json_business_summary",
        (
            "OUTPUT FORMAT — return ONLY this exact JSON structure, nothing else:\n"
            '{"business_summary": {"industry": "sector description", '
            '"products": "main products/services", "geography": "key markets"}}\n'
            "Each field must be a non-empty string of at least 15 characters."
        ),
    ),

    # ── Alpha-Cortex: Consistency Check ───────────────────────────────────────
    (
        r'Return JSON.*?consistency_check|Which risks ARE discussed|'
        r'consistency_check.*?risk.*?Section 7',
        "json_consistency_check",
        (
            "OUTPUT FORMAT — return ONLY this exact JSON structure, nothing else:\n"
            '{"consistency_check": ["exact risk phrase 1", "exact risk phrase 2"]}\n'
            "Include only risks that are ACTUALLY discussed in Section 7 (MD&A).\n"
            "Copy the risk phrases EXACTLY as given in the input list."
        ),
    ),

    # ── FAB++ / AgentBusters: Crypto Trading Decision ────────────────────────
    (
        r'BTCUSDT|trading_decision|ohlcv.*open.*high.*low.*close|'
        r'"action".*BUY.*SELL.*HOLD|open_positions.*balance',
        "json_trading_decision",
        (
            "OUTPUT FORMAT — return ONLY this exact JSON structure, nothing else:\n"
            '{"action": "BUY", "size": 0.1, "stop_loss": 46500.0, '
            '"take_profit": 48000.0, "reasoning": "brief reason", "confidence": 0.7}\n'
            "action must be exactly: BUY, SELL, HOLD, or CLOSE\n"
            "size: position size in base currency (e.g. 0.1 BTC)\n"
            "stop_loss / take_profit: price levels (0.0 if not applicable)\n"
            "confidence: float 0.0-1.0"
        ),
    ),

    # ── FAB++ / AgentBusters: CoT + Answer format ────────────────────────────
    (
        r'(?:chain[- ]of[- ]thought|cot.*answer|Return.*cot.*answer)',
        "json_cot_answer",
        (
            "OUTPUT FORMAT — return ONLY this exact JSON structure, nothing else:\n"
            '{"cot": "step-by-step reasoning here", "answer": "final answer here"}\n'
            "cot: show your full working/reasoning\n"
            "answer: the final answer only (number, word, or short phrase)"
        ),
    ),

    # ── FAB++ Options Trading ─────────────────────────────────────────────────
    (
        r'Black-Scholes|option.*price.*strike|call.*put.*expir|Greeks.*Delta.*Gamma|'
        r'implied volatility.*option|iron condor|covered call|protective put',
        "json_options",
        (
            "OUTPUT FORMAT — include your full calculation then end with JSON:\n"
            "Show your Black-Scholes working (d1, d2, N(d1), N(d2)).\n"
            "Then provide:\n"
            '{"result": {"price": 0.0, "greeks": {"delta": 0.0, "gamma": 0.0, '
            '"theta": 0.0, "vega": 0.0}, "assessment": "fairly priced|overpriced|underpriced"}}\n'
            "All numbers to 3 decimal places. Tolerance: within 5% of Black-Scholes."
        ),
    ),

    # ── OfficeQA: Treasury Bulletin questions ─────────────────────────────────
    (
        r'Treasury Bulletin|FINAL_ANSWER|treasury.*expenditure|'
        r'fiscal year.*(?:million|billion|thousand)|public debt.*outstanding|'
        r'interest.*public debt.*billion',
        "xml_final_answer",
        (
            "OUTPUT FORMAT — end your response with this EXACT tag:\n"
            "<FINAL_ANSWER>\n[numerical value only, no units, no commas]\n</FINAL_ANSWER>\n"
            "Example: <FINAL_ANSWER>\n46.2\n</FINAL_ANSWER>\n"
            "The value inside the tag must be a plain number. "
            "Show your reasoning first, then the tag at the very end."
        ),
    ),

    # ── Data integration / CSV output ─────────────────────────────────────────
    (
        r'TPC-DI|data integration.*CSV|join.*customers.*accounts|'
        r'customer_id.*columns|Return.*CSV.*format',
        "csv_data_integration",
        (
            "OUTPUT FORMAT — return the result as CSV.\n"
            "First line: header row with exact column names.\n"
            "Subsequent lines: data rows.\n"
            "Use comma delimiter, quote strings containing commas.\n"
            "Do not include any text before or after the CSV block."
        ),
    ),

    # ── Portfolio allocation ───────────────────────────────────────────────────
    (
        r'allocat.*portfolio|ticker.*percent.*allocation|VTI.*VXUS|'
        r'retirement.*invest|college.*savings.*invest',
        "portfolio_allocation",
        (
            "OUTPUT FORMAT — provide allocations that sum to exactly 100%.\n"
            "Format each holding as: TICKER: XX% — reason\n"
            "Example:\n  VTI: 50% — broad US market, low cost\n"
            "  VXUS: 30% — international diversification\n"
            "  BND: 20% — fixed income stability\n"
            "Total must equal 100%. Include 3-6 holdings."
        ),
    ),

    # ── General finance JSON (fallback for structured finance tasks) ──────────
    (
        r'Return JSON|return.*json.*format|answer.*json|provide.*json',
        "json_generic",
        (
            "OUTPUT FORMAT — return your answer as valid JSON.\n"
            "Match the exact structure requested in the task.\n"
            "No markdown fences, no prose outside the JSON object."
        ),
    ),
]


# ── Public API ────────────────────────────────────────────────────────────────

def detect_output_format(task_text: str) -> str | None:
    """
    Scan task text for format requirements.
    Returns format_key string or None if no specific format detected.
    Zero API cost — pure regex.
    """
    for pattern, format_key, _ in _FORMAT_RULES:
        if re.search(pattern, task_text, re.IGNORECASE | re.DOTALL):
            return format_key
    return None


def build_format_directive(format_key: str) -> str:
    """
    Return the format directive text for injection into system context.
    Returns empty string if format_key not found.
    """
    for _, key, directive in _FORMAT_RULES:
        if key == format_key:
            return f"\n\n{'='*60}\n{directive}\n{'='*60}\n"
    return ""


def get_format_name(format_key: str) -> str:
    """Human-readable name for logging."""
    names = {
        "json_risk_classification": "Alpha-Cortex Risk JSON",
        "json_business_summary": "Alpha-Cortex Summary JSON",
        "json_consistency_check": "Alpha-Cortex Consistency JSON",
        "json_trading_decision": "Crypto Trading Decision JSON",
        "json_cot_answer": "CoT + Answer JSON",
        "json_options": "Options Pricing JSON",
        "xml_final_answer": "OfficeQA FINAL_ANSWER XML",
        "csv_data_integration": "CSV Data Output",
        "portfolio_allocation": "Portfolio Allocation",
        "json_generic": "Generic JSON",
    }
    return names.get(format_key, format_key)
