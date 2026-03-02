"""
simulate_competition.py
Competition simulation — acts as GREEN SERVER sending tasks to the Finance Purple Agent.

Covers all known Finance track green agents:
  1. AgentBusters / FAB++ — options pricing, crypto trading, chain-of-thought
  2. OfficeQA           — Treasury Bulletin PDF questions (FINAL_ANSWER format)
  3. Alpha-Cortex-AI    — SEC 10-K filing analysis (risk/summary/consistency)
  4. Data Matchmaker    — CSV data integration (TPC-DI style)
  5. Portfolio Advisor  — portfolio optimization, Sharpe ratio

Usage:
  1. Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
  2. python3 simulate_competition.py

What this shows:
  - How the purple server classifies each task
  - Which format directive fires
  - Whether FSM synthesis kicks in
  - The actual answer quality
  - Timing per task
"""
from __future__ import annotations
import asyncio
import json
import os
import sys
import time

# ── Load .env ──────────────────────────────────────────────────────────────────
_ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_ENV_FILE):
    with open(_ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                if v and "PASTE_YOUR_KEY_HERE" not in v:
                    os.environ.setdefault(k.strip(), v.strip())

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: Set ANTHROPIC_API_KEY in .env first")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(__file__))
from src.worker_brain import run_worker

# ── ANSI colors ────────────────────────────────────────────────────────────────
G = "\033[92m"   # green  = green server
P = "\033[95m"   # purple = purple agent
Y = "\033[93m"   # yellow = timing / meta
C = "\033[96m"   # cyan   = answer
R = "\033[91m"   # red    = error / warning
RESET = "\033[0m"
BOLD = "\033[1m"

def green(s): return f"{G}{s}{RESET}"
def purple(s): return f"{P}{s}{RESET}"
def yellow(s): return f"{Y}{s}{RESET}"
def cyan(s): return f"{C}{s}{RESET}"
def bold(s): return f"{BOLD}{s}{RESET}"
def hr(char="─", width=72): return yellow(char * width)

# ── Realistic competition task payloads ────────────────────────────────────────
# Modelled on actual benchmark formats from the three known Finance green agents.

SCENARIOS = [

    # ────────────────────────────────────────────────────────────────────────────
    # GREEN AGENT 1: OfficeQA — 697 Treasury Bulletin PDF questions
    # Format: <FINAL_ANSWER>\n[number]\n</FINAL_ANSWER>
    # ────────────────────────────────────────────────────────────────────────────
    {
        "id": "officeqa-001",
        "green_agent": "OfficeQA",
        "description": "Treasury: total federal expenditure lookup",
        "task": (
            "Using the US Treasury Bulletin data, what was the total federal outlays "
            "(expenditures) for fiscal year 2021 in billions of dollars? "
            "Provide your answer as:\n<FINAL_ANSWER>\n[numerical value only, no units]\n</FINAL_ANSWER>"
        ),
        "tools_endpoint": "",
        "expected_format": "xml_final_answer",
    },
    {
        "id": "officeqa-002",
        "green_agent": "OfficeQA",
        "description": "Treasury: public debt calculation",
        "task": (
            "According to Treasury Bulletin data, what was the total public debt outstanding "
            "at the end of fiscal year 2022, in trillions? Round to 2 decimal places.\n"
            "Provide your answer as:\n<FINAL_ANSWER>\n[numerical value only]\n</FINAL_ANSWER>"
        ),
        "tools_endpoint": "",
        "expected_format": "xml_final_answer",
    },

    # ────────────────────────────────────────────────────────────────────────────
    # GREEN AGENT 2: FAB++ / AgentBusters — options pricing + crypto trading
    # Formats: JSON options, JSON trading decision, JSON CoT
    # ────────────────────────────────────────────────────────────────────────────
    {
        "id": "fabpp-001",
        "green_agent": "FAB++ (AgentBusters)",
        "description": "Black-Scholes call option pricing",
        "task": (
            "Calculate the Black-Scholes price of a European call option with the following parameters:\n"
            "- Current stock price (S): $105.00\n"
            "- Strike price (K): $100.00\n"
            "- Risk-free rate (r): 5% per annum\n"
            "- Volatility (σ): 20% per annum\n"
            "- Time to expiration (T): 1 year\n\n"
            "Also calculate Delta, Gamma, Theta, and Vega.\n\n"
            "Return JSON: {\"result\": {\"price\": 0.0, \"greeks\": {\"delta\": 0.0, "
            "\"gamma\": 0.0, \"theta\": 0.0, \"vega\": 0.0}}}"
        ),
        "tools_endpoint": "",
        "expected_format": "json_options",
    },
    {
        "id": "fabpp-002",
        "green_agent": "FAB++ (AgentBusters)",
        "description": "Crypto trading decision (BTCUSDT OHLCV)",
        "task": (
            "You are a crypto trading agent managing a BTCUSDT position.\n\n"
            "Current market data (last 5 candles, 1h OHLCV):\n"
            "| Time  | Open    | High    | Low     | Close   | Volume   |\n"
            "|-------|---------|---------|---------|---------|----------|\n"
            "| 09:00 | 42100.0 | 42350.0 | 41900.0 | 42280.0 | 1250.5   |\n"
            "| 10:00 | 42280.0 | 42500.0 | 42100.0 | 42450.0 | 987.3    |\n"
            "| 11:00 | 42450.0 | 42600.0 | 42300.0 | 42380.0 | 1102.1   |\n"
            "| 12:00 | 42380.0 | 42420.0 | 41800.0 | 41950.0 | 1580.7   |\n"
            "| 13:00 | 41950.0 | 42050.0 | 41700.0 | 41820.0 | 1423.9   |\n\n"
            "Open positions: []\nAccount balance: 50000 USDT\n\n"
            "Return JSON:\n"
            "{\"action\": \"BUY|SELL|HOLD|CLOSE\", \"quantity\": 0.0, "
            "\"stop_loss\": 0.0, \"take_profit\": 0.0, \"reasoning\": \"...\"}"
        ),
        "tools_endpoint": "",
        "expected_format": "json_trading_decision",
    },
    {
        "id": "fabpp-003",
        "green_agent": "FAB++ (AgentBusters)",
        "description": "Chain-of-thought financial reasoning",
        "task": (
            "A company has the following financial data for Q3 2024:\n"
            "- Revenue: $2.4 billion\n"
            "- COGS: $1.44 billion\n"
            "- Operating expenses: $480 million\n"
            "- Interest expense: $24 million\n"
            "- Tax rate: 21%\n\n"
            "Question: What is the net income margin for Q3 2024? "
            "Show your complete chain-of-thought reasoning, then provide:\n"
            "{\"answer\": \"X.XX%\", \"reasoning\": \"step-by-step calculation\"}"
        ),
        "tools_endpoint": "",
        "expected_format": "json_cot_answer",
    },

    # ────────────────────────────────────────────────────────────────────────────
    # GREEN AGENT 3: Alpha-Cortex-AI — SEC 10-K filing analysis
    # Formats: JSON risk_classification, JSON business_summary, JSON consistency_check
    # ────────────────────────────────────────────────────────────────────────────
    {
        "id": "alphacortex-001",
        "green_agent": "Alpha-Cortex-AI",
        "description": "SEC 10-K risk classification",
        "task": (
            "Below is an excerpt from a company's SEC 10-K annual report Risk Factors section:\n\n"
            "\"The Company's operations are subject to significant cybersecurity threats. "
            "A breach of our systems could expose customer data and result in regulatory fines. "
            "Additionally, our supply chain relies on a single-source semiconductor supplier in Taiwan, "
            "creating concentration risk. Exchange rate fluctuations between USD and EUR impact our "
            "European revenue, which represents 38% of total sales. Rising interest rates have increased "
            "our variable-rate debt servicing costs by approximately $12M annually.\"\n\n"
            "Classify ALL risk types mentioned. Return JSON:\n"
            "{\"risk_classification\": [\"Risk Type 1\", \"Risk Type 2\", ...]}\n"
            "Use these categories: Market Risk, Operational Risk, Financial Risk, "
            "Cybersecurity Risk, Geopolitical Risk, Supply Chain Risk, Regulatory Risk"
        ),
        "tools_endpoint": "",
        "expected_format": "json_risk_classification",
    },
    {
        "id": "alphacortex-002",
        "green_agent": "Alpha-Cortex-AI",
        "description": "SEC 10-K business summary",
        "task": (
            "Analyze the following 10-K filing excerpt and generate a structured business summary:\n\n"
            "\"TechNova Solutions Inc. (NASDAQ: TNSI) is a global provider of enterprise cloud "
            "infrastructure services, headquartered in San Jose, CA. Founded in 2008, the company "
            "operates data centers across 14 countries with over 8,200 employees. Primary revenue "
            "streams include Platform-as-a-Service (58%), Professional Services (27%), and "
            "Hardware (15%). FY2023 revenue reached $1.87 billion, up 23% year-over-year, "
            "driven by 41% growth in PaaS subscriptions. Key customers include Fortune 500 "
            "enterprises in financial services, healthcare, and manufacturing sectors.\"\n\n"
            "Return JSON: {\"business_summary\": {\"industry\": \"\", \"products\": [], "
            "\"geography\": \"\", \"revenue\": \"\", \"growth\": \"\", \"key_customers\": \"\"}}"
        ),
        "tools_endpoint": "",
        "expected_format": "json_business_summary",
    },
    {
        "id": "alphacortex-003",
        "green_agent": "Alpha-Cortex-AI",
        "description": "SEC 10-K consistency check",
        "task": (
            "Review the following 10-K data points and identify any inconsistencies:\n\n"
            "From MD&A section: 'Net revenue increased 18% to $3.2B in FY2023'\n"
            "From Financial Statements: FY2022 net revenue = $2.8B, FY2023 net revenue = $3.2B\n"
            "From CEO Letter: 'We achieved record revenue of $3.4B this fiscal year'\n"
            "From Press Release: '17.6% revenue growth driven by international expansion'\n\n"
            "Return JSON: {\"consistency_check\": {\"is_consistent\": false, "
            "\"discrepancies\": [{\"field\": \"\", \"value_1\": \"\", "
            "\"value_2\": \"\", \"source_1\": \"\", \"source_2\": \"\"}]}}"
        ),
        "tools_endpoint": "",
        "expected_format": "json_consistency_check",
    },

    # ────────────────────────────────────────────────────────────────────────────
    # GREEN AGENT 4: Portfolio optimization
    # Format: JSON portfolio allocation
    # ────────────────────────────────────────────────────────────────────────────
    {
        "id": "portfolio-001",
        "green_agent": "Portfolio Advisor",
        "description": "Portfolio allocation for retirement",
        "task": (
            "A 35-year-old investor with a moderate risk tolerance wants to build a retirement "
            "portfolio targeting a 7% annual return with a 30-year horizon. Available ETFs:\n"
            "- VTI (US Total Market): historical return 10.2%, std dev 15.1%\n"
            "- VXUS (International): historical return 6.8%, std dev 17.3%\n"
            "- BND (US Bonds): historical return 3.2%, std dev 4.8%\n"
            "- GLD (Gold): historical return 5.1%, std dev 13.2%\n\n"
            "Correlations: VTI-VXUS: 0.82, VTI-BND: -0.15, VTI-GLD: 0.02\n\n"
            "Recommend an optimal allocation. Return JSON:\n"
            "{\"portfolio\": {\"VTI\": 0.0, \"VXUS\": 0.0, \"BND\": 0.0, \"GLD\": 0.0}, "
            "\"expected_return\": 0.0, \"expected_volatility\": 0.0, "
            "\"sharpe_ratio\": 0.0, \"rationale\": \"\"}"
        ),
        "tools_endpoint": "",
        "expected_format": "portfolio_allocation",
    },

    # ────────────────────────────────────────────────────────────────────────────
    # GREEN AGENT 5: Loan amortization — tests ACE seeded tool
    # ────────────────────────────────────────────────────────────────────────────
    {
        "id": "calc-001",
        "green_agent": "Financial Calculator",
        "description": "Loan amortization (tests ACE seeded tool)",
        "task": (
            "Calculate the monthly payment for a home mortgage:\n"
            "- Loan principal: $450,000\n"
            "- Annual interest rate: 6.5%\n"
            "- Loan term: 30 years (360 months)\n\n"
            "Also show: total interest paid over life of loan, "
            "and the breakdown for month 1 (principal vs interest).\n\n"
            "Return JSON: {\"monthly_payment\": 0.00, \"total_interest\": 0.00, "
            "\"month_1\": {\"principal\": 0.00, \"interest\": 0.00, \"balance\": 0.00}}"
        ),
        "tools_endpoint": "",
        "expected_format": "json_generic",
    },
]


# ── Simulation runner ──────────────────────────────────────────────────────────

async def run_scenario(scenario: dict, idx: int, total: int) -> dict:
    task_id = scenario["id"]
    session_id = f"sim-session-{task_id}"

    print()
    print(hr("═"))
    print(f"{bold(f'SCENARIO {idx}/{total}')} — {green(scenario['green_agent'])}")
    print(f"ID: {yellow(task_id)}  |  {scenario['description']}")
    print(f"Expected format: {yellow(scenario.get('expected_format', 'auto'))}")
    print(hr())

    print(green("\n[GREEN SERVER → PURPLE AGENT]"))
    task_preview = scenario["task"][:200].replace("\n", " ")
    print(f"  Task: {task_preview}{'...' if len(scenario['task']) > 200 else ''}")
    print(f"  session_id: {session_id}")
    print()

    t0 = time.time()
    try:
        answer = await run_worker(
            task_text=scenario["task"],
            policy_doc="",
            tools_endpoint=scenario.get("tools_endpoint", ""),
            task_id=task_id,
            session_id=session_id,
        )
        elapsed = time.time() - t0
        success = True
    except Exception as e:
        answer = f"EXCEPTION: {e}"
        elapsed = time.time() - t0
        success = False

    print(purple("\n[PURPLE AGENT → GREEN SERVER]"))
    print(f"  Time: {yellow(f'{elapsed:.1f}s')}")
    print(f"  Answer length: {len(answer)} chars")
    print()
    print(cyan("  ── ANSWER ──"))

    # Pretty-print answer with line wrapping
    lines = answer.split("\n")
    for line in lines[:60]:  # cap at 60 lines
        print(f"  {line}")
    if len(lines) > 60:
        print(f"  ... [{len(lines)-60} more lines]")

    # Format compliance check
    fmt = scenario.get("expected_format", "")
    compliance = _check_format_compliance(answer, fmt)
    print()
    print(f"  Format compliance ({fmt}): {green('✓ PASS') if compliance else R+'✗ FAIL'+RESET}")

    return {
        "id": task_id,
        "green_agent": scenario["green_agent"],
        "elapsed_s": round(elapsed, 2),
        "answer_len": len(answer),
        "success": success,
        "format_compliant": compliance,
    }


def _check_format_compliance(answer: str, fmt: str) -> bool:
    """Quick heuristic check that the answer has the expected format."""
    import re
    lower = answer.lower()
    checks = {
        "xml_final_answer":        bool(re.search(r'<final_answer>', lower)),
        "json_risk_classification": bool(re.search(r'"risk_classification"', lower)),
        "json_business_summary":    bool(re.search(r'"business_summary"', lower)),
        "json_consistency_check":   bool(re.search(r'"consistency_check"', lower)),
        "json_trading_decision":    bool(re.search(r'"action"', lower) and re.search(r'(buy|sell|hold|close)', lower)),
        "json_options":             bool(re.search(r'"price"', lower) or re.search(r'"greeks"', lower)),
        "json_cot_answer":          bool(re.search(r'"answer"', lower) or re.search(r'"reasoning"', lower)),
        "json_generic":             bool("{" in answer),
        "portfolio_allocation":     bool(re.search(r'"portfolio"', lower) or re.search(r'"sharpe', lower) or re.search(r'(vti|vxus|bnd)', lower)),
        "csv_data_integration":     "," in answer and "\n" in answer,
    }
    return checks.get(fmt, True)


async def main():
    print()
    print(hr("═", 72))
    print(bold("  FINANCE PURPLE AGENT — COMPETITION SIMULATION"))
    print(bold("  Acting as GREEN SERVER sending tasks to PURPLE AGENT"))
    print(f"  Green agents simulated: AgentBusters/FAB++, OfficeQA, Alpha-Cortex, Portfolio")
    print(f"  Total scenarios: {len(SCENARIOS)}")
    print(hr("═", 72))

    results = []
    t_total = time.time()

    for i, scenario in enumerate(SCENARIOS, 1):
        result = await run_scenario(scenario, i, len(SCENARIOS))
        results.append(result)
        # Small delay between tasks (mimics benchmark pacing)
        if i < len(SCENARIOS):
            await asyncio.sleep(1.0)

    total_elapsed = time.time() - t_total

    # ── Final scoreboard ──────────────────────────────────────────────────────
    print()
    print(hr("═", 72))
    print(bold("  SIMULATION COMPLETE — SCOREBOARD"))
    print(hr("═", 72))
    print(f"  {'ID':<20} {'Agent':<25} {'Time':>6}  {'Len':>6}  {'OK':>4}  {'Fmt':>4}")
    print(f"  {'-'*20} {'-'*25} {'-'*6}  {'-'*6}  {'-'*4}  {'-'*4}")
    for r in results:
        ok_str = green("✓") if r["success"] else R+"✗"+RESET
        fmt_str = green("✓") if r["format_compliant"] else R+"✗"+RESET
        print(f"  {r['id']:<20} {r['green_agent']:<25} {r['elapsed_s']:>5.1f}s  {r['answer_len']:>6}  {ok_str:>4}  {fmt_str:>4}")

    n_ok = sum(1 for r in results if r["success"])
    n_fmt = sum(1 for r in results if r["format_compliant"])
    print(hr())
    print(f"  Tasks completed:   {green(f'{n_ok}/{len(results)}')}")
    print(f"  Format compliant:  {green(f'{n_fmt}/{len(results)}')}  ← judge sees this score")
    print(f"  Total time:        {yellow(f'{total_elapsed:.1f}s')}")
    print(f"  Avg per task:      {yellow(f'{total_elapsed/len(results):.1f}s')}")
    print(hr("═", 72))
    print()


if __name__ == "__main__":
    asyncio.run(main())
