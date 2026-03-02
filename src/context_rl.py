"""
context_rl.py
RL feedback loop for context injection quality.

After every task, REFLECT phase checks whether the pre-computed financial facts
injected during PRIME matched the actual task outcome. Results are stored in
context_quality.json and used to dynamically adjust injection confidence.

Three operating modes per (process_type, context_type) pair:
  HIGH CONFIDENCE  (≥75%)  → inject with confidence annotation
  LOW CONFIDENCE   (55–74%) → inject with uncertainty annotation
  DRIFT DETECTED   (<40% on last 5) → inject DRIFT WARNING instead of value
                              (tells Claude not to trust standard thresholds)

Why drift detection matters:
  Competition benchmarks can change thresholds, formulas, or policies mid-run.
  Competitors using static computation keep getting the wrong answer after the change.
  We detect the change within 3–5 tasks and switch to "compute fresh from tool data."

Accuracy check logic:
  For VARIANCE: check if our predicted action (approve/escalate) matches the answer
  For SLA CREDIT: check if our computed credit amount appears in the answer
  Generic: check if our key numeric value appears in the answer
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

_DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "context_quality.json")

# ── Thresholds ─────────────────────────────────────────────────────────────────
DEFAULT_CONFIDENCE = 0.75     # before enough data — lean optimistic
MIN_INJECT_CONFIDENCE = 0.55  # below this: inject drift warning, not value
DRIFT_THRESHOLD = 0.40        # below this on last 5: flag rule change
WINDOW_SIZE = 10              # rolling window for recent accuracy
MIN_SAMPLES_FOR_CONFIDENCE = 3  # need at least 3 before adjusting from default


# ── Persistence ────────────────────────────────────────────────────────────────

def _load() -> dict:
    try:
        if os.path.exists(_DATA_FILE):
            with open(_DATA_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save(data: dict) -> None:
    try:
        with open(_DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


# ── Core API ───────────────────────────────────────────────────────────────────

def record_context_outcome(
    process_type: str,
    context_type: str,
    was_match: bool,
) -> None:
    """
    Record whether injected context matched task outcome.
    Called from REFLECT phase in worker_brain.py.

    Args:
        process_type: FSM process type (e.g. "invoice_reconciliation")
        context_type: which context was injected (e.g. "variance", "sla_credit")
        was_match:    True if our pre-computed value matched the final answer
    """
    data = _load()
    pt = data.setdefault(process_type, {})
    ct = pt.setdefault(context_type, {
        "attempts": 0,
        "matches": 0,
        "recent": [],
        "last_updated": None,
        "drift_alerts": 0,
    })

    ct["attempts"] += 1
    if was_match:
        ct["matches"] += 1
    ct["recent"] = (ct["recent"] + [1 if was_match else 0])[-WINDOW_SIZE:]
    ct["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Drift alert counter
    recent = ct["recent"]
    if len(recent) >= 5:
        recent_acc = sum(recent) / len(recent)
        if recent_acc < DRIFT_THRESHOLD:
            ct["drift_alerts"] += 1

    _save(data)


def get_confidence(process_type: str, context_type: str) -> float:
    """
    Returns confidence score [0.0, 1.0] for injecting this context type.
    Uses rolling window of last WINDOW_SIZE results when available.
    Falls back to DEFAULT_CONFIDENCE before enough data.
    """
    data = _load()
    ct = data.get(process_type, {}).get(context_type, {})
    if not ct or ct.get("attempts", 0) < MIN_SAMPLES_FOR_CONFIDENCE:
        return DEFAULT_CONFIDENCE
    recent = ct.get("recent", [])
    if len(recent) >= MIN_SAMPLES_FOR_CONFIDENCE:
        return sum(recent) / len(recent)
    attempts = ct["attempts"]
    matches = ct["matches"]
    return matches / attempts if attempts > 0 else DEFAULT_CONFIDENCE


def should_inject(process_type: str, context_type: str) -> bool:
    """Returns True if context should be injected (confidence above minimum)."""
    return get_confidence(process_type, context_type) >= MIN_INJECT_CONFIDENCE


def is_drift_detected(process_type: str, context_type: str) -> bool:
    """Returns True if recent accuracy suggests rule/logic drift in the benchmark."""
    data = _load()
    ct = data.get(process_type, {}).get(context_type, {})
    recent = ct.get("recent", [])
    if len(recent) < 5:
        return False
    return (sum(recent) / len(recent)) < DRIFT_THRESHOLD


def get_confidence_annotation(process_type: str, context_type: str) -> str:
    """
    Returns a brief annotation for the system prompt:
    e.g. "(87% accurate on last 8 tasks)"  or "(drift detected — verify fresh)"
    """
    conf = get_confidence(process_type, context_type)
    data = _load()
    ct = data.get(process_type, {}).get(context_type, {})
    n = len(ct.get("recent", []))

    if n < MIN_SAMPLES_FOR_CONFIDENCE:
        return ""
    if is_drift_detected(process_type, context_type):
        return f" ⚠ DRIFT DETECTED ({conf:.0%} recent accuracy — threshold may have changed)"
    if conf >= 0.75:
        return f" ({conf:.0%} accurate on last {n} tasks — trust this)"
    return f" ({conf:.0%} recent accuracy — verify before relying on this)"


def get_drift_warning(context_type: str) -> str:
    """
    When drift is detected, return a warning to inject INSTEAD of the computed value.
    This tells Claude to compute fresh from tool data rather than trust our pre-computation.
    """
    warnings = {
        "variance": (
            "⚠ COMPUTATION DRIFT ALERT: Recent variance calculations have not matched "
            "task outcomes. The variance threshold may have changed from the standard value. "
            "DO NOT assume standard thresholds — retrieve the current threshold from available "
            "tools and compute variance fresh."
        ),
        "sla_credit": (
            "⚠ COMPUTATION DRIFT ALERT: Recent SLA credit calculations have not matched "
            "task outcomes. The SLA credit formula or cap may have changed. "
            "Retrieve the current SLA terms from available tools and compute fresh."
        ),
        "proration": (
            "⚠ COMPUTATION DRIFT ALERT: Recent proration calculations have been inaccurate. "
            "Retrieve the exact contract terms from available tools before computing proration."
        ),
    }
    return warnings.get(context_type, (
        f"⚠ COMPUTATION DRIFT ALERT: Pre-computed {context_type} values have been inaccurate "
        f"recently. Compute fresh from tool data — do not rely on standard formulas."
    ))


# ── Accuracy detection ─────────────────────────────────────────────────────────

def check_context_accuracy(
    injected_context: str,
    answer: str,
    process_type: str,
) -> list[tuple[str, bool]]:
    """
    Detect whether the injected context matched the actual task answer.
    Returns list of (context_type, was_match) pairs — may be empty if undetermined.

    Called from REFLECT phase after the task answer is finalized.
    """
    if not injected_context or not answer:
        return []

    results: list[tuple[str, bool]] = []
    ans_lower = answer.lower()
    ctx_lower = injected_context.lower()

    # ── Variance accuracy check ─────────────────────────────────────────────
    if "variance" in ctx_lower and process_type in (
        "invoice_reconciliation", "procurement", "expense_approval"
    ):
        # Determine what we recommended
        # finance_tools produces "within {threshold}% threshold → APPROVE" or
        # "exceeds {threshold}% threshold → ESCALATE"
        if (
            "does not exceed" in ctx_lower
            or "within" in ctx_lower
            or "recommended action: approve" in ctx_lower
            or "→ approve" in ctx_lower
        ):
            we_said_approve = True
        elif (
            "requires escalation: true" in ctx_lower
            or "escalate for approval" in ctx_lower
            or ("exceeds" in ctx_lower and "within" not in ctx_lower)
        ):
            we_said_approve = False
        else:
            we_said_approve = None

        if we_said_approve is not None:
            approve_signals = ["approv", "authorized", "payment scheduled", "process payment"]
            escalate_signals = ["escalat", "reject", "denied", "flag", "requires review",
                                "over threshold", "exceeds", "above limit"]
            answer_approved = any(s in ans_lower for s in approve_signals)
            answer_escalated = any(s in ans_lower for s in escalate_signals)

            if answer_approved and not answer_escalated:
                results.append(("variance", we_said_approve))
            elif answer_escalated and not answer_approved:
                results.append(("variance", not we_said_approve))
            # If both signals present or neither: ambiguous — skip

    # ── SLA credit accuracy check ───────────────────────────────────────────
    if "sla credit" in ctx_lower and process_type == "sla_breach":
        m = re.search(r'\$([0-9,]+(?:\.\d{1,2})?)', injected_context)
        if m:
            credit_str = m.group(1).replace(",", "")
            # Check if this credit amount (or close variant) appears in answer
            # Allow for minor rounding differences (within $1)
            try:
                our_val = float(credit_str)
                # Look for any dollar amount in answer within $1 of ours
                answer_amounts = [
                    float(x.replace(",", ""))
                    for x in re.findall(r'\$([0-9,]+(?:\.\d{1,2})?)', answer)
                ]
                matched = any(abs(v - our_val) <= 1.0 for v in answer_amounts)
                results.append(("sla_credit", matched))
            except (ValueError, TypeError):
                pass

    # ── Proration accuracy check ────────────────────────────────────────────
    if "proration" in ctx_lower or "remaining value" in ctx_lower:
        m = re.search(r'\$([0-9,]+(?:\.\d{1,2})?) remaining', injected_context, re.IGNORECASE)
        if m:
            our_str = m.group(1).replace(",", "")
            try:
                our_val = float(our_str)
                answer_amounts = [
                    float(x.replace(",", ""))
                    for x in re.findall(r'\$([0-9,]+(?:\.\d{1,2})?)', answer)
                ]
                matched = any(abs(v - our_val) <= 1.0 for v in answer_amounts)
                results.append(("proration", matched))
            except (ValueError, TypeError):
                pass

    return results


def get_context_stats() -> dict:
    """Return full context quality stats (used by /rl/status endpoint)."""
    data = _load()
    summary: dict = {}
    for pt, ctypes in data.items():
        summary[pt] = {}
        for ctx_type, ct in ctypes.items():
            recent = ct.get("recent", [])
            conf = (sum(recent) / len(recent)) if recent else DEFAULT_CONFIDENCE
            summary[pt][ctx_type] = {
                "confidence": round(conf, 3),
                "attempts": ct.get("attempts", 0),
                "drift_alerts": ct.get("drift_alerts", 0),
                "status": (
                    "drift" if is_drift_detected(pt, ctx_type)
                    else "low" if conf < 0.75
                    else "high"
                ),
            }
    return summary
