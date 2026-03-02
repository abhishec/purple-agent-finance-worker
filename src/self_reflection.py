"""
self_reflection.py
Pre-return answer quality check — agent scores its own answer before submitting.
Ported from BrainOS computeAgentQuality() + structured memory extraction.

Before returning an answer to the benchmark, Haiku evaluates:
  1. Does it address ALL parts of the task?
  2. Are required fields present (amounts, IDs, decisions)?
  3. Is the reasoning complete?

If score < 0.65, Haiku generates a targeted improvement prompt and the
executor gets one more attempt. This catches incomplete answers BEFORE
the benchmark judge sees them.

This is the highest-leverage single addition for benchmark scoring.
"""
from __future__ import annotations
import asyncio
import json
import re

from src.config import ANTHROPIC_API_KEY
from src.token_budget import _is_bracket_format


import re as _re


def _has_structured_completion(answer: str) -> bool:
    """True if answer contains any field:value completion signal — domain-agnostic."""
    # Pattern 1: explicit field:value pairs (catches Risk rating:, Decision:, Total:, etc.)
    field_value_pattern = _re.compile(
        r'^[A-Za-z][A-Za-z\s_]{1,30}:\s*.{3,}',  # "Field name: value"
        _re.MULTILINE
    )
    if field_value_pattern.search(answer):
        return True

    # Pattern 2: expanded completion markers (original + domain-specific additions)
    _COMPLETION_MARKERS = {
        # Original
        "approved", "rejected", "completed", "total:", "amount:", "decision:",
        # Financial
        "credit:", "debit:", "balance:", "variance:", "penalty:", "refund:",
        # Risk/compliance
        "risk:", "rating:", "score:", "level:", "finding:",
        # Status/resolution
        "status:", "resolved:", "closed:", "escalated:", "flagged:",
        "processed:", "authorized:", "denied:", "blocked:",
        # Action
        "recommendation:", "action:", "next step:", "outcome:",
    }
    answer_lower = answer.lower()
    return any(m in answer_lower for m in _COMPLETION_MARKERS)

REFLECTION_MODEL = "claude-haiku-4-5-20251001"
REFLECTION_TIMEOUT = 8.0
IMPROVE_THRESHOLD = 0.65     # reflect + improve if score below this
REFLECTION_ENABLED = True    # set False to skip (e.g. if budget tight)


# ── Reflection ────────────────────────────────────────────────────────────────

async def reflect_on_answer(
    task_text: str,
    answer: str,
    process_type: str,
    tool_count: int,
) -> dict:
    """
    Score the answer quality and return improvement hints if needed.
    Returns:
      {score: float, complete: bool, missing: list[str], improve_prompt: str}
    Fire-and-forget safe — on any failure returns {score: 0.8, complete: True}.
    """
    # Bracket-format = exact_match target — valid by definition, never improve.
    # Haiku would score '["INV-001"]' as "incomplete" triggering a corruption pass.
    # This check must come BEFORE the API-key check so bracket answers always get
    # score=1.0 even when the API key is missing (e.g. in test environments).
    # Use strict JSON-array check: prose like "Rejected. [Reason: ...]" must
    # not skip reflection — only true JSON lists like ["INV-001"] should.
    if _is_bracket_format(answer):
        return {"score": 1.0, "complete": True, "missing": [], "improve_prompt": ""}

    if not REFLECTION_ENABLED or not ANTHROPIC_API_KEY:
        return {"score": 0.8, "complete": True, "missing": [], "improve_prompt": ""}

    # Fast heuristic pre-check — skip Haiku if clearly good
    heuristic = _heuristic_score(answer, task_text, tool_count)
    if heuristic >= 0.85:
        return {"score": heuristic, "complete": True, "missing": [], "improve_prompt": ""}

    try:
        result = await asyncio.wait_for(
            _call_reflection(task_text, answer, process_type, tool_count),
            timeout=REFLECTION_TIMEOUT,
        )
        return result
    except Exception:
        return {"score": heuristic, "complete": heuristic >= IMPROVE_THRESHOLD, "missing": [], "improve_prompt": ""}


def _heuristic_score(answer: str, task_text: str, tool_count: int) -> float:
    """
    Fast zero-API quality score. Mirrors BrainOS computeAgentQuality().
    Penalizes: empty data, no tool calls, very short answers, error phrases.
    Rewards: structured output, tool usage, completeness markers.
    """
    score = 0.5   # conservative baseline (BrainOS uses 0.5 too)

    # Length signals
    answer_stripped = answer.strip()
    length = len(answer_stripped)
    # Strict JSON-array check — prose starting with '[' must not get bracket bonus.
    is_bracket_format = _is_bracket_format(answer_stripped)

    if is_bracket_format:
        # Bracket-format = exact_match target — correct format, not too short
        score += 0.15
    elif length > 800:   score += 0.20
    elif length > 400:   score += 0.15
    elif length > 150:   score += 0.08
    elif length < 50:    score -= 0.20   # suspiciously short

    # Tool usage signals
    if tool_count >= 5:   score += 0.15
    elif tool_count >= 3: score += 0.10
    elif tool_count >= 1: score += 0.05
    elif tool_count == 0: score -= 0.10  # no tools = probably incomplete

    # Structure signals — reward JSON/structured content
    if "{" in answer and "}" in answer:  score += 0.08
    if _has_structured_completion(answer):
        score += 0.08

    # Penalty: error phrases
    error_phrases = ["task failed", "error occurred", "unable to", "cannot access",
                     "no data found", "tool unavailable", "token budget exhausted"]
    if any(p in answer.lower() for p in error_phrases):
        score -= 0.25

    # Penalty: empty data arrays (BrainOS -0.25 rule)
    if re.search(r'"data"\s*:\s*\[\s*\]', answer):
        score -= 0.25
    if re.search(r'"results"\s*:\s*\[\s*\]', answer):
        score -= 0.15

    # Penalty: incomplete markers
    if "TODO" in answer or "placeholder" in answer.lower():
        score -= 0.20

    return max(0.0, min(1.0, score))


async def _call_reflection(
    task_text: str,
    answer: str,
    process_type: str,
    tool_count: int,
) -> dict:
    """Call Haiku to evaluate and identify gaps."""
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    task_snippet = task_text[:400]
    answer_snippet = answer[:600]
    process_label = process_type.replace("_", " ").title()

    resp = await client.messages.create(
        model=REFLECTION_MODEL,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": (
                f"Process: {process_label} | Tools used: {tool_count}\n"
                f"Task: {task_snippet}\n"
                f"Answer: {answer_snippet}\n\n"
                "Evaluate this answer. Does it:\n"
                "1. Address ALL parts of the task?\n"
                "2. Include required fields (amounts, IDs, decisions, reasons)?\n"
                "3. Show evidence of data lookup (not just reasoning)?\n\n"
                "Reply JSON only:\n"
                '{"score": 0.0-1.0, "complete": true/false, '
                '"missing": ["item1", "item2"], '
                '"improve_prompt": "one sentence telling what to add"}'
            ),
        }],
    )
    text = resp.content[0].text if resp.content else ""
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if m:
        parsed = json.loads(m.group())
        return {
            "score": float(parsed.get("score", 0.7)),
            "complete": bool(parsed.get("complete", True)),
            "missing": parsed.get("missing", []),
            "improve_prompt": parsed.get("improve_prompt", ""),
        }
    return {"score": 0.7, "complete": True, "missing": [], "improve_prompt": ""}


# ── Improvement ───────────────────────────────────────────────────────────────

def build_improvement_prompt(reflection: dict, task_text: str) -> str:
    """
    Build a follow-up prompt for Claude to improve the answer.
    Only called when score < IMPROVE_THRESHOLD.
    """
    missing = reflection.get("missing", [])
    hint = reflection.get("improve_prompt", "")

    parts = ["Your previous answer was incomplete. Improve it:"]
    if missing:
        parts.append(f"Missing: {', '.join(missing)}")
    if hint:
        parts.append(f"Specifically: {hint}")
    parts.append("Provide the complete, final answer now.")
    return "\n".join(parts)


def should_improve(reflection: dict) -> bool:
    """True if the answer needs improvement."""
    return (
        not reflection.get("complete", True)
        or reflection.get("score", 1.0) < IMPROVE_THRESHOLD
    ) and bool(reflection.get("missing") or reflection.get("improve_prompt"))
