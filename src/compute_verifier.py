"""
compute_verifier.py  — Compute Math Reflection Gate
Self-reflection gate after COMPUTE state.

Inspired by BrainOS platform/lib/brain/five-phase-executor.ts explicit phase validation.

Problem: After the COMPUTE state, the agent has calculated financial values.
If those values are wrong (bad arithmetic, wrong formula, off-by-one), they
get written to the DB in MUTATE → functional score = 0.

Fix: Before advancing past COMPUTE, run a fast Haiku critique:
  1. Extract numbers from the answer
  2. Ask Haiku: "Do these calculations look correct? List any errors."
  3. If errors detected → return correction prompt → caller re-runs COMPUTE
  4. If clean → proceed to MUTATE

Cost: ~200 tokens (Haiku) per task with a COMPUTE state. Worth it.

Usage in worker_brain.py:
    result = await verify_compute_output(task_text, answer, process_type)
    if result["has_errors"]:
        # Re-run solve_with_claude with result["correction_prompt"]
"""
from __future__ import annotations

import asyncio
import re
from typing import NamedTuple

import anthropic

from src.config import ANTHROPIC_API_KEY

_HAIKU = "claude-haiku-4-5-20251001"
_TIMEOUT = 8.0   # seconds — tight budget, Haiku is fast

# Numeric patterns we care about verifying
_NUMBER_RE = re.compile(
    r'(?:[$£€¥]?\s*\d[\d,]*\.?\d*(?:\s*%)?)',
)

# Process types that NEVER need math verification — explicit exclusion list.
# All other types proceed through verification whenever numbers are present.
# This is the inverted-default pattern: safe by default, opt-out only.
_COMPUTE_LIGHT = frozenset({
    "notification_only",   # pure notification, no calculation
    "status_inquiry",      # lookup only, no calculation
    "document_retrieval",  # fetch only, no calculation
})


class ComputeVerifyResult(NamedTuple):
    has_errors: bool
    confidence: float    # 0–1, how confident we are in the answer
    issues: list[str]    # detected problems
    correction_prompt: str  # non-empty only when has_errors=True


def _extract_numbers(text: str) -> list[str]:
    """Pull significant numeric values from answer text."""
    return _NUMBER_RE.findall(text)[:20]  # cap at 20 to avoid prompt bloat


async def verify_compute_output(
    task_text: str,
    answer: str,
    process_type: str,
) -> ComputeVerifyResult:
    """
    Run Haiku critique on the computed answer.
    Returns ComputeVerifyResult — caller decides whether to re-run.

    Fast-path: if answer has no numbers, or process type doesn't need it,
    returns clean result immediately (no API cost).
    """
    # Fast-path: bracket-format = exact_match target, not a financial computation
    if answer.strip().startswith('['):
        return ComputeVerifyResult(False, 0.95, [], "")

    # Fast-path: explicit exclusion — provably calculation-free process types
    if process_type in _COMPUTE_LIGHT:
        return ComputeVerifyResult(False, 0.85, [], "")

    # Fast-path: no numeric content to verify — number-presence drives verification,
    # not process type. If there are no numbers there is no math to check.
    numbers = _extract_numbers(answer)
    if not numbers or len(answer) < 100:
        return ComputeVerifyResult(False, 0.85, [], "")

    # Proceed with verification for ALL other process types that contain numbers.

    system_prompt = """\
You are a financial calculation auditor. Review computations in an agent's answer.

Your job:
1. Check if numerical results are plausible and internally consistent
2. Spot obvious arithmetic errors, wrong formulas, or impossible values
3. Flag values that contradict each other (e.g. total ≠ sum of parts)

Respond with JSON only:
{
  "has_errors": true/false,
  "confidence": 0.0-1.0,
  "issues": ["description of issue 1", ...],
  "correction_hint": "Specific instruction to fix the error, or empty string"
}

Be concise. Only flag clear errors, not stylistic issues."""

    user_msg = (
        f"TASK:\n{task_text[:800]}\n\n"
        f"AGENT ANSWER (excerpt):\n{answer[:1500]}\n\n"
        f"Key numbers found: {', '.join(numbers[:10])}\n\n"
        "Are the calculations correct? Return JSON."
    )

    try:
        client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        resp = await asyncio.wait_for(
            client.messages.create(
                model=_HAIKU,
                max_tokens=300,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            ),
            timeout=_TIMEOUT,
        )
        raw = resp.content[0].text.strip() if resp.content else "{}"

        # Strip markdown fences
        if raw.startswith("```"):
            raw = re.sub(r"```(?:json)?\n?", "", raw).strip().rstrip("```").strip()

        import json
        data = json.loads(raw)
        has_errors = bool(data.get("has_errors"))
        confidence = float(data.get("confidence", 0.85))
        issues = data.get("issues", [])
        hint = data.get("correction_hint", "")

        correction_prompt = ""
        if has_errors and hint:
            correction_prompt = (
                f"Your previous answer had calculation errors:\n"
                f"{chr(10).join(f'- {i}' for i in issues)}\n\n"
                f"Correction needed: {hint}\n\n"
                f"Please recalculate and provide the corrected answer for:\n{task_text[:600]}"
            )

        return ComputeVerifyResult(has_errors, confidence, issues, correction_prompt)

    except Exception:
        # Never block execution — verification is best-effort
        return ComputeVerifyResult(False, 0.75, [], "")
