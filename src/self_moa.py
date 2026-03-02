"""
self_moa.py
Mixture-of-Agents (MoA) synthesis — zero additional infrastructure.

Two patterns:
  Pattern 1 — Dual top_p synthesis (50 lines, +6% quality):
    Run same query twice in parallel with different sampling temperatures.
    Consensus check → return longer answer or synthesize if divergent.

  Pattern 2 — 3-lens synthesis (for complex multi-part tasks):
    Run 3 Haiku calls in parallel, each with a different analytical lens.
    Sonnet synthesizes all 3 into a final answer.

Public API:
  synthesize_answer(task_text, system_prompt, use_3lens, model) -> (answer, consensus_score)
  quick_synthesize(task_text, system_prompt) -> answer
"""
from __future__ import annotations

import asyncio
from typing import NamedTuple

import anthropic

from src.config import ANTHROPIC_API_KEY, FALLBACK_MODEL

# ── Constants ─────────────────────────────────────────────────────────────────

_HAIKU_MODEL = "claude-haiku-4-5-20251001"
_SYNTH_MODEL = "claude-sonnet-4-6"          # used only for 3-lens synthesis step

_DUAL_TIMEOUT_TOTAL = 15.0    # seconds — total budget for dual top_p
_DUAL_TIMEOUT_EACH  = 12.0    # seconds — each individual call before fallback
_LENS_TIMEOUT_EACH  = 12.0    # seconds — each lens call
_SYNTH_TIMEOUT      = 20.0    # seconds — sonnet synthesis step

_OVERLAP_HIGH = 0.70          # above this: answers agree, return longer one
_OVERLAP_CALL_THRESHOLD = 0.70  # alias for clarity in code


# ── Word-overlap consensus ─────────────────────────────────────────────────────

def _word_set(text: str) -> set[str]:
    """Lowercase word tokens, stop-words stripped for signal clarity."""
    _STOP = frozenset(
        "the a an and or but is are was were be been being have has had do does did "
        "will would could should may might shall to of in on at by for with from as "
        "it its this that these those i we you he she they me us him her them my our "
        "your his its their what which who whom when where why how all any some".split()
    )
    return {w for w in text.lower().split() if w.isalpha() and w not in _STOP}


def compute_overlap(a: str, b: str) -> float:
    """Jaccard-style overlap between two answer strings. Range [0, 1]."""
    set_a = _word_set(a)
    set_b = _word_set(b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def _best_of_two(a: str, b: str) -> str:
    """Return the longer (more complete) of two answers."""
    return a if len(a) >= len(b) else b


# ── Shared Anthropic async client factory ─────────────────────────────────────

def _client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)


# ── Single inference helper ───────────────────────────────────────────────────

async def _call_haiku(
    system_prompt: str,
    user_message: str,
    top_p: float,
    max_tokens: int = 1024,
    timeout: float = _DUAL_TIMEOUT_EACH,
) -> str:
    """Single Haiku inference with configurable top_p and timeout."""
    client = _client()
    response = await asyncio.wait_for(
        client.messages.create(
            model=_HAIKU_MODEL,
            max_tokens=max_tokens,
            top_p=top_p,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        ),
        timeout=timeout,
    )
    return response.content[0].text if response.content else ""


# ── Pattern 1: Dual top_p synthesis ──────────────────────────────────────────

async def _dual_top_p(
    task_text: str,
    system_prompt: str,
    model: str,
) -> tuple[str, float]:
    """
    Run task_text with top_p=0.85 (conservative) and top_p=0.99 (exploratory)
    in parallel.  Compute word-overlap; if high → return longer answer,
    if low → use Haiku to synthesize best of both.

    Returns (answer, consensus_score).
    """
    conservative_coro = _call_haiku(system_prompt, task_text, top_p=0.85, timeout=_DUAL_TIMEOUT_EACH)
    exploratory_coro  = _call_haiku(system_prompt, task_text, top_p=0.99, timeout=_DUAL_TIMEOUT_EACH)

    try:
        results = await asyncio.wait_for(
            asyncio.gather(conservative_coro, exploratory_coro, return_exceptions=True),
            timeout=_DUAL_TIMEOUT_TOTAL,
        )
    except asyncio.TimeoutError:
        # Total budget exceeded — return whichever completed via individual timeout
        results = ["", ""]

    answer_a: str = results[0] if isinstance(results[0], str) else ""
    answer_b: str = results[1] if isinstance(results[1], str) else ""

    # Fallback: if one failed entirely, return the other
    if not answer_a and not answer_b:
        return "", 0.0
    if not answer_a:
        return answer_b, 0.5
    if not answer_b:
        return answer_a, 0.5

    overlap = compute_overlap(answer_a, answer_b)

    if overlap >= _OVERLAP_HIGH:
        # Answers agree — return the longer/more complete one
        return _best_of_two(answer_a, answer_b), overlap

    # Divergent answers — run one Haiku synthesis call
    synthesis_prompt = (
        "You are a synthesis engine. You have received two independent answers to the same task. "
        "Produce a single best answer by taking the most accurate, complete, and useful elements "
        "from both. Do not say 'Answer A says...' — just output the synthesized answer directly."
    )
    synthesis_user = (
        f"TASK:\n{task_text}\n\n"
        f"ANSWER A (conservative):\n{answer_a}\n\n"
        f"ANSWER B (exploratory):\n{answer_b}\n\n"
        "Synthesize the best answer:"
    )

    try:
        synthesized = await asyncio.wait_for(
            _call_haiku(synthesis_prompt, synthesis_user, top_p=0.85, max_tokens=1024),
            timeout=_DUAL_TIMEOUT_EACH,
        )
        return synthesized, overlap
    except Exception:
        # Synthesis failed — return longer individual answer
        return _best_of_two(answer_a, answer_b), overlap


# ── Pattern 2: 3-lens synthesis ───────────────────────────────────────────────

_LENS_PROMPTS = {
    "risk": (
        "You are a risk-focused analyst. For the given task, identify what could go wrong, "
        "any policy violations, missing data, edge cases, and potential failure modes. "
        "Be specific and concrete — not generic. Flag exact amounts, dates, or entities "
        "that present risk."
    ),
    "execution": (
        "You are an execution-focused analyst. For the given task, focus on what actions "
        "need to be taken and in what exact order. Be step-by-step and precise. "
        "List dependencies between steps. Identify which actions are reversible vs. irreversible."
    ),
    "data_quality": (
        "You are a data-quality analyst. For the given task, verify all numbers for consistency, "
        "check for completeness gaps, flag any missing required fields, and identify any "
        "arithmetic inconsistencies. Cross-check figures where possible."
    ),
}


async def _three_lens(
    task_text: str,
    system_prompt: str,
) -> tuple[str, float]:
    """
    Run 3 parallel Haiku calls with different analytical lenses.
    Compute pairwise consensus. Synthesize with Sonnet.

    Returns (answer, mean_pairwise_consensus_score).
    """
    risk_coro      = _call_haiku(_LENS_PROMPTS["risk"],        task_text, top_p=0.85, timeout=_LENS_TIMEOUT_EACH)
    exec_coro      = _call_haiku(_LENS_PROMPTS["execution"],   task_text, top_p=0.85, timeout=_LENS_TIMEOUT_EACH)
    quality_coro   = _call_haiku(_LENS_PROMPTS["data_quality"],task_text, top_p=0.85, timeout=_LENS_TIMEOUT_EACH)

    results = await asyncio.gather(risk_coro, exec_coro, quality_coro, return_exceptions=True)

    lens_a: str = results[0] if isinstance(results[0], str) else ""
    lens_b: str = results[1] if isinstance(results[1], str) else ""
    lens_c: str = results[2] if isinstance(results[2], str) else ""

    # Pairwise overlap for consensus scoring
    pairs = [
        compute_overlap(lens_a, lens_b),
        compute_overlap(lens_b, lens_c),
        compute_overlap(lens_a, lens_c),
    ]
    mean_consensus = sum(pairs) / len(pairs) if pairs else 0.0

    # Build Sonnet synthesis context
    synthesis_system = (
        "You are a senior analyst synthesizing multiple expert perspectives into one comprehensive answer. "
        "Do NOT repeat each lens verbatim. Extract the key insights from each perspective and produce "
        "a single, well-structured, actionable answer. Use headers where appropriate."
    )
    synthesis_user = (
        f"TASK:\n{task_text}\n\n"
        f"SYSTEM CONTEXT:\n{system_prompt}\n\n"
        f"RISK ANALYSIS:\n{lens_a or '(not available)'}\n\n"
        f"EXECUTION PLAN:\n{lens_b or '(not available)'}\n\n"
        f"DATA QUALITY REVIEW:\n{lens_c or '(not available)'}\n\n"
        "Produce a single synthesized answer:"
    )

    client = _client()
    try:
        response = await asyncio.wait_for(
            client.messages.create(
                model=_SYNTH_MODEL,
                max_tokens=1500,
                system=synthesis_system,
                messages=[{"role": "user", "content": synthesis_user}],
            ),
            timeout=_SYNTH_TIMEOUT,
        )
        synthesized = response.content[0].text if response.content else ""
        return synthesized, mean_consensus
    except Exception:
        # Sonnet synthesis failed — fall back to longest lens answer
        best = max([lens_a, lens_b, lens_c], key=len)
        return best, mean_consensus


# ── Public API ────────────────────────────────────────────────────────────────

async def synthesize_answer(
    task_text: str,
    system_prompt: str,
    use_3lens: bool = False,
    model: str = _HAIKU_MODEL,
) -> tuple[str, float]:
    """
    Synthesize the best answer for task_text using MoA patterns.

    Args:
        task_text:     The user task / query.
        system_prompt: System context to inject (agent persona, process info, etc.)
        use_3lens:     If True, use 3-lens Sonnet synthesis (for complex multi-part tasks).
                       If False (default), use dual top_p Haiku synthesis.
        model:         Base model for Haiku calls. Ignored for Sonnet synthesis step.

    Returns:
        (answer, consensus_score) where consensus_score in [0, 1].
        Higher consensus = the two/three answers agreed more closely.
    """
    try:
        if use_3lens:
            return await _three_lens(task_text, system_prompt)
        else:
            return await _dual_top_p(task_text, system_prompt, model)
    except Exception:
        # Total fallback — run a single standard call
        try:
            answer = await _call_haiku(system_prompt, task_text, top_p=0.9)
            return answer, 0.0
        except Exception:
            return "", 0.0


async def quick_synthesize(task_text: str, system_prompt: str) -> str:
    """
    Fast MoA synthesis using dual top_p only.
    Use this for all tasks as a drop-in replacement for a single Claude call.
    Returns the best answer string (consensus score discarded).
    """
    answer, _ = await synthesize_answer(
        task_text=task_text,
        system_prompt=system_prompt,
        use_3lens=False,
    )
    # If synthesis produced nothing, fall back to a plain Haiku call
    if not answer:
        try:
            answer = await _call_haiku(system_prompt, task_text, top_p=0.9)
        except Exception:
            answer = ""
    return answer


# ── Numeric MoA — dual top_p for tool-result tasks ───────────────────────────
#
# Problem: existing quick_synthesize only runs when tool_count == 0.
# Numeric tasks (amortization, NPV, invoice reconciliation) use tools
# but still benefit from dual-interpretation synthesis to catch math errors.
#
# Pattern:
#   1. Run "restate-and-verify" Haiku call (conservative top_p): given the task +
#      collected answer, verify the numbers and restate the conclusion cleanly.
#   2. Run "challenge" Haiku call (exploratory top_p): look for any errors or
#      alternative interpretations of the numeric result.
#   3. Synthesize: if both agree → return longer; if diverge → merge.

_NUMERIC_VERIFY_PROMPT = """\
You are a financial verification specialist. You are given a task and an agent's answer.
Your job: verify the numeric calculations are correct and restate the final answer cleanly.
If numbers look correct: restate the answer more clearly with proper formatting.
If you spot an error: provide the corrected calculation and answer.
Be concise. Show key numbers. Do not repeat the entire answer verbatim."""

_NUMERIC_CHALLENGE_PROMPT = """\
You are a skeptical financial auditor. You are given a task and an agent's answer.
Your job: challenge the answer — look for arithmetic errors, wrong formulas, missed steps.
If the answer is correct: confirm it and add any useful context or caveats.
If you find an error: provide the correct calculation.
Be specific. Focus on the numbers, not the prose."""


async def numeric_moa_synthesize(
    task_text: str,
    initial_answer: str,
    system_context: str = "",
) -> str:
    """
    Dual-path verification MoA for numeric/financial task answers.
    Runs two Haiku interpretations of the answer in parallel, synthesizes the best.

    Args:
        task_text:      Original task
        initial_answer: Answer from main execution (may have math errors)
        system_context: Optional system context for grounding

    Returns: synthesized/verified answer string, or initial_answer on failure
    """
    # Bracket-format answers are exact_match targets — never run numeric MoA on them.
    # IDs like "INV-001" contain digits but are not financial calculations.
    if initial_answer.strip().startswith('['):
        return initial_answer

    # Only run for answers with actual financial numeric content
    import re
    if not re.search(r'\d[\d,.]*', initial_answer):
        return initial_answer

    user_content = (
        f"TASK:\n{task_text[:600]}\n\n"
        f"AGENT ANSWER:\n{initial_answer[:1200]}"
    )

    verify_coro   = _call_haiku(_NUMERIC_VERIFY_PROMPT,    user_content, top_p=0.80, max_tokens=800, timeout=10.0)
    challenge_coro = _call_haiku(_NUMERIC_CHALLENGE_PROMPT, user_content, top_p=0.95, max_tokens=800, timeout=10.0)

    try:
        results = await asyncio.wait_for(
            asyncio.gather(verify_coro, challenge_coro, return_exceptions=True),
            timeout=14.0,
        )
    except asyncio.TimeoutError:
        return initial_answer

    verified: str  = results[0] if isinstance(results[0], str) and results[0] else ""
    challenged: str = results[1] if isinstance(results[1], str) and results[1] else ""

    if not verified and not challenged:
        return initial_answer
    if not verified:
        return challenged if len(challenged) > len(initial_answer) * 0.5 else initial_answer
    if not challenged:
        return verified if len(verified) > len(initial_answer) * 0.5 else initial_answer

    overlap = compute_overlap(verified, challenged)

    if overlap >= _OVERLAP_HIGH:
        # High agreement — return the more complete one
        best = _best_of_two(verified, challenged)
        return best if len(best) > len(initial_answer) * 0.4 else initial_answer

    # Divergent — quick Haiku synthesis
    synth_user = (
        f"ORIGINAL TASK:\n{task_text[:400]}\n\n"
        f"INITIAL ANSWER:\n{initial_answer[:600]}\n\n"
        f"VERIFICATION VIEW:\n{verified[:500]}\n\n"
        f"CHALLENGE VIEW:\n{challenged[:500]}\n\n"
        "Produce a single correct, concise final answer:"
    )
    synth_system = (
        "Synthesize these perspectives into the single most accurate answer. "
        "Prioritize mathematical correctness. Be concise."
    )
    try:
        synthesized = await asyncio.wait_for(
            _call_haiku(synth_system, synth_user, top_p=0.85, max_tokens=900, timeout=10.0),
            timeout=12.0,
        )
        return synthesized if synthesized and len(synthesized) > 50 else initial_answer
    except Exception:
        return _best_of_two(verified, challenged) or initial_answer
