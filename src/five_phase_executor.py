"""
five_phase_executor.py
5-phase task executor — alternative to a single Claude call for complex tasks.

Phases:
  1. PLAN      (Haiku)  — decompose into 2-4 JSON subtasks
  2. GATHER    (tools)  — async tool calls for subtasks requiring data
  3. SYNTHESIZE (Sonnet) — comprehensive analysis from plan + data
  4. ARTIFACT  (Haiku)  — format into clean structured deliverable
  5. INSIGHT   (fire-and-forget) — extract_and_store to knowledge base

Public API:
  five_phase_execute(task_text, system_context, process_type, on_tool_call, tools)
      -> (answer, tool_count, quality_score)
  should_use_five_phase(task_text, tool_count_so_far) -> bool
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Awaitable, Callable

import anthropic

from src.config import ANTHROPIC_API_KEY, FALLBACK_MODEL
from src.knowledge_extractor import extract_and_store

# ── Constants ─────────────────────────────────────────────────────────────────

_HAIKU_MODEL  = "claude-haiku-4-5-20251001"
_SONNET_MODEL = "claude-sonnet-4-6"

_PLAN_TOKENS     = 200
_SYNTH_TOKENS    = 1500
_ARTIFACT_TOKENS = 800

_PLAN_TIMEOUT     = 10.0   # seconds
_GATHER_TIMEOUT   = 30.0   # seconds — entire gather phase
_SYNTH_TIMEOUT    = 45.0   # seconds
_ARTIFACT_TIMEOUT = 20.0   # seconds

# Heuristic thresholds for should_use_five_phase
_COMPLEX_TASK_LENGTH   = 200    # characters
_COMPLEX_TOOL_COUNT    = 3      # previous tool calls
_LOW_QUALITY_THRESHOLD = 0.65   # prior attempt score below this triggers 5-phase

_MULTI_QUESTION_PATTERN = re.compile(r'[?]{1}.*[?]{1}', re.DOTALL)


# ── Anthropic client factory ──────────────────────────────────────────────────

def _client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)


# ── Phase helpers ─────────────────────────────────────────────────────────────

async def _phase_plan(task_text: str, system_context: str) -> dict[str, Any]:
    """
    PLAN phase (Haiku): decompose task into 2-4 JSON subtasks.

    Expected output schema:
    {
      "subtasks": ["...", "..."],
      "process_type": "...",
      "requires_tools": true/false
    }

    Falls through gracefully: returns a minimal plan if Haiku fails or
    produces invalid JSON.
    """
    plan_system = (
        "You are a task decomposition engine. Given a task and system context, "
        "output ONLY valid JSON with this exact structure:\n"
        '{"subtasks": ["subtask1", "subtask2"], "process_type": "string", "requires_tools": true}\n'
        "Rules:\n"
        "- 2 to 4 subtasks, each a concrete actionable step\n"
        "- process_type: one of expense_approval, procurement, compliance_audit, payroll, "
        "  month_end_close, general, or the most relevant type\n"
        "- requires_tools: true if any subtask needs data from external systems\n"
        "Output ONLY the JSON object. No prose, no explanation."
    )
    plan_user = f"SYSTEM CONTEXT:\n{system_context}\n\nTASK:\n{task_text}"

    client = _client()
    try:
        response = await asyncio.wait_for(
            client.messages.create(
                model=_HAIKU_MODEL,
                max_tokens=_PLAN_TOKENS,
                system=plan_system,
                messages=[{"role": "user", "content": plan_user}],
            ),
            timeout=_PLAN_TIMEOUT,
        )
        raw = response.content[0].text.strip() if response.content else ""
        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        plan = json.loads(raw)
        # Validate structure
        if not isinstance(plan.get("subtasks"), list):
            raise ValueError("subtasks missing or not a list")
        return plan
    except Exception as exc:
        # Graceful fallback — minimal plan
        return {
            "subtasks": [task_text],
            "process_type": "general",
            "requires_tools": True,
            "_plan_error": str(exc),
        }


async def _phase_gather(
    subtasks: list[str],
    on_tool_call: Callable[[str, dict], Awaitable[dict]] | None,
    tools: list[dict] | None,
) -> tuple[list[dict], int]:
    """
    GATHER phase: execute subtasks that require tool calls.

    For each subtask, uses Haiku to determine the best tool call, then fires
    the tool via on_tool_call(). Runs subtasks concurrently (bounded by
    asyncio.gather). Caps at 8 tool calls total.

    Returns (gathered_results, tool_count).
    """
    if on_tool_call is None or not tools:
        return [], 0

    tool_names = [t.get("name", "") for t in tools]
    tool_names_str = ", ".join(tool_names[:20])

    gather_system = (
        "You are a tool selector. Given a subtask and available tools, output ONLY valid JSON:\n"
        '{"tool": "tool_name", "params": {"key": "value"}}\n'
        "If no tool is relevant, output: {}\n"
        "Available tools: " + tool_names_str
    )

    async def _execute_subtask(subtask: str) -> dict:
        client = _client()
        try:
            response = await asyncio.wait_for(
                client.messages.create(
                    model=_HAIKU_MODEL,
                    max_tokens=150,
                    system=gather_system,
                    messages=[{"role": "user", "content": f"Subtask: {subtask}"}],
                ),
                timeout=10.0,
            )
            raw = response.content[0].text.strip() if response.content else "{}"
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            tool_call = json.loads(raw)
            if not tool_call or "tool" not in tool_call:
                return {"subtask": subtask, "result": None, "skipped": True}
            result = await on_tool_call(tool_call["tool"], tool_call.get("params", {}))
            return {"subtask": subtask, "tool": tool_call["tool"], "result": result}
        except Exception as exc:
            return {"subtask": subtask, "error": str(exc)}

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*[_execute_subtask(st) for st in subtasks[:8]], return_exceptions=True),
            timeout=_GATHER_TIMEOUT,
        )
    except asyncio.TimeoutError:
        results = []

    gathered = []
    for r in results:
        if isinstance(r, Exception):
            gathered.append({"error": str(r)})
        elif isinstance(r, dict):
            gathered.append(r)

    tool_count = sum(1 for r in gathered if "tool" in r and "error" not in r)
    return gathered, tool_count


async def _phase_synthesize(
    task_text: str,
    system_context: str,
    plan: dict[str, Any],
    gathered: list[dict],
) -> str:
    """
    SYNTHESIZE phase (Sonnet): comprehensive analysis from plan + tool data.
    Falls through to empty string on failure.
    """
    gathered_str = json.dumps(gathered, indent=2, default=str) if gathered else "(no tool data collected)"

    synth_system = (
        "You are a senior analyst producing a comprehensive, accurate analysis. "
        "You have access to the task plan and all data gathered from external systems. "
        "Produce a thorough, well-reasoned answer. Be specific about numbers, dates, and entities. "
        "Do not hedge unnecessarily — the data is in front of you."
    )
    synth_user = (
        f"SYSTEM CONTEXT:\n{system_context}\n\n"
        f"TASK:\n{task_text}\n\n"
        f"EXECUTION PLAN:\n{json.dumps(plan, indent=2)}\n\n"
        f"GATHERED DATA:\n{gathered_str}\n\n"
        "Produce a comprehensive analysis and answer:"
    )

    client = _client()
    try:
        response = await asyncio.wait_for(
            client.messages.create(
                model=_SONNET_MODEL,
                max_tokens=_SYNTH_TOKENS,
                system=synth_system,
                messages=[{"role": "user", "content": synth_user}],
            ),
            timeout=_SYNTH_TIMEOUT,
        )
        return response.content[0].text if response.content else ""
    except Exception:
        return ""


async def _phase_artifact(synthesis: str, task_text: str) -> str:
    """
    ARTIFACT phase (Haiku): reformat synthesis into clean structured deliverable.
    Uses headers, bullet points, formatted dollar amounts and decisions.
    Falls through to raw synthesis on failure.
    """
    if not synthesis:
        return synthesis

    # Bracket-format synthesis = exact_match target — never reformat via artifact phase
    if synthesis.strip().startswith('['):
        return synthesis

    artifact_system = (
        "You are a document formatter. Take the provided analysis and reformat it into a "
        "clean, professional deliverable. Rules:\n"
        "- Use ## headers to separate major sections\n"
        "- Use bullet points for lists of items, steps, or findings\n"
        "- Format dollar amounts as $X,XXX.XX\n"
        "- Format dates as YYYY-MM-DD\n"
        "- Put decisions and recommendations in a ## Decision section at the end\n"
        "- Keep all factual content — do not summarize away any data\n"
        "Output only the formatted document."
    )
    artifact_user = (
        f"ORIGINAL TASK:\n{task_text}\n\n"
        f"ANALYSIS TO FORMAT:\n{synthesis}"
    )

    client = _client()
    try:
        response = await asyncio.wait_for(
            client.messages.create(
                model=_HAIKU_MODEL,
                max_tokens=_ARTIFACT_TOKENS,
                system=artifact_system,
                messages=[{"role": "user", "content": artifact_user}],
            ),
            timeout=_ARTIFACT_TIMEOUT,
        )
        formatted = response.content[0].text if response.content else ""
        return formatted if formatted else synthesis
    except Exception:
        return synthesis  # fall through to raw synthesis


def _estimate_quality(
    answer: str,
    tool_count: int,
    plan: dict[str, Any],
    gathered: list[dict],
) -> float:
    """
    Heuristic quality score [0.25, 0.95] for the 5-phase output.
    Higher score = richer output with more data.
    """
    score = 0.50  # baseline

    # Length signal — longer structured answers tend to be more complete
    if len(answer) > 1000:
        score += 0.10
    if len(answer) > 2000:
        score += 0.05

    # Tool data signal — gathering real data improves quality
    if tool_count > 0:
        score += min(0.15, tool_count * 0.05)

    # Error penalty
    error_count = sum(1 for r in gathered if "error" in r)
    if error_count > 0:
        score -= min(0.15, error_count * 0.05)

    # Plan quality signal
    subtask_count = len(plan.get("subtasks", []))
    if subtask_count >= 2:
        score += 0.05
    if subtask_count >= 3:
        score += 0.05

    # Structural signal — headers and bullets indicate artifact phase succeeded
    if "##" in answer:
        score += 0.05
    if "- " in answer or "• " in answer:
        score += 0.03

    return round(max(0.25, min(0.95, score)), 3)


# ── Public API ────────────────────────────────────────────────────────────────

async def five_phase_execute(
    task_text: str,
    system_context: str,
    process_type: str,
    on_tool_call: Callable[[str, dict], Awaitable[dict]] | None = None,
    tools: list[dict] | None = None,
) -> tuple[str, int, float]:
    """
    Execute a complex task through 5 sequential phases.

    Args:
        task_text:      The user task / query string.
        system_context: Agent persona, process info, knowledge base context.
        process_type:   FSM process type (e.g. "payroll", "compliance_audit").
        on_tool_call:   Async callable(tool_name, params) -> result dict.
                        If None, GATHER phase is skipped.
        tools:          Anthropic-format tool list (for tool selection in GATHER).

    Returns:
        (answer, tool_count, quality_score)
        - answer:       Final formatted deliverable string.
        - tool_count:   Number of successful tool calls made.
        - quality_score: Heuristic quality [0.25, 0.95].

    All phases are graceful — failure in any phase falls through to the next
    using whatever data is available.
    """
    # Phase 1: PLAN
    try:
        plan = await _phase_plan(task_text, system_context)
    except Exception:
        plan = {"subtasks": [task_text], "process_type": process_type, "requires_tools": False}

    # Phase 2: GATHER
    gathered: list[dict] = []
    tool_count = 0
    if plan.get("requires_tools", True) and on_tool_call is not None:
        try:
            gathered, tool_count = await _phase_gather(
                subtasks=plan.get("subtasks", [task_text]),
                on_tool_call=on_tool_call,
                tools=tools,
            )
        except Exception as exc:
            gathered = [{"error": str(exc)}]

    # Phase 3: SYNTHESIZE
    synthesis = await _phase_synthesize(task_text, system_context, plan, gathered)
    if not synthesis:
        # Synthesize phase failed — construct minimal answer from gathered data
        if gathered:
            synthesis = (
                f"## Task\n{task_text}\n\n"
                f"## Gathered Data\n{json.dumps(gathered, indent=2, default=str)}"
            )
        else:
            synthesis = f"Unable to complete analysis for: {task_text}"

    # Phase 4: ARTIFACT
    answer = await _phase_artifact(synthesis, task_text)

    # Phase 5: INSIGHT (fire-and-forget — never blocks return)
    quality = _estimate_quality(answer, tool_count, plan, gathered)
    asyncio.ensure_future(
        extract_and_store(
            task_text=task_text,
            answer=answer,
            domain=process_type,
            quality=quality,
        )
    )

    return answer, tool_count, quality


async def should_use_five_phase(task_text: str, tool_count_so_far: int) -> bool:
    """
    Decide whether the 5-phase executor is warranted.

    Returns True when ANY of the following heuristics fire:
    - Task text is longer than 200 characters (multi-part query)
    - Task contains multiple question marks (multiple sub-questions)
    - A complexity keyword is found in the task text (open-ended, not a frozenset)
    - tool_count_so_far >= 3 (previous attempt already used heavy tooling)

    Does NOT use a Claude call — pure heuristic, zero cost.
    """
    # Length heuristic
    if len(task_text) > _COMPLEX_TASK_LENGTH:
        return True

    # Multi-question heuristic
    if _MULTI_QUESTION_PATTERN.search(task_text):
        return True

    # Keyword complexity heuristic — signal-based, open-ended list.
    # Works for novel process types (supplier_onboarding, debt_restructuring, etc.)
    # without requiring them to appear in a hardcoded frozenset.
    text_lower = task_text.lower()
    complexity_keywords = [
        "month-end", "month end", "financial close", "close the books",
        "compliance audit", "kyc", "sox", "regulatory audit", "gdpr audit",
        "payroll", "pay run", "salary run", "wage processing",
        "subscription migration", "plan migration", "mass upgrade",
        "p1 incident", "p2 incident", "sev 1", "sev1", "major incident",
    ]
    if any(kw in text_lower for kw in complexity_keywords):
        return True

    # Tool saturation heuristic
    if tool_count_so_far >= _COMPLEX_TOOL_COUNT:
        return True

    return False
