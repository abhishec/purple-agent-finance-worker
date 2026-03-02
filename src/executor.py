from __future__ import annotations
import json
import time
from typing import Callable, Awaitable

from src.brainos_client import run_task, BrainOSUnavailableError
from src.claude_executor import solve_with_claude
from src.mcp_bridge import discover_tools, call_tool
from src.policy_checker import evaluate_policy_rules
from src.structured_output import build_policy_section
from src.rl_loop import build_rl_primer, record_outcome
from src.session_context import (
    add_turn, get_context_prompt, is_multi_turn,
    get_schema_cache, save_fsm_checkpoint, get_fsm_checkpoint,
    maybe_compress_async,
)
from src.fsm_runner import FSMRunner
from src.privacy_guard import check_privacy
from src.token_budget import TokenBudget, format_competition_answer
from src.schema_adapter import resilient_tool_call
from src.config import GREEN_AGENT_MCP_URL


def _parse_policy(policy_doc: str) -> tuple[dict | None, str]:
    if not policy_doc:
        return None, ""
    try:
        parsed = json.loads(policy_doc)
        if isinstance(parsed, dict) and "rules" in parsed:
            result = evaluate_policy_rules(parsed["rules"], parsed.get("context", {}))
            return result, build_policy_section(result)
    except (json.JSONDecodeError, TypeError):
        pass
    return None, f"\nPOLICY:\n{policy_doc}\n"


async def handle_task(
    task_text: str,
    policy_doc: str,
    tools_endpoint: str,
    task_id: str,
    session_id: str,
) -> str:
    """
    Full execution pipeline:
    0. Privacy guard      — fast refuse, no DB cost
    1. Token budget       — track 10K limit, Haiku/Sonnet switching
    2. RL primer          — learned patterns from past tasks
    3. Multi-turn context — compressed session history
    4. FSM restore        — resume process state from previous turn
    5. Policy check       — deterministic if JSON rules
    6. Schema-resilient tool calls — retry with column correction
    7. BrainOS → Claude fallback
    8. Async Haiku compression — proper LLM summary when > 20 turns
    9. RL outcome recording
   10. Format competition answer
    """
    start_ms = int(time.time() * 1000)
    ep = tools_endpoint or GREEN_AGENT_MCP_URL
    budget = TokenBudget()

    # ── 0. Privacy guard — immediate refuse, zero DB cost ────────────────────
    privacy = check_privacy(task_text)
    if privacy and privacy.get("refused"):
        return privacy["message"]

    # ── 1. RL primer ─────────────────────────────────────────────────────────
    rl_primer = build_rl_primer(task_text)
    if rl_primer:
        budget.consume(rl_primer, "rl_primer")

    # ── 2. Multi-turn context ────────────────────────────────────────────────
    multi_turn_ctx = ""
    if is_multi_turn(session_id):
        multi_turn_ctx = get_context_prompt(session_id)
        if multi_turn_ctx:
            budget.consume(multi_turn_ctx, "session_context")

    # ── 3. FSM — restore checkpoint or start fresh ───────────────────────────
    checkpoint = get_fsm_checkpoint(session_id)
    fsm = FSMRunner(task_text=task_text, session_id=session_id, checkpoint=checkpoint)
    phase_prompt = fsm.build_phase_prompt()
    budget.consume(phase_prompt, "fsm_phase")

    # ── 4. Policy enforcement ─────────────────────────────────────────────────
    policy_result, policy_section = _parse_policy(policy_doc)
    if policy_result:
        budget.consume(policy_section, "policy")
        # Wire policy into FSM state machine
        if fsm.current_state.value == "POLICY_CHECK":
            fsm.apply_policy(policy_result)
            phase_prompt = fsm.build_phase_prompt()

    # ── 5. Tool discovery ─────────────────────────────────────────────────────
    try:
        tools = await discover_tools(ep, session_id=session_id)
    except Exception:
        tools = []

    # Schema-resilient tool call wrapper (schema drift + retry)
    schema_cache = get_schema_cache(session_id)

    async def on_tool_call(tool_name: str, params: dict) -> dict:
        try:
            return await resilient_tool_call(tool_name, params, _raw_call, schema_cache)
        except Exception as e:
            return {"error": str(e)}

    async def _raw_call(tool_name: str, params: dict) -> dict:
        try:
            return await call_tool(ep, tool_name, params, session_id)
        except Exception as e:
            return {"error": str(e)}

    # Build system context (cap to budget)
    context_parts = [
        f"Task ID: {task_id}",
        f"Session ID: {session_id}",
        f"Tools endpoint: {ep}",
    ]
    if rl_primer:
        context_parts.append(budget.cap_prompt(rl_primer, "rl"))
    if multi_turn_ctx:
        context_parts.append(budget.cap_prompt(multi_turn_ctx, "history"))
    context_parts.append(phase_prompt)
    if policy_section:
        context_parts.append(policy_section)
    context_parts.append(budget.efficiency_hint())

    system_context = "\n\n".join(context_parts)
    budget.consume(system_context, "system_context")

    # ── 6. Execute ───────────────────────────────────────────────────────────
    answer = ""
    tool_count = 0
    error = None

    add_turn(session_id, "user", task_text)

    # Select model based on FSM state + budget
    model = budget.get_model(fsm.current_state.value, task_text)
    max_tokens = budget.get_max_tokens(fsm.current_state.value)

    try:
        answer = await run_task(
            message=task_text,
            system_context=system_context,
            on_tool_call=on_tool_call,
            session_id=session_id,
        )
    except BrainOSUnavailableError:
        if not budget.should_skip_llm:
            try:
                answer, tool_count = await solve_with_claude(
                    task_text=task_text,
                    policy_section=policy_section,
                    policy_result=policy_result,
                    tools=tools,
                    on_tool_call=on_tool_call,
                    session_id=session_id,
                    model=model,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                error = str(e)
                answer = f"Task failed: {error}"
        else:
            answer = "Token budget exhausted. Task incomplete."

    if answer:
        add_turn(session_id, "assistant", answer)
        budget.consume(answer, "answer")

    # ── 7. Save FSM checkpoint for next turn ─────────────────────────────────
    save_fsm_checkpoint(
        session_id,
        process_type=fsm.process_type,
        state_idx=fsm._idx,
        state_history=fsm.ctx.state_history,
        requires_hitl=fsm.ctx.requires_hitl,
    )

    # ── 8. Async Haiku compression — upgrade inline dump to real LLM summary ─
    await maybe_compress_async(session_id)

    # ── 9. RL outcome recording ───────────────────────────────────────────────
    policy_passed = policy_result.get("passed") if policy_result else None
    quality = record_outcome(
        task_text=task_text,
        answer=answer,
        tool_count=tool_count,
        policy_passed=policy_passed,
        error=error,
    )

    # ── 10. Format competition answer ─────────────────────────────────────────
    duration_ms = int(time.time() * 1000) - start_ms
    fsm_summary = fsm.get_summary()

    if fsm_summary.get("requires_hitl") and not answer.strip().startswith('['):
        answer += f"\n\n[Process: {fsm.process_type} | Human approval required]"

    return format_competition_answer(
        answer=answer,
        process_type=fsm.process_type,
        quality=quality,
        duration_ms=duration_ms,
        policy_passed=policy_passed,
    )
