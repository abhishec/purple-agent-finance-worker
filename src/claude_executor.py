from __future__ import annotations
import json
from typing import Callable, Awaitable

import anthropic

from src.config import FALLBACK_MODEL, ANTHROPIC_API_KEY
from src.structured_output import format_final_answer

MAX_ITERATIONS = 20
MAX_TOOL_CALLS = 18


async def solve_with_claude(
    task_text: str,
    policy_section: str,
    policy_result: dict | None,
    tools: list[dict],
    on_tool_call: Callable[[str, dict], Awaitable[dict]],
    session_id: str,
    model: str | None = None,
    max_tokens: int = 4096,
    original_task_text: str = "",
) -> tuple[str, int]:
    """
    Primary Claude execution engine. Returns (answer, tool_count).
    model + max_tokens are set by TokenBudget — Haiku at >80% usage, Sonnet otherwise.
    tool_count fed into RL quality scoring.

    original_task_text: when this call is an improvement pass, pass the original task
    here so Claude has full context. It is prepended to the system prompt.
    """
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    effective_model = model or FALLBACK_MODEL

    # If this is an improvement pass, prepend the original task context so Claude
    # is not working with just the improvement delta ("Your answer was incomplete...")
    original_context_block = ""
    if original_task_text and original_task_text != task_text:
        original_context_block = (
            f"\n\nORIGINAL TASK CONTEXT:\n{original_task_text}\n\n"
            "Use the above for full context while completing the improvement request.\n"
        )

    system_prompt = f"""You are an autonomous business operations agent running in a benchmark evaluation.

CRITICAL RULES:
1. NEVER ask the user for more information. All data is accessible via tools.
2. Start calling tools IMMEDIATELY. Do not ask clarifying questions.
3. If a task mentions specific IDs (e.g. BK-001, ORD-001, EMP-MR), call the relevant tool directly.
4. Complete ALL required actions end-to-end before writing your final summary.
5. For list/ranking answers: return ["Item1", "Item2"] bracket format exactly.
6. confirm_with_user ALWAYS returns "ok" (auto-confirmed). When you call it and get status=ok, IMMEDIATELY call the next mutation tool. Do NOT stop to ask questions.
7. The task text contains ALL information you need. Never ask "which order?" or "what action?" — it is already specified.

EXECUTION ORDER (critical for scoring):
- Phase 1 READ: Call all get_*/check_*/calculate_* tools first to gather data.
- Phase 2 CONFIRM: Call confirm_with_user if required by policy (it auto-confirms, returns ok immediately).
- Phase 3 EXECUTE: Call modify_*/update_*/cancel_*/process_*/create_*/post_*/send_*/approve_*/flag_* mutation tools.
- Phase 4 NOTIFY: Call notification/communication tools last (send_notification, post_status_update, draft_*).
- Always escalate/page BEFORE creating reports. Always calculate BEFORE drafting client communications.
- If escalation is required per task policy: call escalate_*/page_* tools BEFORE notify_*/send_* tools.
- Use EVERY available tool that is relevant to the task — incomplete tool coverage loses points.
{policy_section}{original_context_block}
Execute the task fully and in correct order. After all actions, provide a concise answer."""

    messages: list[dict] = [{"role": "user", "content": task_text}]
    tool_count = 0
    last_meaningful_content = ""

    for _ in range(MAX_ITERATIONS):
        response = await client.messages.create(
            model=effective_model,
            max_tokens=max_tokens,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason == "end_turn":
            for block in assistant_content:
                if hasattr(block, "text") and block.text.strip():
                    raw_answer = block.text.strip()
                    # Bracket-format exact_match answers must pass through unmodified
                    if raw_answer.startswith('['):
                        return raw_answer, tool_count
                    return format_final_answer(block.text, policy_result), tool_count
            return "", tool_count

        if response.stop_reason != "tool_use":
            break

        tool_results = []
        hit_tool_limit = False
        for block in assistant_content:
            if block.type != "tool_use":
                # Capture any text content alongside tool calls for later synthesis
                if hasattr(block, "text") and block.text.strip():
                    last_meaningful_content = block.text.strip()
                continue

            # Enforce MAX_TOOL_CALLS before calling the tool
            if tool_count >= MAX_TOOL_CALLS:
                hit_tool_limit = True
                break

            tool_count += 1
            result = await on_tool_call(
                block.name,
                block.input if isinstance(block.input, dict) else {}
            )
            # Use JSON format instead of Python repr for cleaner Claude parsing
            try:
                content_str = json.dumps(result, default=str)
            except Exception:
                content_str = str(result)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content_str,
                "_tool_name": block.name,
                "_result_raw": result,
            })

        if hit_tool_limit:
            # Append any tool results collected before hitting the limit so
            # _synthesize_from_history can use them (they'd be lost otherwise).
            if tool_results:
                clean_partial = [
                    {k: v for k, v in tr.items() if not k.startswith("_")}
                    for tr in tool_results
                ]
                messages.append({"role": "user", "content": clean_partial})
            break

        if tool_results:
            # Strip internal keys before appending to messages (Claude API doesn't accept them)
            clean_results = [
                {k: v for k, v in tr.items() if not k.startswith("_")}
                for tr in tool_results
            ]
            messages.append({"role": "user", "content": clean_results})

    # Synthesize an answer from whatever we collected
    return _synthesize_from_history(messages, tool_count, last_meaningful_content, policy_result), tool_count


def _synthesize_from_history(
    messages: list[dict],
    tool_count: int,
    last_meaningful_content: str,
    policy_result: dict | None,
) -> str:
    """
    Build a meaningful final answer from message history when MAX_ITERATIONS
    or MAX_TOOL_CALLS is reached without a clean end_turn stop.
    Prefers the last assistant text block; falls back to a tool-result digest.
    """
    # Walk backward through messages looking for useful assistant text
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in reversed(content):
                if hasattr(block, "text") and block.text.strip():
                    raw = block.text.strip()
                    # Bracket-format exact_match answers must pass through unmodified
                    if raw.startswith('['):
                        return raw
                    return format_final_answer(block.text, policy_result)
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "").strip()
                    if text:
                        # Bracket-format exact_match answers must pass through unmodified
                        if text.startswith('['):
                            return text
                        return format_final_answer(text, policy_result)
        elif isinstance(content, str) and content.strip():
            raw = content.strip()
            # Bracket-format exact_match answers must pass through unmodified
            if raw.startswith('['):
                return raw
            return format_final_answer(content, policy_result)

    # Collect tool results for a digest summary
    tool_results_text: list[str] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_result":
                    raw = item.get("content", "")
                    if raw and str(raw) not in ("None", "{}", "[]", ""):
                        tool_results_text.append(str(raw)[:300])

    if last_meaningful_content:
        base = last_meaningful_content
        # Bracket-format exact_match answers must pass through unmodified —
        # do not wrap with "Based on N tool calls: ..." prefix
        if base.strip().startswith('['):
            return base.strip()
    elif tool_results_text:
        digest = " | ".join(tool_results_text[-5:])
        base = f"Collected data from {tool_count} tool calls: {digest}"
    else:
        base = f"Task executed across {tool_count} tool calls. No further data available."

    return format_final_answer(f"Based on {tool_count} tool calls: {base}", policy_result)
