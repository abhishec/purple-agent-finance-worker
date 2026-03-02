"""
memory_compressor.py
Multi-turn memory compression — context gets smarter, not bigger.
Ported from BrainOS copilot context layer.

Differentiator #2: Most agents start fresh every task. This compresses
prior turns and injects them. Context window shrinks, intelligence grows.
"""
from __future__ import annotations
import anthropic
from src.config import ANTHROPIC_API_KEY

MAX_TOKENS = 8_000
CHARS_PER_TOKEN = 4
KEEP_RECENT = 6
COMPRESS_MODEL = "claude-haiku-4-5-20251001"


def count_tokens(messages: list[dict]) -> int:
    total = sum(
        len(m.get("content", "") if isinstance(m.get("content"), str) else "")
        for m in messages
    )
    return -(-total // CHARS_PER_TOKEN)  # ceiling division


async def compress_history(messages: list[dict]) -> tuple[list[dict], str, int]:
    """
    Returns (compressed_messages, summary, tokens_saved).
    Keeps system message + last 6 turns, summarizes middle with Haiku.
    Graceful fallback if Anthropic unavailable.
    """
    original_tokens = count_tokens(messages)
    if original_tokens <= MAX_TOKENS:
        return messages, "", 0

    system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
    non_system = messages[1:] if system_msg else messages
    recent = non_system[-KEEP_RECENT:]
    middle = non_system[:-KEEP_RECENT]

    if not middle:
        return messages, "", 0

    summary = ""
    try:
        client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        history_text = "\n\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {str(m.get('content', ''))[:600]}"
            for m in middle
        )
        resp = await client.messages.create(
            model=COMPRESS_MODEL,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": (
                    "Summarize this conversation excerpt (max 200 words). "
                    "Preserve key goals, facts, decisions, in-progress items:\n\n"
                    + history_text
                ),
            }],
        )
        summary = resp.content[0].text if resp.content and hasattr(resp.content[0], "text") else ""
    except Exception:
        summary = ""

    summary_msg = {
        "role": "system",
        "content": (
            f"[Earlier conversation summary — {len(middle)} messages compressed]\n\n{summary}"
            if summary
            else f"[{len(middle)} earlier messages removed to stay within context limits]"
        ),
    }

    compressed = ([system_msg] if system_msg else []) + [summary_msg] + recent
    tokens_saved = max(0, original_tokens - count_tokens(compressed))
    return compressed, summary, tokens_saved
