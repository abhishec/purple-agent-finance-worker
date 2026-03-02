"""
session_context.py
Multi-turn A2A conversation memory + FSM state persistence.
Inspired by BrainOS memory compression + checkpoint_data system.

Each session_id gets:
- Conversation history (Haiku-compressed when > 20 turns via maybe_compress_async)
- FSM state (restored on next turn — multi-turn process continuity)
- Schema cache (column name corrections persist within session)
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field

MAX_SESSION_AGE = 3600   # 1 hour idle → evict
MAX_RAW_TURNS = 20
KEEP_RECENT = 6

_sessions: dict[str, "SessionContext"] = {}


@dataclass
class Turn:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class FSMCheckpoint:
    """Persisted FSM state for multi-turn continuity. Mirrors BrainOS checkpoint_data."""
    process_type: str
    state_idx: int
    state_history: list[str] = field(default_factory=list)
    requires_hitl: bool = False


@dataclass
class SessionContext:
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    compressed_summary: str = ""
    fsm_checkpoint: FSMCheckpoint | None = None
    schema_cache: dict = field(default_factory=dict)   # column correction cache
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


# ── Public API ────────────────────────────────────────────────────────────────

def get_or_create(session_id: str) -> SessionContext:
    _evict_stale()
    if session_id not in _sessions:
        _sessions[session_id] = SessionContext(session_id=session_id)
    return _sessions[session_id]


def add_turn(session_id: str, role: str, content: str) -> None:
    ctx = get_or_create(session_id)
    ctx.turns.append(Turn(role=role, content=content))
    ctx.last_active = time.time()
    # Inline fallback — replaced by async Haiku compression when executor calls maybe_compress_async()
    if len(ctx.turns) > MAX_RAW_TURNS * 2:
        _compress_inline(ctx)


def get_context_prompt(session_id: str) -> str:
    """Compressed history + recent turns for system prompt injection."""
    ctx = _sessions.get(session_id)
    if not ctx or (not ctx.compressed_summary and not ctx.turns):
        return ""

    parts = []
    if ctx.compressed_summary:
        parts.append(f"## Prior Conversation Summary\n{ctx.compressed_summary}")

    recent = ctx.turns[-KEEP_RECENT:]
    if recent:
        parts.append("## Recent Conversation")
        for t in recent:
            label = "User" if t.role == "user" else "Agent"
            parts.append(f"{label}: {t.content[:400]}")

    return "\n".join(parts)


def is_multi_turn(session_id: str) -> bool:
    ctx = _sessions.get(session_id)
    return bool(ctx and ctx.turns)


def get_schema_cache(session_id: str) -> dict:
    """Per-session schema correction cache — column name fixes persist across turns."""
    return get_or_create(session_id).schema_cache


async def maybe_compress_async(session_id: str) -> None:
    """
    Compress session history using Claude Haiku when > MAX_RAW_TURNS turns.
    Preferred over _compress_inline — produces a real LLM summary, not a text dump.
    Called by executor.py after each assistant turn completes.
    Graceful no-op if compression not needed or Anthropic unavailable.
    """
    ctx = _sessions.get(session_id)
    if not ctx or len(ctx.turns) <= MAX_RAW_TURNS:
        return

    from src.memory_compressor import compress_history

    messages = [
        {"role": t.role, "content": t.content}
        for t in ctx.turns
    ]
    try:
        compressed_msgs, summary, tokens_saved = await compress_history(messages)
    except Exception:
        # Fall back to inline compression rather than failing silently
        _compress_inline(ctx)
        return

    if tokens_saved <= 0 or not summary:
        return

    # Rebuild turns from the compressed message list (drop system/summary messages)
    new_turns = [
        Turn(role=m["role"], content=m["content"])
        for m in compressed_msgs
        if m.get("role") in ("user", "assistant")
    ]
    ctx.turns = new_turns
    ctx.compressed_summary = (
        ctx.compressed_summary + "\n\n" + summary if ctx.compressed_summary else summary
    )


# ── FSM state persistence ─────────────────────────────────────────────────────

def save_fsm_checkpoint(
    session_id: str,
    process_type: str,
    state_idx: int,
    state_history: list[str],
    requires_hitl: bool = False,
) -> None:
    """Save FSM state after each turn so next turn can resume where we left off."""
    ctx = get_or_create(session_id)
    ctx.fsm_checkpoint = FSMCheckpoint(
        process_type=process_type,
        state_idx=state_idx,
        state_history=list(state_history),
        requires_hitl=requires_hitl,
    )


def get_fsm_checkpoint(session_id: str) -> FSMCheckpoint | None:
    """Restore FSM state from previous turn. None if first turn."""
    ctx = _sessions.get(session_id)
    return ctx.fsm_checkpoint if ctx else None


# ── Internal ──────────────────────────────────────────────────────────────────

def _compress_inline(ctx: SessionContext) -> None:
    """Fallback sync compression — used only when async path isn't available."""
    older = ctx.turns[:-KEEP_RECENT]
    keep = ctx.turns[-KEEP_RECENT:]
    if not older:
        return
    lines = [f"{'User' if t.role == 'user' else 'Agent'}: {t.content[:200]}" for t in older]
    block = "\n".join(lines)
    ctx.compressed_summary = (
        ctx.compressed_summary + "\n\n" + block if ctx.compressed_summary else block
    )
    ctx.turns = keep


def _evict_stale() -> None:
    now = time.time()
    stale = [sid for sid, ctx in _sessions.items() if now - ctx.last_active > MAX_SESSION_AGE]
    for sid in stale:
        del _sessions[sid]
