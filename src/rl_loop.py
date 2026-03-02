"""
rl_loop.py
Lightweight RL feedback loop — learns from outcomes, injects case log primer.
Inspired by BrainOS agent-rl.ts + rl-agent-loop.ts.

Two-layer learning (same as BrainOS):
1. Case log: task patterns + outcomes → injected as primer before each task
2. Quality scoring: measures answer quality (tool usage, completeness, policy adherence)

Fixes applied (2026-03-01):
- Bug B: Added import + call of build_benchmark_primer() from report_analyzer
- Added extract_structured_memory() with pure string analysis (no API cost)
- Added _update_case_entry_metadata() for enriching existing case entries
"""
from __future__ import annotations
from src.token_budget import _is_bracket_format
import json
import os
import re
import time
import hashlib
from dataclasses import dataclass, field, asdict

CASE_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "case_log.json")
MAX_CASES = 200
RELEVANT_CASES = 3


@dataclass
class CaseEntry:
    case_id: str
    task_summary: str
    keywords: list[str]
    outcome: str          # "success" | "failure" | "partial"
    quality: float        # 0.0–1.0
    what_worked: str
    what_failed: str
    tool_count: int
    domain: str = ""
    timestamp: float = field(default_factory=time.time)


def _load_cases() -> list[dict]:
    try:
        if os.path.exists(CASE_LOG_PATH):
            with open(CASE_LOG_PATH, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_cases(cases: list[dict]) -> None:
    try:
        with open(CASE_LOG_PATH, "w") as f:
            json.dump(cases[-MAX_CASES:], f, indent=2)
    except Exception:
        pass


def _extract_keywords(text: str) -> list[str]:
    stop = {
        "the","a","an","is","are","was","were","be","been","have","has","had",
        "do","does","did","will","would","could","should","can","for","in","on",
        "at","to","of","and","or","but","with","from","this","that","it","i",
        "you","please","need","want","help","task","make","get","use"
    }
    words = text.lower().split()
    seen, unique = set(), []
    for w in words:
        w = w.strip(".,!?;:\"'()[]")
        if len(w) > 3 and w not in stop and w not in seen:
            seen.add(w)
            unique.append(w)
    return unique[:15]



def _has_structured_completion(answer: str) -> bool:
    """True if answer contains any field:value completion signal — domain-agnostic."""
    # Pattern 1: explicit field:value pairs (catches Risk rating:, Decision:, Total:, etc.)
    field_value_pattern = re.compile(
        r'^[A-Za-z][A-Za-z\s_]{1,30}:\s*.{3,}',  # "Field name: value"
        re.MULTILINE
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


def score_quality(answer: str, tool_count: int, policy_passed: bool | None) -> float:
    """
    Quality score 0–1. Ported from BrainOS computeAgentQuality().
    Conservative baseline 0.5 (matches BrainOS), then adjust by signals.

    Penalizes:
      - Empty data arrays: -0.25 (BrainOS rule)
      - No tool calls: -0.10
      - Short answers: -0.20 if < 50 chars (EXCEPT bracket format exact_match answers)
      - Error phrases: -0.25
      - Policy violation: -0.15

    Rewards:
      - Answer length and structure
      - Tool usage depth
      - Policy compliance
      - Structured output markers
    """
    score = 0.50   # BrainOS conservative baseline

    answer_stripped = answer.strip()
    length = len(answer_stripped)

    # Bracket-format exact_match answers are valid regardless of length —
    # do not penalize '["INV-001"]' as "too short".
    # Use strict JSON-array check: prose like "Rejected. [Reason: ...]" must
    # not receive the bracket bonus — only true JSON lists like ["INV-001"].
    is_bracket_format = _is_bracket_format(answer_stripped)

    if is_bracket_format:
        score += 0.15  # correct format signal
    elif length > 800:  score += 0.20
    elif length > 400:  score += 0.15
    elif length > 150:  score += 0.08
    elif length < 50:   score -= 0.20

    if tool_count >= 5:   score += 0.15
    elif tool_count >= 3: score += 0.10
    elif tool_count >= 1: score += 0.05
    elif tool_count == 0: score -= 0.10

    if policy_passed is True:    score += 0.15
    elif policy_passed is False: score -= 0.15
    # None = unknown, no adjustment

    # BrainOS: empty data array penalty (Bug A verified — raw strings compile correctly)
    if re.search(r'"data"\s*:\s*\[\s*\]', answer):    score -= 0.25
    if re.search(r'"results"\s*:\s*\[\s*\]', answer): score -= 0.15

    # Structure reward
    if _has_structured_completion(answer):
        score += 0.08
    if "{" in answer and "}" in answer:
        score += 0.05

    # Error phrase penalty
    error_phrases = ["task failed", "unable to", "cannot access", "no data found",
                     "token budget exhausted", "tool unavailable"]
    if any(p in answer.lower() for p in error_phrases):
        score -= 0.25

    return round(max(0.0, min(1.0, score)), 3)


# ── Structured memory extraction (pure string, zero API cost) ─────────────────

def _extract_success_pattern(task_text: str, answer: str, domain: str = "") -> str:
    """
    Extract a concise success pattern description from task + answer.
    No API calls — pure string analysis.
    """
    parts = []

    # Count tool references in answer
    tool_mentions = len(re.findall(
        r'\b(?:tool|called|fetched|retrieved|queried|executed|invoked)\b',
        answer.lower()
    ))
    if tool_mentions > 0:
        parts.append(f"Used ~{tool_mentions} tool references")

    # Use domain directly — it's the authoritative process type, already classified
    process_label = f"Process: {domain}" if domain else ""
    if process_label:
        parts.append(process_label)

    # Extract dollar amount if present
    amount_m = re.search(r'\$[\d,]+(?:\.\d{2})?', answer)
    if amount_m:
        parts.append(f"Amount processed: {amount_m.group()}")

    # Detect approval/rejection outcome
    if re.search(r'\b(approved|rejected|completed|resolved)\b', answer.lower()):
        outcome_m = re.search(r'\b(approved|rejected|completed|resolved)\b', answer.lower())
        if outcome_m:
            parts.append(f"Outcome: {outcome_m.group()}")

    return ". ".join(parts) if parts else "Completed successfully"


def _extract_failure_pattern(task_text: str, answer: str) -> str:
    """
    Extract a concise failure pattern description from task + answer.
    No API calls — pure string analysis.
    """
    parts = []

    # Check for common error phrases
    error_patterns = [
        (r"no data found", "No data found in tool response"),
        (r"unable to", "Unable to complete action"),
        (r"cannot access", "Tool access failure"),
        (r"token budget", "Token budget exhausted"),
        (r"tool unavailable", "Required tool unavailable"),
        (r"missing\s+\w+", "Missing required field"),
        (r"timed? out", "Timeout during execution"),
    ]
    for pattern, label in error_patterns:
        if re.search(pattern, answer.lower()):
            parts.append(label)
            break

    # Check for missing data indicators
    if re.search(r'"data"\s*:\s*\[\s*\]', answer):
        parts.append("Empty data response from tool")
    if re.search(r'"results"\s*:\s*\[\s*\]', answer):
        parts.append("Empty results from query")

    # Short answer = incomplete
    if len(answer.strip()) < 100:
        parts.append("Answer too short — likely incomplete")

    return ". ".join(parts) if parts else "Task incomplete or low quality"


def _update_case_entry_metadata(domain: str, what_worked: str = "", what_failed: str = "") -> None:
    """
    Enrich the most recent case log entry for the given domain
    with structured what_worked / what_failed metadata.
    Called after extract_structured_memory() — enriches RL primer quality.
    """
    try:
        cases = _load_cases()
        if not cases:
            return
        # Find the most recent entry for this domain
        for i in range(len(cases) - 1, max(len(cases) - 10, -1), -1):
            if cases[i].get("domain") == domain:
                if what_worked and not cases[i].get("what_worked"):
                    cases[i]["what_worked"] = what_worked
                if what_failed and not cases[i].get("what_failed"):
                    cases[i]["what_failed"] = what_failed
                _save_cases(cases)
                return
    except Exception:
        pass


def extract_structured_memory(task_text: str, answer: str, domain: str, quality: float) -> None:
    """
    Extract 3 structured facts from task outcome using pure string analysis (no API).
    Inspired by BrainOS agent-rl.ts recordAgentOutcome() pattern.

    For successes (quality >= 0.6):
      - What worked: tool count, process type, dollar amount, outcome word
    For failures:
      - What failed: error phrase, missing data, short answer indicator

    Results stored back into the most recent case log entry for this domain,
    enriching the RL primer injected before future similar tasks.
    """
    if quality >= 0.6:
        what_worked = _extract_success_pattern(task_text, answer, domain)
        _update_case_entry_metadata(domain, what_worked=what_worked)
    else:
        what_failed = _extract_failure_pattern(task_text, answer)
        _update_case_entry_metadata(domain, what_failed=what_failed)


# ── Core RL functions ─────────────────────────────────────────────────────────

def record_outcome(
    task_text: str,
    answer: str,
    tool_count: int,
    policy_passed: bool | None = None,
    error: str | None = None,
    domain: str = "",
) -> float:
    """Record task outcome. Returns quality score (dopamine if >=0.6, gaba if <0.6)."""
    cases = _load_cases()
    quality = score_quality(answer, tool_count, policy_passed)
    outcome = "success" if quality >= 0.6 else ("failure" if error else "partial")
    case_id = hashlib.md5(f"{task_text[:50]}{time.time()}".encode()).hexdigest()[:8]

    what_worked = ""
    what_failed = ""
    if outcome == "success":
        if tool_count > 0:
            what_worked = f"Used {tool_count} tool calls"
        if policy_passed:
            what_worked += (". Policy enforced correctly" if what_worked else "Policy enforced correctly")
    else:
        what_failed = error or "Partial/incomplete answer"

    entry = CaseEntry(
        case_id=case_id,
        task_summary=task_text[:120],
        keywords=_extract_keywords(task_text),
        outcome=outcome,
        quality=round(quality, 3),
        what_worked=what_worked,
        what_failed=what_failed,
        tool_count=tool_count,
        domain=domain,
    )
    cases.append(asdict(entry))
    _save_cases(cases)

    # Enrich with structured memory extraction immediately (pure string, no API)
    extract_structured_memory(task_text, answer, domain, quality)

    return quality


def build_rl_primer(task_text: str) -> str:
    """
    Build a case-log primer — injected before task execution.
    Inspired by BrainOS rl-agent-loop.ts injectCaseLogContext().

    Bug B fix (2026-03-01): Now also includes benchmark intelligence
    from report_analyzer.build_benchmark_primer() so the agent knows
    exactly where it lost points in the last benchmark run.
    """
    cases = _load_cases()
    # Prune stale/low-quality/repeated-failure entries before scoring
    try:
        from src.context_pruner import prune_case_log, prune_rl_primer
        cases = prune_case_log(cases, task_text)
    except Exception:
        pass  # Graceful no-op if context_pruner unavailable
    task_kw = set(_extract_keywords(task_text))
    scored = []
    for c in cases:
        overlap = len(task_kw & set(c.get("keywords", [])))
        if overlap > 0:
            scored.append((overlap, c))
    scored.sort(key=lambda x: (-x[0], -x[1].get("quality", 0)))
    relevant = [c for _, c in scored[:RELEVANT_CASES]]

    lines = []

    # ── Benchmark intelligence (Bug B fix) ───────────────────────────────────
    try:
        from src.report_analyzer import build_benchmark_primer
        bench_primer = build_benchmark_primer()
        if bench_primer:
            lines.append(bench_primer)
    except Exception:
        pass  # Graceful no-op — benchmark data may not be available yet

    # ── Case log patterns ─────────────────────────────────────────────────────
    if relevant:
        lines.append("## LEARNED PATTERNS (from similar past tasks — apply these)")
        for c in relevant:
            icon = "✅" if c["outcome"] == "success" else ("❌" if c["outcome"] == "failure" else "⚠️")
            lines.append(f'\n{icon} Past: "{c["task_summary"][:80]}" — quality {c["quality"]:.2f}')
            if c.get("what_worked"):
                lines.append(f'   ✓ Worked: {c["what_worked"]}')
            if c.get("what_failed"):
                lines.append(f'   ✗ Avoid: {c["what_failed"]}')
        lines.append("")

    return "\n".join(lines) if lines else ""
