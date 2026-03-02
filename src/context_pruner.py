"""
context_pruner.py  — Context Rot Pruning
Context rot pruning for RL case log + knowledge base.

Inspired by BrainOS platform/lib/brain/schema-drift-handler.ts filterContextRot().

Problem: The RL case log accumulates entries over the sprint.
Old low-quality entries pollute the PRIME phase prompt with stale patterns,
leading Claude to apply past solutions to different problems.

Fix: Before injecting the RL primer, filter out:
  - Entries with quality < MIN_QUALITY (bad outcomes — don't repeat)
  - Entries older than MAX_AGE_HOURS (stale — world may have changed)
  - Entries marked as repeated failures (3+ failures on same pattern)
  - Keep at least MIN_KEEP entries (don't over-prune)

Conservative: if pruning would remove > 70% of entries, return original.
This mirrors BrainOS filterContextRot() which keeps original if < 20% remain.

Usage in rl_loop.py build_rl_primer():
    cases = prune_case_log(cases, task_text)
"""
from __future__ import annotations

import time
from typing import Any

MIN_QUALITY = 0.35       # discard cases below this quality score
MAX_AGE_HOURS = 72.0     # discard cases older than 3 days
MIN_KEEP = 3             # always keep at least this many cases
MAX_PRUNE_FRACTION = 0.7 # if pruning > 70%, return original (conservative)

# Similarity threshold for "repeated failure" detection
_KEYWORD_OVERLAP_THRESHOLD = 0.5


def _keyword_overlap(kws_a: list[str], kws_b: list[str]) -> float:
    """Jaccard overlap between two keyword lists."""
    if not kws_a or not kws_b:
        return 0.0
    set_a = set(kws_a)
    set_b = set(kws_b)
    return len(set_a & set_b) / len(set_a | set_b)


def _is_repeated_failure(entry: dict, all_entries: list[dict]) -> bool:
    """Return True if this failed entry has 3+ similar failed entries."""
    if entry.get("outcome") != "failure":
        return False
    entry_kws = entry.get("keywords", [])
    similar_failures = sum(
        1 for e in all_entries
        if e is not entry
        and e.get("outcome") == "failure"
        and _keyword_overlap(entry_kws, e.get("keywords", [])) >= _KEYWORD_OVERLAP_THRESHOLD
    )
    return similar_failures >= 2  # this + 2 others = 3 total


def prune_case_log(cases: list[dict], task_text: str = "") -> list[dict]:
    """
    Filter stale, low-quality, and repeated-failure entries from the case log.

    Args:
        cases: list of CaseEntry dicts from rl_loop._load_cases()
        task_text: current task (used to boost relevance of recent similar cases)

    Returns: pruned list (same format, subset of original)
    """
    if len(cases) <= MIN_KEEP:
        return cases  # too few to prune

    now = time.time()
    max_age_secs = MAX_AGE_HOURS * 3600

    kept = []
    for entry in cases:
        quality = entry.get("quality", 0.5)
        timestamp = entry.get("timestamp", now)
        age_secs = now - timestamp
        outcome = entry.get("outcome", "success")

        # Rule 1: drop low-quality failures
        if quality < MIN_QUALITY and outcome == "failure":
            continue

        # Rule 2: drop very old entries
        if age_secs > max_age_secs:
            continue

        # Rule 3: drop repeated failure patterns
        if _is_repeated_failure(entry, cases):
            continue

        kept.append(entry)

    # Conservative guard: if pruning removed too much, return original
    if len(kept) < MIN_KEEP:
        return cases[-MIN_KEEP:]  # keep most recent

    prune_fraction = 1.0 - len(kept) / len(cases)
    if prune_fraction > MAX_PRUNE_FRACTION:
        # Soft fallback: return higher-quality half of originals
        sorted_by_quality = sorted(cases, key=lambda e: e.get("quality", 0.5), reverse=True)
        return sorted_by_quality[:max(MIN_KEEP, len(cases) // 2)]

    return kept


def prune_rl_primer(primer_text: str) -> str:
    """
    Light text-level pruning of the RL primer string.
    Removes lines mentioning stale patterns:
    - Lines containing "(stale)" or "(outdated)" markers
    - Lines with very low signal (just punctuation / < 5 chars)
    """
    if not primer_text:
        return primer_text

    lines = primer_text.splitlines()
    kept_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or len(stripped) < 5:
            continue
        if "(stale)" in stripped.lower() or "(outdated)" in stripped.lower():
            continue
        kept_lines.append(line)

    return "\n".join(kept_lines)


def get_pruner_stats(original: list[dict], pruned: list[dict]) -> dict:
    return {
        "original_count": len(original),
        "pruned_count": len(pruned),
        "removed": len(original) - len(pruned),
        "removal_rate": round((len(original) - len(pruned)) / max(len(original), 1), 3),
    }
