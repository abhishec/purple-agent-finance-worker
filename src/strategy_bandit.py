"""
strategy_bandit.py  — UCB1 Strategy Bandit
UCB1 multi-armed bandit for FSM strategy selection.

Inspired by BrainOS packages/memory-stack/src/causality/causal-method-bandit.ts.

Problem: For each task type we have 3 strategies:
  "fsm"        — 8-state FSM (default, structured)
  "five_phase" — Five-Phase Executor (for complex multi-step tasks)
  "moa"        — Mixture of Agents (for pure-reasoning / numeric tasks)

We don't know which strategy wins for a given process type until we try.
UCB1 learns which strategy works best per process type over the sprint.

UCB1 score: Q(arm) + C * sqrt(ln(N) / n(arm))
  Q    = mean reward for this arm (quality 0–1)
  N    = total pulls across all arms for this process type
  n    = pulls for this arm
  C    = exploration constant (1.41 = sqrt(2))

Persisted to strategy_bandit.json (same dir as tool_registry.json).
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

_BANDIT_FILE = Path(os.environ.get("RL_CACHE_DIR", "/app")) / "strategy_bandit.json"

# Available arms
STRATEGIES = ("fsm", "five_phase", "moa")

# UCB1 exploration constant — sqrt(2) is standard
_C = math.sqrt(2)

# In-memory state: {process_type: {strategy: {q, n}}}
_state: dict[str, dict[str, dict]] = {}
_loaded = False


# ── Persistence ───────────────────────────────────────────────────────────────

def _load() -> None:
    global _state, _loaded
    if _loaded:
        return
    try:
        if _BANDIT_FILE.exists():
            _state = json.loads(_BANDIT_FILE.read_text())
    except Exception:
        _state = {}
    _loaded = True


def _save() -> None:
    try:
        _BANDIT_FILE.write_text(json.dumps(_state, indent=2))
    except Exception:
        pass


def _arms(process_type: str) -> dict[str, dict]:
    """Get or initialise arms for a process type."""
    if process_type not in _state:
        _state[process_type] = {s: {"q": 0.5, "n": 0} for s in STRATEGIES}
    return _state[process_type]


# ── UCB1 selection ────────────────────────────────────────────────────────────

def select_strategy(process_type: str, task_text: str = "") -> str:
    """
    Return the UCB1-optimal strategy for this process type.
    On first call (all n=0), returns 'fsm' (safe default).
    After enough data, converges to the best arm.

    task_text is used for heuristic overrides (e.g. numeric tasks → MoA hint).
    """
    _load()
    arms = _arms(process_type)

    # Always explore unvisited arms first (UCB1 requires n > 0 for all arms)
    unvisited = [s for s, d in arms.items() if d["n"] == 0]
    if unvisited:
        # Prefer fsm for first visit — most reliable default
        return "fsm" if "fsm" in unvisited else unvisited[0]

    N_total = sum(d["n"] for d in arms.values())

    best_score = -1.0
    best_arm = "fsm"
    for strategy, data in arms.items():
        q = data["q"]
        n = data["n"]
        ucb1 = q + _C * math.sqrt(math.log(N_total) / n)
        if ucb1 > best_score:
            best_score = ucb1
            best_arm = strategy

    return best_arm


def record_outcome(process_type: str, strategy: str, quality: float) -> None:
    """
    Update the bandit with the observed quality reward (0–1).
    Uses incremental mean update: Q_new = Q_old + (reward - Q_old) / n
    """
    _load()
    arms = _arms(process_type)
    if strategy not in arms:
        arms[strategy] = {"q": 0.5, "n": 0}

    data = arms[strategy]
    data["n"] += 1
    # Incremental mean
    data["q"] = data["q"] + (quality - data["q"]) / data["n"]
    _save()


def get_stats() -> dict:
    """Return bandit stats for /health endpoint."""
    _load()
    total_pulls = sum(
        d["n"]
        for arms in _state.values()
        for d in arms.values()
    )
    process_types_learned = len(_state)
    best_arms = {
        pt: max(arms.items(), key=lambda x: x[1]["q"])[0]
        for pt, arms in _state.items()
        if any(d["n"] > 0 for d in arms.values())
    }
    return {
        "total_pulls": total_pulls,
        "process_types_learned": process_types_learned,
        "best_arms": best_arms,
    }
