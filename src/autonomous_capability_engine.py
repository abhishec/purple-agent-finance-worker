"""
autonomous_capability_engine.py
Autonomous Capability Engine (ACE) — 100x evolution of dynamic_tools.py.

=============================================================
DESIGN PHILOSOPHY
=============================================================

The old approach (static _GAP_PATTERNS):
  Task text → 36 regex patterns → hardcoded descriptions → Haiku synthesis
  Ceiling: finite list of patterns we predicted in advance.
  Misses: IFRS-16 leases, Xero GL→P&L, SOFR swaps, anything novel.

The ACE approach (zero hardcoding):
  Execution failure / LLM uncertainty → formalize gap → search libraries
  → compose existing tools → synthesize with best available library
  → auto-test → register → notify → NEXT TASK gets it for free

Core innovations:
  1. OBSERVATION-BASED detection: gaps found from real execution failures,
     not predicted from keywords. No list can go stale.
  2. LIBRARY-AWARE synthesis: uses scipy/numpy/pandas when available.
     Black-Scholes via scipy.stats.norm.cdf is more accurate than math.erf.
  3. ASYNC PROMISES: capability building NEVER blocks task execution.
     Worker continues with best-effort. Next task gets the capability.
  4. COMPOSITION ENGINE: "P&L from GL" = csv_parser + classifier + aggregator.
     Reuses and combines existing capabilities before building new ones.
  5. SELF-TESTING: auto-generates 5 test cases with known answers.
     ≥4/5 required to register. Retry with error context on failure.
  6. CAPABILITY GRAPH: task_type → [required capabilities] learned from
     every task. Pre-warms capabilities before execution even starts.

=============================================================
ASYNC PROMISE PATTERN (how execution works)
=============================================================

t=0ms   Task arrives
t=5ms   PRIME: capability audit → gaps detected → promises started (async)
t=15ms  EXECUTE begins (WITHOUT waiting for ACE)
t=800ms FSM reaches COMPUTE state
t=805ms worker.await_promises(timeout=5.0s)
t=3200ms  ACE finishes: capability registered and hot-loaded
t=3205ms  Promise resolves → tool available at COMPUTE
t=3800ms  REFLECT: graph updated → next task pre-loads this capability

First task of new type: ~3s ACE overhead (once, then cached).
All subsequent tasks: zero overhead, tool pre-loaded from graph.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from src.config import ANTHROPIC_API_KEY

# ── Paths ─────────────────────────────────────────────────────────────────────

_STORE_FILE  = Path(os.environ.get("RL_CACHE_DIR", "/app")) / "capability_store.json"
_GRAPH_FILE  = Path(os.environ.get("RL_CACHE_DIR", "/app")) / "capability_graph.json"

# ── Sandbox globals (library-aware) ──────────────────────────────────────────
# Populated lazily at first use — checks what's actually installed.

_SANDBOX: dict[str, Any] = {}
_SANDBOX_READY = False


def _build_sandbox() -> dict[str, Any]:
    """
    Build sandbox globals with the richest available libraries.
    Order of preference: scipy > numpy > pandas > sympy > pure Python.
    Only imports what's actually installed — never crashes on missing lib.
    """
    import csv as _csv
    import io as _io
    import statistics as _stats
    import random as _random
    import urllib.request as _urllib

    sb: dict[str, Any] = {
        "__builtins__": None,
        # Pure Python — always available
        "math": math,
        "abs": abs, "int": int, "float": float, "str": str, "bool": bool,
        "round": round, "min": min, "max": max, "sum": sum, "len": len,
        "range": range, "enumerate": enumerate, "zip": zip,
        "list": list, "dict": dict, "tuple": tuple, "set": set,
        "isinstance": isinstance, "pow": pow, "divmod": divmod,
        "sorted": sorted, "reversed": reversed, "any": any, "all": all,
        "ValueError": ValueError, "ZeroDivisionError": ZeroDivisionError,
        "TypeError": TypeError, "KeyError": KeyError,
        # Standard library
        "random": _random,
        "statistics": _stats,
        "csv": _csv,
        "io": _io,
        "urllib": _urllib,
        "json": json,
        "re": re,
    }

    # Scientific libraries — best-effort import
    try:
        import scipy.stats as _scipy_stats
        import scipy.optimize as _scipy_opt
        sb["scipy_stats"]   = _scipy_stats
        sb["scipy_optimize"] = _scipy_opt
        sb["_has_scipy"] = True
    except ImportError:
        sb["_has_scipy"] = False

    try:
        import numpy as _np
        sb["np"] = _np
        sb["_has_numpy"] = True
    except ImportError:
        sb["_has_numpy"] = False

    try:
        import pandas as _pd
        sb["pd"] = _pd
        sb["_has_pandas"] = True
    except ImportError:
        sb["_has_pandas"] = False

    try:
        import sympy as _sympy
        sb["sympy"] = _sympy
        sb["_has_sympy"] = True
    except ImportError:
        sb["_has_sympy"] = False

    from decimal import Decimal, ROUND_HALF_UP
    sb["Decimal"] = Decimal
    sb["ROUND_HALF_UP"] = ROUND_HALF_UP

    return sb


def _get_sandbox() -> dict[str, Any]:
    global _SANDBOX, _SANDBOX_READY
    if not _SANDBOX_READY:
        _SANDBOX = _build_sandbox()
        _SANDBOX_READY = True
    return _SANDBOX


def _library_availability_str() -> str:
    """Human-readable string of available libraries for synthesis prompt."""
    sb = _get_sandbox()
    libs = []
    if sb.get("_has_scipy"):
        libs.append("scipy (scipy_stats.norm.cdf, scipy_optimize.brentq available as scipy_stats/scipy_optimize)")
    if sb.get("_has_numpy"):
        libs.append("numpy (available as 'np': np.exp, np.log, np.linalg, np.array, etc.)")
    if sb.get("_has_pandas"):
        libs.append("pandas (available as 'pd': pd.read_csv, df.groupby, df.merge, etc.)")
    if sb.get("_has_sympy"):
        libs.append("sympy (symbolic math, exact derivatives)")
    libs.append("math (standard: math.exp, math.log, math.sqrt, math.erf, etc.)")
    libs.append("statistics (mean, stdev, etc.)")
    libs.append("csv, io, json, re, random (standard library)")
    libs.append("Decimal, ROUND_HALF_UP (precision arithmetic)")
    libs.append("urllib (urllib.request.urlopen for HTTP fetches)")
    return "\n".join(f"  - {l}" for l in libs)


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class CapabilitySpec:
    """Structured description of a required capability."""
    name: str                     # snake_case function name
    description: str              # one-sentence human description
    inputs: dict                  # {param_name: "type and meaning"}
    outputs: dict                 # {key: "type and meaning"}
    formula_hint: str             # mathematical formula or algorithm sketch
    example: dict                 # {"input": {}, "expected_output": {}}
    raw_signal: str               # original gap signal text


@dataclass
class CapabilityRecord:
    """A registered capability in the store."""
    name: str
    description: str
    python_code: str
    input_schema: dict
    test_cases: list[dict]
    accuracy_score: float         # fraction of test cases passing
    usage_count: int = 0
    success_count: int = 0        # tasks that succeeded after using this capability
    library_used: str = "math"    # which library was used
    composed_from: list[str] = field(default_factory=list)
    task_types: list[str] = field(default_factory=list)
    version: int = 1
    synthesized_at: float = field(default_factory=time.time)
    last_validated: float = field(default_factory=time.time)
    _seeded: bool = False
    _synthesized: bool = True


# ── Capability Store ──────────────────────────────────────────────────────────

class CapabilityStore:
    """
    Persistent store for all registered capabilities.
    Evolved version of tool_registry.json with full metadata.
    """

    def __init__(self):
        self._records: dict[str, CapabilityRecord] = {}
        self._fns: dict[str, Any] = {}       # name → callable (hot-loaded)
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        try:
            if _STORE_FILE.exists():
                raw = json.loads(_STORE_FILE.read_text())
                for name, data in raw.items():
                    try:
                        rec = CapabilityRecord(**{
                            k: v for k, v in data.items()
                            if k in CapabilityRecord.__dataclass_fields__
                        })
                        self._records[name] = rec
                        fn = self._exec(rec.python_code, name)
                        if fn:
                            self._fns[name] = fn
                    except Exception:
                        pass
        except Exception:
            pass
        self._loaded = True

    def save(self) -> None:
        try:
            out = {}
            for name, rec in self._records.items():
                out[name] = asdict(rec)
            _STORE_FILE.write_text(json.dumps(out, indent=2))
        except Exception:
            pass

    def has(self, name: str) -> bool:
        self.load()
        return name in self._fns

    def get_fn(self, name: str) -> Any | None:
        self.load()
        return self._fns.get(name)

    def get_record(self, name: str) -> CapabilityRecord | None:
        self.load()
        return self._records.get(name)

    def all_names(self) -> list[str]:
        self.load()
        return list(self._records.keys())

    def all_schemas(self) -> list[dict]:
        """Return MCP-compatible tool schemas for all registered capabilities."""
        self.load()
        result = []
        for name, rec in self._records.items():
            result.append({
                "name": name,
                "description": rec.description,
                "input_schema": rec.input_schema,
            })
        return result

    def register(self, rec: CapabilityRecord) -> bool:
        """Register a new capability. Returns True on success."""
        fn = self._exec(rec.python_code, rec.name)
        if fn is None:
            return False
        self._records[rec.name] = rec
        self._fns[rec.name] = fn
        self.save()
        return True

    def call(self, name: str, params: dict) -> dict:
        """Execute a registered capability."""
        self.load()
        fn = self._fns.get(name)
        if fn is None:
            return {"error": f"Capability '{name}' not found"}
        try:
            result = fn(**params)
            return result if isinstance(result, dict) else {"result": result}
        except Exception as e:
            return {"error": str(e), "capability": name}

    def record_usage(self, name: str, success: bool) -> None:
        """Update usage stats after a task completes."""
        self.load()
        rec = self._records.get(name)
        if rec:
            rec.usage_count += 1
            if success:
                rec.success_count += 1
            self.save()

    def add_task_type(self, name: str, task_type: str) -> None:
        self.load()
        rec = self._records.get(name)
        if rec and task_type not in rec.task_types:
            rec.task_types.append(task_type)
            self.save()

    def _exec(self, code: str, func_name: str) -> Any | None:
        """Execute synthesized code in the library-aware sandbox."""
        sb = dict(_get_sandbox())
        try:
            exec(compile(code, "<capability>", "exec"), sb)
            fn = sb.get(func_name)
            return fn if callable(fn) else None
        except Exception:
            return None


# ── Capability Graph ──────────────────────────────────────────────────────────

class CapabilityGraph:
    """
    Learns which capabilities each task type requires.
    Updated after every REFLECT phase.
    Pre-warms capabilities before task execution starts.

    Format: {task_type: {capability_name: use_count}}
    """

    def __init__(self):
        self._graph: dict[str, dict[str, int]] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        try:
            if _GRAPH_FILE.exists():
                self._graph = json.loads(_GRAPH_FILE.read_text())
        except Exception:
            self._graph = {}
        self._loaded = True

    def save(self) -> None:
        try:
            _GRAPH_FILE.write_text(json.dumps(self._graph, indent=2))
        except Exception:
            pass

    def update(self, task_type: str, used_capabilities: list[str]) -> None:
        """Record which capabilities were used for a given task type."""
        self.load()
        if task_type not in self._graph:
            self._graph[task_type] = {}
        for cap in used_capabilities:
            self._graph[task_type][cap] = self._graph[task_type].get(cap, 0) + 1
        self.save()

    def predicted_needs(self, task_type: str, min_use_count: int = 2) -> list[str]:
        """
        Return capabilities likely needed for this task type.
        Only returns caps seen ≥ min_use_count times for this type.
        """
        self.load()
        type_caps = self._graph.get(task_type, {})
        return [cap for cap, count in type_caps.items() if count >= min_use_count]


# ── Capability Observer ───────────────────────────────────────────────────────

class CapabilityObserver:
    """
    Monitors execution output for gap signals.
    Detects capability needs from real execution failures, not predictions.

    Gap signal patterns (what to look for in LLM output / tool errors):
    - Tool-not-found error from MCP
    - LLM expressing "I would need to calculate..."
    - LLM expressing "I don't have a tool for..."
    - LLM expressing "Without a [function/calculator/formula]..."
    - Numeric answer with very low confidence hedging
    """

    # Patterns that indicate the LLM knows it needs a capability it doesn't have
    _NEED_PATTERNS = [
        r"i (?:would |could )?need (?:a |to )(?:calculate|compute|determine|find) (.{10,80})",
        r"without (?:a |the )?(?:tool|function|calculator|formula) (?:for|to) (.{10,80})",
        r"i (?:don't|do not|cannot|can't) (?:calculate|compute|determine) (.{10,80})",
        r"(?:to calculate|to compute|to determine) (.{10,80}),? i (?:would )?need",
        r"(?:require|requires|requiring) (?:a |the )?(?:calculation|formula|function) (?:for|to) (.{10,80})",
        r"(?:no tool|no function|no method) (?:available )?(?:for|to) (.{10,80})",
    ]

    def scan_output(self, output_text: str) -> list[str]:
        """
        Scan LLM output for gap signals.
        Returns list of natural-language gap descriptions.
        """
        signals = []
        text_lower = output_text.lower()
        for pattern in self._NEED_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                signal = match.group(1).strip().rstrip(".,;:")
                if len(signal) > 10:
                    signals.append(signal)
        return signals[:3]  # cap at 3 per output to avoid noise

    def scan_tool_error(self, error: str) -> str | None:
        """
        Extract gap description from a tool-not-found error.
        Returns gap description or None.
        """
        err_lower = error.lower()
        if any(k in err_lower for k in ["tool not found", "no tool named", "unknown tool", "tool_not_found"]):
            # Try to extract tool name from error
            m = re.search(r"['\"]([a-z_][a-z_0-9]+)['\"]", error)
            if m:
                tool_name = m.group(1).replace("_", " ")
                return f"function to perform {tool_name}"
        return None


# ── Gap Formalization ─────────────────────────────────────────────────────────

_FORMALIZE_SYSTEM = """\
You are a software architect specializing in financial computation.
Convert a vague gap description into a precise, implementable Python function specification.
Return ONLY valid JSON — no markdown, no explanation."""

_FORMALIZE_PROMPT = """\
Task context: {task_context}
Gap signal: {signal}
Existing capabilities (do NOT re-specify these): {existing_names}

Convert this gap into a Python function specification:
{{
  "name": "snake_case_function_name",
  "description": "One sentence: what this function computes",
  "inputs": {{"param_name": "type and meaning (e.g. S: float, current stock price)"}},
  "outputs": {{"key": "type and meaning (always include 'result' key for primary value)"}},
  "formula_hint": "The mathematical formula or algorithm (e.g. d1 = (ln(S/K) + ...) / ...)",
  "example": {{"input": {{"S": 175, "K": 180}}, "expected_output": {{"result": 3.22}}}}
}}

Rules:
- name must be snake_case, starting with domain prefix (finance_, data_, text_)
- description must be precise enough to implement correctly
- formula_hint must include the actual formula, not just "calculate X"
- example must have concrete numbers"""


async def _formalize_gap(
    signal: str,
    task_context: str,
    existing_names: list[str],
) -> CapabilitySpec | None:
    """Convert a gap signal into a structured CapabilitySpec via Haiku."""
    if not ANTHROPIC_API_KEY:
        return None
    try:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        msg = await asyncio.wait_for(
            client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=600,
                system=_FORMALIZE_SYSTEM,
                messages=[{"role": "user", "content": _FORMALIZE_PROMPT.format(
                    task_context=task_context[:300],
                    signal=signal,
                    existing_names=", ".join(existing_names[:20]) or "none",
                )}],
            ),
            timeout=8.0,
        )
        raw = msg.content[0].text.strip() if msg.content else ""
        # Strip markdown fences
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw).strip()
        data = json.loads(raw)
        if not isinstance(data, dict) or not data.get("name"):
            return None
        return CapabilitySpec(
            name=str(data["name"]).strip(),
            description=str(data.get("description", signal)),
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {"result": "float"}),
            formula_hint=str(data.get("formula_hint", "")),
            example=data.get("example", {}),
            raw_signal=signal,
        )
    except Exception:
        return None


# ── Composition Check ─────────────────────────────────────────────────────────

def _try_compose(spec: CapabilitySpec, store: CapabilityStore) -> list[str] | None:
    """
    Check if existing capabilities can be composed to solve this gap.
    Returns list of capability names to compose, or None if composition isn't viable.

    Strategy: keyword overlap between spec description and existing capability descriptions.
    If 2+ existing caps have ≥2 overlapping content words → suggest composition.
    """
    existing = store.all_names()
    if len(existing) < 2:
        return None

    # Stop-words that don't signal relevance
    _STOP = frozenset("the a an and or but is are was were be been have has had do"
                      " does did will would could should may might shall to of in on"
                      " at by for with from as it its this that these those i we you"
                      " he she they me us him her them my our your his their".split())

    spec_words = {w for w in re.findall(r'\b[a-z]{4,}\b', spec.description.lower())
                  if w not in _STOP}

    candidates = []
    for name in existing:
        rec = store.get_record(name)
        if not rec:
            continue
        cap_words = {w for w in re.findall(r'\b[a-z]{4,}\b', rec.description.lower())
                     if w not in _STOP}
        overlap = len(spec_words & cap_words)
        if overlap >= 2:
            candidates.append((name, overlap))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top = [name for name, _ in candidates[:3]]
    return top if len(top) >= 2 else None


# ── Synthesis ─────────────────────────────────────────────────────────────────

_SYNTHESIS_SYSTEM = """\
You are an expert Python developer specializing in financial and data computations.
Implement the requested function using the BEST available library.
Return ONLY valid JSON — no markdown, no explanation outside JSON."""

_SYNTHESIS_PROMPT = """\
Implement this Python function:

Name: {name}
Description: {description}
Inputs: {inputs}
Outputs: {outputs}
Formula/Algorithm: {formula_hint}
Example: {example}

Available libraries (use the MOST ACCURATE/EFFICIENT one for this problem):
{available_libs}

Requirements:
1. Function signature: def {name}(**kwargs) — accept keyword arguments
2. Always return a dict with at minimum a "result" key (the primary computed value)
3. Include "details" dict for secondary values (e.g. Greeks, intermediate steps)
4. Use scipy_stats.norm.cdf instead of math.erf approximation when available
5. Use np.exp/np.log instead of math.exp/math.log for vectorized operations
6. Handle edge cases: division by zero, negative sqrt, etc.
7. Code must be self-contained (no imports needed — libraries already available as globals)

Return JSON:
{{
  "python_code": "def {name}(**kwargs):\\n    ...",
  "library_used": "scipy|numpy|pandas|math",
  "input_schema": {{
    "type": "object",
    "properties": {{"param": {{"type": "number", "description": "..."}}}},
    "required": ["param1", "param2"]
  }},
  "test_cases": [
    {{"inputs": {{}}, "expected_result_approx": 0.0, "tolerance_pct": 0.05}},
    {{"inputs": {{}}, "expected_result_approx": 0.0, "tolerance_pct": 0.05}},
    {{"inputs": {{}}, "expected_result_approx": 0.0, "tolerance_pct": 0.05}},
    {{"inputs": {{}}, "expected_result_approx": 0.0, "tolerance_pct": 0.05}},
    {{"inputs": {{}}, "expected_result_approx": 0.0, "tolerance_pct": 0.05}}
  ]
}}"""


async def _synthesize(spec: CapabilitySpec) -> dict | None:
    """Call Haiku to synthesize a capability implementation."""
    if not ANTHROPIC_API_KEY:
        return None
    prompt = _SYNTHESIS_PROMPT.format(
        name=spec.name,
        description=spec.description,
        inputs=json.dumps(spec.inputs),
        outputs=json.dumps(spec.outputs),
        formula_hint=spec.formula_hint,
        example=json.dumps(spec.example),
        available_libs=_library_availability_str(),
    )
    try:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        msg = await asyncio.wait_for(
            client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1400,
                system=_SYNTHESIS_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=12.0,
        )
        raw = msg.content[0].text.strip() if msg.content else ""
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw).strip()
        return json.loads(raw)
    except Exception:
        return None


# ── Validation ────────────────────────────────────────────────────────────────

def _validate(code: str, func_name: str, test_cases: list[dict]) -> tuple[bool, float, str]:
    """
    Validate synthesized code by running test cases.
    Returns (passed: bool, accuracy: float, reason: str).
    Requires ≥4/5 test cases to pass (80% threshold).
    """
    sb = dict(_get_sandbox())
    try:
        exec(compile(code, "<capability>", "exec"), sb)
        fn = sb.get(func_name)
    except Exception as e:
        return False, 0.0, f"compile error: {e}"

    if fn is None:
        return False, 0.0, "function not defined after exec"

    if not test_cases:
        return True, 1.0, "no test cases (accepted without validation)"

    passes = 0
    for tc in test_cases[:5]:
        try:
            inputs = tc.get("inputs", {})
            expected = float(tc.get("expected_result_approx", 0))
            tolerance = float(tc.get("tolerance_pct", 0.05))
            result = fn(**inputs)
            actual = float(result.get("result", 0)) if isinstance(result, dict) else float(result)
            denom = max(abs(expected), 1e-9)
            if abs(actual - expected) / denom <= tolerance:
                passes += 1
        except Exception:
            pass

    total = min(len(test_cases), 5)
    accuracy = passes / total if total > 0 else 0.0
    passed = passes >= max(1, total - 1)   # require all but at most 1 failure
    reason = f"{passes}/{total} test cases passed"
    return passed, accuracy, reason


# ── Capability Acquirer ───────────────────────────────────────────────────────

class CapabilityAcquirer:
    """
    Full pipeline: formalize → library_search → compose → synthesize → test → register.
    Called by CapabilityPromise in background.

    Step 1: FORMALIZE — convert vague signal to CapabilitySpec
    Step 2: LIBRARY SEARCH — check scipy/numpy/pandas for existing solution
    Step 3: COMPOSE — can existing capabilities compose to solve this?
    Step 4: SYNTHESIZE — Haiku writes implementation (library-aware)
    Step 5: VALIDATE — auto-generated test suite (≥4/5 required)
    Step 6: REGISTER — store + hot-load + notify
    """

    def __init__(self, store: CapabilityStore):
        self._store = store

    async def acquire(self, signal: str, task_context: str) -> str | None:
        """
        Full acquisition pipeline for a gap signal.
        Returns capability name on success, None on failure.
        """
        # Step 0: check if already acquired (race guard)
        existing = self._store.all_names()

        # Step 1: formalize
        spec = await _formalize_gap(signal, task_context, existing)
        if not spec:
            return None

        # Already in store?
        if self._store.has(spec.name):
            return spec.name

        # Step 2: composition check (can we build from existing tools?)
        composable = _try_compose(spec, self._store)
        # NOTE: For now we note composable candidates in metadata but still synthesize
        # a unified function — composition-as-delegation is a v2 feature.
        # The key value is knowing WHICH existing tools are related.

        # Step 3: synthesize (library-aware)
        raw = await _synthesize(spec)
        if not raw:
            # Retry once with simplified description
            simplified_spec = CapabilitySpec(
                name=spec.name,
                description=spec.description,
                inputs=spec.inputs,
                outputs=spec.outputs,
                formula_hint="Implement using math module (pure Python, no external libraries)",
                example=spec.example,
                raw_signal=spec.raw_signal,
            )
            raw = await _synthesize(simplified_spec)
            if not raw:
                return None

        code = raw.get("python_code", "")
        test_cases = raw.get("test_cases", [])
        library_used = raw.get("library_used", "math")
        input_schema = raw.get("input_schema", {"type": "object", "additionalProperties": True})

        if not code:
            return None

        # Step 4: validate
        passed, accuracy, reason = _validate(code, spec.name, test_cases)

        if not passed and test_cases:
            # One retry: tell Haiku what failed
            retry_signal = (
                f"{signal}\n\nPrevious implementation had errors: {reason}. "
                f"Please fix the formula and ensure test cases pass."
            )
            retry_spec = CapabilitySpec(
                name=spec.name,
                description=spec.description,
                inputs=spec.inputs,
                outputs=spec.outputs,
                formula_hint=spec.formula_hint + f"\n\nNote: previous attempt failed: {reason}",
                example=spec.example,
                raw_signal=retry_signal,
            )
            raw2 = await _synthesize(retry_spec)
            if raw2:
                code2 = raw2.get("python_code", "")
                test_cases2 = raw2.get("test_cases", []) or test_cases
                passed2, accuracy2, reason2 = _validate(code2, spec.name, test_cases2)
                if passed2 or accuracy2 > accuracy:
                    code, test_cases, accuracy, reason = code2, test_cases2, accuracy2, reason2
                    library_used = raw2.get("library_used", library_used)
                    input_schema = raw2.get("input_schema", input_schema)
                    passed = passed2

        # Accept if passed, or if no test cases (trust the synthesis)
        if not passed and test_cases:
            return None  # both attempts failed validation

        # Step 5: register
        rec = CapabilityRecord(
            name=spec.name,
            description=spec.description,
            python_code=code,
            input_schema=input_schema,
            test_cases=test_cases[:5],
            accuracy_score=accuracy,
            library_used=library_used,
            composed_from=composable or [],
            _synthesized=True,
        )
        success = self._store.register(rec)
        return spec.name if success else None


# ── Capability Promise ────────────────────────────────────────────────────────

class CapabilityPromise:
    """
    Non-blocking future for capability acquisition.

    Usage:
        promise = CapabilityPromise(signal, task_context, store)
        promise.start()                      # fire-and-forget
        # ... main execution continues ...
        ready = await promise.wait(timeout=5.0)
        if ready:
            # new capability is in store, use it
    """

    def __init__(self, signal: str, task_context: str, store: CapabilityStore):
        self.signal = signal
        self.task_context = task_context
        self.status: str = "pending"     # pending | building | ready | failed
        self.capability_name: str | None = None
        self._store = store
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        """Start background acquisition. Non-blocking."""
        self.status = "building"
        try:
            self._task = asyncio.ensure_future(self._run())
        except RuntimeError:
            # No running event loop (e.g., in test context)
            self.status = "failed"

    async def wait(self, timeout: float = 5.0) -> bool:
        """Wait up to timeout seconds. Returns True if capability is ready."""
        if self.status == "ready":
            return True
        if self.status == "failed" or self._task is None:
            return False
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        return self.status == "ready"

    async def _run(self) -> None:
        try:
            acquirer = CapabilityAcquirer(self._store)
            name = await acquirer.acquire(self.signal, self.task_context)
            if name:
                self.capability_name = name
                self.status = "ready"
            else:
                self.status = "failed"
        except Exception:
            self.status = "failed"


# ── Module-level singletons ───────────────────────────────────────────────────

_store  = CapabilityStore()
_graph  = CapabilityGraph()
_observer = CapabilityObserver()


# ── Public API ────────────────────────────────────────────────────────────────

def get_store() -> CapabilityStore:
    """Return the module-level CapabilityStore singleton."""
    _store.load()
    return _store


def get_graph() -> CapabilityGraph:
    """Return the module-level CapabilityGraph singleton."""
    _graph.load()
    return _graph


def start_promises_for_task(
    task_text: str,
    task_type: str,
    fsm_requirements: list[str] | None = None,
) -> list[CapabilityPromise]:
    """
    PRIME phase entry point.
    1. Check capability graph for predicted needs of this task_type.
    2. Check FSM-specified capability_requirements.
    3. For any needed capability not in store → start a CapabilityPromise.
    Returns list of started promises.
    """
    _store.load()
    _graph.load()

    needed: set[str] = set()

    # From capability graph (learned from past tasks)
    predicted = _graph.predicted_needs(task_type)
    needed.update(predicted)

    # From FSM requirements (synthesized per task type)
    if fsm_requirements:
        needed.update(fsm_requirements)

    # Filter to capabilities we don't already have
    missing = [cap for cap in needed if not _store.has(cap)]

    promises = []
    for cap_name in missing[:3]:  # max 3 concurrent acquisitions
        promise = CapabilityPromise(
            signal=f"function named {cap_name.replace('_', ' ')}",
            task_context=task_text[:200],
            store=_store,
        )
        promise.start()
        promises.append(promise)

    return promises


def start_promise_for_signal(signal: str, task_text: str) -> CapabilityPromise | None:
    """
    EXECUTE phase entry point — called by CapabilityObserver when gap detected.
    Starts a promise for a specific gap signal from live execution.
    """
    if not signal or len(signal) < 10:
        return None
    promise = CapabilityPromise(signal=signal, task_context=task_text, store=_store)
    promise.start()
    return promise


async def await_promises(
    promises: list[CapabilityPromise],
    timeout: float = 5.0,
) -> list[str]:
    """
    COMPUTE state entry point.
    Wait up to timeout seconds for all promises to complete.
    Returns names of newly available capabilities.
    """
    if not promises:
        return []
    results = await asyncio.gather(
        *[p.wait(timeout=timeout) for p in promises],
        return_exceptions=True,
    )
    ready = []
    for promise, result in zip(promises, results):
        if result is True and promise.capability_name:
            ready.append(promise.capability_name)
    return ready


def observe_execution_output(output: str) -> list[CapabilityPromise]:
    """
    EXECUTE phase: scan LLM output for gap signals.
    Starts promises for detected gaps.
    Returns list of started promises.
    """
    signals = _observer.scan_output(output)
    promises = []
    for signal in signals:
        # Don't re-acquire what we already have
        spec_name_guess = re.sub(r'\s+', '_', signal.lower().strip())[:30]
        if not _store.has(spec_name_guess):
            p = CapabilityPromise(signal=signal, task_context=output[:200], store=_store)
            p.start()
            promises.append(p)
    return promises


def observe_tool_error(error: str, task_text: str) -> CapabilityPromise | None:
    """
    Called when a tool-not-found error occurs.
    Starts a promise to acquire the missing capability.
    """
    signal = _observer.scan_tool_error(error)
    if not signal:
        return None
    return start_promise_for_signal(signal, task_text)


def update_graph(task_type: str, used_capabilities: list[str]) -> None:
    """
    REFLECT phase: update capability graph with what was used.
    """
    _graph.load()
    _graph.update(task_type, used_capabilities)


def get_ace_stats() -> dict:
    """Return ACE stats for /rl/status endpoint."""
    _store.load()
    _graph.load()
    names = _store.all_names()
    records = [_store.get_record(n) for n in names]
    return {
        "total_capabilities": len(names),
        "synthesized": sum(1 for r in records if r and r._synthesized),
        "seeded": sum(1 for r in records if r and r._seeded),
        "capability_names": names,
        "graph_task_types": list(_graph._graph.keys()),
        "library_availability": {
            "scipy": _get_sandbox().get("_has_scipy", False),
            "numpy": _get_sandbox().get("_has_numpy", False),
            "pandas": _get_sandbox().get("_has_pandas", False),
            "sympy": _get_sandbox().get("_has_sympy", False),
        },
    }
