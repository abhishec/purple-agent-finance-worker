"""
report_analyzer.py
Downloads benchmark reports from S3/HTTP, extracts dimension scores and
failure patterns, generates benchmark_intelligence.json.

This intelligence is injected into build_rl_primer() so the agent knows
EXACTLY where it lost points last run and what to do differently.
"""
from __future__ import annotations
import json
import os
import time

from src.config import (
    S3_TRAINING_BUCKET,
    S3_REPORTS_PREFIX,
    BENCHMARK_API_URL,
    BRAINOS_API_KEY,
)

INTELLIGENCE_PATH = os.path.join(os.path.dirname(__file__), "..", "benchmark_intelligence.json")
STALE_HOURS = 6   # refresh reports more frequently than training data


# ── Download helpers ──────────────────────────────────────────────────────────

def _latest_report_from_s3() -> dict | None:
    """Download the most-recent report JSON from S3 reports prefix."""
    try:
        import boto3  # type: ignore
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        objects = []
        for page in paginator.paginate(Bucket=S3_TRAINING_BUCKET, Prefix=S3_REPORTS_PREFIX):
            objects.extend(page.get("Contents", []))

        # Sort by LastModified, pick newest .json
        json_objs = [o for o in objects if o["Key"].endswith(".json")]
        if not json_objs:
            return None
        latest = max(json_objs, key=lambda o: o["LastModified"])
        resp = s3.get_object(Bucket=S3_TRAINING_BUCKET, Key=latest["Key"])
        return json.loads(resp["Body"].read().decode("utf-8"))
    except Exception:
        return None


def _latest_report_from_http() -> dict | None:
    """Fallback: trigger a fresh report via HTTP."""
    try:
        import urllib.request
        url = f"{BENCHMARK_API_URL}/report/now?hours=4"
        req = urllib.request.Request(
            url,
            method="POST",
            headers={
                "Authorization": f"Bearer {BRAINOS_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


# ── Report parsing ────────────────────────────────────────────────────────────

def _parse_dimension_scores(report: dict) -> dict[str, float]:
    """
    Extract per-dimension scores from benchmark report.
    Handles both flat and nested report formats.
    """
    scores: dict[str, float] = {}

    # Format 1: top-level "dimensions" key
    dims = report.get("dimensions", {})
    if isinstance(dims, dict):
        for k, v in dims.items():
            if isinstance(v, (int, float)):
                scores[k] = float(v)
            elif isinstance(v, dict):
                scores[k] = float(v.get("score", v.get("average", 0)))

    # Format 2: "results" list with per-task scores
    results = report.get("results", [])
    if isinstance(results, list):
        dim_totals: dict[str, list[float]] = {}
        for result in results:
            for k, v in result.items():
                if k.endswith("_score") or k in ("quality", "correctness", "tool_use", "policy", "format"):
                    if isinstance(v, (int, float)):
                        dim_totals.setdefault(k, []).append(float(v))
        for k, vals in dim_totals.items():
            scores[k] = round(sum(vals) / len(vals), 3)

    # Format 3: "summary" block
    summary = report.get("summary", {})
    if isinstance(summary, dict):
        for k, v in summary.items():
            if isinstance(v, (int, float)) and k not in scores:
                scores[k] = float(v)

    return scores


def _extract_failure_patterns(report: dict) -> list[dict]:
    """
    Extract actionable failure patterns from report.
    Returns list of {task, dimension, score, guidance} dicts.
    """
    patterns = []

    results = report.get("results", [])
    if not isinstance(results, list):
        return patterns

    for result in results:
        task = result.get("task", result.get("task_id", ""))
        errors = result.get("errors", result.get("failures", []))
        score = result.get("score", result.get("quality", None))

        if isinstance(errors, list):
            for err in errors:
                if isinstance(err, str):
                    patterns.append({
                        "task": str(task)[:80],
                        "dimension": "general",
                        "score": score,
                        "guidance": _error_to_guidance(err),
                    })
                elif isinstance(err, dict):
                    dim = err.get("dimension", err.get("type", "general"))
                    msg = err.get("message", err.get("error", str(err)))
                    patterns.append({
                        "task": str(task)[:80],
                        "dimension": dim,
                        "score": score,
                        "guidance": _error_to_guidance(msg),
                    })

        # Low-score tasks without explicit errors
        if score is not None and float(score) < 0.7 and not errors:
            patterns.append({
                "task": str(task)[:80],
                "dimension": "quality",
                "score": score,
                "guidance": f"Low score ({score:.2f}) — review tool usage and answer completeness",
            })

    return patterns[:30]   # cap to avoid bloat


def _error_to_guidance(error_msg: str) -> str:
    """Convert raw error message to actionable agent guidance."""
    msg = str(error_msg).lower()

    if "float" in msg or "precision" in msg or "rounding" in msg:
        return "Use integer cents for all financial math — never raw floats"
    if "policy" in msg or "approval" in msg or "unauthorized" in msg:
        return "Always check policy_checker before any mutation tool call"
    if "hitl" in msg or "human" in msg or "gate" in msg:
        return "Trigger APPROVAL_GATE state before executing mutation tools"
    if "tool" in msg and ("timeout" in msg or "error" in msg or "fail" in msg):
        return "Use resilient_tool_call with retry — tools can timeout transiently"
    if "paginate" in msg or "cursor" in msg or "page" in msg:
        return "Use paginated_fetch for large result sets — never assume single page"
    if "schema" in msg or "column" in msg or "field" in msg:
        return "Use schema_adapter fuzzy matching when column names vary"
    if "format" in msg or "json" in msg or "output" in msg:
        return "Return structured JSON answer — not plain text"
    if "privacy" in msg or "pii" in msg or "sensitive" in msg:
        return "Run privacy_guard check before exposing PII fields in output"
    if "timeout" in msg or "deadline" in msg:
        return "Check token_budget before each LLM call — skip if over 80%"
    if "fsm" in msg or "state" in msg:
        return "Ensure FSM progresses through all required states for this process type"

    return error_msg[:120]


def _score_to_guidance(dim: str, score: float) -> str:
    """Generate improvement guidance based on low dimension score."""
    if score >= 0.8:
        return ""

    guidance_map = {
        "tool_use": "Increase tool call depth — use at least 3 tool calls for data tasks",
        "policy": "Run policy_checker deterministically before any mutation",
        "format": "Return answers in structured JSON with all required fields",
        "quality": "Provide comprehensive answers with supporting data from tools",
        "correctness": "Cross-check calculations — use financial_calculator for all math",
        "completeness": "Ensure all sub-tasks in multi-step requests are addressed",
        "hitl": "Gate mutation tools behind APPROVAL_GATE FSM state",
        "privacy": "Filter PII from output — run privacy_guard on all field names",
    }
    for k, g in guidance_map.items():
        if k in dim.lower():
            return g
    return f"Improve {dim} (current: {score:.2f}) — review task handling for this dimension"


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_and_save(force: bool = False) -> dict:
    """
    Download latest benchmark report, extract intelligence, save to disk.
    Returns {refreshed, overall_score, weak_dimensions, failure_count}.
    """
    # Check staleness
    if not force and os.path.exists(INTELLIGENCE_PATH):
        try:
            mtime = os.path.getmtime(INTELLIGENCE_PATH)
            if time.time() - mtime < STALE_HOURS * 3600:
                with open(INTELLIGENCE_PATH) as f:
                    intel = json.load(f)
                return {
                    "refreshed": False,
                    "overall_score": intel.get("overall_score", 0),
                    "weak_dimensions": intel.get("weak_dimensions", []),
                    "failure_count": len(intel.get("failure_patterns", [])),
                }
        except Exception:
            pass

    # Download report
    report = _latest_report_from_s3()
    if not report:
        report = _latest_report_from_http()
    if not report:
        return {"refreshed": False, "overall_score": 0, "weak_dimensions": [], "failure_count": 0}

    # Parse
    dim_scores = _parse_dimension_scores(report)
    failure_patterns = _extract_failure_patterns(report)

    overall = report.get("overall_score", report.get("score", report.get("pass_rate", 0)))
    if not isinstance(overall, (int, float)):
        overall = sum(dim_scores.values()) / len(dim_scores) if dim_scores else 0

    # Identify weak dimensions (score < 0.8)
    weak_dims = [
        {"dimension": k, "score": round(v, 3), "guidance": _score_to_guidance(k, v)}
        for k, v in sorted(dim_scores.items(), key=lambda x: x[1])
        if v < 0.8
    ]

    intelligence = {
        "generated_at": time.time(),
        "overall_score": round(float(overall), 3),
        "dimension_scores": {k: round(v, 3) for k, v in dim_scores.items()},
        "weak_dimensions": weak_dims,
        "failure_patterns": failure_patterns,
        "run_count": report.get("run_count", report.get("total_runs", len(report.get("results", [])))),
    }

    try:
        with open(INTELLIGENCE_PATH, "w") as f:
            json.dump(intelligence, f, indent=2)
    except Exception:
        pass

    return {
        "refreshed": True,
        "overall_score": intelligence["overall_score"],
        "weak_dimensions": [d["dimension"] for d in weak_dims],
        "failure_count": len(failure_patterns),
    }


def load_intelligence() -> dict:
    """Load cached benchmark intelligence. Returns empty dict if unavailable."""
    try:
        if os.path.exists(INTELLIGENCE_PATH):
            with open(INTELLIGENCE_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def build_benchmark_primer() -> str:
    """
    Build a benchmark-focused primer from the intelligence report.
    Injected into agent system prompt by worker_brain.py PRIME phase.
    """
    intel = load_intelligence()
    if not intel:
        return ""

    lines = ["## BENCHMARK INTELLIGENCE (apply to this task)"]

    overall = intel.get("overall_score", 0)
    if overall > 0:
        lines.append(f"Last run overall score: {overall:.1%}")

    weak = intel.get("weak_dimensions", [])
    if weak:
        lines.append("\n### Areas needing improvement:")
        for w in weak[:5]:
            guidance = w.get("guidance", "")
            if guidance:
                lines.append(f"  ⚠ {w['dimension']} ({w['score']:.0%}): {guidance}")

    failures = intel.get("failure_patterns", [])
    if failures:
        lines.append("\n### Known failure patterns to avoid:")
        for fp in failures[:5]:
            guidance = fp.get("guidance", "")
            if guidance:
                lines.append(f"  ✗ {guidance}")

    lines.append("")
    return "\n".join(lines)
