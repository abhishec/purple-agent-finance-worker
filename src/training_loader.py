"""
training_loader.py
Pulls benchmark training data from S3/HTTP → seeds rl_loop case_log.
Positive runs (passing benchmark examples) → quality=1.0 CaseEntry records.

Two data paths:
1. S3 direct: boto3 download of JSONL from agentbench-training-data/
2. HTTP fallback: POST https://benchmark.usebrainos.com/training-data/export?hours=N
"""
from __future__ import annotations
import json
import os
import time
import hashlib
from typing import Any

from src.config import (
    S3_TRAINING_BUCKET,
    S3_TRAINING_PREFIX,
    BENCHMARK_API_URL,
    BRAINOS_API_KEY,
)
from src.rl_loop import CaseEntry, _load_cases, _save_cases, _extract_keywords

SEED_MARKER_PATH = os.path.join(os.path.dirname(__file__), "..", ".training_seeded")
STALE_HOURS = 12   # re-seed after 12h
MAX_SEED_ENTRIES = 100


# ── S3 download ───────────────────────────────────────────────────────────────

def _download_from_s3(prefix: str) -> list[dict]:
    """Download all JSONL objects under prefix. Returns list of parsed records."""
    try:
        import boto3  # type: ignore
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        records: list[dict] = []

        for page in paginator.paginate(Bucket=S3_TRAINING_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not (key.endswith(".jsonl") or key.endswith(".json")):
                    continue
                resp = s3.get_object(Bucket=S3_TRAINING_BUCKET, Key=key)
                body = resp["Body"].read().decode("utf-8")
                for line in body.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        # Non-JSONL file (whole-JSON) — try single parse
                        try:
                            records.append(json.loads(body))
                        except Exception:
                            pass
                        break
        return records
    except ImportError:
        return []
    except Exception:
        return []


def _download_from_http(hours: int = 24) -> list[dict]:
    """Fallback: pull training data via HTTP export endpoint."""
    try:
        import urllib.request
        url = f"{BENCHMARK_API_URL}/training-data/export?hours={hours}"
        req = urllib.request.Request(
            url,
            method="POST",
            headers={
                "Authorization": f"Bearer {BRAINOS_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
        records = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass
        # Maybe it was a JSON array
        if not records:
            try:
                records = json.loads(raw)
                if isinstance(records, dict):
                    records = records.get("data", [records])
            except Exception:
                pass
        return records
    except Exception:
        return []


# ── JSONL → CaseEntry conversion ──────────────────────────────────────────────

def _messages_to_task_summary(messages: list[dict]) -> str:
    """Extract first user message as the task summary."""
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                # Claude Messages API content blocks
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return block.get("text", "")[:120]
                return str(content)[:120]
            return str(content)[:120]
    return ""


def _extract_tool_count(messages: list[dict]) -> int:
    """Count tool_use blocks in an assistant turn."""
    count = 0
    for m in messages:
        if m.get("role") == "assistant":
            content = m.get("content", "")
            if isinstance(content, list):
                count += sum(1 for b in content if isinstance(b, dict) and b.get("type") == "tool_use")
    return count


def _extract_answer(messages: list[dict]) -> str:
    """Extract last assistant text message as the answer."""
    answer = ""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            content = m.get("content", "")
            if isinstance(content, str):
                answer = content
                break
            if isinstance(content, list):
                for block in reversed(content):
                    if isinstance(block, dict) and block.get("type") == "text":
                        answer = block.get("text", "")
                        break
                if answer:
                    break
    return answer


def _record_to_case_entry(record: dict) -> CaseEntry | None:
    """
    Convert one training record (Claude Messages API multi-turn) to CaseEntry.
    Training data = positive examples → quality forced to 1.0.
    """
    # Top-level fields
    task_id = record.get("task_id") or record.get("id") or ""
    messages = record.get("messages", [])
    metadata = record.get("metadata", {})

    if not messages:
        return None

    task_summary = (
        metadata.get("task_summary")
        or metadata.get("task")
        or _messages_to_task_summary(messages)
    )
    if not task_summary:
        return None

    tool_count = _extract_tool_count(messages)
    answer = _extract_answer(messages)

    # Positive training examples → always mark success / quality 1.0
    what_worked = metadata.get("what_worked", "")
    if not what_worked:
        parts = []
        if tool_count > 0:
            parts.append(f"Used {tool_count} tool calls")
        process_type = metadata.get("process_type", "")
        if process_type:
            parts.append(f"Process: {process_type}")
        domain = metadata.get("domain", "")
        if domain:
            parts.append(f"Domain: {domain}")
        what_worked = ". ".join(parts) if parts else "Training example — positive run"

    case_id = hashlib.md5(f"seed:{task_id}:{task_summary[:30]}".encode()).hexdigest()[:8]

    return CaseEntry(
        case_id=case_id,
        task_summary=task_summary[:120],
        keywords=_extract_keywords(task_summary),
        outcome="success",
        quality=1.0,
        what_worked=what_worked,
        what_failed="",
        tool_count=tool_count,
        timestamp=time.time(),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def seed_from_training_data(force: bool = False) -> dict:
    """
    Download benchmark training data and seed the RL case log.
    Returns stats dict: {seeded, skipped, source, total_cases}.
    Idempotent — skips if seeded within STALE_HOURS unless force=True.
    """
    # Check staleness
    if not force and os.path.exists(SEED_MARKER_PATH):
        try:
            mtime = os.path.getmtime(SEED_MARKER_PATH)
            if time.time() - mtime < STALE_HOURS * 3600:
                return {"seeded": 0, "skipped": True, "source": "cache", "total_cases": len(_load_cases())}
        except Exception:
            pass

    # Try S3 first, fall back to HTTP
    source = "s3"
    raw_records = _download_from_s3(S3_TRAINING_PREFIX)
    if not raw_records:
        source = "http"
        raw_records = _download_from_http()

    if not raw_records:
        return {"seeded": 0, "skipped": False, "source": "none", "total_cases": len(_load_cases())}

    # Convert to CaseEntry
    new_entries = []
    for rec in raw_records[:MAX_SEED_ENTRIES]:
        entry = _record_to_case_entry(rec)
        if entry:
            new_entries.append(entry)

    if not new_entries:
        return {"seeded": 0, "skipped": False, "source": source, "total_cases": len(_load_cases())}

    # Merge with existing cases — seed entries go first (highest priority primer)
    from dataclasses import asdict
    existing = _load_cases()
    existing_ids = {c.get("case_id") for c in existing}
    fresh = [asdict(e) for e in new_entries if e.case_id not in existing_ids]
    merged = fresh + existing  # seed entries at front → highest retrieval priority
    _save_cases(merged)

    # Write seed marker
    try:
        with open(SEED_MARKER_PATH, "w") as f:
            f.write(str(time.time()))
    except Exception:
        pass

    return {
        "seeded": len(fresh),
        "skipped": False,
        "source": source,
        "total_cases": len(merged),
    }


def is_stale() -> bool:
    """True if training data needs refresh."""
    if not os.path.exists(SEED_MARKER_PATH):
        return True
    try:
        return time.time() - os.path.getmtime(SEED_MARKER_PATH) > STALE_HOURS * 3600
    except Exception:
        return True
