"""
schema_adapter.py
Schema drift resilience for ASSESS state.
Inspired by BrainOS brain/schema-drift-handler.ts.

When a tool call fails with "column not found":
1. Run schema introspection
2. Fuzzy-match closest column name (Levenshtein via difflib)
3. Retry with corrected name
4. Cache mapping for this session

Also handles empty results that indicate column filter drift (Fix A).
"""
from __future__ import annotations
import re
from difflib import get_close_matches, SequenceMatcher
from typing import Callable, Awaitable

# Canonical → known aliases (mirrors BrainOS KNOWN_COLUMN_ALIASES)
KNOWN_COLUMN_ALIASES: dict[str, list[str]] = {
    "client_name":      ["customer_name", "account_name", "company_name", "org_name"],
    "amount":           ["value", "total", "price", "cost", "sum", "total_amount"],
    "user_id":          ["creator_id", "employee_id"],
    "name":             ["title", "label", "display_name", "full_name"],
    "category":         ["type", "classification", "group", "kind"],
    # Extended aliases (Fix C)
    "email":            ["em", "e_mail", "email_address", "contact_email", "mail"],
    "status":           ["st", "stat", "state", "state_code", "current_status"],
    "owner_id":         ["oid", "owner", "asgn", "assigned_to", "assignee_id"],
    "created_at":       ["created", "create_date", "creation_date", "ts",
                         "created_date", "date_created", "timestamp"],
    "updated_at":       ["updated", "update_date", "modified_at", "last_modified",
                         "updated_date", "modification_date"],
    "description":      ["desc", "descr", "detail", "details", "notes",
                         "comments", "body", "content"],
}

SCHEMA_ERROR_PATTERNS = [
    r"column[s]?\s+['\"]?(\w+)['\"]?\s+(?:not found|does not exist|unknown|not recognized)",
    r"no such column[s]?:?\s+['\"]?(\w+)['\"]?",
    r"invalid column name[s]?\s+['\"]?(\w+)['\"]?",
    r"unknown column[s]?[:\s]+['\"]?(\w+)['\"]?",
    r"field[s]?\s+['\"]?(\w+)['\"]?\s+(?:not found|does not exist)",
    r"KeyError:\s+['\"]?(\w+)['\"]?",
]

# Keys whose empty value signals a filtered-but-drifted query (Fix A)
_EMPTY_RESULT_KEYS = ("data", "items", "records", "results", "rows", "list")

# Signal words that identify schema introspection tools when scanning the available
# tool list dynamically.  Any tool whose name contains one of these substrings is
# considered a schema tool candidate.
_SCHEMA_TOOL_SIGNALS = [
    "schema", "column", "describe", "inspect", "metadata", "structure", "table_info",
]

# Hard-coded fallback names — always tried even when not in the available tools list,
# so existing behaviour is preserved when no tool list is supplied.
_KNOWN_SCHEMA_TOOLS = ["describe_table", "get_schema", "list_columns", "schema_introspect"]


def _find_schema_tools(available_tools: list[str]) -> list[str]:
    """
    Return an ordered list of schema introspection tool names to try.

    Strategy:
      1. Scan ``available_tools`` for names containing any of ``_SCHEMA_TOOL_SIGNALS``.
      2. Append the 4 known fallback names so they are always attempted, even if the
         live tool list is empty or does not include them.

    Callers should try tools in the returned order and stop on the first success.
    """
    found: list[str] = []
    for tool_name in available_tools:
        name_lower = tool_name.lower()
        if any(signal in name_lower for signal in _SCHEMA_TOOL_SIGNALS):
            if tool_name not in found:
                found.append(tool_name)

    # Always append the known fallback names so we never regress on envs where the
    # tool list is unavailable.
    for kt in _KNOWN_SCHEMA_TOOLS:
        if kt not in found:
            found.append(kt)

    return found


def _result_is_empty_due_to_drift(result: dict) -> bool:
    """
    Return True when a successful (non-error) tool result has an empty collection
    that may indicate a column filter matched nothing due to schema drift.
    """
    if not isinstance(result, dict):
        return False
    if "error" in str(result).lower():
        return False
    for key in _EMPTY_RESULT_KEYS:
        val = result.get(key)
        if val is not None and isinstance(val, (list, dict)) and len(val) == 0:
            # Don't treat as drift if total/count says real data exists (filtered result)
            if result.get("total", result.get("count", result.get("total_count", 0))) > 0:
                continue
            return True
    return False


def detect_schema_error(error_text: str) -> str | None:
    """Extract the bad column name from an error message. None if not a schema error."""
    text = error_text.lower()
    for pattern in SCHEMA_ERROR_PATTERNS:
        m = re.search(pattern, text)
        if m and m.lastindex >= 1:
            return m.group(1)
    return None


def fuzzy_match_column(bad_col: str, candidates: list[str]) -> str | None:
    """
    Match bad_col to closest candidate.
    Order:
      Tier 1: exact match
      Tier 2: known alias lookup
      Tier 3: difflib close match (cutoff 0.6, or 0.5 for short abbreviations)
      Tier 4: Levenshtein ratio fallback (>0.7)
      Tier 5: prefix matching (bad_col is prefix of candidate, or vice versa)

    Mirrors BrainOS fuzzyMatchColumn().
    """
    if not candidates:
        return None

    # Tier 1: Exact match
    if bad_col in candidates:
        return bad_col

    # Tier 2: Known alias lookup (both directions)
    for canonical, aliases in KNOWN_COLUMN_ALIASES.items():
        if bad_col in aliases and canonical in candidates:
            return canonical
        if bad_col == canonical:
            match = next((a for a in aliases if a in candidates), None)
            if match:
                return match

    # Tier 3: difflib close match
    # Lower cutoff for short abbreviations (len <= 3) — e.g. "em", "st", "ts"
    cutoff = 0.5 if len(bad_col) <= 3 else 0.6
    matches = get_close_matches(bad_col, candidates, n=1, cutoff=cutoff)
    if matches:
        return matches[0]

    # Tier 4: Levenshtein ratio fallback (>0.7)
    best, best_ratio = None, 0.0
    for c in candidates:
        ratio = SequenceMatcher(None, bad_col, c).ratio()
        if ratio > best_ratio:
            best_ratio, best = ratio, c
    if best and best_ratio > 0.7:
        return best

    # Tier 5: Prefix matching
    # bad_col is a prefix of a candidate (e.g. "own" → "owner_id")
    prefix_candidates = [c for c in candidates if c.startswith(bad_col)]
    if prefix_candidates:
        # Prefer shortest match (most specific prefix hit)
        return min(prefix_candidates, key=len)
    # Or a candidate is a prefix of bad_col (e.g. "email" candidate, "email_addr" bad_col)
    suffix_candidates = [c for c in candidates if bad_col.startswith(c)]
    if suffix_candidates:
        return max(suffix_candidates, key=len)

    return None


def _replace_in_params(params: dict, bad: str, good: str) -> dict:
    result = {}
    for k, v in params.items():
        if isinstance(v, str):
            result[k] = v.replace(bad, good)
        elif isinstance(v, dict):
            result[k] = _replace_in_params(v, bad, good)
        elif isinstance(v, list):
            result[k] = [
                _replace_in_params(i, bad, good) if isinstance(i, dict)
                else i.replace(bad, good) if isinstance(i, str)
                else i
                for i in v
            ]
        else:
            result[k] = v
    return result


async def _attempt_schema_correction(
    tool_name: str,
    params: dict,
    error_text: str,
    on_tool_call: Callable[[str, dict], Awaitable[dict]],
    schema_cache: dict,
    available_tool_names: list[str] | None = None,
) -> dict | None:
    """
    Core correction logic: introspect schema, fuzzy-match the bad column,
    retry with corrected params. Returns corrected result or None if unable.

    ``available_tool_names`` — names of tools available in the current environment.
    When supplied, schema tool discovery is dynamic (see ``_find_schema_tools``).
    When omitted, falls back to the 4 hard-coded names (no regression).
    """
    bad_col = detect_schema_error(error_text) if error_text else None

    # For empty-result drift we attempt correction even without a named bad column.
    # Try each param key that looks like a column filter.
    cols_to_try: list[str] = []
    if bad_col:
        cols_to_try = [bad_col]
    else:
        # Heuristic: string-valued params other than id/session fields may be column filters
        cols_to_try = [
            k for k, v in params.items()
            if isinstance(v, str) and k not in ("table", "table_name", "resource",
                                                  "session_id", "organization_id")
        ]

    if not cols_to_try:
        return None

    # Introspect schema once — try tools in dynamically-discovered order
    table = params.get("table") or params.get("table_name") or params.get("resource", "")
    schema_result = None
    schema_tools = _find_schema_tools(available_tool_names or [])
    for schema_tool in schema_tools:
        try:
            r = await on_tool_call(schema_tool, {"table": table} if table else {})
            if not (isinstance(r, dict) and "error" in r):
                schema_result = r
                break
        except Exception:
            continue

    if not schema_result:
        return None

    schema_text = str(schema_result)
    columns = list(set(re.findall(r'\b([a-z_][a-z0-9_]{2,})\b', schema_text.lower())))

    corrected_params = dict(params)
    made_correction = False
    for col in cols_to_try:
        cache_key = f"{tool_name}:{col}"
        if cache_key in schema_cache:
            corrected = schema_cache[cache_key]
        else:
            corrected = fuzzy_match_column(col, columns)
            if corrected:
                schema_cache[cache_key] = corrected

        if corrected and corrected != col:
            corrected_params = _replace_in_params(corrected_params, col, corrected)
            made_correction = True

    if not made_correction:
        return None

    return await on_tool_call(tool_name, corrected_params)


async def resilient_tool_call(
    tool_name: str,
    params: dict,
    on_tool_call: Callable[[str, dict], Awaitable[dict]],
    schema_cache: dict,
    available_tool_names: list[str] | None = None,
) -> dict:
    """
    Wrapper around on_tool_call with schema drift retry.

    Error path: if call fails with a column error → introspect → fuzzy-match → retry.
    Empty result path (Fix A): if call succeeds but returns empty collection
    → also attempt column correction (filter may reference a drifted column name).

    ``available_tool_names`` — optional list of tool names available in the current
    environment.  When provided, schema introspection tool discovery is dynamic
    (non-standard names like ``inspect_table`` or ``table_metadata`` are found
    automatically).  When omitted the 4 known fallback names are used (same
    behaviour as the previous hard-coded implementation).
    """
    result = await on_tool_call(tool_name, params)
    error_text = str(result.get("error", "")) if isinstance(result, dict) else str(result)
    # has_error: True whenever the "error" key is present and non-empty.
    # The old check (`"error" in error_text.lower()`) falsely skipped correction
    # for messages like "column 'foo' not found" or "no such column: foo"
    # that don't contain the word "error" in the value itself.
    has_error = bool(error_text)

    if has_error:
        # Standard error correction path
        corrected = await _attempt_schema_correction(
            tool_name, params, error_text, on_tool_call, schema_cache,
            available_tool_names=available_tool_names,
        )
        if corrected is not None:
            return corrected
        return result

    # Fix A: empty result path — may indicate column filter drift
    if _result_is_empty_due_to_drift(result):
        corrected = await _attempt_schema_correction(
            tool_name, params, "", on_tool_call, schema_cache,
            available_tool_names=available_tool_names,
        )
        if corrected is not None and not _result_is_empty_due_to_drift(corrected):
            return corrected

    return result
