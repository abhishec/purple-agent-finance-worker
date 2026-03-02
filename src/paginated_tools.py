"""
paginated_tools.py
Cursor-based pagination for bulk data tool calls.
Ported from BrainOS batch-ingestion-engine.ts BatchIngestionConfig pattern.

Most benchmark tools cap at 10-100 records per call.
Tasks 11-15 have 287 transactions, 156 invoices, 312 PRs — need looping.

Supports: page/limit, cursor, offset, has_more patterns.
"""
from __future__ import annotations
from typing import Callable, Awaitable

MAX_PAGES = 20
DEFAULT_PAGE_SIZE = 100

# Common result container keys across different tool APIs
_RESULT_KEYS = ("data", "results", "records", "items", "rows",
                "transactions", "invoices", "tickets", "accounts",
                "issues", "users", "deals", "contacts", "entries")


async def paginated_fetch(
    tool_name: str,
    base_params: dict,
    on_tool_call: Callable[[str, dict], Awaitable[dict]],
    max_pages: int = MAX_PAGES,
    result_key: str | None = None,
) -> list:
    """
    Call a tool repeatedly until all pages exhausted.
    Mirrors BrainOS batch-ingestion-engine cursor-based loop.

    Returns aggregated list of all records across all pages.
    Stops on: empty result, cursor exhausted, error, max_pages reached.
    """
    all_records: list = []
    page = 1
    cursor: str | None = None
    offset = 0

    for _ in range(max_pages):
        params = {**base_params}

        if cursor:
            params["cursor"] = cursor
        else:
            params.setdefault("page", page)
            params.setdefault("limit", DEFAULT_PAGE_SIZE)
            params.setdefault("per_page", DEFAULT_PAGE_SIZE)
            params.setdefault("offset", offset)

        result = await on_tool_call(tool_name, params)

        if isinstance(result, dict) and "error" in result:
            break  # tool error — stop gracefully

        records = _extract_records(result, result_key)
        if not records:
            break

        all_records.extend(records)

        # Detect next page signal
        if isinstance(result, dict):
            next_cursor = result.get("next_cursor") or result.get("cursor")
            has_more = result.get("has_more") or result.get("next_page") or result.get("has_next_page")
            total = result.get("total") or result.get("total_count") or result.get("count")

            if next_cursor:
                cursor = next_cursor
                continue

            if total and len(all_records) >= int(total):
                break  # got everything

            if has_more:
                page += 1
                offset += len(records)
                continue

            if len(records) < params.get("limit", DEFAULT_PAGE_SIZE):
                break  # returned fewer than page size → last page
        else:
            break  # non-dict result, single page

        page += 1
        offset += len(records)

    return all_records


async def fetch_all_matching(
    tool_name: str,
    base_params: dict,
    on_tool_call: Callable[[str, dict], Awaitable[dict]],
    filter_fn: Callable[[dict], bool] | None = None,
    max_pages: int = MAX_PAGES,
) -> list:
    """
    Paginated fetch with optional filter function applied per record.
    Useful for: "get all invoices overdue > 90 days" from a mixed result set.
    """
    all_records = await paginated_fetch(tool_name, base_params, on_tool_call, max_pages)
    if filter_fn:
        return [r for r in all_records if filter_fn(r)]
    return all_records


def group_by(records: list[dict], key: str) -> dict[str, list[dict]]:
    """Group a list of records by a field value. Useful for AR aging buckets."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        k = str(r.get(key, "unknown"))
        groups.setdefault(k, []).append(r)
    return groups


def sum_field(records: list[dict], field: str) -> float:
    """Sum a numeric field across records. Safe — ignores non-numeric."""
    total = 0.0
    for r in records:
        try:
            total += float(r.get(field, 0) or 0)
        except (TypeError, ValueError):
            pass
    return round(total, 2)


def deduplicate(records: list[dict], key: str) -> list[dict]:
    """Remove duplicate records by a key field (e.g., invoice_id). Keeps first occurrence."""
    seen: set = set()
    result = []
    for r in records:
        v = r.get(key)
        if v is not None and v not in seen:
            seen.add(v)
            result.append(r)
    return result


def _extract_records(result, result_key: str | None) -> list:
    """Extract record list from various tool response shapes."""
    if isinstance(result, list):
        return result
    if not isinstance(result, dict):
        return []
    if result_key and result_key in result:
        v = result[result_key]
        return v if isinstance(v, list) else []
    for key in _RESULT_KEYS:
        if key in result and isinstance(result[key], list):
            return result[key]
    # Last resort: find any list value
    for v in result.values():
        if isinstance(v, list) and v:
            return v
    return []
