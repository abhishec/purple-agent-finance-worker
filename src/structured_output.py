"""
structured_output.py
Structured output + exact_match answer format discipline.
Inspired by BrainOS memory-stack structured-output.ts.

Differentiator #3: Return ["Answer1", "Answer2"] sorted lists — not prose.
12% of benchmark score depends on exact format compliance.
"""
from __future__ import annotations
import json
import re


# Keywords that indicate a list/ranking answer is expected
LIST_TASK_KEYWORDS = [
    "list", "rank", "order", "sort", "top", "best", "worst",
    "prioritize", "enumerate", "which", "candidates", "options",
    "recommend", "select", "choose", "identify all",
]

# Keywords that indicate a yes/no or single-value answer
SCALAR_TASK_KEYWORDS = [
    "approve", "deny", "reject", "flag", "should i", "is this",
    "can i", "am i allowed", "does this", "is it", "was it",
]

# Quantity patterns that signal list output regardless of "list" keyword.
# Covers novel phrasings like "Give me the three highest-risk vendors" or
# "Which invoices exceed the approved amount".
_QUANTITY_LIST_PATTERNS = [
    re.compile(r'\b(top|best|worst|highest|lowest|first|last)\s+\d+\b', re.I),
    re.compile(
        r'\b\d+\s+(vendors?|invoices?|items?|orders?|employees?|records?|cases?|'
        r'tickets?|candidates?|options?|risks?|issues?)\b',
        re.I,
    ),
    re.compile(
        r'\b(all|every|each)\s+(vendors?|invoices?|items?|orders?|employees?|records?)\b',
        re.I,
    ),
    re.compile(r'\bwhich\s+\w+\s+(are|have|should|need|exceed|violate|qualify)\b', re.I),
    re.compile(r'\b(list|enumerate|identify all|show me all|find all|give me)\b', re.I),
]


def _is_list_task_dynamic(task_text: str) -> bool:
    """Dynamic list task detection — works for novel phrasings."""
    task_lower = task_text.lower()
    # Original keyword check (fast path)
    if any(kw in task_lower for kw in LIST_TASK_KEYWORDS):
        return True
    # Quantity signal — novel phrasing
    for pattern in _QUANTITY_LIST_PATTERNS:
        if pattern.search(task_text):
            return True
    return False


def is_list_task(task_text: str) -> bool:
    return _is_list_task_dynamic(task_text)


def is_scalar_task(task_text: str) -> bool:
    text = task_text.lower()
    return any(kw in text for kw in SCALAR_TASK_KEYWORDS)


def extract_ranked_items(text: str) -> list[str]:
    """
    Extract a list of items from LLM response text.
    Tries: JSON array → numbered list → bullet list → comma-separated.
    """
    # 1. JSON array (most reliable)
    json_match = re.search(r'\[([^\[\]]+)\]', text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(f"[{json_match.group(1)}]")
            if isinstance(parsed, list) and all(isinstance(i, (str, int, float)) for i in parsed):
                return [str(i).strip() for i in parsed if str(i).strip()]
        except json.JSONDecodeError:
            pass

    # 2. Numbered list: "1. Item" or "1) Item"
    numbered = re.findall(r'^\s*\d+[.)]\s+(.+)$', text, re.MULTILINE)
    if len(numbered) >= 2:
        return [i.strip() for i in numbered]

    # 3. Bullet list: "- Item" "* Item" "• Item"
    bulleted = re.findall(r'^\s*[-*•]\s+(.+)$', text, re.MULTILINE)
    if len(bulleted) >= 2:
        return [i.strip() for i in bulleted]

    # 4. Comma-separated on one line — STRICT guards to avoid corrupting financial amounts.
    # Only trigger when items are all short word-like strings with no digits/currency.
    _NUMERIC_PATTERN = re.compile(r'[\d$€£¥]')
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    for line in lines[:3]:  # check first 3 lines
        if ',' in line and len(line) < 200:
            items = [i.strip() for i in line.split(',') if i.strip()]
            if len(items) >= 2:
                # Reject if any item contains a number or currency symbol (financial data)
                if any(_NUMERIC_PATTERN.search(item) for item in items):
                    continue
                # Reject if any item is too long (full sentences, not list items)
                if any(len(item) > 50 for item in items):
                    continue
                return items

    return []



def _answer_is_json_array(answer: str) -> bool:
    """True if answer IS a JSON array — the strongest signal for bracket format."""
    s = answer.strip()
    if not (s.startswith('[') and s.endswith(']')):
        return False
    try:
        import json as _json
        return isinstance(_json.loads(s), list)
    except (ValueError, _json.JSONDecodeError):
        return False


def enforce_bracket_format(items: list[str]) -> str:
    """
    Format items as a clean JSON array string — exact_match benchmark format.
    Sorts alphabetically unless items appear to be ranked (numbered).
    """
    if not items:
        return "[]"
    clean = [re.sub(r'^\d+[.)]\s*', '', i).strip() for i in items]
    clean = [i for i in clean if i]
    return json.dumps(clean, ensure_ascii=False)


def format_final_answer(answer: str, task_text: str = "", policy_result: dict | None = None) -> str:
    """
    Post-process the final answer:
    1. Post-hoc bracket detection — if the answer IS a valid JSON array, return immediately.
       The answer itself is ground truth; overrides task-level classification.
    2. Prepend policy outcome if failed (prose answers only — never corrupt bracket format)
    3. Try to detect and enforce bracket format for list answers
    4. Post-hoc correction: if the answer contains a parseable JSON array but task was
       not originally classified as a list task, still enforce bracket format — the
       answer itself is ground truth.
    """
    answer_stripped = answer.strip()

    # Post-hoc bracket detection — answer itself is ground truth.
    # Overrides task-level classification if the agent returned a valid JSON array.
    # Checked BEFORE any other formatting logic — no metadata, no prose wrapping.
    if _answer_is_json_array(answer_stripped):
        return answer_stripped

    # Bracket-format answers are exact_match targets.
    # NEVER add a policy prefix — it would break string comparison.
    # Also skip re-extraction to preserve the original ordering.
    if answer_stripped.startswith('['):
        return answer_stripped

    parts = []

    if policy_result and not policy_result.get("passed"):
        parts.append(f"[POLICY: {policy_result.get('summary', 'Policy check failed')}]")

    # Try to extract and reformat as bracket list
    items = extract_ranked_items(answer_stripped)
    if items:
        bracket = enforce_bracket_format(items)
        parts.append(bracket)
    elif task_text and _is_list_task_dynamic(task_text):
        # Task was classified as a list task but LLM returned prose — still pass through
        # so the caller can decide to re-prompt.  We don't corrupt a valid prose answer.
        parts.append(answer_stripped)
    else:
        parts.append(answer_stripped)

    return "\n".join(parts)


def build_policy_section(policy_result: dict) -> str:
    """Format deterministic policy result for system prompt injection."""
    if not policy_result:
        return ""

    status = "PASSED" if policy_result.get("passed") else "FAILED"
    lines = [
        "",
        "## POLICY ENFORCEMENT RESULT (deterministic — do not override)",
        f"Status: {status}",
        f"Summary: {policy_result.get('summary', '')}",
    ]
    triggered = policy_result.get("triggeredRules", [])
    if triggered:
        lines.append(f"Triggered: {', '.join(r['ruleId'] for r in triggered)}")
    if policy_result.get("escalationLevel"):
        lines.append(f"Escalation: {policy_result['escalationLevel']}")
    lines.append("Your response MUST reflect this outcome exactly.")
    lines.append("")
    return "\n".join(lines)
