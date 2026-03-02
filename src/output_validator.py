"""
output_validator.py
Per-process expected output schema validation.
Checks the agent's answer contains required fields before returning.

Each process type has a set of required output signals.
If any are missing, the validator flags them so self_reflection.py
can request a targeted improvement.

Zero API cost — pure string/regex matching.
"""
from __future__ import annotations
import re

# ── Per-process expected output fields ───────────────────────────────────────
# Format: list of (field_name, patterns_that_match_it)

REQUIRED_OUTPUTS: dict[str, list[tuple[str, list[str]]]] = {
    "expense_approval": [
        ("decision",      ["approved", "rejected", "denied", "approval"]),
        ("amount",        [r'\$[\d,]+', r'\d+\.\d{2}', "amount"]),
        ("requester",     ["requester", "employee", "submitted by", "claimed by"]),
        ("reason",        ["reason", "because", "justification", "policy"]),
    ],
    "invoice_reconciliation": [
        ("decision",      ["approved", "rejected", "matched", "3-way match", "reconciled"]),
        ("amount",        [r'\$[\d,]+', "amount", "total"]),
        ("vendor",        ["vendor", "supplier", "from"]),
        ("variance",      ["variance", "difference", "discrepancy", "match"]),
    ],
    "procurement": [
        ("po_number",     ["po", "purchase order", "po#", "order number"]),
        ("vendor",        ["vendor", "supplier"]),
        ("amount",        [r'\$[\d,]+', "amount", "total"]),
        ("approval",      ["approved", "approval required", "pending approval"]),
    ],
    "hr_offboarding": [
        ("access",        ["access revoked", "deactivated", "suspended", "removed"]),
        ("systems",       ["github", "slack", "jira", "okta", "aws", "systems", "accounts"]),
        ("equipment",     ["equipment", "laptop", "hardware", "return"]),
    ],
    "payroll": [
        ("gross",         ["gross", "gross pay", "total gross"]),
        ("net",           ["net", "net pay", "take-home"]),
        ("deductions",    ["deductions", "tax", "withholding"]),
        ("headcount",     ["employees", "headcount", "staff"]),
    ],
    "sla_breach": [
        ("credit",        [r'\$[\d,]+', "credit", "sla credit", "compensation"]),
        ("breach",        ["breach", "violation", "downtime", "duration"]),
        ("customer",      ["customer", "client", "account"]),
    ],
    "month_end_close": [
        ("period",        ["period", "month", "quarter", "closed", "locked"]),
        ("balance",       ["balance", "trial balance", "p&l", "net"]),
        ("approval",      ["approved", "signed off", "cfo", "controller"]),
    ],
    "ar_collections": [
        ("amount",        [r'\$[\d,]+', "overdue", "outstanding", "balance"]),
        ("action",        ["email sent", "called", "notice", "payment plan", "collection"]),
        ("aging",         ["30", "60", "90", "days", "aging"]),
    ],
    "compliance_audit": [
        ("findings",      ["finding", "findings", "issue", "non-compliance", "control"]),
        ("score",         ["score", "compliant", "pass", "fail", "rating"]),
        ("actions",       ["remediation", "action", "deadline", "owner"]),
    ],
    "dispute_resolution": [
        ("decision",      ["approved", "rejected", "partial", "resolved", "credit"]),
        ("amount",        [r'\$[\d,]+', "credit amount", "refund"]),
        ("reason",        ["reason", "because", "evidence", "determination"]),
    ],
    "subscription_migration": [
        ("plan",          ["plan", "tier", "subscription"]),
        ("billing",       [r'\$[\d,]+', "charge", "refund", "credit", "billing"]),
        ("effective",     ["effective", "date", "migration date", "starting"]),
    ],
    "order_management": [
        ("order_id",      ["order", "order#", "order number", "confirmation"]),
        ("total",         [r'\$[\d,]+', "total", "amount charged"]),
        ("fulfillment",   ["ship", "deliver", "fulfillment", "estimated"]),
    ],
    "incident_response": [
        ("severity",      ["p1", "p2", "p3", "sev", "severity", "critical"]),
        ("resolution",    ["resolved", "mitigated", "fixed", "closed", "restored"]),
        ("impact",        ["affected", "customers", "impact", "duration"]),
    ],
    "customer_onboarding": [
        ("account_id",    ["account", "id", "provisioned", "created"]),
        ("csm",           ["csm", "success manager", "assigned", "owner"]),
        ("next_step",     ["next", "kickoff", "milestone", "schedule"]),
    ],
}

# Fields always expected regardless of process type
UNIVERSAL_REQUIRED = [
    ("summary",   ["summary", "completed", "result", "outcome", "in summary", "to summarize"]),
]

# Synthesized / unknown process type: these fields indicate a complete business answer.
# When a process_type is not in REQUIRED_OUTPUTS, we validate these instead of silently
# skipping validation (the original bug — unknown types got no validation at all).
#
# Fix (HIGH): Previously, REQUIRED_OUTPUTS.get(process_type, []) returned [] for any
# novel process type. Only UNIVERSAL_REQUIRED ("summary") was checked.
# Result: a malformed or empty answer could return valid=True, score=1.0 for novel types,
# meaning missing fields were never detected and self_reflection never triggered a retry.
#
# Fix: Unknown types get a set of cross-domain "business answer" fields that any
# complete business process outcome should contain. These are intentionally broad
# (not process-specific) so they add real signal without over-constraining novel types.
SYNTHESIZED_REQUIRED: list[tuple[str, list[str]]] = [
    ("decision_or_status", [
        "approved", "rejected", "completed", "resolved", "processed",
        "escalated", "pending", "failed", "success", "done",
    ]),
    ("action_taken", [
        "action", "executed", "performed", "updated", "created", "sent",
        "notified", "scheduled", "applied", "recorded",
    ]),
    ("key_entity", [
        r'\b[A-Z]{2,}-\d+\b',   # ticket/order IDs like ORD-123, TKT-456
        r'\$[\d,]+',              # any dollar amount
        r'\b\d{4}-\d{2}-\d{2}\b', # any date
        "invoice", "order", "request", "ticket", "account", "customer", "vendor",
    ]),
]


def validate_output(answer: str, process_type: str) -> dict:
    """
    Check if the answer contains required output fields for this process type.
    Returns {valid: bool, missing: list[str], present: list[str], score: float}
    Zero API cost — pure string matching.

    Fix: Novel/synthesized process types now get SYNTHESIZED_REQUIRED validation
    instead of an empty required list. This ensures self_reflection detects
    incomplete answers for unknown process types.
    """
    # Bracket-format answers are exact_match targets — field validation doesn't apply.
    # Running improvement passes on them would corrupt the bracket format.
    if answer.strip().startswith('['):
        return {"valid": True, "coverage": 1.0, "present": [], "missing": [], "score": 1.0}

    answer_lower = answer.lower()

    process_specific = REQUIRED_OUTPUTS.get(process_type)
    if process_specific is not None:
        # Known process type — use its specific required fields
        required = process_specific + UNIVERSAL_REQUIRED
    else:
        # Unknown / synthesized process type — use cross-domain business answer fields.
        # This replaces the original [] fallback that caused silent under-validation.
        required = SYNTHESIZED_REQUIRED + UNIVERSAL_REQUIRED

    present, missing = [], []

    for field_name, patterns in required:
        found = False
        for pattern in patterns:
            # Regex patterns start with r'\' or contain special chars
            try:
                if re.search(pattern, answer_lower):
                    found = True
                    break
            except re.error:
                if pattern.lower() in answer_lower:
                    found = True
                    break
        if found:
            present.append(field_name)
        else:
            missing.append(field_name)

    total = len(required)
    coverage = len(present) / total if total > 0 else 1.0

    return {
        "valid": len(missing) == 0,
        "coverage": round(coverage, 2),
        "present": present,
        "missing": missing,
        "score": round(coverage, 2),
    }


def get_missing_fields_prompt(missing: list[str], process_type: str) -> str:
    """Format missing fields as an improvement prompt."""
    if not missing:
        return ""
    label = process_type.replace("_", " ").title()
    fields = ", ".join(missing)
    return f"Your {label} answer is missing required fields: {fields}. Add them now."
