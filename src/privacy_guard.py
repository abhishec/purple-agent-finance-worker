"""
privacy_guard.py
Privacy refusal — fires at DECOMPOSE state before any tool calls.
Inspired by BrainOS policy-checker.ts privacy rules.

Fast path: detect → ESCALATE immediately, no DB queries, no LLM cost.
Correct: never leaks private/confidential data through the agent.
"""
from __future__ import annotations

# Explicit task types → immediate refusal
PRIVATE_TASK_TYPES = {
    "private_customer_information",
    "confidential_company_knowledge",
    "internal_operation_data",
    "sensitive_personal_data",
    "restricted_financial_data",
}

# Keywords that trigger privacy refusal
PRIVACY_KEYWORDS = [
    "password", "passwd", "credentials",
    "api key", "api_key", "secret key", "private key",
    "ssn", "social security", "national id", "tax id",
    "credit card", "card number", "cvv", "bank account", "routing number",
    "medical record", "health record", "phi", "hipaa",
    "personal health", "diagnosis", "prescription",
    "private information", "internal only",
    "not for distribution",
    "home address", "personal address",
    "date of birth", " dob ",
]

# Safe context — prevents false positives
SAFE_CONTEXTS = [
    "anonymized", "aggregated", "redacted", "masked",
    "hashed", "encrypted at rest", "tokenized",
    "sample data", "test data", "dummy data", "mock",
    # Business process contexts — never refuse these
    "process payroll", "run payroll", "payroll for",
    "compliance audit", "compliance report",
    "salary adjustment", "salary range", "compensation review",
    "internal controls", "restricted-access controls",
]


def check_privacy(task_text: str, task_type: str | None = None) -> dict | None:
    """
    Returns refusal dict if task is requesting private data, None if safe.
    Runs before any tool calls — zero DB cost on refusal path.
    """
    # Explicit task type match
    if task_type and task_type.lower() in PRIVATE_TASK_TYPES:
        return _refusal(task_type, "task_type_match")

    text = task_text.lower()

    # Safe context check — prevent false positives
    if any(safe in text for safe in SAFE_CONTEXTS):
        return None

    # Keyword match
    triggered = [kw for kw in PRIVACY_KEYWORDS if kw in text]
    if triggered:
        return _refusal(", ".join(triggered[:3]), "keyword_match")

    return None


def _refusal(trigger: str, method: str) -> dict:
    return {
        "refused": True,
        "trigger": trigger,
        "method": method,
        "escalation_level": "ciso",
        "message": (
            "I cannot provide this information as it contains confidential "
            "or private data. This request has been flagged and escalated "
            "per policy requirements."
        ),
    }
