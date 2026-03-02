"""
entity_extractor.py
Cross-task entity memory — ported from BrainOS entity-extraction.ts.

Extracts named entities from every task (vendors, people, amounts, IDs,
dates) and persists them to entity_memory.json. Before each task, relevant
past entities are injected — so the agent already knows about Acme Corp
from task 3 when it appears in task 7.

Zero API cost — pure regex extraction (fast path only).
"""
from __future__ import annotations
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict

ENTITY_PATH = os.path.join(os.path.dirname(__file__), "..", "entity_memory.json")
MAX_ENTITIES = 1000
ENTITY_TTL = 86400 * 7   # 7 days — entities expire


@dataclass
class EntityRecord:
    entity_id: str
    entity_type: str   # vendor | person | amount | id | date | email | percentage | product
    raw_value: str     # as it appeared in the task
    normalized: str    # cleaned/standardized form
    context: str       # snippet of text where it appeared
    domain: str        # process type where first seen
    seen_count: int = 1
    last_seen: float = field(default_factory=time.time)
    first_seen: float = field(default_factory=time.time)


# ── Extraction patterns ───────────────────────────────────────────────────────

_PATTERNS: list[tuple[str, str, str]] = [
    # (type, pattern, group_to_normalize)
    ("amount",      r'\$[\d,]+(?:\.\d{1,2})?(?:\s*[KMB](?:illion)?)?', "dollar"),
    ("percentage",  r'\b\d+(?:\.\d+)?%', "pct"),
    ("id",          r'\b[A-Z]{2,8}-\d+\b', "ref_id"),          # JIRA-123, INV-456, EMP-789
    ("email",       r'\b[\w.+-]+@[\w.-]+\.\w{2,}\b', "email"),
    ("date",        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,?\s+\d{4})?\b', "date"),
    ("date",        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', "date_num"),
    ("vendor",      r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:Corp|Inc|LLC|Ltd|Co|Group|Holdings|Technologies|Services|Solutions|Systems|Consulting|Partners)\.?)\b', "company"),
    ("person",      r'\b((?:Mr|Ms|Mrs|Dr)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', "person_title"),
    ("product",     r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?\s+(?:Plan|Tier|Package|License|Subscription|Module|Suite))\b', "product"),
]

# Proper nouns (Title Case 2+ words) — catch vendors/people not matching above
_TITLE_CASE = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b')

_STOP_TITLES = {
    "The Task", "This Process", "Last Day", "First Name", "Last Name",
    "New Customer", "End Date", "Start Date", "Due Date", "Net Terms",
}



def _infer_entity_type_from_context(entity_text: str, surrounding_text: str) -> str:
    """Infer entity type from how the entity is used in context — no API call."""
    context_lower = surrounding_text.lower()

    # Person signals: appears after person-role verbs or in from:/to:/by: positions
    _PERSON_CONTEXT = [
        "submitted by", "approved by", "assigned to", "managed by", "reported by",
        "reviewed by", "from:", "to:", "cc:", "contact:", "employee:", "manager:",
        "requested by", "authorized by", "emailed by", "signed by",
    ]
    if any(signal in context_lower for signal in _PERSON_CONTEXT):
        return "person"

    # Two capitalized words (First Last) without company suffixes = likely person
    if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', entity_text):
        # Check it's not preceded by company context
        _COMPANY_CONTEXT = ["vendor", "supplier", "client", "company", "corp", "inc", "ltd"]
        if not any(signal in context_lower for signal in _COMPANY_CONTEXT):
            return "person"

    # Location signals
    _LOCATION_SIGNALS = ["office in", "based in", "located in", "headquarters", "branch at", "region:"]
    if any(signal in context_lower for signal in _LOCATION_SIGNALS):
        return "location"

    # Product/system signals
    _PRODUCT_CONTEXT = ["system", "platform", "software", "tool", "application", "license", "subscription"]
    if any(signal in context_lower for signal in _PRODUCT_CONTEXT):
        return "product"

    # Default: vendor (unchanged for ambiguous cases)
    return "vendor"


def extract_entities(text: str, domain: str = "general") -> list[EntityRecord]:
    """Extract all entities from text. Returns list of EntityRecord objects."""
    import hashlib
    found: dict[str, EntityRecord] = {}

    for etype, pattern, _ in _PATTERNS:
        for m in re.finditer(pattern, text):
            raw = m.group().strip()
            if len(raw) < 2:
                continue
            norm = _normalize(etype, raw)
            eid = hashlib.md5(f"{etype}:{norm}".encode()).hexdigest()[:10]
            ctx_start = max(0, m.start() - 30)
            ctx_end = min(len(text), m.end() + 30)
            if eid not in found:
                found[eid] = EntityRecord(
                    entity_id=eid,
                    entity_type=etype,
                    raw_value=raw,
                    normalized=norm,
                    context=text[ctx_start:ctx_end].replace("\n", " "),
                    domain=domain,
                )

    # Title case catch-all for unmatched proper nouns
    for m in _TITLE_CASE.finditer(text):
        raw = m.group().strip()
        if raw in _STOP_TITLES or len(raw) < 5:
            continue
        # Skip if already captured
        if any(raw in r.raw_value for r in found.values()):
            continue
        import hashlib
        eid = hashlib.md5(f"noun:{raw}".encode()).hexdigest()[:10]
        if eid not in found:
            ctx_start = max(0, m.start() - 25)
            ctx_end = min(len(text), m.end() + 25)
            surrounding = text[ctx_start:ctx_end].replace("\n", " ")
            entity_type = _infer_entity_type_from_context(raw, surrounding)
            found[eid] = EntityRecord(
                entity_id=eid,
                entity_type=entity_type,
                raw_value=raw,
                normalized=raw,
                context=surrounding,
                domain=domain,
            )

    return list(found.values())


def _normalize(etype: str, raw: str) -> str:
    """Normalize entity to canonical form."""
    if etype == "amount":
        s = raw.replace("$", "").replace(",", "").strip()
        if s.upper().endswith("K"):
            try:
                return f"${float(s[:-1]) * 1000:.0f}"
            except Exception:
                pass
        if s.upper().endswith("M"):
            try:
                return f"${float(s[:-1]) * 1_000_000:.0f}"
            except Exception:
                pass
        return f"${s}"
    if etype == "email":
        return raw.lower()
    return raw.strip()


# ── Storage ───────────────────────────────────────────────────────────────────

def _load() -> list[dict]:
    try:
        if os.path.exists(ENTITY_PATH):
            with open(ENTITY_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save(records: list[dict]) -> None:
    try:
        # Evict expired entries before saving
        now = time.time()
        active = [r for r in records if now - r.get("last_seen", 0) < ENTITY_TTL]
        with open(ENTITY_PATH, "w") as f:
            json.dump(active[-MAX_ENTITIES:], f, indent=2)
    except Exception:
        pass


# ── Public API ────────────────────────────────────────────────────────────────

def record_task_entities(task_text: str, answer: str, domain: str) -> int:
    """
    Extract and persist entities from task + answer.
    Called in REFLECT phase. Fire-and-forget safe.
    Returns count of new/updated entities.
    """
    try:
        combined = f"{task_text}\n{answer}"
        new_records = extract_entities(combined, domain)
        if not new_records:
            return 0

        existing = _load()
        by_id = {r["entity_id"]: r for r in existing}
        updated = 0

        for rec in new_records:
            eid = rec.entity_id
            if eid in by_id:
                by_id[eid]["seen_count"] = by_id[eid].get("seen_count", 1) + 1
                by_id[eid]["last_seen"] = time.time()
                by_id[eid]["domain"] = domain   # update to most recent domain
            else:
                by_id[eid] = asdict(rec)
                updated += 1

        _save(list(by_id.values()))
        return updated
    except Exception:
        return 0


def get_entity_context(task_text: str, top_k: int = 6) -> str:
    """
    Look up past entity memory relevant to this task.
    Returns formatted string for PRIME injection, or "" if nothing relevant.
    """
    try:
        records = _load()
        if not records:
            return ""

        # Extract entities from the incoming task
        task_entities = extract_entities(task_text)
        if not task_entities:
            return ""

        task_vals = {e.normalized.lower() for e in task_entities}
        task_raws = {e.raw_value.lower() for e in task_entities}

        # Find stored entities that match anything in the task
        matches: list[dict] = []
        for stored in records:
            s_norm = stored.get("normalized", "").lower()
            s_raw = stored.get("raw_value", "").lower()
            if s_norm in task_vals or s_raw in task_raws:
                # Only surface entities seen more than once (genuinely recurring)
                if stored.get("seen_count", 1) >= 2:
                    matches.append(stored)

        # Also surface high-frequency entities from same domain
        domain_freq = sorted(
            [r for r in records if r.get("seen_count", 1) >= 3],
            key=lambda r: -r.get("seen_count", 1),
        )[:3]
        for r in domain_freq:
            if r not in matches:
                matches.append(r)

        if not matches:
            return ""

        seen_ids, unique = set(), []
        for m in matches[:top_k]:
            if m["entity_id"] not in seen_ids:
                seen_ids.add(m["entity_id"])
                unique.append(m)

        lines = ["## ENTITY MEMORY (known entities from past tasks)"]
        for m in unique:
            etype = m.get("entity_type", "entity")
            val = m.get("normalized") or m.get("raw_value", "")
            ctx = m.get("context", "")[:80]
            seen = m.get("seen_count", 1)
            lines.append(f"  • [{etype}] {val}  (seen {seen}x — context: \"{ctx}\")")
        lines.append("")
        return "\n".join(lines)
    except Exception:
        return ""
