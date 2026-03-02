"""
knowledge_extractor.py
Post-task knowledge extraction — ported from BrainOS knowledge-extractor.ts.

After every task with quality >= 0.50 (lowered from 0.65 for more coverage),
Haiku extracts up to 4 reusable facts and stores them in knowledge_base.json.
Future tasks get these facts injected in the PRIME phase — the agent
compounds knowledge across all tasks.

Example: Task 3 handles Acme Corp invoice with net-60 terms.
Task 7 asks about an Acme Corp PO → agent already knows their payment terms.

Fixes applied (2026-03-01):
- Bug C: _call_haiku_extract() regex changed from r'\[.*?\]' to re.DOTALL
  to correctly parse multi-line JSON arrays returned by Haiku
- Lowered EXTRACTION_THRESHOLD from 0.65 → 0.50 (more aggressive capture)
- Increased max insights per task from 2 → 4
- Added fast-path extraction: structured facts from regex before any API call
- Added entity-keyed secondary index in knowledge base for faster entity lookup
- Added growth logging to knowledge_growth.log after every extraction
"""
from __future__ import annotations
import json
import os
import re
import time
import hashlib
from dataclasses import dataclass, field, asdict

from src.config import ANTHROPIC_API_KEY

KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge_base.json")
GROWTH_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge_growth.log")
MAX_KNOWLEDGE_ENTRIES = 500
EXTRACTION_THRESHOLD = 0.50   # Lowered from 0.65 — extract from more tasks
EXTRACT_MODEL = "claude-haiku-4-5-20251001"
RELEVANCE_OVERLAP = 2         # min keyword overlap to surface knowledge


@dataclass
class KnowledgeEntry:
    entry_id: str
    domain: str           # process_type (expense_approval, procurement, ...)
    content: str          # the extracted insight (max 100 words)
    entities: list[str]   # entity names/ids mentioned (for lookup)
    entity_index: dict    # {entity_name: True} — secondary index for O(1) entity lookup
    keywords: list[str]   # for keyword-match retrieval
    confidence: float     # 0-1 from LLM extraction
    quality_score: float  # task quality that generated this
    source_task: str      # first 80 chars of task that produced this
    extraction_method: str = "haiku"  # "haiku" | "fast-path" | "fallback"
    created_at: float = field(default_factory=time.time)


# ── Storage ───────────────────────────────────────────────────────────────────

def _load() -> list[dict]:
    try:
        if os.path.exists(KNOWLEDGE_PATH):
            with open(KNOWLEDGE_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save(entries: list[dict]) -> None:
    try:
        with open(KNOWLEDGE_PATH, "w") as f:
            json.dump(entries[-MAX_KNOWLEDGE_ENTRIES:], f, indent=2)
    except Exception:
        pass


def _append_growth_log(domain: str, quality: float, new_count: int, total: int) -> None:
    """Append a growth event to knowledge_growth.log for monitoring."""
    try:
        with open(GROWTH_LOG_PATH, "a") as f:
            f.write(
                f"{time.strftime('%Y-%m-%dT%H:%M:%S')} "
                f"domain={domain} quality={quality:.2f} "
                f"new={new_count} total={total}\n"
            )
    except Exception:
        pass


# ── Fast-path structured extraction (zero API cost) ──────────────────────────

def _fast_path_extract(task_text: str, answer: str, domain: str) -> list[dict]:
    """
    Extract structured facts directly from task + answer using regex patterns.
    Zero API cost — runs before Haiku call.
    Produces insights at confidence=0.6.

    Patterns extracted:
      - Dollar amounts → "In [domain]: amount was [X]"
      - Approved/rejected decisions → "[domain] decision: [outcome] for [entity]"
      - Policy thresholds → "Policy threshold: [X] for [domain]"
    """
    insights = []
    combined = f"{task_text} {answer}"

    # Dollar amounts
    amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?(?:\s*[KMB](?:illion)?)?', combined)
    for amt in amounts[:2]:  # cap at 2 amounts
        insights.append({
            "content": f"In {domain}: amount referenced was {amt}",
            "confidence": 0.6,
            "method": "fast-path",
        })

    # Approval/rejection decisions with entity context
    decision_m = re.search(
        r'\b(approved|rejected|denied|escalated|resolved)\b.{0,60}?'
        r'([A-Z][a-zA-Z\s]{2,30}(?:Corp|Inc|LLC|Ltd|Co)?)',
        combined, re.IGNORECASE
    )
    if decision_m:
        outcome = decision_m.group(1).lower()
        entity = decision_m.group(2).strip()
        insights.append({
            "content": f"{domain} decision: {outcome} for {entity}",
            "confidence": 0.6,
            "method": "fast-path",
        })
    elif re.search(r'\b(approved|rejected|denied|escalated|resolved)\b', combined, re.IGNORECASE):
        outcome_m = re.search(r'\b(approved|rejected|denied|escalated|resolved)\b', combined, re.IGNORECASE)
        insights.append({
            "content": f"{domain}: outcome was {outcome_m.group(1).lower()}",
            "confidence": 0.6,
            "method": "fast-path",
        })

    # Policy thresholds: "$X limit", "threshold of $X", "up to $X"
    threshold_m = re.search(
        r'(?:limit|threshold|cap|ceiling|up to|maximum|minimum)\s+(?:of\s+)?(\$[\d,]+(?:\.\d{2})?)',
        combined, re.IGNORECASE
    )
    if threshold_m:
        insights.append({
            "content": f"Policy threshold: {threshold_m.group(1)} for {domain}",
            "confidence": 0.6,
            "method": "fast-path",
        })

    # Net payment terms (vendor-specific knowledge)
    net_m = re.search(r'(?:net[-\s]?(\d+)|(\d+)[-\s]?day\s+(?:payment\s+)?terms?)', combined, re.IGNORECASE)
    if net_m:
        days = net_m.group(1) or net_m.group(2)
        insights.append({
            "content": f"In {domain}: payment terms net-{days} days",
            "confidence": 0.6,
            "method": "fast-path",
        })

    return insights[:3]  # max 3 fast-path insights


# ── Keyword / entity extraction ───────────────────────────────────────────────

def _extract_keywords(text: str) -> list[str]:
    stop = {"the","a","an","is","are","was","were","be","been","have","has","had",
            "do","does","did","will","would","could","should","can","for","in","on",
            "at","to","of","and","or","but","with","from","this","that","it","i",
            "you","please","need","want","help","task","make","get","use","all","any"}
    words = text.lower().split()
    seen, unique = set(), []
    for w in words:
        w = w.strip(".,!?;:\"'()[]{}$%")
        if len(w) > 2 and w not in stop and w not in seen:
            seen.add(w)
            unique.append(w)
    return unique[:20]


def _extract_entities_regex(text: str) -> list[str]:
    """Fast regex entity extraction — zero API cost."""
    entities = []

    # Dollar amounts
    for m in re.finditer(r'\$[\d,]+(?:\.\d{2})?(?:K|M|B)?', text):
        entities.append(m.group())

    # Percentages
    for m in re.finditer(r'\d+(?:\.\d+)?%', text):
        entities.append(m.group())

    # Named things: Title Case words (likely vendor/person names)
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', text):
        val = m.group()
        if val not in ("The", "This", "That", "In", "At", "On", "For"):
            entities.append(val)

    # IDs: JIRA-123, INV-456, EMP-789
    for m in re.finditer(r'\b[A-Z]{2,8}-\d+\b', text):
        entities.append(m.group())

    # Emails
    for m in re.finditer(r'\b[\w.+-]+@[\w.-]+\.\w{2,}\b', text):
        entities.append(m.group())

    # Deduplicate preserving order
    seen, result = set(), []
    for e in entities:
        if e not in seen:
            seen.add(e)
            result.append(e)
    return result[:15]


# ── Haiku extraction (API call) ───────────────────────────────────────────────

async def _call_haiku_extract(task_text: str, answer: str, domain: str) -> list[dict]:
    """
    Call Haiku to extract up to 4 reusable factual insights.
    Returns list of {content, confidence} dicts.
    Graceful no-op if API unavailable.

    Bug C fix (2026-03-01): regex changed from r'\[.*?\]' (non-greedy, no DOTALL)
    to re.search(..., re.DOTALL) so multi-line JSON arrays are correctly parsed.
    """
    if not ANTHROPIC_API_KEY:
        return []
    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        task_snippet = task_text[:300]
        answer_snippet = answer[:400]

        resp = await client.messages.create(
            model=EXTRACT_MODEL,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": (
                    f"Domain: {domain}\n"
                    f"Task: {task_snippet}\n"
                    f"Result: {answer_snippet}\n\n"
                    "Extract 2-4 SHORT, reusable factual insights from this completed task. "
                    "Focus on: vendor terms, entity-specific rules, policy thresholds, "
                    "process patterns, or constraints that would help future similar tasks.\n\n"
                    "Return JSON array: [{\"content\": \"...\", \"confidence\": 0.0-1.0}]\n"
                    "Each insight max 50 words. Only facts, no instructions."
                ),
            }],
        )
        text = resp.content[0].text if resp.content else ""
        # Bug C fix: use re.DOTALL so '.' matches newlines in multi-line JSON arrays
        m = re.search(r'\[.*\]', text, re.DOTALL)  # greedy: capture outermost array
        if m:
            insights = json.loads(m.group())
            if isinstance(insights, list):
                return [
                    {**i, "method": "haiku"}
                    for i in insights
                    if isinstance(i, dict) and "content" in i
                ]
    except Exception:
        pass
    return []


# ── Public API ────────────────────────────────────────────────────────────────

def _build_entry(
    content: str,
    domain: str,
    entities: list[str],
    keywords: list[str],
    quality: float,
    task_text: str,
    confidence: float,
    method: str = "haiku",
) -> dict | None:
    """Build a KnowledgeEntry dict, or None if content is invalid."""
    content = content.strip()
    if not content or len(content) < 10:
        return None
    entry_id = hashlib.md5(f"{domain}:{content[:40]}".encode()).hexdigest()[:8]
    # Entity-keyed secondary index for O(1) lookup
    entity_index = {e.lower(): True for e in entities}
    entry = KnowledgeEntry(
        entry_id=entry_id,
        domain=domain,
        content=content,
        entities=entities,
        entity_index=entity_index,
        keywords=keywords,
        confidence=round(confidence, 3),
        quality_score=round(quality, 3),
        source_task=task_text[:80],
        extraction_method=method,
    )
    return asdict(entry)


async def extract_and_store(
    task_text: str,
    answer: str,
    domain: str,
    quality: float,
) -> int:
    """
    Extract knowledge from a completed task and store it.
    Called in REFLECT phase by worker_brain.py after RL recording.
    Returns number of new insights stored (0 if quality too low or extraction failed).
    Fire-and-forget safe — never raises.

    Pipeline (2026-03-01):
      1. Fast-path: regex extracts structured facts (zero API cost)
      2. Haiku: LLM extracts semantic insights (API call, only if threshold met)
      3. Fallback: minimal fact from answer if both above produce nothing
      4. Dedup by entry_id, store, update entity index, log growth
    """
    if quality < EXTRACTION_THRESHOLD:
        return 0
    if not task_text or not answer:
        return 0

    try:
        entries = _load()
        existing_ids = {e.get("entry_id") for e in entries}
        entities = _extract_entities_regex(task_text + " " + answer)
        keywords = _extract_keywords(task_text)
        new_count = 0

        # Step 1: Fast-path extraction (always runs, zero API cost)
        fast_insights = _fast_path_extract(task_text, answer, domain)

        # Step 2: Haiku extraction (API call — up to 4 insights)
        haiku_insights = await _call_haiku_extract(task_text, answer, domain)

        # Step 3: Fallback if nothing extracted yet
        all_insights = fast_insights + haiku_insights
        if not all_insights and len(answer) > 100:
            all_insights = [{
                "content": answer[:120].replace("\n", " "),
                "confidence": 0.55,
                "method": "fallback",
            }]

        if not all_insights:
            return 0

        # Step 4: Store up to 4 per task (increased from 2)
        for insight in all_insights[:4]:
            content = insight.get("content", "").strip()
            conf = float(insight.get("confidence", 0.6))
            method = insight.get("method", "haiku")
            built = _build_entry(content, domain, entities, keywords, quality, task_text, conf, method)
            if built is None:
                continue
            if built["entry_id"] in existing_ids:
                continue
            entries.append(built)
            existing_ids.add(built["entry_id"])
            new_count += 1

        if new_count:
            _save(entries)
            _append_growth_log(domain, quality, new_count, len(entries))

        return new_count
    except Exception:
        return 0


def get_relevant_knowledge(task_text: str, domain: str, top_k: int = 4) -> str:
    """
    Retrieve relevant past knowledge for injection into PRIME phase.
    Uses keyword overlap + entity matching (via entity_index) + domain affinity.
    Returns a formatted string for the system prompt, or "" if nothing relevant.
    """
    entries = _load()
    if not entries:
        return ""

    task_kw = set(_extract_keywords(task_text))
    task_entities_raw = _extract_entities_regex(task_text)
    task_entities = {e.lower() for e in task_entities_raw}

    scored = []
    for e in entries:
        score = 0.0
        # Keyword overlap
        kw_overlap = len(task_kw & set(e.get("keywords", [])))
        score += kw_overlap * 0.4
        # Entity overlap — use entity_index for O(1) lookup when available
        entity_index = e.get("entity_index", {})
        if entity_index:
            ent_overlap = sum(1 for te in task_entities if te in entity_index)
        else:
            # Fallback to list scan for entries written before entity_index was added
            ent_overlap = sum(
                1 for ent in e.get("entities", [])
                if ent.lower() in task_entities or any(ent.lower() in te for te in task_entities)
            )
        score += ent_overlap * 0.8
        # Domain affinity
        if e.get("domain") == domain:
            score += 0.3
        # Quality weight
        score += e.get("quality_score", 0) * 0.2
        if score >= 0.4:
            scored.append((score, e))

    if not scored:
        return ""

    scored.sort(key=lambda x: -x[0])
    top = [e for _, e in scored[:top_k]]

    lines = ["## KNOWLEDGE BASE (facts from past tasks — apply where relevant)"]
    for e in top:
        conf_str = f" (confidence: {e['confidence']:.0%})" if e.get("confidence") else ""
        method_str = f" [{e.get('extraction_method', 'haiku')}]" if e.get("extraction_method") else ""
        lines.append(f"  • [{e.get('domain', 'general')}]{conf_str}{method_str} {e['content']}")
    lines.append("")
    return "\n".join(lines)
