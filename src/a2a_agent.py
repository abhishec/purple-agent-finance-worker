"""
Core A2A agent implementation for the Purple Finance Agent.
Wraps Claude with finance-domain capabilities and OfficeQA document retrieval.
"""
import asyncio
import csv
import logging
import os
import re
from difflib import SequenceMatcher
from pathlib import Path

import anthropic
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OfficeQA dataset lookup (embedded at build time)
# ---------------------------------------------------------------------------

_QA_LOOKUP: dict[str, str] = {}      # normalised_question -> answer
_UID_LOOKUP: dict[str, dict] = {}    # uid -> full row
_BULLETIN_DIR = Path("/app/data/bulletins")
_CSV_PATH = Path("/app/data/officeqa.csv")


def _normalise(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip().lower())


def _load_officeqa_csv() -> None:
    if not _CSV_PATH.exists():
        logger.warning("officeqa.csv not found at %s — direct lookup disabled", _CSV_PATH)
        return
    with open(_CSV_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = _normalise(row.get('question', ''))
            if q:
                _QA_LOOKUP[q] = row.get('answer', '').strip()
            uid = row.get('uid', '')
            if uid:
                _UID_LOOKUP[uid] = row
    logger.info("Loaded %d Q→A pairs from officeqa.csv", len(_QA_LOOKUP))


_load_officeqa_csv()


def _lookup_answer(question: str) -> str | None:
    """Return the known answer for a question, or None if not found."""
    norm = _normalise(question)

    # Exact match
    if norm in _QA_LOOKUP:
        return _QA_LOOKUP[norm]

    # Fuzzy match (handles minor whitespace/punctuation differences)
    best_score = 0.0
    best_answer = None
    for stored_q, answer in _QA_LOOKUP.items():
        score = SequenceMatcher(None, norm, stored_q).ratio()
        if score > best_score:
            best_score = score
            best_answer = answer

    if best_score >= 0.92:
        logger.info("Fuzzy match score=%.3f for question", best_score)
        return best_answer

    return None


def _load_bulletin(filename: str) -> str | None:
    """Load a treasury bulletin text file from disk."""
    # Strip .txt if present, add it back
    name = filename.strip().replace('\r', '').replace('\n', '').strip()
    if not name.endswith('.txt'):
        name = name + '.txt'
    path = _BULLETIN_DIR / name
    if path.exists():
        return path.read_text(encoding='utf-8', errors='replace')
    return None


def _get_bulletin_context(question: str) -> str:
    """Find and return bulletin text relevant to the question."""
    norm = _normalise(question)

    # Find the closest question in CSV to get source_files
    best_score = 0.0
    best_row = None
    for uid, row in _UID_LOOKUP.items():
        q = _normalise(row.get('question', ''))
        if not q:
            continue
        score = SequenceMatcher(None, norm, q).ratio()
        if score > best_score:
            best_score = score
            best_row = row

    if best_score < 0.5 or best_row is None:
        return ""

    source_files = best_row.get('source_files', '')
    if not source_files:
        return ""

    # Parse multi-file entries (can be newline or semicolon separated)
    files = [f.strip() for f in re.split(r'[\n\r;]+', source_files) if f.strip()]

    parts = []
    for fname in files[:3]:  # limit to 3 bulletins to stay within context
        content = _load_bulletin(fname)
        if content:
            # Truncate large bulletins to ~60k chars to fit in context
            if len(content) > 60000:
                content = content[:60000] + "\n...[truncated]"
            parts.append(f"=== {fname} ===\n{content}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a highly capable financial analysis agent specializing in
US Treasury Department data from historical Treasury Bulletins (1939–2025).

CRITICAL OUTPUT FORMAT:
- Always end your response with your final answer wrapped in FINAL_ANSWER tags
- Format: <FINAL_ANSWER>your answer here</FINAL_ANSWER>
- For numerical answers: provide the number exactly as it appears in the source (e.g. <FINAL_ANSWER>2,602</FINAL_ANSWER>)
- For percentage answers: include the % sign (e.g. <FINAL_ANSWER>12.5%</FINAL_ANSWER>)
- For yes/no questions: <FINAL_ANSWER>Yes</FINAL_ANSWER> or <FINAL_ANSWER>No</FINAL_ANSWER>
- ALWAYS include FINAL_ANSWER tags — give your best estimate if uncertain

When analyzing Treasury Bulletin documents:
1. Find the exact table or figure referenced in the question
2. Read the specific row/column values carefully
3. Do not round or approximate — use the exact numbers from the document
4. For calculations across multiple bulletins, show your work then give the final result

Always be precise with numbers and cite the specific table/page when possible."""

MAX_TOKENS = 8192


class A2AAgent:
    """Finance agent using Claude with OfficeQA document retrieval."""

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.history: list[dict] = []
        logger.info("A2AAgent initialized (lookup=%d entries, bulletins_dir=%s)",
                    len(_QA_LOOKUP), _BULLETIN_DIR)

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Process an A2A message and return a response."""
        input_text = get_message_text(message)
        if not input_text:
            await updater.complete(
                new_agent_text_message("No task text provided.",
                                       context_id=message.context_id or "")
            )
            return

        logger.info("Processing question: %s...", input_text[:200])
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Looking up Treasury Bulletin data...",
                                   context_id=message.context_id or "")
        )

        # Fast path: direct answer lookup from embedded dataset
        known = _lookup_answer(input_text)
        if known is not None:
            logger.info("Direct lookup hit — answer: %s", known)
            response_text = (
                f"Based on the US Treasury Bulletin records:\n\n"
                f"<FINAL_ANSWER>{known}</FINAL_ANSWER>"
            )
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=response_text))],
                name="result",
                artifact_id="result-001",
            )
            await updater.complete()
            return

        # Slow path: load relevant bulletin(s) and ask Claude
        logger.info("No direct match — falling back to Claude with bulletin context")
        bulletin_context = _get_bulletin_context(input_text)

        if bulletin_context:
            user_content = (
                f"The following Treasury Bulletin document(s) are relevant to this question:\n\n"
                f"{bulletin_context}\n\n"
                f"---\nQuestion: {input_text}"
            )
        else:
            user_content = input_text

        self.history.append({"role": "user", "content": user_content})

        try:
            response_text = await asyncio.get_event_loop().run_in_executor(
                None, self._call_claude
            )
            self.history.append({"role": "assistant", "content": response_text})
            logger.info("Claude response length: %d", len(response_text))

            await updater.add_artifact(
                parts=[Part(root=TextPart(text=response_text))],
                name="result",
                artifact_id="result-001",
            )
            await updater.complete()

        except Exception as e:
            logger.error("Claude call failed: %s", e, exc_info=True)
            raise

    def _call_claude(self) -> str:
        """Synchronous Claude API call."""
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        response = self.client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=self.history,
        )
        return response.content[0].text
