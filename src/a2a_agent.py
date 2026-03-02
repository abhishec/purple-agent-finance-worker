"""
Core A2A agent implementation for the Purple Finance Agent.
Wraps Claude 3.5 Sonnet with finance-domain capabilities.
"""
import asyncio
import logging
import os

import anthropic
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a highly capable financial analysis and trading agent. You excel at:
- SEC filing analysis (10-K, 10-Q, 8-K filings, financial statements)
- Options and derivatives trading decisions (pricing, Greeks, strategies)
- Cryptocurrency trading and market analysis
- Risk classification and portfolio risk management
- Treasury operations and fixed income analysis
- Business financial summaries and consistency checks
- Quantitative finance calculations (amortization, NPV, IRR, Black-Scholes)
- Data integration across financial datasets

When given a task:
1. Understand the financial context and question clearly
2. Apply rigorous quantitative reasoning where appropriate
3. Provide structured, precise responses with numerical results when needed
4. For trading decisions, include rationale, risk considerations, and confidence
5. For document analysis, extract key metrics and flag anomalies

Always be precise with numbers, cite sources when analyzing documents, and
structure outputs clearly (JSON when format is specified)."""

MAX_TOKENS = 4096
MAX_TURNS = 20


class A2AAgent:
    """Finance agent using Claude 3.5 Sonnet."""

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.history: list[dict] = []
        logger.info("A2AAgent (Finance) initialized with Claude 3.5 Sonnet")

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Process an A2A message and return a response."""
        input_text = get_message_text(message)
        if not input_text:
            await updater.complete(
                new_agent_text_message("No task text provided.",
                                       context_id=message.context_id or "")
            )
            return

        logger.info(f"Processing finance task: {input_text[:200]}...")
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Analyzing financial data...",
                                   context_id=message.context_id or "")
        )

        self.history.append({"role": "user", "content": input_text})

        try:
            response_text = await asyncio.get_event_loop().run_in_executor(
                None, self._call_claude
            )

            self.history.append({"role": "assistant", "content": response_text})

            logger.info(f"Finance task completed. Response length: {len(response_text)}")

            await updater.add_artifact(
                parts=[Part(root=TextPart(text=response_text))],
                name="result",
                artifact_id="result-001",
            )
            await updater.complete()

        except Exception as e:
            logger.error(f"Claude call failed: {e}", exc_info=True)
            raise

    def _call_claude(self) -> str:
        """Synchronous Claude API call."""
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=self.history,
        )
        return response.content[0].text
