"""
A2A-compatible server wrapper for the Purple Finance Agent.
Exposes the agent via the A2A (Agent-to-Agent) protocol for AgentBeats assessments.
"""
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from a2a_executor import A2AExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_agent_card(host: str, port: int, card_url: str | None) -> AgentCard:
    url = card_url or f"http://{host}:{port}/"
    skill = AgentSkill(
        id="finance",
        name="Financial Analysis and Trading",
        description=(
            "AI agent for financial analysis, trading decisions, and document processing. "
            "Handles SEC filing analysis (10-K, 10-Q), options and crypto trading decisions, "
            "risk classification, treasury operations, portfolio management, and quantitative "
            "finance calculations."
        ),
        tags=["finance", "trading", "options", "crypto", "sec", "risk", "treasury", "portfolio"],
        examples=[
            "Analyze the risk profile of this SEC 10-K filing",
            "What is the fair value of this call option using Black-Scholes?",
            "Should I buy or sell BTC given current market conditions?",
            "Classify the credit risk for this corporate bond portfolio",
        ],
    )
    return AgentCard(
        name="Purple Finance Agent",
        description=(
            "A powerful financial analysis and trading agent built on Claude 3.5 Sonnet. "
            "Specializes in SEC filing analysis, options pricing, crypto trading, risk "
            "classification, and quantitative finance for enterprise financial workflows."
        ),
        url=url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


def main():
    parser = argparse.ArgumentParser(description="Purple Finance Agent - A2A Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind to")
    parser.add_argument("--card-url", type=str, default=None, help="Public URL for agent card")
    args = parser.parse_args()

    logger.info(f"Starting Finance A2A server on {args.host}:{args.port}")

    agent_card = build_agent_card(args.host, args.port, args.card_url)
    request_handler = DefaultRequestHandler(
        agent_executor=A2AExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
    uvicorn.run(app.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
