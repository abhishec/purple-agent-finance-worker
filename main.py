"""
Finance Mini AI Worker — AgentBeats-compatible entry point.

Accepts --host, --port, --card-url CLI args (required by AgentBeats platform).
Runs our FastAPI server with those settings.
"""
from __future__ import annotations
import argparse
import os
import uvicorn


def main():
    parser = argparse.ArgumentParser(
        description="Finance Mini AI Worker — AgentX-AgentBeats Phase 2 Finance Track"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind server on")
    parser.add_argument("--port", type=int, default=9010,
                        help="Port to bind server on")
    parser.add_argument("--card-url", type=str, default=None,
                        help="Public URL advertised in AgentCard (e.g. https://purple.agentbench.usebrainos.com)")
    args = parser.parse_args()

    if args.card_url:
        os.environ["PURPLE_AGENT_CARD_URL"] = args.card_url

    os.environ["PORT"] = str(args.port)

    print(f"[finance-agent] Starting on {args.host}:{args.port}", flush=True)
    if args.card_url:
        print(f"[finance-agent] Advertising card URL: {args.card_url}", flush=True)

    uvicorn.run(
        "src.server:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
