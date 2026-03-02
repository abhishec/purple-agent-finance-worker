from __future__ import annotations
import asyncio
import json
from typing import Callable, Awaitable

import httpx

from src.config import BRAINOS_API_URL, BRAINOS_API_KEY, BRAINOS_ORG_ID, TASK_TIMEOUT


class BrainOSUnavailableError(Exception):
    pass


async def run_task(
    message: str,
    system_context: str,
    on_tool_call: Callable[[str, dict], Awaitable[dict]],
    session_id: str,
) -> str:
    """
    Stream a task through BrainOS copilot API.
    Handles SSE stream, detects tool_call events, calls on_tool_call, injects results.
    Raises BrainOSUnavailableError on connection failure or timeout.
    """
    if not BRAINOS_API_KEY or not BRAINOS_ORG_ID:
        raise BrainOSUnavailableError("BrainOS credentials not configured")

    url = f"{BRAINOS_API_URL}/api/copilot/chat"
    headers = {
        "Authorization": f"Bearer {BRAINOS_API_KEY}",
        "X-Organization-Id": BRAINOS_ORG_ID,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {
        "message": message,
        "conversationId": session_id,
        "extraParams": {"systemContext": system_context},
    }

    try:
        async with httpx.AsyncClient(timeout=TASK_TIMEOUT) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                if resp.status_code >= 400:
                    raise BrainOSUnavailableError(f"BrainOS returned {resp.status_code}")

                final_answer = ""
                tool_results = []
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if "tool_call" in event:
                        tc = event["tool_call"]
                        tool_result = await on_tool_call(tc.get("name", ""), tc.get("params", {}))
                        # Collect tool results to send back in follow-up POST
                        tool_results.append({
                            "name": tc.get("name", ""),
                            "result": tool_result
                        })

                    if "answer" in event:
                        final_answer = event["answer"]
                    elif "text" in event:
                        final_answer += event["text"]

                # If we have tool results but no final answer, send them back via follow-up POST.
                # Reuse the same AsyncClient — we're still inside the outer `async with` block.
                if tool_results and not final_answer:
                    followup_payload = {
                        "message": "Tool results:",
                        "conversationId": session_id,
                        "toolResults": tool_results,
                    }
                    async with client.stream("POST", url, headers=headers, json=followup_payload) as followup_resp:
                        if followup_resp.status_code >= 400:
                            raise BrainOSUnavailableError(f"BrainOS follow-up returned {followup_resp.status_code}")

                        async for line in followup_resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data_str = line[6:].strip()
                            if not data_str or data_str == "[DONE]":
                                continue
                            try:
                                event = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            if "answer" in event:
                                final_answer = event["answer"]
                            elif "text" in event:
                                final_answer += event["text"]

                return final_answer

    except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as e:
        raise BrainOSUnavailableError(str(e)) from e
