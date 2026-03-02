from __future__ import annotations
import httpx
from src.config import TOOL_TIMEOUT


def validate_tool_call(
    tool_name: str,
    params: dict,
    tools_list: list[dict],
) -> tuple[bool, str]:
    """Pre-flight: verify tool exists and required params are present.

    Returns (valid, error_msg). If tools_list is empty we cannot validate —
    allow through rather than blocking legitimate calls.
    """
    if not tools_list:
        return True, ""  # can't validate without schema — allow through

    tool_schema = next((t for t in tools_list if t.get("name") == tool_name), None)
    if tool_schema is None:
        # Tool not in discovered list — likely hallucinated
        available = [t.get("name") for t in tools_list[:10]]
        return False, f"Tool '{tool_name}' not in available tools. Available: {available}"

    # Check required params — handle both input_schema (Anthropic) and inputSchema (MCP spec)
    input_schema = tool_schema.get("input_schema") or tool_schema.get("inputSchema", {})
    required = input_schema.get("required", [])
    missing = [r for r in required if r not in params]
    if missing:
        return False, f"Tool '{tool_name}' missing required params: {missing}"

    return True, ""


def _is_empty_result(result: dict) -> bool:
    """Return True when the tool result carries no useful data."""
    if "error" in result:
        return False  # errors are meaningful — not "empty"
    for key in ("data", "result", "items", "records", "rows"):
        val = result.get(key)
        if val is not None:
            if isinstance(val, (list, dict)) and len(val) == 0:
                continue  # empty container — keep checking other keys
            return False  # non-empty value found
    # If none of the expected keys had content, consider it empty
    return all(
        (result.get(k) is None or result.get(k) == [] or result.get(k) == {})
        for k in ("data", "result", "items", "records", "rows")
    )


async def discover_tools(tools_endpoint: str, session_id: str = "") -> list[dict]:
    """GET {tools_endpoint}/mcp/tools — returns Anthropic-format tool list.

    Pass session_id to get only the tools registered for that specific task session
    (prevents Claude from seeing all 130+ tools from every scenario at once).
    """
    async with httpx.AsyncClient(timeout=TOOL_TIMEOUT) as client:
        url = f"{tools_endpoint}/mcp/tools"
        if session_id:
            url = f"{url}?session_id={session_id}"
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


async def call_tool(
    tools_endpoint: str,
    tool_name: str,
    params: dict,
    session_id: str,
    tools_list: list[dict] | None = None,
) -> dict:
    """POST {tools_endpoint}/mcp — calls a tool and returns result.

    Runs pre-flight validation via validate_tool_call() when tools_list is
    provided.  Invalid calls return immediately without a network round-trip.
    """
    # Pre-flight validation
    valid, error_msg = validate_tool_call(tool_name, params, tools_list or [])
    if not valid:
        return {"error": error_msg, "validation_failed": True}

    async with httpx.AsyncClient(timeout=TOOL_TIMEOUT) as client:
        resp = await client.post(
            f"{tools_endpoint}/mcp",
            json={"tool": tool_name, "params": params, "session_id": session_id},
        )
        resp.raise_for_status()
        return resp.json()
