"""Daemon proxy: detect running MCP server and proxy queries to it.

When the MCP server is running (e.g., via systemd), CLI query commands
can proxy to it instead of creating a new local session (avoiding
MIGraphX compilation overhead on each CLI call).
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any


def _server_json_path() -> Path:
    """Return path to server.json."""
    from .config import get_settings
    return get_settings().data_dir / "server.json"


def _read_endpoint() -> str | None:
    """Read endpoint from server.json if it exists."""
    path = _server_json_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data.get("endpoint")
    except (json.JSONDecodeError, OSError):
        return None


_MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def _parse_sse_response(raw: str) -> dict | None:
    """Parse SSE response to extract JSON-RPC result."""
    for line in raw.split("\n"):
        if line.startswith("data: "):
            try:
                return json.loads(line[6:])
            except json.JSONDecodeError:
                continue
    # Try parsing as plain JSON (in case json_response mode is enabled)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _check_server(endpoint: str, timeout: float = 2.0) -> bool:
    """Check if MCP server is responsive at the given endpoint."""
    try:
        # Send a minimal JSON-RPC initialize request
        req = urllib.request.Request(
            endpoint,
            data=json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "pf2e-codex-cli", "version": "0.1.0"}
                }
            }).encode(),
            headers=_MCP_HEADERS,
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=timeout)
        raw = resp.read().decode()
        data = _parse_sse_response(raw)
        return data is not None and "result" in data
    except Exception:
        return False


def _call_tool(endpoint: str, tool_name: str, arguments: dict[str, Any], timeout: float = 30.0) -> dict[str, Any] | None:
    """Call an MCP tool via JSON-RPC and return the result."""
    try:
        # Initialize
        init_req = urllib.request.Request(
            endpoint,
            data=json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "pf2e-codex-cli", "version": "0.1.0"}
                }
            }).encode(),
            headers=_MCP_HEADERS,
            method="POST",
        )
        init_resp = urllib.request.urlopen(init_req, timeout=timeout)
        init_raw = init_resp.read().decode()
        init_data = _parse_sse_response(init_raw)
        session_id = init_data.get("result", {}).get("sessionId")

        # Send initialized notification
        notif_headers = dict(_MCP_HEADERS)
        if session_id:
            notif_headers["Mcp-Session-Id"] = session_id
        notif_req = urllib.request.Request(
            endpoint,
            data=json.dumps({
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }).encode(),
            headers=notif_headers,
            method="POST",
        )
        urllib.request.urlopen(notif_req, timeout=timeout)

        # Call tool
        call_headers = dict(_MCP_HEADERS)
        if session_id:
            call_headers["Mcp-Session-Id"] = session_id
        call_req = urllib.request.Request(
            endpoint,
            data=json.dumps({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments,
                }
            }).encode(),
            headers=call_headers,
            method="POST",
        )
        call_resp = urllib.request.urlopen(call_req, timeout=timeout)
        call_raw = call_resp.read().decode()
        call_data = _parse_sse_response(call_raw)

        # Extract result from MCP response
        result = call_data.get("result", {})
        content = result.get("content", [])
        if content and content[0].get("type") == "text":
            return json.loads(content[0]["text"])
        return None
    except Exception:
        return None


def proxy_search(query: str, top_k: int = 5, hybrid: bool = True, **kwargs: Any) -> dict | None:
    """Try to proxy a search query to the running MCP server."""
    endpoint = _read_endpoint()
    if not endpoint or not _check_server(endpoint):
        return None
    args = {"query": query, "top_k": top_k, "hybrid": hybrid, **kwargs}
    return _call_tool(endpoint, "pf2e_search", args)


def proxy_get_entry(entry_id: str) -> dict | None:
    """Try to proxy a get_entry query to the running MCP server."""
    endpoint = _read_endpoint()
    if not endpoint or not _check_server(endpoint):
        return None
    return _call_tool(endpoint, "pf2e_get_entry", {"entry_id": entry_id})


def proxy_related(entry_id: str, direction: str = "both", limit: int = 10) -> dict | None:
    """Try to proxy a related query to the running MCP server."""
    endpoint = _read_endpoint()
    if not endpoint or not _check_server(endpoint):
        return None
    return _call_tool(endpoint, "pf2e_related", {"entry_id": entry_id, "direction": direction, "limit": limit})


def proxy_status() -> dict | None:
    """Try to proxy a status query to the running MCP server."""
    endpoint = _read_endpoint()
    if not endpoint or not _check_server(endpoint):
        return None
    return _call_tool(endpoint, "pf2e_index_status", {})


def proxy_catalog() -> dict | None:
    """Try to proxy a catalog query to the running MCP server."""
    endpoint = _read_endpoint()
    if not endpoint or not _check_server(endpoint):
        return None
    return _call_tool(endpoint, "pf2e_catalog", {})
