"""FastMCP server for PF2E rules knowledge base."""

from __future__ import annotations

import atexit
import json
import signal
import socket
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP  # type: ignore[import-untyped]

from .config import Settings, get_settings
from .index import SearchIndex


# Server endpoint file — CLI reads this to detect running server
_SERVER_JSON: Path | None = None


def _check_port_free(host: str, port: int) -> bool:
    """Check if a port is free on the given host."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except OSError:
        return False


def _pick_port(host: str, preferred: int) -> int:
    """Return preferred port if free, otherwise find a free one."""
    if _check_port_free(host, preferred):
        return preferred
    # Preferred port occupied — find any free port
    import random
    for _ in range(50):
        candidate = random.randint(10000, 60000)
        if _check_port_free(host, candidate):
            print(
                f"Port {preferred} in use, using {candidate} instead."
                f" Update your MCP config to http://127.0.0.1:{candidate}/mcp",
                file=sys.stderr,
            )
            return candidate
    # Last resort
    return _find_free_port(host)


def _write_server_json(host: str, port: int, transport: str) -> Path:
    """Write server endpoint to server.json for CLI auto-detection."""
    settings = get_settings()
    path = settings.data_dir / "server.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "endpoint": f"http://{host}:{port}/mcp",
        "host": host,
        "port": port,
        "transport": transport,
        "pid": __import__("os").getpid(),
    }
    path.write_text(json.dumps(data, indent=2))
    return path


def _cleanup_server_json() -> None:
    """Remove server.json on shutdown."""
    global _SERVER_JSON
    if _SERVER_JSON and _SERVER_JSON.exists():
        _SERVER_JSON.unlink(missing_ok=True)
        _SERVER_JSON = None


def _register_cleanup() -> None:
    """Register cleanup handlers for graceful shutdown."""
    atexit.register(_cleanup_server_json)
    for sig in (signal.SIGTERM, signal.SIGINT):
        prev = signal.getsignal(sig)
        def _handler(signum, frame, _prev=prev):
            _cleanup_server_json()
            if callable(_prev):
                _prev(signum, frame)
            elif _prev == signal.SIG_DFL:
                sys.exit(128 + signum)
        signal.signal(sig, _handler)


# Recent search history (for result flagging)
_RECENT: deque[dict] = deque(maxlen=5)
_FLAGGED_PATH = (get_settings().data_dir if get_settings().data_dir else Path.home() / ".local" / "share" / "pf2e-codex") / "flagged_results.jsonl"


def _flagged_path() -> Path:
    settings = get_settings()
    return settings.data_dir / "flagged_results.jsonl"


def _store_recent(query: str, results: list[dict], **params: str | int | bool | None) -> None:
    _RECENT.append({
        "query": query,
        "results": results,
        "params": params,
        "ts": datetime.now(timezone.utc).isoformat(),
    })


def create_mcp_app(settings: Settings | None = None, host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    settings = settings or get_settings()

    # Centralized model manager — single owner of all ONNX sessions.
    # start() blocks until both models are compiled and ready.
    from .model_manager import ModelManager  # noqa: PLC0415
    manager = ModelManager(
        model_name=settings.model,
        reranker_model=settings.reranker_model,
        provider=settings.provider,
        onnx_provider=settings.onnx_provider,
    )
    import threading as _threading
    _threading.Thread(target=manager.start, daemon=True).start()

    search = SearchIndex(settings.db, manager)

    mcp = FastMCP("pf2e", host=host, port=port)
    mcp._search_index = search  # prevent GC — keep provider alive across requests

    @mcp.tool()
    def pf2e_search(
        query: str,
        top_k: int = 5,
        hybrid: bool = False,
        rerank: bool = True,
        license: str | None = None,
        content_type: str | None = None,
        pack: str | None = None,
        remaster: bool | None = None,
    ) -> str:
        """Search the PF2E rules database for entries matching a query.

        Returns results with cross-references (refs), legacy names, confidence,
        and license. Supports filtering by license, content type, or pack.

        IMPORTANT: Default to current (Remaster) rules unless the user explicitly
        asks for legacy/pre-remaster rules. ORC license ≠ Remaster; they are
        orthogonal. Use license='OGL' only for legacy content.

        Args:
            query: Natural language question or keywords.
            top_k: Number of results (default 5, max 20).
            hybrid: If true, boosts exact name matches.
            rerank: If true, applies cross-encoder reranker for better relevance.
            license: Filter by license: 'ORC' (newer license), 'OGL' (older), or null.
            content_type: Filter by type: 'feat', 'spell', 'condition', 'journal_page', etc.
            pack: Filter by pack name: 'spells', 'feats', 'pathfinder-bestiary', etc.
            remaster: Filter by remaster status: true=current rules, false=legacy rules, null=both.

        Returns:
            JSON with results: id, name, type, pack, text, license, refs,
            legacy_name, confidence.

        Tip: 'refs' lists entries this result references — useful for chaining.
        'legacy_name' shows the pre-remaster name (e.g. 'flat-footed' for 'Off-Guard').
        """
        if not search.warmup_ready.is_set():
            return json.dumps({"error": "Server is warming up (8-10 min on first start). Retry shortly."})
        top_k = max(1, min(top_k, 20))
        results = search.search(query, top_k, hybrid=hybrid, rerank=rerank,
                                license=license, content_type=content_type,
                                pack=pack, remaster=remaster)
        _store_recent(query, results, top_k=top_k, hybrid=hybrid, rerank=rerank,
                      license=license, content_type=content_type, pack=pack, remaster=remaster)
        return json.dumps({"query": query, "results": results}, indent=2)

    @mcp.tool()
    def pf2e_flag_result(
        result_index: int,
        note: str = "",
    ) -> str:
        """Flag a search result as incorrect or low-quality.

        Call this right after pf2e_search when you notice a bad result.
        result_index: 1-based index of the bad result in the last search.
        note: Optional description of what's wrong (e.g. "wrong entry type", "outdated rule").

        Returns:
            Confirmation message.
        """
        if not _RECENT:
            return json.dumps({"error": "No recent search to flag"}, indent=2)

        last = _RECENT[-1]
        idx = result_index - 1  # convert to 0-based
        if idx < 0 or idx >= len(last["results"]):
            return json.dumps({
                "error": f"result_index out of range (1-{len(last['results'])})"}, indent=2)

        flagged = {
            **last,
            "flagged_result": last["results"][idx],
            "result_index": result_index,
            "note": note,
            "flagged_at": datetime.now(timezone.utc).isoformat(),
        }
        del flagged["results"]  # keep only the flagged one, not the full list

        path = _flagged_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(flagged) + "\n")

        return json.dumps({"ok": True, "flagged": flagged["flagged_result"]["name"]}, indent=2)
        return json.dumps({"query": query, "results": results}, indent=2)

    @mcp.tool()
    def pf2e_get_entry(entry_id: str) -> str:
        """Fetch the full text of a PF2E entry by ID, slug, name, or UUID.

        Use when you already know the entry name or have a UUID from a
        search result (shown as 'id' in results). This returns the complete
        entry text including mechanical effects, which search may truncate.

        Accepts:
            - Internal ID: "feats:Fury-Instinct" or "conditions:blinded"
            - Foundry UUID: "Compendium.pf2e.feats.Item.Fury-Instinct"
            - Bare slug: "fury-instinct"
            - Exact name: "Fury Instinct"

        Args:
            entry_id: The ID, slug, name, or UUID of the entry.

        Returns:
            JSON with the full entry text.
        """
        if not search.warmup_ready.is_set():
            return json.dumps({"error": "Server is warming up (8-10 min on first start). Retry shortly."})
        result = search.fetch_by_id(entry_id)
        if result:
            return json.dumps(result, indent=2)
        return json.dumps({"error": f"Entry not found: {entry_id}"}, indent=2)

    @mcp.tool()
    def pf2e_rules_explain(
        topic: str,
        top_k: int = 3,
        license: str | None = None,
        content_type: str | None = None,
        remaster: bool | None = None,
    ) -> str:
        """Get core rules explanations for a topic using deep semantic search.

        Use for rules-phrased questions where journal pages and conditions
        should be prioritized. Examples:
          - "how does flanking work"
          - "do status and circumstance penalties stack"
          - "what happens when dying"
          - "cover rules"

        For specific named entries (feats, spells, items), use pf2e_search.
        For exact entry text, combine search results with pf2e_get_entry.

        Args:
            topic: The rules topic to explain.
            top_k: Number of results (default 3, max 10).
            license: Filter by license: 'ORC', 'OGL', or null.
            content_type: Filter by type: 'condition', 'journal_page', 'feat', etc.
            remaster: Filter by remaster status: true=current rules, false=legacy rules, null=both.

        Returns:
            JSON with prioritized results from journal pages and conditions.
        """
        if not search.warmup_ready.is_set():
            return json.dumps({"error": "Server is warming up (8-10 min on first start). Retry shortly."})
        top_k = max(1, min(top_k, 10))
        results = search.rules_explain(topic, top_k, license=license, content_type=content_type, remaster=remaster)
        return json.dumps({"topic": topic, "results": results}, indent=2)

    @mcp.tool()
    def pf2e_related(entry_id: str, direction: str = "both", limit: int = 10) -> str:
        """Find entries related by cross-references (e.g., 'what grants Power Attack?').

        Args:
            entry_id: ID, slug, name, or UUID (e.g., "fury-instinct" or "Power Attack").
            direction: "outgoing" (what this entry references),
                       "incoming" (what references this entry),
                       or "both" (default).
            limit: Max results per direction (default 10, max 50).

        Returns:
            JSON with "outgoing" and "incoming" lists of related entries.
        """
        if not search.warmup_ready.is_set():
            return json.dumps({"error": "Server is warming up (8-10 min on first start). Retry shortly."})
        limit = max(1, min(limit, 50))
        results = search.related(entry_id, direction, limit)
        return json.dumps({"entry_id": entry_id, "direction": direction, "results": results}, indent=2)

    @mcp.tool()
    def pf2e_catalog() -> str:
        """Show the structure of the PF2E database.

        Returns counts of entries by type, license, and pack. Use this to
        discover what content is available before filtering searches.

        Returns:
            JSON with total_chunks, types (feat, spell, condition, etc.),
            licenses (ORC, OGL, NONE), and packs (spells, feats, etc.).
        """
        if not search.warmup_ready.is_set():
            return json.dumps({"error": "Server is warming up (8-10 min on first start). Retry shortly."})
        return json.dumps(search.catalog(), indent=2)

    @mcp.tool()
    def pf2e_index_status() -> str:
        """Check the status of the PF2E index (model, chunk count, date)."""
        if not search.warmup_ready.is_set():
            return json.dumps({"error": "Server is warming up (8-10 min on first start). Retry shortly."})
        meta = search.status()
        return json.dumps(meta, indent=2)

    return mcp


DEFAULT_PORT = 14141


def serve(settings: Settings | None = None, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
    settings = settings or get_settings()
    transport = settings.transport

    # For HTTP transports, check port availability and write server.json
    if transport in ("streamable-http", "sse"):
        port = _pick_port(host, port)
        global _SERVER_JSON
        _SERVER_JSON = _write_server_json(host, port, transport)
        _register_cleanup()

    mcp = create_mcp_app(settings, host=host, port=port)

    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "streamable-http":
        print(f"MCP server on http://{host}:{port}/mcp  (streamable-http)", file=sys.stderr)
        mcp.run(transport="streamable-http")
    else:
        print(f"MCP server on http://{host}:{port}/sse  (SSE)", file=sys.stderr)
        mcp.run(transport="sse")
