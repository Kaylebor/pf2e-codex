"""FastMCP server for PF2E rules knowledge base."""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP  # type: ignore[import-untyped]

from .config import Settings, get_settings
from .index import SearchIndex


def create_mcp_app(settings: Settings | None = None) -> FastMCP:
    settings = settings or get_settings()
    search = SearchIndex(settings.db, settings.model, settings.provider, settings.onnx_provider)

    mcp = FastMCP("pf2e")

    @mcp.tool()
    def pf2e_search(query: str, top_k: int = 5, hybrid: bool = False) -> str:
        """Search the PF2E rules database for entries matching a query.

        Use this for finding specific feats, spells, conditions, items, or
        when you need a broad rules search. For deep rules explanations (e.g.
        "how does flanking work", "do penalties stack"), prefer pf2e_rules_explain.

        Args:
            query: Natural language question or keywords.
            top_k: Number of results (default 5, max 20).
            hybrid: If true, boosts exact name matches. Useful when you
                    know or half-remember the entry name.

        Returns:
            JSON with results containing id, name, type, pack, text, license.
        """
        top_k = max(1, min(top_k, 20))
        results = search.search(query, top_k, hybrid=hybrid)
        return json.dumps({"query": query, "results": results}, indent=2)

    @mcp.tool()
    def pf2e_get_entry(entry_id: str) -> str:
        """Fetch the full text of a PF2E entry by ID, slug, name, or UUID.

        Use this when you already know the entry name or have a UUID
        from a search result (shown as 'id' in results).

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
        result = search.fetch_by_id(entry_id)
        if result:
            return json.dumps(result, indent=2)
        return json.dumps({"error": f"Entry not found: {entry_id}"}, indent=2)

    @mcp.tool()
    def pf2e_rules_explain(topic: str, top_k: int = 3) -> str:
        """Get core rules explanations for a topic using deep semantic search.

        Use this for rules-phrased questions where you want journal pages
        and condition entries prioritized (e.g. "how does flanking work",
        "do status and circumstance penalties stack", "what happens when dying").

        Args:
            topic: The rules topic to explain.
            top_k: Number of results (default 3, max 10).

        Returns:
            JSON with prioritized results from journal pages and conditions.
        """
        top_k = max(1, min(top_k, 10))
        results = search.rules_explain(topic, top_k)
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
        limit = max(1, min(limit, 50))
        results = search.related(entry_id, direction, limit)
        return json.dumps({"entry_id": entry_id, "direction": direction, "results": results}, indent=2)

    @mcp.tool()
    def pf2e_index_status() -> str:
        """Check the status of the PF2E index (model, chunk count, date)."""
        meta = search.status()
        return json.dumps(meta, indent=2)

    return mcp


def serve(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    mcp = create_mcp_app(settings)
    if settings.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse")
