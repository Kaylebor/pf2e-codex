"""FastMCP server for PF2E rules knowledge base."""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP  # type: ignore[import-untyped]

from .config import Settings, get_settings
from .index import SearchIndex


def create_mcp_app(settings: Settings | None = None) -> FastMCP:
    settings = settings or get_settings()
    search = SearchIndex(settings.db, settings.model)

    mcp = FastMCP("pf2e")

    @mcp.tool()
    def pf2e_search(query: str, top_k: int = 5) -> str:
        """Search the PF2E rules database for entries matching a natural language query.

        Args:
            query: A natural language question or keywords about PF2E rules,
                   mechanics, feats, spells, conditions, etc.
            top_k: Number of top results to return (default 5, max 20).

        Returns:
            JSON string with search results containing id, name, type, pack, text, distance.
        """
        top_k = max(1, min(top_k, 20))
        results = search.search(query, top_k)
        return json.dumps({"query": query, "results": results}, indent=2)

    @mcp.tool()
    def pf2e_lookup(name: str) -> str:
        """Look up a PF2E entry by exact or near-exact name.

        Args:
            name: The exact name of a feat, spell, condition, action, etc.

        Returns:
            JSON string with matching entries.
        """
        results = search.lookup(name)
        return json.dumps({"name": name, "results": results}, indent=2)

    @mcp.tool()
    def pf2e_rules_explain(topic: str, top_k: int = 3) -> str:
        """Get core rules explanations for a topic (e.g. 'flanking', 'stacking penalties').

        Args:
            topic: The rules topic to explain.
            top_k: Number of top results to return (default 3, max 10).

        Returns:
            JSON string with prioritized results from journal pages and conditions.
        """
        top_k = max(1, min(top_k, 10))
        results = search.rules_explain(topic, top_k)
        return json.dumps({"topic": topic, "results": results}, indent=2)

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
