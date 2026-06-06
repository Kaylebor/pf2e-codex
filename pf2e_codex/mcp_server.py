"""FastMCP server for PF2E rules knowledge base."""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP  # type: ignore[import-untyped]

from .config import Settings, get_settings
from .index import SearchIndex


def create_mcp_app(settings: Settings | None = None, host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    settings = settings or get_settings()
    search = SearchIndex(settings.db, settings.model, settings.provider, settings.onnx_provider)

    mcp = FastMCP("pf2e", host=host, port=port)

    @mcp.tool()
    def pf2e_search(
        query: str,
        top_k: int = 5,
        hybrid: bool = False,
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
        top_k = max(1, min(top_k, 20))
        results = search.search(query, top_k, hybrid=hybrid, license=license, content_type=content_type, pack=pack, remaster=remaster)
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
        return json.dumps(search.catalog(), indent=2)

    @mcp.tool()
    def pf2e_index_status() -> str:
        """Check the status of the PF2E index (model, chunk count, date)."""
        meta = search.status()
        return json.dumps(meta, indent=2)

    return mcp


def serve(settings: Settings | None = None, host: str = "127.0.0.1", port: int = 8000) -> None:
    settings = settings or get_settings()
    mcp = create_mcp_app(settings, host=host, port=port)
    transport = settings.transport
    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "streamable-http":
        print(f"MCP server on http://{host}:{port}/mcp  (streamable-http)")
        mcp.run(transport="streamable-http")
    else:
        print(f"MCP server on http://{host}:{port}/sse  (SSE)")
        mcp.run(transport="sse")
