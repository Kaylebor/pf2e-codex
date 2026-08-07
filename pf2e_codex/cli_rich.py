"""Rich CLI output helpers for pf2e-codex."""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(highlight=False)


def print_search_results(results: list[dict], query: str) -> None:
    """Render search results as a Rich table."""
    table = Table(
        title=f"Results for: {query}",
        box=box.SIMPLE_HEAVY,
        show_lines=True,
        title_style="bold cyan",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Name", style="bold")
    table.add_column("Type", style="green")
    table.add_column("Pack", style="dim")
    table.add_column("Source", style="dim")
    table.add_column("License", style="yellow")
    table.add_column("Confidence", justify="center")

    conf_colors = {"high": "bold green", "medium": "yellow", "low": "dim"}

    for i, r in enumerate(results, 1):
        name = r.get("name", "")
        legacy = r.get("legacy_name")
        if legacy:
            name += f"  [dim](formerly {legacy})[/dim]"

        refs = r.get("refs", [])
        ref_str = ""
        if refs:
            ref_names = [ref["name"] for ref in refs[:5]]
            ref_str = f"\n[dim]refs: {', '.join(ref_names)}{'...' if len(refs) > 5 else ''}[/dim]"

        conf = r.get("confidence", "?")
        conf_style = conf_colors.get(conf, "")
        provenance = r.get("provenance") or {}
        local_provenance = provenance.get("provenance") or {}
        source = (
            local_provenance.get("title")
            or provenance.get("product")
            or provenance.get("source")
            or ""
        )
        page_start = provenance.get("source_page_start")
        page_end = provenance.get("source_page_end")
        if page_start is not None:
            pages = str(page_start) if page_end in (None, page_start) else f"{page_start}-{page_end}"
            source = f"{source}\nPDF p. {pages}".strip()

        table.add_row(
            str(i),
            name + ref_str,
            r.get("type", "?"),
            r.get("pack", "?"),
            source,
            r.get("license", "?"),
            f"[{conf_style}]{conf}[/{conf_style}]" if conf_style else conf,
        )

    console.print(table)

    # Footer with summary
    n = len(results)
    types = {r.get("type", "") for r in results}
    licenses = {r.get("license", "") for r in results}
    console.print(f"\n  {n} result{'s' if n != 1 else ''} | types: {', '.join(sorted(types))} | licenses: {', '.join(sorted(licenses))}\n")


def print_catalog(cat: dict) -> None:
    """Render catalog as Rich panels."""
    # Summary
    console.print(Panel(
        f"Total chunks: [bold]{cat['total_chunks']:,}[/bold] | "
        f"Total references: [bold]{cat['total_references']:,}[/bold]",
        title="PF2E Database",
        style="cyan",
    ))

    # Types
    types_table = Table(title="Content Types", box=box.SIMPLE)
    types_table.add_column("Type", style="bold")
    types_table.add_column("Count", justify="right")
    for t, count in cat["types"].items():
        types_table.add_row(t, f"{count:,}")
    console.print(types_table)

    # Licenses
    console.print("\n")
    lic_table = Table(title="Licenses", box=box.SIMPLE)
    lic_table.add_column("License", style="bold")
    lic_table.add_column("Count", justify="right")
    for lic, count in cat["licenses"].items():
        lic_table.add_row(lic, f"{count:,}")
    console.print(lic_table)

    # Remaster status
    console.print("\n")
    rem_table = Table(title="Remaster Status", box=box.SIMPLE)
    rem_table.add_column("Status", style="bold")
    rem_table.add_column("Count", justify="right")
    for status, count in cat.get("remaster", {}).items():
        rem_table.add_row(status, f"{count:,}")
    console.print(rem_table)

    # Packs
    console.print("\n")
    pack_table = Table(title="Top Packs", box=box.SIMPLE)
    pack_table.add_column("Pack", style="bold")
    pack_table.add_column("Count", justify="right")
    for p, count in list(cat["packs"].items())[:15]:
        pack_table.add_row(p, f"{count:,}")
    console.print(pack_table)


def print_status(meta: dict) -> None:
    """Render index status as a panel."""
    items = "\n".join(f"  [bold]{k}[/bold]: {v}" for k, v in meta.items())
    console.print(Panel(items, title="Index Status", style="cyan"))


def print_validation(result: dict) -> None:
    """Render validation results."""
    table = Table(
        title=f"Validation: {result['n_queries']} queries",
        box=box.SIMPLE_HEAVY,
        show_lines=True,
        title_style="bold cyan",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Query")
    table.add_column("Expected", style="bold")
    table.add_column("Rank", justify="center")
    table.add_column("Top 3")

    for i, r in enumerate(result["results"], 1):
        rank = r["rank"]
        if rank == 1:
            rank_str = "[bold green]✓ 1[/bold green]"
        elif rank and rank <= 3:
            rank_str = f"[green]{rank}[/green]"
        elif rank:
            rank_str = f"[yellow]{rank}[/yellow]"
        else:
            rank_str = "[red]✗[/red]"

        table.add_row(
            str(i),
            r["query"][:45],
            r["expected"],
            rank_str,
            ", ".join(r.get("top_3", [])[:3])[:50],
        )

    console.print(table)

    # Summary
    mrr = result["mrr"]
    console.print(
        f"\n  MRR: [bold]{mrr:.3f}[/bold] | "
        f"Perfect: [green]{result['perfect']}/{result['n_queries']}[/green] | "
        f"Top 3: [yellow]{result['top3']}/{result['n_queries']}[/yellow] | "
        f"Not found: [red]{result['not_found']}[/red]\n"
    )
