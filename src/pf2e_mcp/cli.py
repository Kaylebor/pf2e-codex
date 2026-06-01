"""CLI entry point: pf2e-mcp index | search | serve | status."""

from __future__ import annotations

from pathlib import Path

import typer

from .config import Settings, get_settings
from .index import SearchIndex
from .models import list_models, recommend
from .pipeline import build_chunks, embed_and_index, index_all, save_chunks

app = typer.Typer(name="pf2e-mcp", help="PF2E rules knowledge base")


def _settings(
    db: str | None = None,
    model: str | None = None,
    release: str | None = None,
) -> Settings:
    kwargs: dict[str, str] = {}
    if db:
        kwargs["db"] = db
    if model:
        kwargs["model"] = model
    if release:
        kwargs["release"] = release
    return get_settings(**kwargs)


@app.command()
def fetch(
    version: str | None = typer.Option(None, "--version", "-v", help="PF2E release version"),
) -> None:
    """Download json-assets.zip from GitHub releases."""
    settings = _settings(release=version)
    from .fetcher import get_cached_zip
    zip_path = get_cached_zip(settings)
    typer.echo(f"Ready: {zip_path}")


@app.command()
def build(
    output: Path = typer.Option(Path("chunks.json"), "--output", "-o", help="Output JSON file"),
    version: str | None = typer.Option(None, "--version", "-v"),
) -> None:
    """Build enriched chunks from PF2E pack data."""
    settings = _settings(release=version)
    chunks = build_chunks(settings)
    save_chunks(chunks, output)


@app.command()
def index(
    chunks_file: Path | None = typer.Argument(None, help="Optional pre-built chunks.json"),
    model: str | None = typer.Option(None, "--model", "-m"),
    db: str | None = typer.Option(None, "--db", "-d"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Replace existing index"),
) -> None:
    """Embed chunks and build sqlite-vec index."""
    settings = _settings(db=db, model=model)
    if chunks_file:
        import json
        chunks: list[dict] = json.loads(chunks_file.read_text())
        typer.echo(f"Loaded {len(chunks)} chunks from {chunks_file}")
        embed_and_index(chunks, settings, rebuild=rebuild)
    else:
        index_all(settings, rebuild=rebuild)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    db: str | None = typer.Option(None, "--db", "-d"),
    model: str | None = typer.Option(None, "--model", "-m"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
) -> None:
    """Search the PF2E index."""
    settings = _settings(db=db, model=model)
    search_idx = SearchIndex(settings.db, settings.model)
    results = search_idx.search(query, top_k)
    typer.echo(f"Query: '{query}'")
    typer.echo(f"Results ({len(results)}):\n")
    for i, r in enumerate(results, 1):
        typer.echo(f"--- Result {i} (dist={r['distance']:.4f}) ---")
        typer.echo(f"[{r['type']}] {r['name']} ({r['pack']})")
        preview = r["text"][:600] + ("..." if len(r["text"]) > 600 else "")
        typer.echo(preview)
        typer.echo()


@app.command()
def status(
    db: str | None = typer.Option(None, "--db", "-d"),
    model: str | None = typer.Option(None, "--model", "-m"),
) -> None:
    """Show index status."""
    settings = _settings(db=db, model=model)
    search_idx = SearchIndex(settings.db, settings.model)
    meta = search_idx.status()
    for k, v in meta.items():
        typer.echo(f"  {k}: {v}")


@app.command()
def config(
    show_file: bool = typer.Option(False, "--file", help="Show which config file is active"),
) -> None:
    """Show effective configuration and source."""
    from .config import _CONFIG_PATHS, _load_toml
    s = get_settings()
    typer.echo("Effective configuration:")
    typer.echo(f"  model: {s.model}")
    typer.echo(f"  db: {s.db}")
    typer.echo(f"  release: {s.release}")
    typer.echo(f"  cache_dir: {s.cache_dir}")
    typer.echo(f"  transport: {s.transport}")
    if show_file:
        for path in _CONFIG_PATHS:
            if path.exists():
                typer.echo(f"\nActive config file: {path}")
                for k, v in _load_toml().items():
                    typer.echo(f"  {k}: {v}")
                return
        typer.echo("\nNo config file found. Checked:")
        for path in _CONFIG_PATHS:
            typer.echo(f"  {path}")


@app.command()
def models(
    hardware: str = typer.Option("cpu", "--hardware", help="Recommend for: cpu, gpu"),
    all: bool = typer.Option(False, "--all", help="List all available models"),
) -> None:
    """List embedding models with hardware recommendations."""
    if all:
        for info in list_models():
            typer.echo(f"\n{info.name}")
            typer.echo(f"  Params: {info.params}, Dim: {info.dim}, CPU time: {info.cpu_time_28k}")
            typer.echo(f"  Quality: {info.quality}")
            typer.echo(f"  Query prefix: '{info.query_prefix}'" if info.query_prefix else "  No prefixing")
            typer.echo(f"  {info.notes}")
    else:
        recs = recommend(hardware)
        typer.echo(f"Recommendations for {hardware}:")
        for name in recs:
            info = next((m for m in list_models() if m.name == name or m.name.endswith(name)), None)
            if info:
                typer.echo(f"  {info.name} ({info.params}, {info.dim}d, {info.quality})")


@app.command()
def serve(
    db: str | None = typer.Option(None, "--db", "-d"),
    model: str | None = typer.Option(None, "--model", "-m"),
    transport: str = typer.Option("stdio", "--transport", "-t"),
) -> None:
    """Start the MCP server (stdio or sse)."""
    settings = _settings(db=db, model=model)
    settings.transport = transport
    from .mcp_server import serve as _serve
    _serve(settings)


def main() -> None:
    app()
