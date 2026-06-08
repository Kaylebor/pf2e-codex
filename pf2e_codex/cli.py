"""CLI entry point: pf2e-codex index | search | serve | status."""

from __future__ import annotations

from pathlib import Path

import typer

from .config import Settings, get_settings
from .index import SearchIndex
from .models import list_models, recommend
from .pipeline import build_chunks, embed_and_index, index_all, save_chunks
from .cli_rich import print_search_results, print_catalog, print_status, print_validation

app = typer.Typer(name="pf2e-codex", help="PF2E rules knowledge base")


def _settings(
    data_dir: str | None = None,
    model: str | None = None,
    release: str | None = None,
) -> Settings:
    kwargs: dict[str, str] = {}
    if data_dir:
        kwargs["data_dir"] = data_dir
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
    model: str | None = typer.Option(None, "--model", "-m", help="Embedding model"),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Data directory (overrides XDG default)"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Replace existing index"),
    update: bool = typer.Option(False, "--update", "-u", help="Incremental update (diff vs indexed release)"),
) -> None:
    """Embed chunks and build sqlite-vec index."""
    settings = _settings(data_dir=data_dir, model=model)
    if update:
        from .pipeline import update_index
        update_index(settings)
        return
    if chunks_file:
        import json
        chunks: list[dict] = json.loads(chunks_file.read_text())
        typer.echo(f"Loaded {len(chunks)} chunks from {chunks_file}")
        typer.echo(f"Target DB: {settings.db}")
        embed_and_index(chunks, settings, rebuild=rebuild)
    else:
        typer.echo(f"Target DB: {settings.db}")
        index_all(settings, rebuild=rebuild)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    model: str | None = typer.Option(None, "--model", "-m", help="Embedding model"),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Data directory (overrides XDG default)"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    rerank: bool = typer.Option(False, "--rerank", help="Enable cross-encoder reranker"),
) -> None:
    """Search the PF2E index."""
    settings = _settings(data_dir=data_dir, model=model)
    search_idx = SearchIndex(settings.db, settings.model, settings.provider, settings.onnx_provider)
    results = search_idx.search(query, top_k, hybrid=True, rerank=rerank)
    print_search_results(results, query)


@app.command()
def get(
    entry_id: str = typer.Argument(..., help="Entry slug, name, UUID, or pack:id"),
    model: str | None = typer.Option(None, "--model", "-m", help="Embedding model"),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Data directory (overrides XDG default)"),
) -> None:
    """Fetch a single entry by its ID or Foundry UUID."""
    settings = _settings(data_dir=data_dir, model=model)
    search_idx = SearchIndex(settings.db, settings.model, settings.provider, settings.onnx_provider)
    result = search_idx.fetch_by_id(entry_id)
    if result:
        typer.echo(f"[{result['type']}] {result['name']} ({result['pack']})")
        typer.echo(f"ID: {result['id']}")
        typer.echo()
        typer.echo(result["text"])
    else:
        typer.echo(f"Entry not found: {entry_id}", err=True)
        raise typer.Exit(code=1)


@app.command()
def related(
    entry_id: str = typer.Argument(..., help="Entry slug, name, or UUID"),
    direction: str = typer.Option("both", "--direction", help="outgoing, incoming, or both"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results per direction"),
    model: str | None = typer.Option(None, "--model", "-m", help="Embedding model"),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Data directory (overrides XDG default)"),
) -> None:
    """Find entries related by cross-references."""
    settings = _settings(data_dir=data_dir, model=model)
    search_idx = SearchIndex(settings.db, settings.model, settings.provider, settings.onnx_provider)
    results = search_idx.related(entry_id, direction, limit)
    if results.get("outgoing"):
        typer.echo(f"\n{entry_id} references:")
        for r in results["outgoing"]:
            typer.echo(f"  [{r['type']}] {r['name']} ({r['pack']}) — {r.get('context', '')[:100]}")
    if results.get("incoming"):
        typer.echo(f"\nEntries referencing {entry_id}:")
        for r in results["incoming"]:
            typer.echo(f"  [{r['type']}] {r['name']} ({r['pack']}) — {r.get('context', '')[:100]}")
    if not results.get("outgoing") and not results.get("incoming"):
        typer.echo("No related entries found.")


@app.command()
def status(
    model: str | None = typer.Option(None, "--model", "-m", help="Embedding model"),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Data directory (overrides XDG default)"),
) -> None:
    """Show index status."""
    settings = _settings(data_dir=data_dir, model=model)
    search_idx = SearchIndex(settings.db, settings.model, settings.provider, settings.onnx_provider)
    meta = search_idx.status()
    meta["model"] = settings.model
    meta["db"] = str(settings.db)
    print_status(meta)


@app.command()
def catalog(
    model: str | None = typer.Option(None, "--model", "-m", help="Embedding model"),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Data directory"),
) -> None:
    """Show the structure of the PF2E database."""
    settings = _settings(data_dir=data_dir, model=model)
    search_idx = SearchIndex(settings.db, settings.model, settings.provider, settings.onnx_provider)
    cat = search_idx.catalog()
    print_catalog(cat)


@app.command()
def config(
    show_file: bool = typer.Option(False, "--file", help="Show which config file is active"),
) -> None:
    """Show effective configuration and source."""
    from .config import _CONFIG_PATHS, _load_toml
    s = get_settings()
    typer.echo("Effective configuration:")
    typer.echo(f"  model: {s.model}")
    typer.echo(f"  data_dir: {s.data_dir}")
    typer.echo(f"  db: {s.db}  (derived)")
    typer.echo(f"  provider: {s.provider}")
    typer.echo(f"  onnx_provider: {s.onnx_provider}")
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
@app.command("mcp")
def mcp_cmd(
    model: str | None = typer.Option(None, "--model", "-m", help="Embedding model"),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Data directory (overrides XDG default)"),
    transport: str = typer.Option("stdio", "--transport", "-t", help="MCP transport: stdio, sse, or streamable-http"),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address (streamable-http / sse)"),
    port: int = typer.Option(8000, "--port", "-p", help="Port (streamable-http / sse)"),
) -> None:
    """Start the MCP server (for Claude, pi, Cursor, etc.)."""
    settings = _settings(data_dir=data_dir, model=model)
    settings.transport = transport
    from .mcp_server import serve as _serve
    _serve(settings, host=host, port=port)


@app.command()
def benchmark(
    models: str = typer.Option(
        "all-MiniLM-L6-v2,snowflake-arctic-embed-xs,snowflake-arctic-embed-s,intfloat/e5-small-v2",
        "--models", "-m",
        help="Comma-separated model names to benchmark",
    ),
    providers: str = typer.Option(
        "onnx",
        "--providers", "-p",
        help="Comma-separated providers to benchmark (default: onnx)",
    ),
    chunks: int = typer.Option(200, "--chunks", "-c", help="Number of chunks to benchmark with"),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Data directory"),
) -> None:
    """Benchmark embedding speed across models and providers."""
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    prov_list = [p.strip() for p in providers.split(",") if p.strip()]
    from .benchmark import run_benchmark, print_results
    data = run_benchmark(models=model_list, providers=prov_list, chunks=chunks)
    print_results(data)


@app.command()
def validate(
    model: str | None = typer.Option(None, "--model", "-m", help="Embedding model"),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Data directory"),
    onnx_provider: str = typer.Option("auto", "--onnx-provider", help="ONNX provider override"),
    mode: str = typer.Option("hybrid", "--mode", help="Search mode: hybrid or semantic"),
    rerank: bool = typer.Option(True, "--rerank", help="Enable cross-encoder reranker"),
) -> None:
    """Validate retrieval quality against standard query suite."""
    from .validate import run_validation, load_queries
    settings = _settings(data_dir=data_dir, model=model)
    queries = load_queries()
    typer.echo(f"Model: {settings.model}")
    typer.echo(f"DB:    {settings.db}")
    typer.echo(f"Mode:  {mode}")
    typer.echo(f"Suite: {len(queries)} queries\n")

    result = run_validation(
        settings.db, settings.model,
        hybrid=(mode == "hybrid"),
        provider=settings.provider,
        onnx_provider=onnx_provider,
        rerank=rerank,
    )

    print_validation(result)


@app.command()
def export(
    model: str = typer.Option("Snowflake/snowflake-arctic-embed-xs", "--model", "-m", help="Model to export to ONNX"),
) -> None:
    """Export a model to ONNX (one-time, needs optimum+torch)."""
    print(f"Exporting {model} to ONNX...")
    print("This requires optimum + torch. Install with: pip install optimum[onnxruntime]")
    print()
    try:
        from .embeddings import ONNXProvider
        provider = ONNXProvider(model, force_provider="cpu")
        print(f"Done! Model cached at ~/.cache/pf2e-codex/onnx/")
        print(f"Dimension: {provider.dim}")
    except RuntimeError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def embed(
    all_models: bool = typer.Option(False, "--all-models", "-A", help="Embed all supported models"),
    models: list[str] | None = typer.Option(None, "--models", "-m", help="Specific models to embed"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Max parallel models (GPU compile spikes — increase carefully)"),
    update: bool = typer.Option(False, "--update", "-u", help="Incremental update existing DBs instead of skipping"),
    rebuild: bool = typer.Option(False, "--rebuild", "-f", help="Rebuild existing DBs from scratch (for chunker changes)"),
    latest: bool = typer.Option(False, "--latest", "-l", help="Fetch and update to the latest PF2E release from GitHub"),
    release: str | None = typer.Option(None, "--release", "-r", help="Specific PF2E release version (e.g. pf2e-8.2.0)"),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Data directory"),
) -> None:
    """Embed chunks for one or more models (shared fetch + chunk phase)."""
    from .models import ALL_MODEL_NAMES

    if all_models:
        model_list = list(ALL_MODEL_NAMES.values())
    elif models:
        model_list = models
    else:
        typer.echo("Specify --all-models or --models MODEL [MODEL...]")
        raise typer.Exit(1)

    if latest and release:
        typer.echo("--latest and --release are mutually exclusive")
        raise typer.Exit(1)

    settings = _settings(data_dir=data_dir)

    if latest:
        from .fetcher import get_latest_release
        typer.echo(f"Detecting latest PF2E release...")
        latest_rel = get_latest_release()
        typer.echo(f"Latest: {latest_rel}")
        from .config import Settings as S
        settings = S(
            data_dir=str(settings.data_dir),
            model=settings.model,
            release=latest_rel,
            provider=settings.provider,
            onnx_provider=settings.onnx_provider,
        )

    if release:
        from .config import Settings as S
        settings = S(
            data_dir=str(settings.data_dir),
            model=settings.model,
            release=release,
            provider=settings.provider,
            onnx_provider=settings.onnx_provider,
        )

    from .pipeline import embed_all_models
    embed_all_models(settings, model_list, concurrency=concurrency, update=update, rebuild=rebuild)


def main() -> None:
    app(prog_name="pf2e-codex")


if __name__ == "__main__":
    main()
