# pf2e-mcp

PF2E rules knowledge base with MCP, CLI, and SDK interfaces.

- **Data source**: Official FoundryVTT PF2E system JSON releases (`json-assets.zip`)
- **Indexing**: Rules-aware chunking + semantic embeddings in sqlite-vec
- **Interfaces**: MCP server, CLI commands, Python SDK

## Quick Start

```bash
# Install
uv venv
uv pip install -e "."

# Download PF2E data and build index
pf2e-mcp index

# Search from CLI
pf2e-mcp search "flat-footed while flanking"

# Run MCP server (stdio for Claude Desktop / Cursor / pi)
pf2e-mcp serve
```

## Architecture

```
pf2e-mcp/
├── src/pf2e_mcp/
│   ├── __init__.py        # SDK exports
│   ├── config.py          # Settings (Pydantic + TOML file)
│   ├── fetcher.py         # Download json-assets.zip from GitHub releases
│   ├── chunker.py         # Rules-aware chunk builder
│   ├── models.py          # Embedding model registry & recommendations
│   ├── embeddings.py      # Pluggable embedding providers
│   ├── index.py           # sqlite-vec storage + search
│   ├── pipeline.py        # Orchestration: fetch → chunk → embed → index
│   ├── mcp_server.py      # FastMCP server
│   └── cli.py             # pf2e-mcp CLI
├── pyproject.toml         # Package config (uv-compatible)
└── pf2e_v2.db             # sqlite-vec database (generated)
```

### Chunking Strategy

One entry = one chunk, with `system.rules` flattened into plain English:

- `FlatModifier` → "-4 status to perception"
- `GrantItem` → "Grants item: Power Attack"
- `DamageDice` → "+1d6 fire damage to melee-strike-damage"
- `RollOption` → toggleable options with choices
- `Strike` → unarmed attack stats

Cross-references (`@UUID[...]{Name}`) resolved to human-readable names.
Expression simplification: `ternary(gte(@actor.level,13),...)` → "+X at level 13+".

### Journal Pages

Journals (GM Screen, Classes, Domains, etc.) split into per-page chunks so core
rules explanations are retrievable.

## CLI Commands

| Command | Description |
|---|---|
| `pf2e-mcp fetch` | Download json-assets.zip |
| `pf2e-mcp build` | Build enriched chunks (JSON output) |
| `pf2e-mcp index` | Full pipeline: fetch → chunk → embed → index |
| `pf2e-mcp search "query"` | Semantic search |
| `pf2e-mcp status` | Show index stats |
| `pf2e-mcp config` | Show effective configuration |
| `pf2e-mcp config --file` | Show active config file contents |
| `pf2e-mcp models` | List embedding models with recommendations |
| `pf2e-mcp serve` | Start MCP server (stdio or sse) |

## Configuration

Priority (highest wins):
1. Command-line flags / kwargs
2. Environment variables (`PF2E_DB`, `PF2E_MODEL`, etc.)
3. Config file (`~/.config/pf2e-mcp/config.toml` or `./pf2e-mcp.toml`)
4. Built-in defaults

### Config file

Create `~/.config/pf2e-mcp/config.toml`:

```toml
model = "snowflake-arctic-embed-s"
db = "~/pf2e/pf2e_v2.db"
release = "pf2e-8.1.2"
```

Or use a project-local `pf2e-mcp.toml` (gitignored by default).

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PF2E_CACHE_DIR` | `~/.cache/pf2e-mcp` | Download cache |
| `PF2E_DB` | `pf2e_v2.db` | sqlite-vec database |
| `PF2E_MODEL` | `snowflake-arctic-embed-xs` | Embedding model |
| `PF2E_RELEASE` | `pf2e-8.1.2` | PF2E system version |

## MCP Tools

| Tool | Description |
|---|---|
| `pf2e_search(query, top_k)` | Semantic search across all rules entries |
| `pf2e_lookup(name)` | Exact name lookup (e.g. "Fireball") |
| `pf2e_rules_explain(topic, top_k)` | Prioritized search favoring journal pages |
| `pf2e_index_status()` | Show model, chunk count, date |

## Embedding Models

All models work out of the box. Query/document prefixing is handled automatically.

### CPU recommendations

Tested on AMD Ryzen 7 7800X3D, indexing 28,837 chunks:

| Model | Params | Dim | Index Time | Quality |
|-------|--------|-----|------------|---------|
| `snowflake-arctic-embed-xs` | 22M | 384 | ~35s | Good |
| `snowflake-arctic-embed-s` | 33M | 384 | ~70s | Good |
| `all-MiniLM-L6-v2` | 22M | 384 | ~50s | Good |
| `e5-small-v2` | 33M | 384 | ~135s | Good |

**Default: `snowflake-arctic-embed-xs`** — fastest on CPU, equivalent quality to MiniLM, automatic prefixing handled.

### GPU recommendations

| Model | Params | Dim | Quality |
|-------|--------|-----|---------|
| `snowflake-arctic-embed-m` | 110M | 768 | Better |
| `nomic-embed-text-v1.5` | 137M | 768 | Better |

Switch models by setting the environment variable:

```bash
PF2E_MODEL="snowflake-arctic-embed-s" pf2e-mcp index --rebuild
```

Or in your config file:
```toml
model = "snowflake-arctic-embed-m"
```

## SDK

```python
from pf2e_mcp.config import get_settings
from pf2e_mcp.index import SearchIndex

settings = get_settings(db="pf2e_v2.db")
search = SearchIndex(settings.db, settings.model)

results = search.search("flat-footed while flanking", top_k=3)
for r in results:
    print(r["name"], r["distance"])
```

## Legal

- Code: MIT (this repo)
- PF2E game content: ORC / OGL (not redistributed; user fetches from official releases)
- No pre-computed embeddings or PF2E data shipped
