# pf2e-codex

PF2E rules knowledge base with MCP, CLI, and SDK interfaces.

- **Data source**: Official FoundryVTT PF2E system JSON releases (`json-assets.zip`)
- **Indexing**: Rules-aware chunking + semantic embeddings in sqlite-vec
- **Interfaces**: MCP server, CLI commands, Python SDK

## Quick Start

### Install

**One-liner (recommended):**
```bash
curl -sSL https://raw.githubusercontent.com/Kaylebor/pf2e-codex/main/install.sh | bash
```

**Arch Linux (AUR):**
```bash
git clone https://aur.archlinux.org/pf2e-codex.git
cd pf2e-codex
makepkg -si
```

**Manual (uv):**
```bash
git clone https://github.com/Kaylebor/pf2e-codex.git
cd pf2e-codex
uv venv
uv pip install -e "."
```

### First run

```bash
# Download PF2E data and build index (~35s on CPU, ~5s on GPU)
pf2e-codex index

# Search from CLI
pf2e-codex search "flat-footed while flanking"

# Run MCP server (stdio for Claude Desktop / Cursor / pi)
pf2e-codex serve
```

## Architecture

```
pf2e-codex/
├── pf2e_codex/
│   ├── __init__.py        # SDK exports
│   ├── config.py          # Settings (Pydantic + TOML file)
│   ├── fetcher.py         # Download json-assets.zip from GitHub releases
│   ├── chunker.py         # Rules-aware chunk builder
│   ├── models.py          # Embedding model registry & recommendations
│   ├── embeddings.py      # Pluggable embedding providers
│   ├── index.py           # sqlite-vec storage + search
│   ├── pipeline.py        # Orchestration: fetch → chunk → embed → index
│   ├── mcp_server.py      # FastMCP server
│   └── cli.py             # pf2e-codex CLI
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
| `pf2e-codex fetch` | Download json-assets.zip |
| `pf2e-codex build` | Build enriched chunks (JSON output) |
| `pf2e-codex index` | Full pipeline: fetch → chunk → embed → index |
| `pf2e-codex search "query"` | Hybrid search (semantic + FTS5) |
| `pf2e-codex status` | Show index stats |
| `pf2e-codex config` | Show effective configuration |
| `pf2e-codex config --file` | Show active config file contents |
| `pf2e-codex get "fireball"` | Fetch a single entry by slug, name, or UUID |
| `pf2e-codex related "off-guard" --direction incoming` | Cross-reference graph |
| `pf2e-codex models` | List embedding models with recommendations |
| `pf2e-codex serve` | Start MCP server (stdio or sse) |

## Configuration

Priority (highest wins):
1. Command-line flags / kwargs
2. Environment variables (`PF2E_DATA_DIR`, `PF2E_MODEL`, etc.)
3. Config file (`~/.config/pf2e-codex/config.toml` or `./pf2e-codex.toml`)
4. Built-in defaults

### Config file

Create `~/.config/pf2e-codex/config.toml`:

```toml
model = "snowflake-arctic-embed-s"
db = "~/pf2e/pf2e_v2.db"
release = "pf2e-8.1.2"
```

Or use a project-local `pf2e-codex.toml` (gitignored by default).

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PF2E_CACHE_DIR` | `~/.cache/pf2e-codex` | Download cache |
| `PF2E_DATA_DIR` | `~/.local/share/pf2e-codex` | Data directory |
| `PF2E_MODEL` | `snowflake-arctic-embed-xs` | Embedding model |
| `PF2E_PROVIDER` | `auto` | `auto`, `onnx`, `sentence_transformers` |
| `PF2E_RELEASE` | `pf2e-8.1.2` | PF2E system version |

## MCP Tools

| Tool | Description |
|---|---|
| `pf2e_search(query, top_k)` | Hybrid search (semantic + FTS5 name/text matching) |
| `pf2e_get_entry(entry_id)` | Fetch full entry by ID, slug, name, or Foundry UUID |
| `pf2e_related(entry_id, direction, limit)` | Cross-reference graph: outgoing/incoming/both |
| `pf2e_rules_explain(topic, top_k)` | Prioritized search favoring journal pages |
| `pf2e_index_status()` | Show model, chunk count, date |

## ONNX Acceleration (automatic)

The tool proactively tries ONNX Runtime for faster inference. It works out of the box on CPU.

**For GPU acceleration, install the matching `onnxruntime` variant:**

| GPU | Install | Source |
|-----|---------|-------|
| AMD (ROCm) | `uv pip install onnxruntime-rocm` | PyPI (official) |
| NVIDIA (CUDA) | `uv pip install onnxruntime-gpu` | PyPI (official) |
| CPU | `uv pip install onnxruntime` | PyPI (official) |

> **Note:** On Arch Linux, use `sudo pacman -S python-onnxruntime-opt-rocm` for AMD or `python-onnxruntime-cuda` for NVIDIA.

**Performance (steady-state, 7900 XTX):**

| Model | PyTorch CPU (batch=100) | ONNX GPU (batch=100) | Speedup |
|-------|----------------------:|---------------------:|-------:|
| all-MiniLM-L6-v2 | 520ms | 8.3ms | 63× |
| snowflake-arctic-embed-xs | 612ms | 8.3ms | 74× |
| intfloat/e5-small-v2 | 1212ms | 13.4ms | 90× |
| Single query (any) | ~5ms | ~1ms | 5× |

If ONNX fails for any reason, the tool silently falls back to sentence-transformers.

To force the fallback:
```bash
PF2E_PROVIDER=sentence_transformers pf2e-codex index
```

To force ONNX (fail if unavailable):
```bash
PF2E_PROVIDER=onnx pf2e-codex index
```

**Provider priority:** ROCm → CUDA → CPU. ZLUDA (CUDA-on-AMD emulation) is correctly deprioritized — native ROCm is preferred.

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
PF2E_MODEL="snowflake-arctic-embed-s" pf2e-codex index --rebuild
```

Or in your config file:
```toml
model = "snowflake-arctic-embed-m"
```

## SDK

```python
from pf2e_codex.config import get_settings
from pf2e_codex.index import SearchIndex

settings = get_settings(db="pf2e_v2.db")
search = SearchIndex(settings.db, settings.model)

results = search.search("flat-footed while flanking", top_k=3)
for r in results:
    print(r["name"], r["distance"])
```

## Usage Examples

```bash
# Quick start: build your local rules database
pf2e-codex index

# Search (hybrid semantic + name match)
pf2e-codex search "fireball" --remaster-only -k 5

# Embed all supported models (shared chunk phase, concurrency-aware)
pf2e-codex embed --all-models

# Update all DBs to latest PF2E release
pf2e-codex embed --all-models -u --latest

# Look up an entry
pf2e-codex get fury-instinct

# Start MCP server (for Claude Desktop, Cursor, pi)
pf2e-codex serve
```

## Legal

- Code: MIT (this repo)
- PF2E game content: ORC / OGL (not redistributed; user fetches from official releases)
- No pre-computed embeddings or PF2E data shipped
