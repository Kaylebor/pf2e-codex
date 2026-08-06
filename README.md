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
pf2e-codex mcp
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
│   ├── index.py           # sqlite-vec storage + semantic/FTS5 search
│   ├── pipeline.py        # Orchestration: fetch → chunk → embed → index/update
│   ├── model_manager.py   # Process-scoped embedding/reranker lifecycle
│   ├── daemon_proxy.py    # CLI proxy for a running MCP daemon
│   ├── mcp_server.py      # MCPServer and read-only SQL tool
│   └── cli.py             # pf2e-codex CLI
├── scripts/               # Benchmark and data-maintenance tools
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
| `pf2e-codex validator` | Validate search quality against 25-query suite |
| `pf2e-codex warmup` | Manually precompile models into the current user's cache |
| `pf2e-codex pull` | Download pre-built embedding DB from GitHub Releases |
| `pf2e-codex mcp` | Start MCP server (--transport streamable-http for daemon) |

## Daemon (systemd)

For persistent GPU inference, run the MCP server as a systemd user service.
The daemon loads both models synchronously before accepting requests and
auto-downloads the embedding DB on first query.

The Streamable HTTP endpoint uses MCP Python SDK 2 and accepts both modern
stateless MCP requests and legacy stateful sessions. The built-in CLI proxy
currently uses the legacy stateful path.

```bash
# Enable and start
systemctl --user enable --now pf2e-codex

# Check status
systemctl --user status pf2e-codex

# CLI commands auto-detect the daemon and proxy to it
pf2e-codex search "fireball"
```

On first start, the daemon downloads the embedding DB (~97MB) and compiles
ONNX models for MIGraphX GPU. Subsequent starts use cached `.mxr` files and
avoid recompilation.

`pf2e-codex warmup` can populate that cache for the current user ahead of a
service restart. Package installation must not run it as root, because the
result would be written to root's cache rather than the daemon user's cache.

### First-query auto-download

When the daemon receives its first search and no embedding DB exists,
it auto-downloads a pre-computed sqlite-vec DB from GitHub Releases.
To rebuild from scratch instead:

```bash
pf2e-codex embed
```

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
release = "pf2e-8.2.0"
```

Or use a project-local `pf2e-codex.toml` (gitignored by default).

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PF2E_CACHE_DIR` | `~/.cache/pf2e-codex` | Download cache |
| `PF2E_DATA_DIR` | `~/.local/share/pf2e-codex` | Data directory |
| `PF2E_MODEL` | `Snowflake/snowflake-arctic-embed-xs` | Embedding model |
| `PF2E_PROVIDER` | `auto` | Embedding provider selection |
| `PF2E_QUERY_PROVIDER` | `cpu` | ONNX provider for daemon queries |
| `PF2E_RELEASE` | `pf2e-8.2.0` | PF2E system version |

## MCP Tools

| Tool | Description |
|---|---|
| `pf2e_search(query, top_k)` | Semantic + sentence-tolerant FTS5 search with named-entry anchoring, filters, reranking, and references |
| `pf2e_flag_result(result_index, note)` | Record an incorrect or low-quality result |
| `pf2e_get_entry(entry_id)` | Fetch full entry by ID, slug, name, or Foundry UUID |
| `pf2e_related(entry_id, direction, limit)` | Cross-reference graph: outgoing/incoming/both |
| `pf2e_rules_explain(topic, top_k)` | Prioritized search favoring journal pages and conditions |
| `pf2e_catalog()` | Show type, license, remaster, and pack counts |
| `pf2e_index_status()` | Show model, chunk count, and release metadata |
| `pf2e_query_db(sql, limit, query_text)` | Execute bounded, read-only SELECT queries |

## ONNX Acceleration (automatic)

The tool proactively tries ONNX Runtime for faster inference. It works out of the box on CPU.

**For GPU acceleration, install the matching `onnxruntime` variant:**

| GPU | Install | Source |
|-----|---------|-------|
| AMD (ROCm) | `pip install onnxruntime-migraphx -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/` | AMD official repo |
| NVIDIA (CUDA) | `pip install onnxruntime-gpu` | PyPI (official) |
| CPU | `pip install onnxruntime` | PyPI (official) |

> **Note:** On Arch Linux, use `sudo pacman -S python-onnxruntime-opt-rocm` for AMD or `python-onnxruntime-cuda` for NVIDIA.

**Performance (steady-state, 7900 XTX):**

| Model | PyTorch CPU (batch=100) | ONNX GPU (batch=100) | Speedup |
|-------|----------------------:|---------------------:|-------:|
| all-MiniLM-L6-v2 | 520ms | 8.3ms | 63× |
| snowflake-arctic-embed-xs | 612ms | 8.3ms | 74× |
| intfloat/e5-small-v2 | 1212ms | 13.4ms | 90× |
| Single query (any) | ~5ms | ~1ms | 5× |

ONNX Runtime is the supported inference backend; missing or broken ONNX
providers produce an explicit startup/indexing error.

To force ONNX (fail if unavailable):
```bash
PF2E_PROVIDER=onnx pf2e-codex index
```

**Provider priority:** ROCm → CUDA → CPU. ZLUDA (CUDA-on-AMD emulation) is correctly deprioritized — native ROCm is preferred.

## Multilingual Search

For Spanish (and other languages), pf2e-codex supports fetching a community-maintained
Babele translation module at index time.

### Configuration

```toml
# Enables Spanish translations at index time
languages = ["en", "es"]

# Use multilingual embedding model (supports 100+ languages)
model = "intfloat/multilingual-e5-small"
reranker_model = "Kaylebor/pf2e-codex-reranker-minilm"
```

When `languages` includes non-English languages, `pf2e-codex build` or `pf2e-codex index`
will download the corresponding translation module and merge translations into chunks.
Each chunk gains a `translations` field with per-language name/text.

### Searching in a language

```bash
# Return results with Spanish names/descriptions
pf2e-codex search "bola de fuego" --lang es

# Keep English as default
pf2e-codex search "fireball"
```

If `--lang es` is set and no Spanish translation exists for an entry, the English text
is shown as fallback.

### Candidate models (under evaluation)

| Model | Type | Size | Languages | Status |
|-------|------|------|-----------|--------|
| `intfloat/multilingual-e5-small` | Embedding | 118M, 384d | 100+ | Recommended |
| `Kaylebor/pf2e-codex-reranker` | Cross-encoder reranker | 2.2GB | 100+ | Works but large |
| `Kaylebor/pf2e-codex-reranker-minilm` | Cross-encoder reranker | 88MB | EN only | Default |

For a lightweight multilingual setup, use the e5-small embedding (handles cross-lingual
retrieval natively) without a reranker. For optimal quality, use the 2.2GB reranker
(compile once, cached). Future work includes quantizing the 2.2GB model to ~1.1GB.

The tool proactively tries ONNX Runtime for faster inference. It works out of the box on CPU.

**For GPU acceleration, install the matching `onnxruntime` variant:**

| GPU | Install | Source |
|-----|---------|-------|
| AMD (ROCm) | `pip install onnxruntime-migraphx -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/` | AMD official repo |
| NVIDIA (CUDA) | `pip install onnxruntime-gpu` | PyPI (official) |
| CPU | `pip install onnxruntime` | PyPI (official) |

> **Note:** On Arch Linux, use `sudo pacman -S python-onnxruntime-opt-rocm` for AMD or `python-onnxruntime-cuda` for NVIDIA.

**Performance (steady-state, 7900 XTX):**

| Model | PyTorch CPU (batch=100) | ONNX GPU (batch=100) | Speedup |
|-------|----------------------:|---------------------:|-------:|
| all-MiniLM-L6-v2 | 520ms | 8.3ms | 63× |
| snowflake-arctic-embed-xs | 612ms | 8.3ms | 74× |
| intfloat/e5-small-v2 | 1212ms | 13.4ms | 90× |
| Single query (any) | ~5ms | ~1ms | 5× |

ONNX Runtime is the supported inference backend; missing or broken ONNX
providers produce an explicit startup/indexing error.

To force ONNX (fail if unavailable):
```bash
PF2E_PROVIDER=onnx pf2e-codex index
```

**Provider priority:** ROCm → CUDA → CPU. ZLUDA (CUDA-on-AMD emulation) is correctly deprioritized — native ROCm is preferred.

## Embedding Models

All models work out of the box. Query/document prefixing is handled automatically.
Reranking is enabled by default and uses a fine-tuned MiniLM cross-encoder
(88MB, Kaylebor/pf2e-codex-reranker-minilm) that significantly boosts quality.

### Search quality (with fine-tuned MiniLM reranker, MRR metric)

Tested on AMD Ryzen 7 7800X3D + Radeon RX 7900 XTX, 25-query PF2E validation suite:

| Model | Params | Dim | DB Size | MRR | Perfect |
|-------|--------|-----|---------|-----|---------|
| `all-MiniLM-L6-v2` | 22M | 384 | 97MB | **0.980** | 24/25 |
| `snowflake-arctic-embed-xs` | 22M | 384 | 97MB | 0.960 | 24/25 |
| `snowflake-arctic-embed-s` | 33M | 384 | 97MB | 0.960 | 23/25 |
| `snowflake-arctic-embed-m` | 110M | 768 | 141MB | 0.960 | 23/25 |
| `BAAI/bge-m3` | 568M | 1024 | 170MB | 0.920 | 23/25 |
| `intfloat/e5-small-v2` | 33M | 384 | 97MB | 0.880 | 22/25 |

### CPU indexing performance

Tested on AMD Ryzen 7 7800X3D, indexing 28,837 chunks:

| Model | Index Time |
|-------|----------|
| `snowflake-arctic-embed-xs` | ~35s |
| `all-MiniLM-L6-v2` | ~50s |
| `snowflake-arctic-embed-s` | ~70s |
| `intfloat/e5-small-v2` | ~135s |
| `snowflake-arctic-embed-m` | ~1h |
| `BAAI/bge-m3` | ~3h+ |

**Default: `snowflake-arctic-embed-m`** — best balance of quality, size, and GPU performance.

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
pf2e-codex mcp

# MCP over HTTP (remote clients)
pf2e-codex mcp -t streamable-http --host 0.0.0.0 --port 8080
```

## Legal

- Code: MIT (this repo)
- PF2E game content: ORC / OGL (not redistributed; user fetches from official releases)
- No pre-computed embeddings or PF2E data shipped
