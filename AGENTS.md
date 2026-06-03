# Agent Guide — pf2e-codex

This document is for LLMs, AI agents, and human collaborators who need to work on this codebase.

## What This Is

A standalone tool for Pathfinder 2E rules lookup. It downloads PF2E data from the official FoundryVTT system releases, chunks and enriches it, embeds it, and stores it in a local sqlite-vec database. Then it exposes that database via:
- **MCP server** (`stdio`/`sse`) — for Claude Desktop, Cursor, pi, etc.
- **CLI** (`pf2e-codex`) — for indexing, searching, configuring
- **Python SDK** (`from pf2e_codex...`) — for programmatic use

No pre-computed embeddings or PF2E data is shipped. Users run `pf2e-codex index` once to build their local DB.

## Architecture

```
pf2e_codex/
├── config.py       # Settings: env vars → TOML file → defaults (Pydantic)
├── fetcher.py      # Download json-assets.zip from GitHub releases
├── chunker.py      # Parse PF2E JSON → enriched text chunks
├── models.py       # Embedding model registry + hardware recommendations
├── embeddings.py   # Provider abstraction (sentence-transformers, ONNX, remote)
├── index.py        # sqlite-vec DB init + SearchIndex class
├── pipeline.py     # Orchestration: fetch → extract → chunk → embed → index
├── mcp_server.py   # FastMCP server with 4 tools
└── cli.py          # Typer CLI entry point
```

### Key Flow

```
GitHub release → json-assets.zip
       ↓
extract_all_packs() → {pack_name: [entries]}
       ↓
ChunkBuilder.build_all(entry, pack_name) → [{id, name, type, pack, text, ...}]
       ↓
embed_chunks() → embeddings
       ↓
sqlite-vec (vec0 virtual table) + chunks table
       ↓
SearchIndex.search(query) → top-k results
```

## Design Decisions

### One entry = one chunk
Each Foundry pack entry becomes one text chunk. Journal entries are split into per-page chunks (essential for core rules retrieval).

### Rules-aware flattening
`system.rules` arrays are flattened into plain English via 30+ rule-specific flatteners. Examples:
- `FlatModifier {value: -2, type: "circumstance", selector: "ac"}` → "-2 circumstance to ac"
- `GrantItem {uuid: "Compendium...", level: 6}` → "Grants (at level 6) item: Sneak Attack"

### Expression simplification
Ternaries like `ternary(gte(@actor.level,13),4,2)` are simplified to "+4 at level 13+ (else 2)". Variables like `@actor.level` map to "character level".

### UUID cross-reference resolution
`@UUID[Compendium.pf2e.feats.Item.ABC123]{Power Attack}` is resolved to "Power Attack" during chunking so embeddings capture relationships.

### Pack-prefixed IDs
Chunk IDs are `pack:entry_id` (e.g. `feats:Fury-Instinct`) because the same `_id` appears across multiple bestiary packs.

### Default model: `snowflake-arctic-embed-xs`
22M params, 384 dims, ~35s to index 28K chunks on Ryzen 7 7800X3D. Faster and equivalent quality to `all-MiniLM-L6-v2`.

### OGL vs ORC vs pre/post remaster
These are orthogonal — ORC predates the Remaster:
- **License** (`chunks.license`): OGL (older) vs ORC (newer). Legal distinction.
  - ORC does NOT mean remaster! Some ORC content predates the Remaster.
  - OGL content can be remaster (619 renamed entries: Force Barrage, etc.).
- **Remaster** (`publication.remaster`): True/False/None. Mechanical distinction.
  - `remaster=True` = current rules (what users want 90% of the time).
  - `remaster=False` = legacy rules.
- **Data split**: ORC+remaster=13,366 | OGL+remaster=False=6,446 | OGL+remaster=True=619.
- **Default behavior**: MCP tools guide LLM to default to remaster=True content.

### Search enrichment
Results include: `refs` (outgoing cross-references), `legacy_name` (pre-remaster
name for renamed entries), `confidence` (high/medium/low from score thresholds),
`license` (ORC/OGL/NONE).

## Gotchas

### sqlite-vec `vec0` syntax
The virtual table uses `float[dim]` syntax, not `vec_float32(dim)`. Embeddings are inserted via `vec_f32(blob)`.

### sqlite-vec `MATCH` requires `k` constraint
```python
# Correct:
WHERE embedding MATCH vec_f32(?) AND k = ?

# WRONG (returns nothing):
WHERE embedding MATCH vec_f32(?)
```

### Model prefixing is automatic
The `SentenceTransformersProvider` handles query/document prefixes via `models.py` registry. Don't add prefixes manually.

### Config priority (highest wins)
1. CLI kwargs / function args
2. Environment variables (`PF2E_MODEL`, etc.)
3. Config file (`~/.config/pf2e-codex/config.toml` or `./pf2e-codex.toml`)
4. Class defaults

### Git-ignored data files
```
pf2e-codex.toml    # local config
*.db               # sqlite-vec databases
chunks*.json       # intermediate chunk files
```

## Testing / Validation

Quick smoke test after changes:
```bash
uv pip install -e "."
pf2e-codex status                     # should show 28,837 chunks
pf2e-codex search "flat-footed" -k 3   # hybrid search, should return Darting Monkey
pf2e-codex get "fury-instinct"         # should return full Fury Instinct entry
pf2e-codex related "off-guard" --direction incoming  # should show feats referencing Off-Guard
pf2e-codex models                      # should list all models
```

MCP server test:
```bash
# In one terminal
pf2e-codex serve

# In another (send JSON-RPC init + tools/list)
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | pf2e-codex serve
```

## Data Source

- **URL**: `https://github.com/foundryvtt/pf2e/releases/download/{version}/json-assets.zip`
- **Contents**: All compendium packs as JSON + `lang/en.json`
- **Size**: ~34MB zip, ~28,837 chunks after processing
- **License**: Foundry code Apache 2.0; PF2E content ORC/OGL

## Common Modification Patterns

### Adding a new embedding model
1. Add `ModelInfo` to `models.py` with correct `query_prefix`/`doc_prefix`
2. It works automatically — `get_provider()` uses the registry

### Adding a new CLI command
1. Add function to `cli.py` with `@app.command()`
2. Use `get_settings(**kwargs)` for config resolution

### Adding a new MCP tool
1. Add `@mcp.tool()` function in `mcp_server.py`
2. Use `search.search()` / `search.fetch_by_id()` / `search.rules_explain()`

### Adding a new rule flattener
1. Define function in `chunker.py` taking `rule: dict`
2. Register in `_RULE_FLATTENERS` dict
3. Return plain English string (or `None` to skip)

## ONNX Acceleration (dev setup)

ONNX is auto-detected. Install in order of preference:

```bash
# CPU (always works)
uv pip install 'optimum[onnxruntime]'

# AMD ROCm 7.x (MIGraphX EP, onnxruntime 1.23+)
uv pip install optimum
uv pip install https://github.com/Looong01/onnxruntime-rocm-build/releases/download/v1.25.0/onnxruntime_migraphx-1.25.0-cp313-cp313-manylinux_2_34_x86_64.whl
# Also need MIGraphX system lib: yay -S migraphx (AUR) or equivalent

# AMD ROCm 6.x (ROCm EP, onnxruntime ≤ 1.22)
uv pip install -e ".[rocm]"

# NVIDIA CUDA
uv pip install -e ".[cuda]"
```

On first use per model, ONNX exports once (cached at `~/.cache/pf2e-codex/onnx/{model}/`).
Subsequent loads skip export.
MIGraphX also compiles the model to GPU kernels on first inference (~10-30s per batch shape).
After compile, steady-state throughput is 50-500× faster than PyTorch CPU.

**Key provider order:** MIGraphX → ROCm → CUDA → CPU
**Per-batch-shape compile:** MIGraphX compiles once per unique batch size. For a running
MCP server this happens once at startup.

**Test ONNX is working:**
```bash
.venv/bin/python -c "
from pf2e_codex.embeddings import _has_onnx, _detect_onnx_provider
print('ONNX:', _has_onnx(), 'Provider:', _detect_onnx_provider())
"
```

**Force fallback:**
```bash
PF2E_PROVIDER=sentence_transformers pf2e-codex index
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `sentence-transformers` | Local embedding models |
| `optimum[onnxruntime]` | ONNX model export (optional) |
| `onnxruntime[-rocm,-gpu]` | ONNX runtime (optional) |
| `sqlite-vec` | Vector storage + similarity search |
| `mcp` | FastMCP server |
| `pydantic` + `pydantic-settings` | Config + validation |
| `typer` | CLI framework |

Python 3.12+, uv recommended.
