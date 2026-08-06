# Agent Guide — pf2e-codex

> **Update this file** whenever you change architecture, add/remove modules, switch deps, or alter design decisions. Keep ROADMAP.md and README.md in sync too. Stale docs are worse than no docs.

## Hard Lessons (don't repeat these)

### Thread safety: always lock, never check-then-act outside lock
Python threads can see stale values without synchronization. Every lazy-init pattern must
use `with self._lock:` wrapping BOTH the check AND the assignment. A double-checked locking
pattern where the first check is outside the lock is still broken — the first check can see
a stale cached value from another thread's write.

```python
# BROKEN — first check outside lock can see stale None
if self._provider is None:
    with self._lock:
        if self._provider is None:
            self._provider = get_provider(...)
return self._provider

# CORRECT — lock acquired before first check
with self._lock:
    if self._provider is None:
        self._provider = get_provider(...)
    return self._provider
```

Applies to: `_provider`, `_reranker`, `_conn` (connection setup must be INSIDE the lock).

### Background warmup removed — models load before the server starts

`ModelManager.start()` is synchronous and blocks until both models are
loaded. The HTTP server does not start listening until `start()` returns,
so by the time any request arrives, models are ready.

On CPU this takes ~5s; on GPU (cached .mxr) ~10-15s.
No background thread, no gate, no race conditions.

Models are process-scoped — held by ModelManager, referenced by
SearchIndex, kept alive for the daemon's lifetime.

### CLI must never fall back to local inference when daemon exists
If the daemon is registered (server.json exists), the CLI must NEVER create its own
SearchIndex. Doing so triggers a separate MIGraphX compile that competes with the
daemon's warmup, causing crashes or OOM.

```python
# Check: if daemon registered, don't fall back
from .daemon_proxy import _server_json_path
if _server_json_path().exists():
    typer.echo("Daemon is registered but not responding.", err=True)
    return
```

### MCP 2 serves both protocol eras

The Streamable HTTP server uses `MCPServer` from MCP Python SDK 2. Its default
`stateless_http=False` must remain unchanged while `daemon_proxy.py` uses the
legacy initialize/session flow. MCP 2 still routes modern protocol requests
statelessly on the same endpoint. Setting `stateless_http=True` would also force
legacy clients into stateless handling and requires separate proxy validation.

### CLI fallback SearchIndex must use config values
When the CLI runs without a daemon, all SearchIndex constructors must receive
`settings.reranker_model` — otherwise the reranker defaults to the untuned base model
(empty repo name).

### Always validate before blaming dependencies
When a crash trace points into a dependency (MIGraphX, PyTorch, etc.), assume our code
is at fault first. Validate with a controlled test before reporting upstream.

### Logging for systemd
Systemd services capture stderr reliably. `print(flush=True)` to stdout may not reach
the journal depending on the service type. Use `sys.stderr.write()` for daemon logs.

### System cache dir is a development artifact
Only the user cache at `~/.cache/pf2e-codex/onnx/migraphx_cache/` should exist. System
dirs like `/usr/share/pf2e-codex/migraphx_cache/` are stale PKGBUILD artifacts and
must be removed.

### Root warmup is wasted
During PKGBUILD install, `post_install` runs as root. `Path.home()` resolves to
`/root/`, so warmup writes .mxr to `/root/.cache/` which the user's daemon never sees.
Warmup should be removed from PKGBUILD entirely; it happens on first daemon start.

## What This Is

A standalone tool for Pathfinder 2E rules lookup. It downloads PF2E data from the official FoundryVTT system releases, chunks and enriches it, embeds it, and stores it in a local sqlite-vec database. Then it exposes that database via:
- **MCP server** (`stdio`/`sse`) — for Claude Desktop, Cursor, pi, etc.
- **CLI** (`pf2e-codex`) — for indexing, searching, configuring
- **Python SDK** (`from pf2e_codex...`) — for programmatic use

No pre-computed embeddings or PF2E data is shipped. Users run `pf2e-codex index` once to build their local DB.

## Packaging (PKGBUILD)

All Python deps bundled via `pip install --target` into `/usr/share/pf2e-codex/lib/`.
Torch is pinned to CPU-only via pip constraint file to avoid NVIDIA bloat (~1.3GB total).
No system package dependencies beyond `python` and `patchelf`.

### ONNX: NEVER install CPU onnxruntime

The CPU onnxruntime variant is NEVER installed in the PKGBUILD. This is intentional and
mandatory — it causes file collisions (cp314-suffixed `.so` wins over GPU variant) and wastes
build time.

The correct pattern:
```bash
# 1. Install pf2e-codex WITHOUT transitive deps (--no-deps)
/usr/bin/pip3 install --no-cache-dir --no-deps --target "$lib" "$startdir"

# 2. Install non-onnxruntime deps from pyproject.toml
/usr/bin/pip3 install --no-cache-dir --target "$lib" \
    --constraint /tmp/pf2e-torch-constraint.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    $(python3 -c "import tomllib, json; d=tomllib.load(open('$startdir/pyproject.toml','rb'))['project']['dependencies']; print(' '.join(json.load(open('/tmp/pf2e-non-ort-deps.json'))))")

# 3. Install GPU-specific onnxruntime variant
if [ amd ]; then
    pip install onnxruntime-migraphx -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/
elif [ nvidia ]; then
    pip install onnxruntime-gpu
else
    pip install onnxruntime
fi
```

Why `--no-deps`? Because `pf2e-codex` has `onnxruntime>=1.20` as a dependency. Without
`--no-deps`, pip installs CPU onnxruntime as a transitive dep, then the GPU variant
installs on top of it — but the cp314-suffixed `.so` files from the CPU variant remain
and Python loads those instead of the GPU ones.

If the PKGBUILD ever starts failing with "onnxruntime not installed" or
"MIGraphX not available", check that CPU onnxruntime was NOT pulled in by the
deps install step.

## Architecture

```
pf2e_codex/
├── config.py       # Settings: env vars → TOML file → defaults (Pydantic)
├── fetcher.py      # Download json-assets.zip from GitHub releases
├── chunker.py      # Parse PF2E JSON → enriched text chunks, UUID resolution, OGL→ORC aliases
├── models.py       # Embedding model registry + hardware recommendations
├── embeddings.py   # ONNX-only provider (automatic export via optimum, inference via onnxruntime)
├── index.py        # sqlite-vec DB init + SearchIndex class (semantic + FTS5 hybrid search)
├── pipeline.py     # Orchestration: fetch → extract → chunk → embed → index (+ incremental update)
├── benchmark.py    # Cross-model embedding speed benchmarks
├── cli_rich.py     # Rich table formatting for CLI output
├── validate.py     # Retrieval quality validation suite (25 queries, MRR)
├── mcp_server.py   # MCPServer with 8 tools (search, fetch, related, SQL, etc.)
└── cli.py          # Typer CLI entry point (fetch, index, search, serve, export, etc.)
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

### Search and update safety
`SearchIndex` initializes its writable and read-only SQLite connections under one lock;
all lazy connection checks happen while holding that lock. The query connection uses
`mode=ro`, and MCP SQL queries add a SQLite authorizer plus a VM-step/deadline budget
that rejects writes, attachment, pragma changes, extension loading, and runaway reads.

Incremental updates group all journal pages by entry before replacing them, process
entry deletions even when there are no changed entries, rebuild the external-content FTS5
index after data mutations, and commit the complete update atomically.

Legacy references store bare Foundry IDs, which can collide across packs. The
`ambiguous_ref_targets` table remembers those collisions across incremental updates so a
deleted duplicate's references are never reassigned to the surviving entry. Full rebuilds
clear and recompute the table from their single source snapshot.

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

Hybrid retrieval tokenizes natural-language questions, removes common question
filler, and uses OR across the remaining FTS5 terms. Semantic and lexical RRF
weights are balanced. A separate name-only pass detects complete entry names in
the query; those entries are promoted before and after reranking so a question
that explicitly names Fireball cannot lose the Fireball entry.

## Training Data Pipeline

Training data for the cross-encoder reranker is generated by pi subagents reading
raw pack JSONs. Each agent generates ~100 `(query, pos, neg)` triplets from one pack.

### Subagent: `triplet-gen`

A restricted agent (`~/.agents/triplet-gen.md`) with tools `read, grep, find, ls, write` only.
No `bash`, no `subagent` — agents that can run code get stuck in infinite loops.
The prompt tells them to read the JSON, pick entry pairs with hard negatives, and write
JSONL using the write tool.

### Workflow

```bash
# Generate triplets (subagents write to training_data/raw/)
# ...

# Merge, validate, deduplicate
python3 scripts/merge-training-data.py
# Clean raw files consumed, appends to training_data/dataset.jsonl

# (Optional) Round-trip validation — flags triplets where reranker scores neg >= pos
python3 scripts/merge-training-data.py --validate
```

### Output

`training_data/dataset.jsonl` — one JSON object per line:
```json
{"query": "What does the Blinded condition do?", "pos": "...", "neg": "..."}
```

### Quality notes

- Agents generate genuinely confusable negatives (Hide↔Sneak, Blinded↔Dazzled, Heal↔Harm, same-class feats)
- 1 invalid entry per ~1500 is normal (empty neg field, flagged in `.errors` file)
- Merge script auto-deletes 100%-valid raw files
- Feat/spell packs yield the richest triplets (high variety, many confusable entries)
- Bestiary packs are lower value (stat blocks, less diverse queries)

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
The `ONNXProvider` handles query/document prefixes via `models.py` registry. Don't add prefixes manually.

### Protobuf 34 is bundled for MIGraphX
On systems with protobuf 35+ (CachyOS 2026+), `onnxruntime-migraphx` needs `libprotobuf.so.34`
which is not in the system. The PKGBUILD downloads `protobuf-34.1-1-x86_64.pkg.tar.zst` from
the Arch archive and extracts the `.so` files into `onnxruntime/capi/`.
At runtime, `pf2e_codex/_preload_onnx.py` pre-loads these with `ctypes.CDLL(RTLD_GLOBAL)`
before any onnxruntime import, making them visible to transitive deps.

If onnxruntime shows "MIGraphX not available", verify `libprotobuf.so.34.1.0` exists in
`/usr/share/pf2e-codex/lib/onnxruntime/capi/`.

### MIGraphX compiled-model caching (automatic directory cache)

The old per-model `migraphx_save_compiled_*` / `migraphx_load_compiled_*` options were removed
in ROCm 6.4. They were replaced by a single **automatic directory cache**:
- **Provider option**: `migraphx_model_cache_dir`
- **Environment variable**: `ORT_MIGRAPHX_MODEL_CACHE_PATH`

When set, MIGraphX automatically caches compiled `.mxr` files in that directory with filenames
based on model hash + GPU arch + input shapes. On subsequent loads, it skips compilation.

Our code sets this to `~/.cache/pf2e-codex/onnx/migraphx_cache/`.

Source: `onnxruntime/core/providers/migraphx/migraphx_execution_provider.cc` (v1.25.0).
Docs are outdated — see https://github.com/microsoft/onnxruntime/issues/25379.

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

Quick smoke test after changes (requires package installed via PKGBUILD):
```bash
pf2e-codex status                     # should show 28,837 chunks
pf2e-codex search "flat-footed" -k 3   # hybrid search, should return Darting Monkey
pf2e-codex get "fury-instinct"         # should return full Fury Instinct entry
pf2e-codex related "off-guard" --direction incoming  # should show feats referencing Off-Guard
pf2e-codex models                      # should list all models
```

MCP server test:
```bash
# stdio (Claude Desktop, pi, Cursor)
pf2e-codex mcp

# streamable-http (remote clients)
pf2e-codex mcp -t streamable-http --host 0.0.0.0 --port 8080

# SSE
pf2e-codex mcp -t sse
```

Test via stdio:
```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"pf2e-codex-smoke","version":"1.0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' |
  pf2e-codex mcp
```

Test modern stateless MCP via HTTP:
```bash
curl -sS -X POST http://127.0.0.1:8080/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'MCP-Protocol-Version: 2026-07-28' \
  -H 'Mcp-Method: tools/list' \
  --data '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {
      "_meta": {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {
          "name": "pf2e-codex-smoke",
          "version": "1.0"
        }
      }
    }
  }'
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

ONNX is auto-detected at runtime. MIGraphX is first priority on AMD.

For local development (outside PKGBUILD):

```bash
# CPU (default, bundled by PKGBUILD when no GPU detected)
pip install onnxruntime

# AMD GPU (official AMD repo, ROCm 7.2+)
pip install onnxruntime-migraphx -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/

# NVIDIA GPU
pip install onnxruntime-gpu
```

The PKGBUILD auto-detects your GPU at build time and installs only the relevant variant.

On first use per model, ONNX exports once (cached at `~/.cache/pf2e-codex/onnx/{model}/`).
Subsequent loads skip export.
MIGraphX also compiles the model to GPU kernels on first inference (~10-30s per batch shape).
After compile, steady-state throughput is 50-500× faster than PyTorch CPU.

**Key provider order:** MIGraphX → ROCm → CUDA → CPU
**Per-batch-shape compile:** MIGraphX compiles once per unique batch size. For a running
MCP server this happens once at startup.

## Common Commands

```bash
# Check status & discover DB
pf2e-codex status

# Build everything from scratch
pf2e-codex index

# Embed all supported models (shared chunk phase)
pf2e-codex embed --all-models

# Update all DBs to the latest PF2E release
pf2e-codex embed --all-models -u --latest

# Incremental update after FoundryVTT PF2E module upgrades
pf2e-codex embed --all-models -u

# Specific model
pf2e-codex index -m BAAI/bge-m3

# Search (hybrid: semantic + name match)
pf2e-codex search "flat-footed" -k 5
pf2e-codex search "fireball" --license ORC --remaster-only

# Look up a specific entry
pf2e-codex get fury-instinct
pf2e-codex get "Compendium.pf2e.feats.Item.ABC123"

# Find related entries (cross-references)
pf2e-codex related off-guard --direction incoming

# Start MCP server (stdio, for Claude/Cursor/pi)
pf2e-codex mcp

# Start MCP server (streamable-http, auto-pick port)
pf2e-codex mcp -t streamable-http

# List all supported models
pf2e-codex models

# Validate retrieval quality
pf2e-codex validate
pf2e-codex validate --mode semantic

# Benchmark model speed
pf2e-codex benchmark
pf2e-codex benchmark --models "all-MiniLM-L6-v2,bge-m3" --providers "cpu"
```

### Daemon proxy (auto-detection)

CLI query commands (search, get, related, status, catalog) auto-detect a running
MCP server and proxy queries to it. This avoids MIGraphX compilation overhead (~7s)
for each CLI invocation.

```bash
# Start server in background (auto-picks port, writes server.json)
pf2e-codex mcp -t streamable-http &

# CLI queries now proxy to the server (~0.1s instead of ~7s)
pf2e-codex search fireball
pf2e-codex get fury-instinct
```

SystemD user service (optional):
```bash
systemctl --user enable pf2e-codex
systemctl --user start pf2e-codex
```

The server writes `~/.local/share/pf2e-codex/server.json` with the endpoint.
CLI reads this file to detect the server. If the registered server is not
responsive, query commands report the failure instead of starting local inference.

## Dependencies

| Package | Purpose |
|---------|---------|
| `optimum[onnxruntime]` | ONNX model export + runtime inference (bundled in package) |
| `sqlite-vec` | Vector storage + similarity search |
| `mcp` | MCP Python SDK 2 server |
| `pydantic` + `pydantic-settings` | Config + validation |
| `typer` + `rich` | CLI framework + formatting |

All Python deps are bundled in the PKGBUILD via `pip install --target`.
The only system dependency is `python`.
