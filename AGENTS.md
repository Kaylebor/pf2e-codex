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

### GPU is the primary inference path; never silently fall back

Both indexing and daemon queries default to automatic ONNX provider selection:
MIGraphX, ROCm, CUDA, then CPU. If Linux DRM reports supported AMD or NVIDIA
hardware but ONNX Runtime exposes only CPU, startup must fail with an actionable
error. CPU remains valid only on a machine without supported GPU hardware or
when the user explicitly selects `cpu`.

Optimum requires Torch for ONNX export, so `uv` must keep Torch pinned to the
explicit `pytorch-cpu` index in `pyproject.toml`. Do not remove that source or
let PyPI resolve a CUDA-flavoured Torch wheel: Torch is not the inference
backend and those libraries break AMD development environments. `make
setup-dev` auto-detects AMD/NVIDIA and installs exactly one matching ONNX
Runtime; `PF2E_DEV_ACCELERATOR` is the explicit override.

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

### Local rulebook PDFs use native text, never OCR

The purchased rulebook PDFs are born-digital and have a reliable selectable
text layer. `pf2e_codex/pdf_export.py` is the single PDF extraction path and
uses the optional `pdfplumber` dependency. It streams versioned JSON containing
native words, font metadata, coordinates, image bounds, and the source hash.

Do not use Docling, Marker, OCR, Markdown conversion, or a `pdftotext`
subprocess in the product pipeline. Those either degraded correct source text
or would make exploration differ from the shipped extractor. PDF export and
corpus parsing are intentionally separate: first generate the raw JSON, then
design and test Paizo-specific parsing against that exact artifact.

The licensed-review path may use `pdf_layout.py` to rasterize pages for
PP-DocLayoutV3, but only as structural evidence. Its ONNX runtime emits region
boxes and reading order, never OCR text; bindings contain opaque native anchors.
Layout evidence may add review/stitch flags but must not alter source text or
the complete native-word inventory. Automatic inference remains GPU-first and
must refuse an unavailable requested provider or accidental CPU-only runtime on
supported GPU hardware. The one-time Torch/Transformers exporter stays in the
isolated PEP 723 script and lockfile because its Transformers 5 requirements
conflict with the embedding exporter's current Optimum-ONNX constraint. Upgrade
that lock periodically, but retain the pinned checkpoint and require ONNX
validation plus provider/output-parity checks before accepting an upgrade.

Paizo source recognition belongs to that parser stage. Preserve original PDF
basenames such as `PZO12001E.pdf` and match a small explicit catalog of known
`PZO` product codes and split-PDF filename patterns. Do not identify products
by a fixed SHA-256: purchased PDFs are personalized with an account watermark,
so hashes differ by customer. The exported source hash is local integrity and
provenance metadata only. Treat raw PDF text and exporter JSON as potentially
containing watermark PII: never commit or log extracted watermark values, and
strip any detected watermark text before indexing.

Persist parsed legacy and Remaster books in full, with source book, rules era,
license, and page provenance. Link overlapping or renamed rules for retrieval,
but do not discard one era as a duplicate. Default search may prefer Remaster
content while retaining legacy-only rules and explicit legacy lookup.

Revision selection compares normalized native-text fingerprints before using
mtime. Persisted state may stabilize differently packaged or watermarked copies
only inside the winning equivalent-content group; it must never pin an older
same-printing errata revision.

### Purchased corpus is private; reviewed licensed mechanics are a separate source

Seed scope defaults to `redistributable`, which never reads user-owned PDF
sources even when `.local-corpus/` exists. Clean seeds merge Foundry with the
tracked `licensed-core` SQLite projection; that projection contains only
independently approved functional mechanics plus sanitized product/reprinting,
page, license, parser, policy, and notice provenance. Its ignored review DB may
contain complete purchased text, claims, prompts, and AON corroboration, but
none of those private review fields enter the tracked projection.

The clean and private scopes use
different files for every model: `pf2e_<model>.db` and
`pf2e_<model>.local.db`. `local-full` is an explicit seed opt-in and must write
`distribution_scope=local-full` into the private DB metadata. Corpus sync can
only target that private slot. Query processes use `database_scope=auto`, which
prefers the private DB when present; MCP and validation accept `--db-scope` to
force one slot.

Pull, automatic download, and release tooling only write the clean slot. They
must stage downloads as siblings, audit before atomic activation, and reject a
DB whose ownership metadata does not match its physical slot. Never convert or
taint a clean DB in place. A default clean seed and a `local-full` seed may share
source inputs and model caches, but never their SQLite file.

Pre-built upload and pull paths must call the distribution audit. Publication
requires an explicit `redistributable` marker; rows may have `origin='foundry'`
or `origin='licensed-core'`, while `origin='corpus'` and unknown ownership are
always rejected. Every licensed-core row must match the embedded approved
section manifest, revision, content hash, policy version, and OGL/ORC notice.
Pull may accept legacy Foundry-only DBs that predate the marker. These guards
prevent known private-PDF leakage but do not replace legal review.

Clean Foundry ownership is necessary but not sufficient. New redistributable
seeds include only entries whose owning publication title is one of Pathfinder
Core Rulebook, Player Core, GM Core, Monster Core, or Player Core 2 and whose
entry-level license is OGL or ORC. Resolve ownership from `system.publication`,
falling back only to `system.details.publication`; never infer it from pack names
or recursively nested citations. Persist `publication_title` on every Foundry
chunk. Missing/blank titles and all other products fail closed in clean seeds,
while local-full retains the upstream-complete Foundry snapshot.

Strict publication audit must recompute the licensed-core IDs, content hashes,
pages/headings, revision/policy provenance, and notices and match that contract
to the packaged `licensed_core.sqlite3`. Hashes stored only inside the model DB
are not a trust anchor. The versioned selection contract lives in
`data/licensed_core_policy_v1.json`; public submissions require extraction
metadata and pass the structural email/path/watermark guard before review.

Complete parser runs are staged only through the trusted native-export bridge:
it recomputes a canonical, watermark-independent word inventory from the raw
export before segmentation, binds only opaque private anchors, and permits
only constrained repeated-margin-furniture or printed-page-number ignores.
Every other source word must occur in exactly one parser section. Do not add a
caller-supplied inventory/manifest CLI; it would let a parser certify a
truncated source. Anchors, paths, raw PDF hashes, and watermark data never
enter the public projection.

Use the quad-state draft screen before asking agents to reconstruct public
rules. Ordinary screening workers read one claimed private section at a time
and submit only `ADD`, `REJECT`, or `DEFER` with one bounded reason. `ADD`
retains a section for later work; it is never a license decision, public
candidate, approval, or publication authorization. `DEFER` is nonterminal and
moves the record out of the ordinary queue into a separate escalation queue;
an escalation worker must resolve it to `ADD` or `REJECT`. Preserve the
original deferrer, reason, and timestamp after resolution. Screening decisions
are scoped to an active trusted parser run, identical retries are idempotent,
terminal conflicts fail, and a noncanonical exact-text duplicate is stored as
`REJECT` with its canonical section reference. The public builder must continue
to ignore all screening tables.

The frozen `paizo-native-v1`, `paizo-native-v2`, `paizo-native-v3`, and
`paizo-native-v4` profiles must not change. The licensed-review runner defaults
to `paizo-native-v5`.
V4 binds GPU-produced PP-DocLayout regions to the authoritative native words,
then emits ordered structural blocks, active sections, and bounded quarantine
records. Every native anchor must occur exactly once across active blocks,
quarantine, or constrained ignores, and each active section's text must equal
the normalized projection of its ordered blocks. Ambiguous order, indivisible
oversize blocks, unresolved tables, and unsupported detected regions fail
closed into quarantine rather than disappearing or being guessed into public
text.
When the model omits a dense text region entirely (notably Monster Core stat
blocks), V4 reconstructs those authoritative native words with the frozen
native geometry and brackets them between nearby detected regions. Such
sections carry `native-layout-fallback`, which remains visible to classification
and independent review. It is not itself a forced mixed-extraction decision:
the fallback is exact native text, and forcing every partly recovered section
to Terra would create a large, unjustified review cost.

V5 retains V4's text and anchor authority but deterministically recovers
rule-bearing quarantine when geometry provides one stable interpretation.
Stable tables retain a grid; ambiguous tables, unsupported regions, order
conflicts, continuations, heading artifacts, and complete-boundary oversize
splits remain exact native text with explicit review flags. Only page numbers,
repeated furniture, contents/index, and credits/legal matter may remain in
nonblocking quarantine. A V5 identity collision changes only the colliding
opaque identities; every other V4 identity remains reusable.

V5 workspace preparation is a one-way schema upgrade, not a compatibility
layer. It builds from a read-only SQLite backup in a sibling file, carries
forward only exact unchanged terminal work, applies content-free structural
and aggregate rule-probe gates, and atomically replaces the live workspace only
after validation. Screening is append-only: rejection remains auditable and a
maintainer may explicitly reopen it without deleting its prior decision.
V5's historical comparison counts a V4 active defect plus the corresponding V4
rule-bearing quarantine as one conserved repair budget. Do not reuse the older
V3-to-V4 percentage-reduction comparator: activating exact recovered anchors
would make its active-only conflict and fragment metrics report improvement as
a regression. The V5 gate instead requires identical anchor inventories, zero
rule-bearing quarantine, lower quarantine ratios, and bounded per-product
section/conflict/heading/fragment movement.

Before screening, versioned private-text normalization groups exact duplicate
heading/body pairs only within one license and rules era. The canonical
occurrence is chosen by catalog order, page, stable identity, and section key;
all shadow occurrences remain auditable provenance and reopen through the
canonical group. A validated vector-free clean Foundry snapshot supplies at
most three deterministic exact/lexical candidates. Exact normalized identities
are suppressed deterministically. Local Qwen then performs a compact gate-first
triage for non-exact candidates through a loopback-only llama.cpp endpoint using
the model's documented non-thinking sampling profile. It cannot authorize suppression: only matching
Qwen and independent Sol `covered` judgments may create the final Foundry proof.
Qwen additions, model disagreement, uncertainty, stale evidence, malformed IDs,
and residual layout/context flags fail open into retained ordinary work. Images,
neighbors, and OCR-derived structure never enter the compact coverage packet or
replace authoritative native text. The public builder repeats duplicate
normalization over approved public text and emits one rule with all source
occurrences.

When a local-full seed has a complete PDF for a product, suppress the bundled
licensed-core rows for that product; keep bundled rows for missing products.
Corpus sync must apply the same rule in both directions. Archives of Nethys is
review evidence only: a match can corroborate that an entry is public rules
content, absence proves nothing, and AON text is never copied into the corpus.

### The licensed review runner owns workflow state

`review_runner.py` is the only automated supervisor for the licensed-core
workflow. Codex workers never schedule work or mutate SQLite. They run through
schema-constrained noninteractive `codex exec` in an isolated read-only sandbox
with user configuration ignored. The supervisor validates the exact submitted
ID set before any mutation, retains content-free attempt/session audit rows,
and retries transport/schema failures no more than three times.
Hosted Codex structured outputs do not accept JSON Schema `uniqueItems`.
Strip that keyword only from the hosted API schema and enforce it against the
unchanged authoritative schema after the response; llama.cpp may continue to
receive the full schema.
The supervisor classifies a Codex model usage-limit response as a sanitized,
non-retryable `model-usage-limit` transport failure. Never burn the remaining
attempt budget on an unchanged quota block or silently substitute a more
expensive model.

Parser activation and semantic scheduling are separate. Every trusted active
parser run remains available for structural validation and deterministic
duplicate/Foundry preparation. `review_product_scope` is the persistent
semantic scheduling and projection boundary: claims, AON, pilots, completion,
and base construction use only enabled products. Holding a product must never
delete its source rows, decisions, candidates, reviews, or deterministic
evidence. `prepare-review` performs only model-free preparation;
`preview --queue screen` uses the production serializer and packer while
remaining read-only and must never instantiate a Codex executor.

Worker evidence goes only through `review_evidence.py`. Its claim context fixes
the allowed section IDs, pre-authorized neighbors, review workspace, and one
validated clean Foundry DB. It exposes no arbitrary SQL or path argument and
must never query a preferred/private model DB or load embeddings. AON network
access remains supervisor-owned, rate-limited, and body-free; cache only
status/title/URL, and treat no-match/failure as inconclusive.

Keep candidate-producer and reviewer session pools disjoint. Rotate a session
after four completed batches, 256 KiB of submitted evidence, or any CLI,
model, prompt, schema, or policy change. Local Qwen performs compact gate-first
triage, Sol independently confirms non-exact suppression proposals, Luna
classifies and reviews ordinary work, and Terra extracts mixed mechanics and
performs first rework. Sol is also the one final rework tier. Exhaustion,
stitch disagreement, or
overlapping approved stitch groups must create `needs-maintainer` and block the
base build. Use `inspect-maintainer` for the bounded private evidence and
`resolve-maintainer` for the explicit decision; do not put private source text
in `status`. Repaired runs must reuse only exact unchanged stitch judgments and
repeat layout discovery/review to a fixed point before any screening begins.
Exact unchanged stitch identity is the product plus ordered stable section-key
set, not mutable proposal evidence such as the section offset. Carry both model
votes and any explicit maintainer resolution; otherwise repaired runs repeatedly
reopen already resolved interleavings and waste worker quota.
Use `run --queue QUEUE --pilot` for the mandatory at-most-one-batch-per-enabled-product
live pilots; a screening pilot must refuse to run while active layout work
remains. Use `run --queue layout --sources PATH` to drain layout review and
trusted repairs to a fixed point without entering semantic screening. Screening
rejections remain as private source-scoped decisions and EXCLUDE candidates;
never delete their source rows merely because the public projection omits them.

The base is not an embedding database. `build-base` produces an ignored,
model-independent audited sibling using public schema v3: canonical
`licensed_rules`, normalized `licensed_rule_sources`, and
`required_foundry_rows` for rules suppressed by confirmed Foundry coverage.
The base also commits to its ordered covered-product list and scope digest. The
build revalidates those rows against the selected clean snapshot and must
produce byte-identical repeated output.
`promote-base` is a separate explicit maintainer action. Never put Foundry rows,
vectors, FTS, model names, worker prompts, private paths, or raw PDF hashes in
the base, and never use it as a mutable template for final model databases.

### Daemon registration owns one configured data directory

HTTP/SSE startup writes `server.json` beneath the exact `Settings.data_dir`
passed to `serve()`. Registration uses exclusive creation, records its PID and
unique ownership token, refuses a live owner, and replaces only a confirmed
stale PID. Cleanup removes only the token it created. Do not reconstruct
settings inside registration helpers or let a second daemon overwrite a live
marker. Ancillary daemon state, including `flagged_results.jsonl`, must also use
the served settings directory rather than a freshly resolved global default.

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

Why `--no-deps`? The runtime variants all provide the same Python package and
must never be layered. Core dependencies intentionally omit ONNX Runtime;
`cpu`, `rocm`, and `cuda` are explicit extras, while `make setup-dev` installs
AMD's MIGraphX wheel directly from the official ROCm index. Adding plain
`onnxruntime` back to core makes `uv sync` silently regress an AMD development
environment to CPU execution.

If the PKGBUILD ever starts failing with "onnxruntime not installed" or
"MIGraphX not available", check that CPU onnxruntime was NOT pulled in by the
deps install step.

## Architecture

```
pf2e_codex/
├── config.py       # Settings: env vars → TOML file → defaults (Pydantic)
├── fetcher.py      # Download json-assets.zip from GitHub releases
├── pdf_export.py   # Native PDF words/geometry → versioned ignored JSON
├── pdf_layout.py   # GPU-first ONNX regions/order bound to opaque native anchors
├── corpus.py       # PZO discovery/revision selection + Paizo rulebook parsing
├── corpus_quality.py # Content-free parser audits and acceptance gates
├── licensed_corpus.py # Ignored review workspace + deterministic public builder
├── licensed_core.py # Validate/load the bundled reviewed mechanics projection
├── licensed_policy.py # Tracked mechanics-selection policy and digest
├── review_runner.py # Deterministic Codex queues, sessions, retries, and base lifecycle
├── review_evidence.py # Claimed-ID-only read-only local evidence executable
├── foundry_scope.py # Owning-publication allowlist for clean Foundry rows
├── distribution.py # DB seed-scope audit for publication and pull boundaries
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
In local-full, each Foundry pack entry becomes one text chunk and journal entries
are split into pages. Clean seeds first restrict entries by owning core
publication and declared OGL/ORC license; journals without owning publication
metadata are excluded.

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

Every chunk has an explicit `origin` and stable `source_id`. Foundry updates
select and delete only `origin='foundry'`; corpus errata refreshes select and
delete only `origin='corpus'`, embed changed section hashes before mutation,
and commit the source refresh atomically. Full rebuilds write a sibling staging
database, validate chunks/vectors/FTS/integrity/provenance, and use `os.replace`
only after validation. Corpus mutation and production rebuilds refuse to run
while `server.json` indicates a daemon may hold the database.

Clean and private complete databases are physically separate. Search prefers
the private file when it exists, while clean seed/release paths always select
the clean file explicitly. Corpus-owned rows must never appear in the clean
slot; opening a canonical slot validates this ownership boundary.

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
that explicitly names Fireball cannot lose the Fireball entry. Remaster ordering
is bounded to exact normalized-name overlaps; it never globally demotes
legacy-only results.

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
.local-corpus/      # purchased PDFs and generated native-text JSON
*.db               # sqlite-vec databases
chunks*.json       # intermediate chunk files
```

## Testing / Validation

Quick smoke test after changes (requires package installed via PKGBUILD):
```bash
pf2e-codex status                     # count depends on clean vs local-full scope
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
- **Size**: ~34MB zip; upstream-complete and clean-filtered chunk counts differ
- **License**: Foundry code Apache 2.0; clean PF2E rows require allowlisted
  owning core publication plus explicit OGL/ORC metadata
- **Optional local corpus**: user-owned PDFs under `.local-corpus/` (never committed)
- **Extraction**: native text only via `corpus-export` or automatic `corpus-sync`
- **Parsing**: `corpus.py` recognizes the built-in PZO catalog, strips watermark-like
  PII and repeated furniture, and emits stable provenance-rich sections

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
# Auto-detect AMD/NVIDIA; CPU only if neither is usable
make setup-dev

# Explicit AMD GPU (official AMD repo, ROCm 7.2+)
PF2E_DEV_ACCELERATOR=amd make setup-dev

# Explicit NVIDIA GPU
PF2E_DEV_ACCELERATOR=nvidia make setup-dev

# Explicit CPU fallback
PF2E_DEV_ACCELERATOR=cpu make setup-dev
```

The PKGBUILD auto-detects your GPU at build time and installs only the relevant variant.

On first use per model, ONNX exports once (cached at `~/.cache/pf2e-codex/onnx/{model}/`).
Subsequent loads skip export.
MIGraphX also compiles the model to GPU kernels on first inference (~10-30s per batch shape).
After compile, steady-state throughput is 50-500× faster than PyTorch CPU.

**Key provider order:** MIGraphX → ROCm → CUDA → CPU (CPU only when
no supported GPU is detected or explicitly selected)
**Per-batch-shape compile:** MIGraphX compiles once per unique batch size. For a running
MCP server this happens once at startup.

## Common Commands

```bash
# Check status & discover DB
pf2e-codex status

# Display or export embedded data-license notices
pf2e-codex licenses
pf2e-codex licenses --output-dir ./data-licenses

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
| `optimum` / `optimum-onnx` | ONNX model export support |
| ONNX Runtime hardware extra | Exactly one of CPU, MIGraphX/ROCm, or CUDA |
| `sqlite-vec` | Vector storage + similarity search |
| `mcp` | MCP Python SDK 2 server |
| `pydantic` + `pydantic-settings` | Config + validation |
| `typer` + `rich` | CLI framework + formatting |
| `pdfplumber` (`corpus` extra) | Native text, font, and geometry export from local PDFs |

All core Python deps are bundled in the PKGBUILD via `pip install --target`.
The PKGBUILD also bundles the `corpus` extra: its wrapper uses `python -S`, so
an Arch `python-pdfplumber` package would not be visible. The extra remains
optional for ordinary pip/uv installs that only query a Foundry-only database.
The only system dependency is `python`.

Hatch source distributions use an explicit `only-include` allowlist. Do not
remove it or rely solely on `.gitignore`: a misconfigured build can traverse
ignored `.local-corpus/` PDFs/review artifacts or recursively include its own
`dist/` output. Every packaging check must build into a fresh external staging
directory, list the archive members, and fail if private/ignored path markers
or unexpectedly large artifacts appear.
