# pf2e-codex

PF2E rules knowledge base with MCP, CLI, and SDK interfaces.

- **Primary data source**: Official FoundryVTT PF2E system JSON releases (`json-assets.zip`)
- **Reviewed rules supplement**: Bundled, provenance-rich licensed mechanics from core books
- **Optional local source**: User-owned rulebook PDFs exported to ignored native-text JSON
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
uv pip install -e ".[cpu]"             # CPU runtime, Foundry-only use
uv pip install -e ".[cpu,corpus]"      # CPU runtime plus local rulebook PDFs
make setup-dev                          # GPU-first dev setup + corpus
```

The Arch package bundles corpus extraction support because its isolated
launcher cannot use Python packages installed outside the package directory.

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
│   ├── pdf_export.py      # Native PDF words/geometry → versioned local JSON
│   ├── corpus.py          # PZO discovery, revision choice, and Paizo parsing
│   ├── corpus_quality.py  # Content-free parser audits and acceptance gates
│   ├── licensed_corpus.py # Private swarm review DB + public projection builder
│   ├── review_runner.py   # Deterministic Codex queue/session supervisor
│   ├── review_evidence.py # Claimed-ID-only read-only evidence boundary
│   ├── licensed_core.py   # Validate/load the bundled reviewed projection
│   ├── licensed_policy.py # Versioned mechanics-selection policy and digest
│   ├── foundry_scope.py   # Clean owning-publication/license allowlist
│   ├── distribution.py    # Publication and physical-slot ownership audits
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

### Local Rulebook Corpus

Place user-owned Paizo PDFs or chapter ZIPs recursively under
`.local-corpus/sources/`. The built-in catalog currently recognizes `PZO2101`
(legacy Core Rulebook), `PZO12001` (Player Core), `PZO12002` (GM Core), and
`PZO12003` (Monster Core), and `PZO12004` (Player Core 2).
Combined `PZO…E.pdf` files take precedence over their split chapter copies.
Printing metadata, normalized content revisions, modification time, and a
deterministic filename tie-break select one active revision per product. A
persisted choice stabilizes equivalent-content copies only; it never pins an
older materially different errata revision.

The native-text exporter runs automatically when its ignored JSON artifact is
missing or stale. It preserves word coordinates, fonts, sizes, action glyphs,
image bounds, and a local source SHA-256. The PDF's selectable words are always
the authoritative text; OCR and image-to-text conversion are never used. The
Paizo parser removes repeated page furniture and watermark-like email text,
reconstructs reading order, and emits stable page/heading-anchored rule
sections. A watermark-independent normalized fingerprint identifies actual
rules content, while the raw hash is used only to detect changes to that local
file.

The normal local-full flow remains pinned to the frozen `paizo-native-v1`
profile. The older v2 and v3 review profiles are also frozen. Licensed review
preparation uses `paizo-native-v5`: it combines the authoritative native words
with the structural layout artifact to produce ordered blocks, active sections,
and bounded quarantine records. V5 deterministically recovers rule-bearing V4
quarantine as exact native text with explicit layout flags; only page numbers,
repeated furniture, contents/index, and credits/legal matter may remain as
nonblocking quarantine. Section text must be the exact normalized
projection of its blocks, and every native anchor must occur exactly once in an
active block, quarantine record, or constrained ignore. Dense native stat-block
text omitted by the layout model is ordered by native PDF geometry between the
nearest detected regions and marked `native-layout-fallback`; it remains visible
to classification and independent review. The flag is not by itself a forced
mixed-extraction path because the recovered text and anchors are exact; workers
still decide whether the section is clean, mixed, or excluded.
The licensed-review runner additionally rasterizes pages for an ONNX-only
PP-DocLayoutV3 pass. Its separate private artifact contains boxes and reading
order, not recognized text. Those regions bind back to opaque native-word
anchors and may add review/stitch flags, but cannot replace, rewrite, omit, or
certify the native text inventory. Automatic provider selection is GPU-first
and refuses an accidental CPU-only runtime on supported GPU hardware; CPU is
available only on hardware without a supported GPU or by explicit request.

For a manual pip/uv installation, install the optional extraction dependency.
The Arch package already includes it. Then inspect discovery and build or
refresh the corpus:

```bash
uv pip install -e ".[corpus]"
uv run --script scripts/export-layout-model.py  # one-time pinned ONNX export
pf2e-codex corpus-status
pf2e-codex corpus-layout-export BOOK.pdf .local-corpus/layout/check.json
pf2e-codex index                              # licensed core-Foundry subset + reviewed-core DB
pf2e-codex index --corpus-scope local-full   # separate private complete DB
pf2e-codex corpus-sync                       # refreshes only the private DB
```

`.local-corpus/` is ignored so purchased sources and extracted text are never
committed. Full rebuilds use a sibling staging database and replace the live DB
only after chunk, vector, FTS, integrity, and provenance validation. Corpus-only
mutation also parses and embeds before one transaction, and refuses to run
while the daemon registration exists. Stop the daemon before either operation.
The two scopes never share a physical database. For each model, the clean slot
is `pf2e_<model>.db` and the private complete slot is
`pf2e_<model>.local.db`. Queries and the MCP daemon prefer the private slot when
it exists; use `pf2e-codex mcp --db-scope clean` or
`pf2e-codex validate --db-scope clean` to force the clean slot. Corpus sync can
only mutate the private slot, while pull, automatic download, and release
tooling can only activate an audited clean artifact.

Full seeds default to `--corpus-scope redistributable`, which selects the clean
slot and never reads purchased PDFs even when `.local-corpus/` exists. It merges
only Foundry entries whose owning publication is one of the five cataloged core
rulebooks and whose entry declares OGL/ORC, plus the bundled `licensed-core`
projection. Owning publication comes from `system.publication`, with
`system.details.publication` as the sole fallback; pack names and nested source
citations never grant clean-seed eligibility. Missing/blank publications,
journals, adventures, PFS, Lost Omens, and other products fail closed. The
reviewed projection contains only mechanics text that passed an independent
private review, with product/reprinting,
page, license, parser, policy, content-hash, and notice provenance. The complete
private PDF text never enters the package or clean database. A local-full seed
suppresses bundled rows product-by-product when the corresponding complete PDF
is present, avoiding duplicate results, and restores them if that local product
is later removed.

`pf2e-codex audit-db DB --strict` requires the explicit redistributable and
core-publication markers, rejects private/unknown/out-of-scope Foundry rows,
recomputes the complete licensed-core contract, and compares it with the
packaged trusted projection. The contract covers IDs, text hashes, pages,
headings, revision/policy provenance, and notices. Use `pf2e-codex licenses` to
display embedded notices or `--output-dir` to export them. This is a technical
fail-closed boundary, not legal advice.

The ignored review workspace is a WAL-mode SQLite database with atomic leases
and a deterministic supervisor. `scripts/licensed-corpus-runner.py` owns PDF
discovery, queue scheduling, prompt evidence, bounded retries, session rotation,
exact-ID/schema validation, and every database mutation. Schema-constrained
`codex exec` workers make semantic judgments only. They run in an isolated
read-only sandbox with user configuration ignored and can inspect only their
claimed IDs through `scripts/licensed-corpus-evidence.py`; no worker-controlled
network or SQLite write path exists.

The five source products have explicit, independent era metadata: `PZO2101` is
legacy/pre-Remaster, while `PZO12001` through `PZO12004` are the current
post-Remaster set. License never determines era. `prepare` stages all five
selected combined PDFs through `paizo-native-v5` plus a source-bound layout
artifact in a fresh sibling workspace. One GPU session is reused across all
five books. It validates complete native-anchor coverage, block/text equality,
privacy, bounded quarantine, structural metrics, and aggregate general-rule
probes. Preparation starts from a read-only backup when a workspace exists,
carries forward only exact unchanged terminal work, and replaces the live file
only after all five runs pass. Deterministic adjacent-section stitch proposals
permit only complete groups of two or three; Luna selection needs independent
Terra confirmation, and any disagreement or overlapping approved group fails
closed as `needs-maintainer`. After a repair, exact unchanged no-merge decisions
carry into the sibling parser run and layout review repeats to a fixed point
before screening may begin.
The V5 history gate compares repaired structure against the V4 active records
plus the exact V4 rule-bearing quarantine being recovered; it also requires an
identical native inventory, zero remaining rule-bearing quarantine, and a lower
per-product quarantine ratio. This avoids treating newly active exact text as a
regression merely because V4 had hidden those anchors in quarantine.
Parser activation and semantic scheduling are separate. All five trusted runs
remain active and auditable, while a persistent product scope may hold a book
without deleting its sections, duplicate mappings, decisions, or review history.
Queue claims, pilots, completion checks, and public projection use only enabled
products. A scoped base records its exact covered-product list and digest.

```bash
# These admin artifacts remain below ignored .local-corpus/ paths.
scripts/licensed-corpus-runner.py prepare \
  .local-corpus/licensed-review.sqlite3 .local-corpus/sources
scripts/licensed-corpus-runner.py status .local-corpus/licensed-review.sqlite3
scripts/licensed-corpus-runner.py quality .local-corpus/licensed-review.sqlite3
# Enable the four Remaster books while retaining legacy PZO2101 on hold.
scripts/licensed-corpus-runner.py set-scope .local-corpus/licensed-review.sqlite3 \
  --include PZO12001 --include PZO12002 --include PZO12003 --include PZO12004 \
  --held-reason legacy-study
# Prepare deterministic duplicate/Foundry evidence without starting Codex.
scripts/licensed-corpus-runner.py prepare-review \
  .local-corpus/licensed-review.sqlite3 \
  --foundry-database /path/to/validated-clean.db
# Serialize and size the exact future Spark envelopes without claims or writes.
scripts/licensed-corpus-runner.py preview \
  .local-corpus/licensed-review.sqlite3 --queue screen \
  --foundry-database /path/to/validated-clean.db
# Compare explicit historical and candidate parser-run selections when needed.
# scripts/licensed-corpus-runner.py compare-quality WORKSPACE \
#   --baseline-runs baseline.json --candidate-runs candidate.json
# If status reports a disagreement, inspect it locally and resolve explicitly:
# scripts/licensed-corpus-runner.py inspect-maintainer WORKSPACE ITEM_ID --include-text
# scripts/licensed-corpus-runner.py resolve-maintainer WORKSPACE ITEM_ID no-merge
# Qualify one Luna selection batch and one Terra confirmation batch per book.
scripts/licensed-corpus-runner.py run .local-corpus/licensed-review.sqlite3 \
  --queue stitch-select --pilot
scripts/licensed-corpus-runner.py run .local-corpus/licensed-review.sqlite3 \
  --queue stitch-confirm --pilot
# Drain layout review/repairs to a fixed point, then stop before screening.
scripts/licensed-corpus-runner.py run .local-corpus/licensed-review.sqlite3 \
  --queue layout --sources .local-corpus/sources
# After layout review is terminal, screen one batch from each enabled book.
scripts/licensed-corpus-runner.py run .local-corpus/licensed-review.sqlite3 \
  --queue screen --pilot --foundry-database /path/to/validated-clean.db
scripts/licensed-corpus-runner.py run .local-corpus/licensed-review.sqlite3 \
  --sources .local-corpus/sources --foundry-database /path/to/validated-clean.db
scripts/licensed-corpus-runner.py verify \
  .local-corpus/licensed-review.sqlite3 --complete \
  --foundry-database /path/to/validated-clean.db
scripts/licensed-corpus-runner.py build-base \
  .local-corpus/licensed-review.sqlite3 \
  .local-corpus/licensed_core.sqlite3 /path/to/reviewed-notices.json \
  --foundry-database /path/to/validated-clean.db
```

Bulk screening uses Spark; Luna handles ordinary classification and review;
Terra handles mixed mechanics extraction, difficult review, and first rework;
Sol is reserved for one final rework. Sessions are disjoint between producers
and reviewers and rotate after four batches, 256 KiB of evidence, or any
model/prompt/schema/policy/CLI change. AON searches run outside Codex through a
rate-limited cache and retain only status, title, and URL; no match or failure
is inconclusive. Pilot mode processes at most one selected-queue batch per
enabled product and refuses to screen while the active parser run has unresolved
stitch work.
An exhausted Codex model quota is recorded as sanitized `model-usage-limit`
metadata and stops after one attempt; the runner never silently substitutes a
different, more expensive model.
Before screening, exact normalized PDF duplicates are grouped within one
license and rules era; only the canonical occurrence enters semantic queues and
all shadow occurrences remain as source provenance. A vector-free snapshot of
the validated clean Foundry database supplies up to three deterministic
same-era/same-license matches to Spark. Confirmed complete coverage is retained
as a revalidated proof and required Foundry-row contract; uncertainty and stale
rows fail open into ordinary review.
Rejected screening records remain in the private workspace with their source
section, decision, and provenance. They are excluded from the public projection,
not deleted. Decisions are append-only, so an explicit maintainer reopen records
a new event while preserving the old rejection and reason. Activating a reparsed
source carries exact unchanged terminal decisions into the new scope; changed,
deferred, or reopened sections return to review while the retired run remains
available for audit or later reconsideration.

The model-independent base uses schema v3. It stores one canonical
`licensed_rules` row per approved public rule, every occurrence in
`licensed_rule_sources`, and any Foundry dependency in
`required_foundry_rows`, plus the ordered covered-product scope and digest. It
contains no embeddings, FTS, Foundry text, private
paths, prompts, or source-PDF hashes, and the runner requires two byte-identical
builds before activation of the ignored artifact.

```bash
scripts/licensed-corpus-runner.py reopen-screening \
  .local-corpus/licensed-review.sqlite3 SECTION_KEY \
  --maintainer MAINTAINER --reason parser-quality
```
`build-base` stops at a validated, model-independent ignored
SQLite artifact. `promote-base` is a separate explicit command; embedding DB
builds, commits, pushes, and releases are later operations.

Only the deterministic approved projection is tracked and bundled. Raw
sections, prompts, decisions, rejected candidates, local paths, file hashes,
native-word anchors, and watermark-derived data remain private. Complete parser
runs still enter only through the direct-PDF bridge: cached JSON exports,
caller-supplied inventories, and caller-supplied completeness manifests are
intentionally not accepted.

Before the expensive candidate-writing pass, a quad-state private screen can reduce
the workload without generating rewritten rules. A screening worker claims a
batch, reads exactly one parsed section at a time, and records `add`, `reject`,
or a bounded `defer` reason:

```bash
scripts/licensed-corpus.py screen-status WORKSPACE
scripts/licensed-corpus.py screen-claim WORKSPACE WORKER --product-code PZO12001
scripts/licensed-corpus.py screen-next WORKSPACE SHARD_ID WORKER
scripts/licensed-corpus.py screen-step WORKSPACE SHARD_ID WORKER INDEX add
scripts/licensed-corpus.py screen-step WORKSPACE SHARD_ID WORKER INDEX defer \
  --defer-reason complex-rule
scripts/licensed-corpus.py screen-claim WORKSPACE SENIOR --queue deferred
scripts/licensed-corpus.py screen-release WORKSPACE SHARD_ID WORKER
```

Decisions are scoped to the active trusted parser run. Identical retries are
no-ops, conflicting retries fail, and exact duplicate source text has one
deterministic canonical section; a requested `add` for another copy is stored
as a duplicate rejection. `screen-next` skips already decided records when a
batch resumes, while `screen-step` returns the next eligible record along with
the persisted result, avoiding a second CLI launch per section. Screening
`add` means only "retain in the private draft." It never creates a public
candidate, satisfies independent review, or authorizes inclusion in the
bundled projection. Deferred sections leave the ordinary queue and enter a
separate escalation queue; the resolving worker must choose `add` or `reject`.
The original worker, bounded reason, and timestamp remain recorded after
resolution. Terminal decisions cannot be changed without an explicit future
invalidation or policy migration.

The exact selection contract is tracked as
`pf2e_codex/data/licensed_core_policy_v1.json`. Public candidate text must have
an extraction method and reason tags and is rejected if it contains obvious
email, watermark, control-character, or local-path material. Clear review
aliases distinguish `APPROVE_PUBLIC` from `CONFIRM_EXCLUSION`; both still
require a separately claimed reviewer.

Foundry and corpus rows have explicit ownership, so Foundry incremental updates
cannot remove PDF-derived rules. Legacy and Remaster books coexist; default
ranking prefers Remaster only when candidates have the same normalized name,
leaving legacy-only material fully searchable. Search and fetch results include
book and PDF-page provenance.

No per-book manifest is required. Optional ambiguity controls can be placed in
`pf2e-codex.toml`:

```toml
corpus_dir = ".local-corpus"
database_scope = "auto" # prefer local when present; clean/local force one slot
corpus_include = ["PZO2101", "PZO12001", "PZO12002", "PZO12003", "PZO12004"]
corpus_exclude = []
corpus_prefer = { PZO12001 = "preferred-copy/PZO12001E.pdf" }
```

## CLI Commands

| Command | Description |
|---|---|
| `pf2e-codex fetch` | Download json-assets.zip |
| `pf2e-codex build` | Build enriched chunks (JSON output) |
| `pf2e-codex corpus-export PDF JSON` | Export native PDF words/geometry without OCR |
| `pf2e-codex corpus-layout-export PDF JSON` | Export private ONNX regions/order without recognizing text |
| `pf2e-codex corpus-status` | Show discovered products and persisted revision choices |
| `pf2e-codex corpus-sync` | Export, parse, and atomically refresh the private DB |
| `pf2e-codex audit-db DB --strict` | Reject private/unmarked DBs before publication |
| `pf2e-codex index` | Full pipeline: fetch → chunk → embed → index |
| `pf2e-codex search "query"` | Hybrid search (semantic + FTS5) |
| `pf2e-codex status` | Show index stats |
| `pf2e-codex config` | Show effective configuration |
| `pf2e-codex config --file` | Show active config file contents |
| `pf2e-codex get "fireball"` | Fetch a single entry by slug, name, or UUID |
| `pf2e-codex related "off-guard" --direction incoming` | Cross-reference graph |
| `pf2e-codex models` | List embedding models with recommendations |
| `pf2e-codex validate` | Validate search quality against 25-query suite |
| `pf2e-codex warmup` | Manually precompile models into the current user's cache |
| `pf2e-codex pull` | Download pre-built embedding DB from GitHub Releases |
| `pf2e-codex mcp` | Start MCP server (--transport streamable-http for daemon) |

## Daemon (systemd)

For persistent GPU inference, run the MCP server as a systemd user service.
The daemon loads both models synchronously before accepting requests. If no
private or clean database exists, it auto-downloads an audited clean database
on first query.

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
it stages a pre-computed sqlite-vec DB from GitHub Releases, verifies its clean
ownership plus PF2E release and embedding model, then activates it atomically.
`pf2e-codex pull --release ...` applies the same checks and replaces an existing
artifact when it is stale. Automatic downloads never create or overwrite the
private `.local.db` slot.
To rebuild from scratch instead:

```bash
pf2e-codex embed
```

Release builds use `.release-dbs/<tag>/` by default, isolated from the live
query databases. Set `PF2E_RELEASE_DB_DIR` to choose another staging directory.

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
release = "pf2e-8.4.1"
```

Or use a project-local `pf2e-codex.toml` (gitignored by default).

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PF2E_CACHE_DIR` | `~/.cache/pf2e-codex` | Download cache |
| `PF2E_DATA_DIR` | `~/.local/share/pf2e-codex` | Data directory |
| `PF2E_MODEL` | `Snowflake/snowflake-arctic-embed-xs` | Embedding model |
| `PF2E_PROVIDER` | `auto` | Embedding provider selection |
| `PF2E_QUERY_PROVIDER` | `auto` | GPU-first ONNX provider for daemon queries; `cpu` is an explicit fallback |
| `PF2E_RELEASE` | `pf2e-8.4.1` | PF2E system version |

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

ONNX Runtime is selected explicitly so CPU and GPU wheels can never overwrite
one another. `make setup-dev` detects AMD or NVIDIA hardware and installs the
matching GPU runtime. It selects CPU only when no supported GPU is detected; an
explicit `PF2E_DEV_ACCELERATOR=cpu` remains available for intentional fallback.

The project locks Torch to its CPU wheel because Optimum uses it only while
exporting models to ONNX. Embedding and reranker inference are ONNX-only and
GPU-first. On a machine with AMD or NVIDIA hardware, automatic provider
selection refuses a CPU-only ONNX installation instead of silently degrading.

**For GPU acceleration, install the matching `onnxruntime` variant:**

| GPU | Install | Source |
|-----|---------|-------|
| Auto-detect | `make setup-dev` | AMD MIGraphX or NVIDIA CUDA |
| AMD (ROCm) | `PF2E_DEV_ACCELERATOR=amd make setup-dev` | AMD official MIGraphX repo |
| NVIDIA (CUDA) | `PF2E_DEV_ACCELERATOR=nvidia make setup-dev` | PyPI (official) |
| CPU fallback | `PF2E_DEV_ACCELERATOR=cpu make setup-dev` | PyPI (official) |

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

**Provider priority:** MIGraphX → ROCm → CUDA → CPU. CPU is accepted
automatically only when no supported GPU is detected. ZLUDA (CUDA-on-AMD
emulation) is correctly deprioritized — native ROCm is preferred.

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

ONNX Runtime is selected explicitly; it is intentionally absent from the core
dependency set so a normal `uv sync` cannot overwrite a GPU runtime with CPU.
Project-managed Torch is pinned to the CPU index because it performs export,
not inference; ONNX Runtime owns hardware acceleration.

**For GPU acceleration, install the matching `onnxruntime` variant:**

| GPU | Install | Source |
|-----|---------|-------|
| Auto-detect | `make setup-dev` | AMD MIGraphX or NVIDIA CUDA |
| AMD (ROCm) | `PF2E_DEV_ACCELERATOR=amd make setup-dev` | AMD official MIGraphX repo |
| NVIDIA (CUDA) | `PF2E_DEV_ACCELERATOR=nvidia make setup-dev` | PyPI (official) |
| CPU fallback | `PF2E_DEV_ACCELERATOR=cpu make setup-dev` | PyPI (official) |

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
- Complete purchased-PDF exports and parses: local-only and never released
- Reviewed core mechanics: redistributed only through the sanitized,
  independently approved `licensed-core` projection with per-row OGL/ORC
  provenance and notices
- Foundry and licensed-core database releases retain applicable OGL/ORC/Paizo
  notices; the distribution audit prevents known private-corpus leakage but is
  not legal advice
- The public clean seed includes only allowlisted core-publication Foundry rows;
  the local-full seed retains the upstream-complete Foundry data privately
