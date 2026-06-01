# Roadmap — pf2e-codex

Ordered by impact. Check items off as completed.

## High Impact

- [ ] **Hybrid search: semantic + FTS5 exact name matching**
  - Problem: Exact names like "Fireball" or "Flurry of Blows" don't always rank #1 in pure semantic search.
  - Approach: Add `name_fts` virtual table with `fts5(name, content=chunks, content_rowid=rowid)`. Blend semantic and FTS scores via reciprocal rank fusion.
  - Files: `index.py`, `cli.py`, `mcp_server.py`

- [ ] **Model benchmarking & selection command**
  - Problem: Users don't know which model suits their hardware without reading docs.
  - Approach: `pf2e-codex benchmark` downloads 3-4 models, indexes 1000 chunks with each, runs a standard query set, prints quality × speed tradeoffs. Auto-recommend based on measured speed.
  - Files: `cli.py`, `models.py`, new `benchmark.py`

- [ ] **ONNX export for Arctic / Nomic on CPU/GPU**
  - Problem: `snowflake-arctic-embed-m` (~110M, 768d) and `nomic-embed-text-v1.5` (~137M, 768d) are impractical on CPU (~1h index time).
  - Approach: Add `optimum[onnxruntime]` provider. Export ONNX once, use `onnxruntime` for 10-20× faster inference. Support ROCm on AMD 7900 XTX via `onnxruntime-rocm`.
  - Files: `embeddings.py` (new `ONNXProvider`)

- [ ] **Incremental updates (diff since last release)**
  - Problem: `pf2e-codex index` re-downloads and re-embeds everything even on patch releases.
  - Approach: Check GitHub releases API for newer version. Diff entry IDs against existing DB. Only download, chunk, and embed changed/new entries. Reuse existing chunks for unchanged entries.
  - Files: `fetcher.py`, `pipeline.py`, `index.py`

## Medium Impact

- [ ] **Cross-reference graph (bidirectional)**
  - Problem: "What feats reference Dread Striker?" requires manual search.
  - Approach: Build a `references` table (`source_id → target_id`) from all `@UUID[...]` links in descriptions and rules. Expose via `SearchIndex.related(id)` or new MCP tool.
  - Files: `chunker.py` (extract refs), `index.py` (store + query)

- [ ] **MCP streamable-http transport**
  - Problem: `stdio` is the only reliable transport. SSE is deprecated in MCP spec.
  - Approach: Add `streamable-http` support once the `mcp` library supports it, or implement a small ASGI wrapper.
  - Files: `mcp_server.py`, `config.py`

- [ ] **Validation suite for retrieval quality**
  - Problem: No automated way to know if a model change or code refactor hurts search quality.
  - Approach: `tests/retrieval.yaml` with 20-30 known-good query→expected-result pairs. Run after every index build. `pytest` integration.
  - Files: `tests/retrieval_test.py`, `tests/retrieval.yaml`

## Low Impact / Polish

- [ ] **Pretty CLI output (Rich tables)**
  - `pf2e-codex search` currently dumps raw text. Rich tables with color-coded types (feat/spell/condition/journal_page) would be much nicer.
  - Files: `cli.py`

- [ ] **Docker image**
  - For users who don't want to install Python/uv. Single container with `pf2e-codex` pre-installed, volume-mount for DB.
  - Files: `Dockerfile`, `.dockerignore`

- [ ] **Web UI (optional, later)**
  - Simple Gradio/Streamlit interface for non-technical users. Lower priority since MCP + CLI already cover most use cases.
  - Files: new `web/` directory

- [ ] **AGENTS.md auto-update hook**
  - Keep `AGENTS.md` in sync with code changes. Could be a CI check or pre-commit hook that verifies the doc reflects current module structure.
  - Files: `.github/workflows/` or `.pre-commit-config.yaml`

## Done

- [x] Package structure: `src/pf2e_codex/` with proper modules
- [x] CLI entry point with Typer
- [x] Config system: env vars + TOML file + Pydantic Settings
- [x] Pluggable embedding providers with model registry
- [x] Automatic query/document prefixing for model-specific models
- [x] MCP server with 4 tools (search, lookup, rules_explain, status)
- [x] Hardware-aware model recommendations
- [x] Default model: `snowflake-arctic-embed-xs` (fast, good quality)
