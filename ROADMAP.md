# Roadmap — pf2e-codex

Ordered by impact. Check items off as completed.

## High Impact

- [x] **Hybrid search: semantic + FTS5 exact name matching**
  - Done: FTS5 external-content table, lazy creation for existing DBs, RRF blending in `SearchIndex.search()`. `distance` for semantic, `rrf_score` for blended results.
  - Files: `index.py`, `cli.py`, `mcp_server.py`

- [ ] **Model benchmarking & selection command**
  - Problem: Users don't know which model suits their hardware without reading docs.
  - Approach: `pf2e-codex benchmark` downloads 3-4 models, indexes 1000 chunks with each, runs a standard query set, prints quality × speed tradeoffs. Auto-recommend based on measured speed.
  - Files: `cli.py`, `models.py`, new `benchmark.py`

- [x] **ONNX export for Arctic / Nomic on CPU/GPU**
  - Done: `ONNXProvider` with lazy export via `optimum`, runtime provider auto-detection (ROCm → CUDA → CPU), graceful fallback to `SentenceTransformersProvider`. `PF2E_PROVIDER` env override. `install.sh` and PKGBUILD handle system-level GPU detection.
  - Files: `embeddings.py`, `config.py`, `pyproject.toml`, `install.sh`, `PKGBUILD`

- [ ] **Incremental updates (diff since last release)**
  - Problem: `pf2e-codex index` re-downloads and re-embeds everything even on patch releases.
  - Approach: Check GitHub releases API for newer version. Diff entry IDs against existing DB. Only download, chunk, and embed changed/new entries. Reuse existing chunks for unchanged entries.
  - Files: `fetcher.py`, `pipeline.py`, `index.py`

## Medium Impact

- [x] **UUID fetch tool**
  - Done: `pf2e_get_entry` accepts internal IDs (`pack:id`), Foundry UUIDs, bare slugs, or names. Tries exact ID → bare ID suffix → slug → name. `pf2e-codex get` CLI command.
  - Files: `index.py`, `mcp_server.py`, `cli.py`

- [x] **Cross-reference graph (bidirectional)**
  - Done: `refs` table stores `source_id → target_uuid` from both description `@UUID[...]` links and rule element UUID fields (GrantItem, EphemeralEffect, Aura effects). Lazy-created for existing DBs. `SearchIndex.related(id, direction, limit)` with `pf2e_related` MCP tool and `pf2e-codex related` CLI command.
  - Files: `chunker.py`, `index.py`, `pipeline.py`, `mcp_server.py`, `cli.py`

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
- [x] MCP server with 5 tools (search, get_entry, related, rules_explain, status)
- [x] Hardware-aware model recommendations
- [x] Default model: `snowflake-arctic-embed-xs` (fast, good quality)
- [x] Hybrid search: semantic + FTS5 via reciprocal rank fusion
- [x] UUID fetch tool: `pf2e_get_entry` + `pf2e-codex get`
- [x] Cross-reference graph: bidirectional outgoing/incoming via `pf2e_related` + `pf2e-codex related`
