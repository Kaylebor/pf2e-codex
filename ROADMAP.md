# Roadmap — pf2e-codex

Ordered by impact. Check items off as completed.

## High Impact

- [x] **Supplement Foundry with user-owned rulebook prose**
  - Native no-OCR exporter plus recursive PZO catalog discovery for combined
    PDFs, split PDFs, and chapter ZIPs under the ignored `.local-corpus/` tree.
  - Persisted revision selection, local raw-hash staleness checks, normalized
    watermark-independent rules fingerprints, and stable page/heading IDs.
  - Paizo layout parser with column ordering, repeated-furniture/watermark
    removal, action-glyph preservation, cross-page sections, and book/page
    provenance. Core Rulebook, Player Core, GM Core, Monster Core, and Player
    Core 2 are cataloged.
  - Foundry/corpus ownership boundaries, versioned sources, tri-state era,
    section-scoped errata refresh, staged full rebuilds, atomic swaps, and
    daemon mutation refusal. Legacy and Remaster remain side by side; only
    exact-name overlaps receive a bounded Remaster preference.
  - Distribution boundary: clean and private complete seeds use separate DB
    files per model. Queries prefer the private file, while pull, auto-download,
    and release tooling can only activate audited clean files. Strict release
    audits reject private rows, stale release/model metadata, or missing
    provenance markers. Required notices remain a release packaging concern;
    the database audit does not fabricate a legal-approval marker.
  - Constraint: the parser consumes exporter JSON; it must not invoke a second
    PDF extraction tool or silently substitute OCR.
  - Source identity: preserve original `PZO` PDF basenames and recognize a
    small explicit catalog of known product-code/split-file patterns. Never use
    a fixed source hash for product identity because purchased PDFs contain
    customer-specific watermarks; hashes are local provenance only. Detect and
    remove watermark text before indexing without logging its value.

- [x] **Pre-built clean DB download (`pull`)** — `pull` and first-query
  auto-download stage artifacts beside the clean slot, verify ownership,
  requested PF2E release, and embedding model, then atomically activate them.
  Existing stale files are replaced; the private `.local.db` slot is never a
  download target.

- [x] **Hybrid search: semantic + FTS5 via RRF**
  - Done: Weighted RRF (0.85 semantic / 0.15 name-LIKE). Stop-word filtering, bag-of-words name matching.

- [x] **Model benchmarking & selection command**
  - Done: `pf2e-codex benchmark` across models × providers (PyTorch CPU, ONNX CPU, ONNX GPU).

- [x] **ONNX export for GPU inference**
  - Done: MIGraphX on 7900 XTX, 50-90x speedup. `PF2E_ONNX_PROVIDER` override.

- [x] **Incremental updates**
  - Done: Content-hash diffing via `pf2e-codex index --update`. Only re-processes changed entries.
  - Correctness hardening: journal pages are replaced as a group, deletion-only releases are applied, external FTS5 is rebuilt, and the update commits atomically.

- [x] **Safe query and lazy-index access**
  - Done: SearchIndex initialization/FTS creation is locked, query connections are read-only, and MCP SQL rejects writes, attachment, pragma changes, extension loading, and runaway VM work.

- [x] **Natural-language entity retrieval**
  - Done: Hybrid search removes question filler, matches lexical terms with OR, balances lexical and semantic candidates, and keeps explicitly named entries such as Fireball in the final results even when reranking is enabled.

- [x] **MCP Python SDK 2 compatibility**
  - Done: Streamable HTTP serves modern stateless requests and legacy stateful sessions on the same endpoint while the CLI proxy retains its existing session flow.

- [x] **UUID fetch tool**
  - Done: `pf2e_get_entry` accepts IDs, slugs, names, UUIDs. `pf2e-codex get` CLI.

- [x] **Cross-reference graph (bidirectional)**
  - Done: `pf2e_related` + `pf2e-codex related`. Outgoing/incoming refs from description @UUID links and rule element fields.

- [x] **OGL→ORC alias injection**
  - Done: 256 aliases auto-extracted from Foundry wiki. Stored in chunk `name` field.

- [x] **License tracking**
  - Done: ORC/OGL/NONE per chunk. Search filters by license.

- [x] **Search enrichment**
  - Done: refs, legacy_name, confidence in every search result.

- [x] **Catalog tool**
  - Done: `pf2e_catalog` / `pf2e-codex catalog` — discover types, licenses, packs.

- [x] **Search filters**
  - Done: `license`, `content_type`, `pack` on `pf2e_search` and `pf2e_rules_explain`.

- [x] **Validation suite**
  - Done: 25-query suite, MRR 0.893 hybrid, **0.960 with reranker**. `pf2e-codex validate`.

## Medium Impact

- [x] **Release pipeline (embed-all)** — `pf2e-codex embed` builds clean model
  DBs with bounded concurrency. `scripts/release-dbs.sh` requires an exact PF2E
  release, propagates any model failure, verifies scope/license/release/model
  metadata, and uploads only clean-slot files from an isolated `.release-dbs/`
  staging directory.

- [ ] **Qualify the exact-one-runtime setup on NVIDIA** — exercise the CUDA
  extra, PKGBUILD detection, model export, cache behavior, bulk indexing, and
  MCP queries on real NVIDIA hardware. Preserve the same no-overlapping-runtime
  invariant used by the AMD/MIGraphX path.

- [ ] **Pretty CLI output (Rich tables)** — search results, status, catalog in rich formatting.

- [x] **MCP streamable-http transport** — `pf2e-codex mcp -t streamable-http --host 0.0.0.0 --port 8080`. Supports stdio, SSE, and streamable-http. Remote clients connect via HTTP POST to `/mcp`. Host/port configurable.

- [ ] **Cross-encoder reranker fine-tuning** — train `bge-reranker-v2-m3` on PF2E rules domain.
  - Data: subagents read raw pack JSONs (`~/.cache/pf2e-codex/extract-*/`), generate `(query, positive_chunk, negative_chunk)` triplets per pack (~2000-3000 total).
  - Sampling: per-pack strategy, parallel subagents via `pi subagents`, output JSONL.
  - Training: custom PyTorch script via `uv run` (margin ranking loss, 3 epochs), ~30min on 7900 XTX.
  - Storage: Push ONNX model to HuggingFace Hub (`kaylebor/pf2e-codex-reranker-minilm`).
  - Config: `reranker_model = "kaylebor/pf2e-codex-reranker-minilm"` — default, auto-downloaded.
  - Achieved: **0.960 MRR** (was 0.893 baseline). 23/25 perfect, 25/25 Top 3.

- [ ] **Docker image** — pre-built env, volume-mount for DB.

- [ ] **Dual DB for OGL/ORC** — index two releases into separate DBs, query both.

- [ ] **Validation suite expansion** — more queries covering edge cases, interactions.

## Low Impact / Polish

- [ ] **Web UI** — Gradio/Streamlit for non-MCP users.

- [ ] **ONNX auto-download of ONNX runtime via install.sh** — currently user must install manually.

## Hardware Platform Expansion

Current: AMD (ROCm/MIGraphX), NVIDIA (CUDA). Potential additions:

| Hardware | EP | Package | Status |
|---|---|---|---|
| Intel CPU/GPU/NPU | OpenVINO | `onnxruntime-openvino` | Production |
| Qualcomm Snapdragon X/X2 | QNN | `onnxruntime-qnn` | Production |
| Apple Silicon | CoreML | `onnxruntime-coreml` | Preview |
| MediaTek Dimensity | NNAPI (generic) | Built-in | Preview |
| ARM (generic) | ACL / Arm NN | Community | Preview |
| Huawei Ascend | CANN | Community | Preview |

OpenVINO (Intel) and QNN (Qualcomm) are production-ready. Would require:
- install.sh GPU detection for Intel/Qualcomm
- PKGBUILD extras for each provider
- `PF2E_PROVIDER` / `PF2E_ONNX_PROVIDER` config values
