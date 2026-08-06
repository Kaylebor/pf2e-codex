# Roadmap — pf2e-codex

Ordered by impact. Check items off as completed.

## High Impact

- [ ] **Pre-built DB download (`pull`)** — `pf2e-codex pull` downloads a pre-computed `sqlite-vec` DB from GitHub Releases (`pf2e-codex-data` repo) into `$PF2E_DATA_DIR`. Auto-triggered on first search if no local DB exists. Shares the same file path as `index` — they're a single "slot" per model. `pull` replaces; `index` rebuilds; `index --update` incrementally patches (handles additions, changes, and deletions). Tagged per PF2E release + model, checksum-verified. Pattern: llama.cpp / GPT4All. Removes "wait 5 minutes" onboarding.

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

- [x] **Release pipeline (embed-all)** — `scripts/embed-all.py`: runs `pf2e-codex index -m <model>` for all supported models in parallel (configurable concurrency), with proper Ctrl+C signal handling via Python `subprocess`. Skips existing DBs. Logs per-model to `/tmp`.
  2. `scripts/push-release.sh` (TODO) — takes generated DBs, creates tagged GitHub Releases on `pf2e-codex-data`.

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
