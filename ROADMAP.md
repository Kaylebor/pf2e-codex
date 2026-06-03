# Roadmap — pf2e-codex

Ordered by impact. Check items off as completed.

## High Impact

- [x] **Hybrid search: semantic + FTS5 via RRF**
  - Done: Weighted RRF (0.85 semantic / 0.15 name-LIKE). Stop-word filtering, bag-of-words name matching.

- [x] **Model benchmarking & selection command**
  - Done: `pf2e-codex benchmark` across models × providers (PyTorch CPU, ONNX CPU, ONNX GPU).

- [x] **ONNX export for GPU inference**
  - Done: MIGraphX on 7900 XTX, 50-90x speedup. `PF2E_ONNX_PROVIDER` override.

- [x] **Incremental updates**
  - Done: Content-hash diffing via `pf2e-codex index --update`. Only re-processes changed entries.

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
  - Done: 25-query suite, MRR 0.850 hybrid. `pf2e-codex validate`.

## Medium Impact

- [ ] **Pretty CLI output (Rich tables)** — search results, status, catalog in rich formatting.

- [ ] **MCP streamable-http transport** — for remote clients. stdio works locally.

- [ ] **Docker image** — pre-built env, volume-mount for DB.

- [ ] **Dual DB for OGL/ORC** — index two releases into separate DBs, query both.

- [ ] **Validation suite expansion** — more queries covering edge cases, interactions.

## Low Impact / Polish

- [ ] **Web UI** — Gradio/Streamlit for non-MCP users.

- [ ] **AGENTS.md auto-update hook** — CI check that docs match module structure.

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
