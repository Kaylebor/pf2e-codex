"""Embedding model registry with hardware-aware recommendations.

Benchmarks on AMD Ryzen 7 7800X3D (8c/16t) + Radeon RX 7900 XTX (24GB) ROCm 7.2.3.
ONNX inference via MIGraphXExecutionProvider (onnxruntime-migraphx 1.25.0).
Steady-state throughput (post-compile), average of 50-100 runs:

| Model | Params | Dim | Index Time (CPU) | Query (CPU batch=100) | Query (GPU batch=100) |
|---|---|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | 22M | 384 | ~50s | 520ms | 8.3ms |
| snowflake-arctic-embed-xs | 22M | 384 | ~35s | 612ms | 8.3ms |
| snowflake-arctic-embed-s | 33M | 384 | ~69s | 1204ms | — |
| intfloat/e5-small-v2 | 33M | 384 | ~135s | 1212ms | 13.4ms |
| snowflake-arctic-embed-m | 110M | 768 | ~1h | — | — |
| nomic-embed-text-v1.5 | 137M | 768 | ~1h+ | — | — |
| bge-m3 | 568M | 1024 | ~3h+ | — | — |

bge-m3: GPU strongly recommended (>500M params). No query prefixes needed.
Best quality model available — 1024d, 8192 token context.
Single-query GPU latency: 1.1ms (MiniLM / Arctic xs), 2.4ms (e5-small).
Compile time: ~10-30s one-time per model, cached in ~/.cache/pf2e-codex/onnx/
* e5-small requires "query:" / "passage:" prefixing.

* Requires "query:" / "passage:" prefixing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    name: str
    params: str
    dim: int
    cpu_time_28k: str
    quality: str
    query_prefix: str = ""
    doc_prefix: str = ""
    notes: str = ""


MODELS: dict[str, ModelInfo] = {
    "all-MiniLM-L6-v2": ModelInfo(
        name="all-MiniLM-L6-v2",
        params="22M",
        dim=384,
        cpu_time_28k="~50s",
        quality="Good",
        notes="Proven baseline. No prefixing needed. Slightly slower than Arctic xs.",
    ),
    "snowflake-arctic-embed-xs": ModelInfo(
        name="Snowflake/snowflake-arctic-embed-xs",
        params="22M",
        dim=384,
        cpu_time_28k="~35s",
        quality="Good",
        query_prefix="Represent this sentence for searching relevant passages: ",
        doc_prefix="",
        notes="Fastest on CPU. Retrieval-focused training.",
    ),
    "snowflake-arctic-embed-s": ModelInfo(
        name="Snowflake/snowflake-arctic-embed-s",
        params="33M",
        dim=384,
        cpu_time_28k="~70s",
        quality="Good",
        query_prefix="Represent this sentence for searching relevant passages: ",
        doc_prefix="",
        notes="Best 384d quality/speed tradeoff.",
    ),
    "e5-small-v2": ModelInfo(
        name="intfloat/e5-small-v2",
        params="33M",
        dim=384,
        cpu_time_28k="~135s",
        quality="Good",
        query_prefix="query: ",
        doc_prefix="passage: ",
        notes="E5 family. Requires strict prefixing.",
    ),
    "snowflake-arctic-embed-m": ModelInfo(
        name="Snowflake/snowflake-arctic-embed-m",
        params="110M",
        dim=768,
        cpu_time_28k="~1h",
        quality="Better",
        query_prefix="Represent this sentence for searching relevant passages: ",
        doc_prefix="",
        notes="Best quality 768d. Slow on CPU, great on GPU.",
    ),
    "nomic-embed-text-v1.5": ModelInfo(
        name="nomic-ai/nomic-embed-text-v1.5",
        params="137M",
        dim=768,
        cpu_time_28k="~1h+",
        quality="Better",
        query_prefix="search_query: ",
        doc_prefix="search_document: ",
        notes="Long context (8192). Slow on CPU, great on GPU.",
    ),
    "bge-m3": ModelInfo(
        name="BAAI/bge-m3",
        params="568M",
        dim=1024,
        cpu_time_28k="~3h+",
        quality="Best",
        notes="Multi-granularity (dense+sparse+colbert). 1024d, 8192 context. No prefixes needed. GPU strongly recommended. MTEB v1: 59.56.",
    ),
}


def get_model_info(name: str) -> ModelInfo | None:
    for key, info in MODELS.items():
        if info.name == name or key == name:
            return info
    return None


def list_models() -> list[ModelInfo]:
    return list(MODELS.values())


ALL_MODEL_NAMES: dict[str, str] = {
    key: info.name for key, info in MODELS.items()
}


def recommend(hardware: str = "cpu") -> list[str]:
    if hardware == "cpu":
        return ["snowflake-arctic-embed-xs", "snowflake-arctic-embed-s", "all-MiniLM-L6-v2"]
    elif hardware == "gpu":
        return ["bge-m3", "snowflake-arctic-embed-m", "nomic-embed-text-v1.5", "snowflake-arctic-embed-s"]
    return ["snowflake-arctic-embed-xs"]
