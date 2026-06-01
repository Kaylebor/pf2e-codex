"""Embedding model registry with hardware-aware recommendations.

Benchmarks on AMD Ryzen 7 7800X3D (8c/16t), batch_size=32, 28.8K chunks:
┌─────────────────────────────────────────┬───────┬─────┬──────────┬─────────┐
│ Model                                   │ Params│ Dim │ Time     │ Quality │
├─────────────────────────────────────────┼───────┼─────┼──────────┼─────────┤
│ all-MiniLM-L6-v2                        │  22M  │ 384 │ ~50s     │ Good    │
│ Snowflake/snowflake-arctic-embed-xs     │  22M  │ 384 │ ~35s     │ Good    │
│ Snowflake/snowflake-arctic-embed-s      │  33M  │ 384 │ ~69s     │ Good    │
│ intfloat/e5-small-v2                    │  33M  │ 384 │ ~135s    │ Good*   │
│ Snowflake/snowflake-arctic-embed-m      │ 110M  │ 768 │ ~1h      │ Better  │
│ nomic-ai/nomic-embed-text-v1.5          │ 137M  │ 768 │ ~1h+     │ Better  │
└─────────────────────────────────────────┴───────┴─────┴──────────┴─────────┘

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
}


def get_model_info(name: str) -> ModelInfo | None:
    for key, info in MODELS.items():
        if info.name == name or key == name:
            return info
    return None


def list_models() -> list[ModelInfo]:
    return list(MODELS.values())


def recommend(hardware: str = "cpu") -> list[str]:
    if hardware == "cpu":
        return ["snowflake-arctic-embed-xs", "snowflake-arctic-embed-s", "all-MiniLM-L6-v2"]
    elif hardware == "gpu":
        return ["snowflake-arctic-embed-m", "nomic-embed-text-v1.5", "snowflake-arctic-embed-s"]
    return ["snowflake-arctic-embed-xs"]
