"""Validation suite: measure retrieval quality against standard queries."""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import yaml

from .embeddings import get_provider
from .index import load_vec_extension


def load_queries(path: Path | None = None) -> list[dict]:
    """Load query suite from YAML."""
    if path is None:
        path = Path(__file__).parent / "queries.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return list(data.values())


def run_validation(
    db_path: Path,
    model_name: str,
    hybrid: bool = True,
    provider: str = "auto",
    onnx_provider: str | None = None,
    top_k: int = 10,
    rerank: bool = True,
) -> dict:
    """Run the query suite and return metrics."""
    from .index import SearchIndex
    search = SearchIndex(db_path, model_name, provider=provider, onnx_provider=onnx_provider)

    queries = load_queries()
    results = []

    for q in queries:
        hits = search.search(q["query"], top_k=top_k, hybrid=hybrid, rerank=rerank)
        names = [r["name"] for r in hits]
        rank = next((i + 1 for i, n in enumerate(names) if q["expected"].lower() in n.lower()), None)
        results.append({
            "query": q["query"],
            "expected": q["expected"],
            "rank": rank,
            "top_3": names[:3],
        })

    ranks = [r["rank"] for r in results]
    mrrs = [1.0 / r if r else 0.0 for r in ranks]

    return {
        "n_queries": len(results),
        "mrr": round(sum(mrrs) / len(mrrs), 4),
        "perfect": sum(1 for r in ranks if r == 1),
        "top3": sum(1 for r in ranks if r and r <= 3),
        "top5": sum(1 for r in ranks if r and r <= 5),
        "not_found": sum(1 for r in ranks if not r),
        "results": results,
    }
