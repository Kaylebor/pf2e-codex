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
    provider: str = "auto",
    onnx_provider: str | None = None,
    top_k: int = 10,
) -> dict:
    """Run the query suite and return metrics."""
    prov = get_provider(model_name, provider=provider, onnx_provider=onnx_provider)
    conn = sqlite3.connect(str(db_path))
    load_vec_extension(conn)

    queries = load_queries()
    results = []

    for q in queries:
        emb = prov.embed_query(q["query"])
        vec = struct.pack(f"{len(emb)}f", *emb)
        rows = conn.execute(f"""
            SELECT name, distance FROM vec_chunks
            JOIN chunks ON vec_chunks.id = chunks.id
            WHERE vec_chunks.embedding MATCH vec_f32(?)
              AND k = ? ORDER BY distance
        """, (vec, top_k)).fetchall()
        names = [r[0] for r in rows]
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
