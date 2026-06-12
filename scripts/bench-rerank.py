"""Benchmark reranker speed vs quality across candidate counts.

Usage:
    uv run python scripts/bench-rerank.py
    uv run python scripts/bench-rerank.py --rerank-candidates 10 25 50
    uv run python scripts/bench-rerank.py --no-rerank  # baseline
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pf2e_codex.model_manager import ModelManager
from pf2e_codex.config import get_settings
from pf2e_codex.index import SearchIndex

QUERIES = [
    "fireball",
    "power attack",
    "sneak attack",
    "blinded",
    "heal",
    "flat-footed",
    "dwarf",
    "elf",
    "wizard",
    "rogue",
    "prismatic ray",
    "how do I remove the stunned condition",
    "what spells deal cold damage",
    "crafting requirements for magic items",
    "treat wounds medicine check",
]


def main():
    settings = get_settings()
    print(f"Model: {settings.model}")
    print(f"Reranker: {settings.reranker_model}")
    print(f"Provider: {settings.query_provider}")

    manager = ModelManager(
        model_name=settings.model,
        reranker_model=settings.reranker_model,
        onnx_provider=settings.query_provider,
    )
    manager.start()
    search = SearchIndex(settings.db, manager)

    candidates = [5, 10, 15, 20, 25, 35, 50]
    print(f"\n{'cands':>6} {'avg_s':>7} {'min_s':>7} {'max_s':>7}")
    print("-" * 35)

    for nc in candidates:
        times: list[float] = []
        for q in QUERIES:
            t0 = time.monotonic()
            results = search.search(
                q, top_k=5, hybrid=True, rerank=True, rerank_candidates=nc,
            )
            dt = time.monotonic() - t0
            times.append(dt)

        avg = sum(times) / len(times)
        print(f"{nc:>6} {avg:>7.2f} {min(times):>7.2f} {max(times):>7.2f}")

    # Baseline: no rerank
    times_no: list[float] = []
    for q in QUERIES:
        t0 = time.monotonic()
        search.search(q, top_k=5, hybrid=True, rerank=False)
        dt = time.monotonic() - t0
        times_no.append(dt)
    avg_no = sum(times_no) / len(times_no)
    print(f"{'none':>6} {avg_no:>7.2f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
