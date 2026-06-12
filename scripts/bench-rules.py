"""Compare rules_explain quality across approaches.

Usage:
    uv run python scripts/bench-rules.py

Scores: 1 = correct rule page at #1, 0.5 at #2, 0.25 at #3, 0 = not found.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pf2e_codex.model_manager import ModelManager
from pf2e_codex.config import get_settings
from pf2e_codex.index import SearchIndex

# (topic, expected_type, expected_keyword_in_name_or_text)
QUERIES = [
    # Conditions — rules_explain should find the condition page
    ("flanking", "condition", ["off-guard", "flanking"]),
    ("off-guard", "condition", ["off-guard"]),
    ("blinded", "condition", ["blinded"]),
    ("stunned", "condition", ["stunned"]),
    ("dying", "condition", ["dying"]),
    ("concealed", "condition", ["concealed"]),
    ("frightened", "condition", ["frightened"]),
    ("paralyzed", "condition", ["paralyzed"]),
    ("cover", "condition", ["cover"]),
    ("prone", "condition", ["prone"]),
    # Mixed — general rules questions
    ("treat wounds", "action", ["treat wounds"]),
    ("craft magic items", "action", ["craft", "item"]),
    ("what is the dying condition", "condition", ["dying"]),
]


def score_result(result: dict, expected_type: str, keywords: list[str]) -> float:
    """Check if result matches expected type and contains keywords."""
    actual_type = result.get("type", "")
    name = result.get("name", "").lower()
    text = result.get("text", "").lower()[:200]
    combined = name + " " + text
    type_ok = actual_type == expected_type or actual_type == "journal_page"
    kw_ok = any(k.lower() in combined for k in keywords)
    if type_ok and kw_ok:
        return 1.0
    if type_ok or kw_ok:
        return 0.5
    return 0.0


def main():
    print(f"Approach: {Path(sys.argv[0]).stem.replace('bench-rules-', '') or 'baseline'}")

    settings = get_settings()
    manager = ModelManager(model_name=settings.model, reranker_model=settings.reranker_model, onnx_provider="cpu")
    manager.start()
    search = SearchIndex(settings.db, manager)

    scores: list[float] = []
    times: list[float] = []

    for topic, exp_type, keywords in QUERIES:
        t0 = time.monotonic()
        results = search.rules_explain(topic, top_k=5)
        dt = time.monotonic() - t0
        times.append(dt)

        best = 0.0
        for i, r in enumerate(results):
            s = score_result(r, exp_type, keywords)
            rank_penalty = 1.0 / (i + 1)
            weighted = s * rank_penalty
            if weighted > best:
                best = weighted

        # Print first result for manual inspection
        first = results[0] if results else {}
        print(f"  {topic:15s} → Rank: {first.get('name','?')[:40]:40s} ({first.get('type','?'):15s}) score={best:.2f}")

        scores.append(best)

    avg = sum(scores) / len(scores)
    avg_t = sum(times) / len(times)
    print(f"\n  Average quality: {avg:.2f}")
    print(f"  Average time: {avg_t:.2f}s")
    print(f"  Perfect (1.0): {sum(1 for s in scores if s >= 0.8)}/{len(scores)}")
    print(f"  Found (>{0}): {sum(1 for s in scores if s > 0)}/{len(scores)}")


if __name__ == "__main__":
    main()
