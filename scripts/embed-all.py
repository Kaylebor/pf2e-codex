#!/usr/bin/env python3
"""embed-all.py — Generate sqlite-vec DBs for all officially-supported models.

Usage:
  ./scripts/embed-all.py [--data-dir DIR] [--concurrency N] [MODEL...]

Assumes pf2e-codex is on PATH. Runs models in parallel with configurable
concurrency. Ctrl+C cleanly terminates all running jobs.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# All officially-supported embedding models
ALL_MODELS = [
    "Snowflake/snowflake-arctic-embed-xs",
    "Snowflake/snowflake-arctic-embed-s",
    "Snowflake/snowflake-arctic-embed-m",
    "all-MiniLM-L6-v2",
    "intfloat/e5-small-v2",
    "nomic-ai/nomic-embed-text-v1.5",
    "BAAI/bge-m3",
]

DATA_DIR = Path(os.environ.get("PF2E_DATA_DIR", Path.home() / ".local" / "share" / "pf2e-codex"))
CONCURRENCY = 2

# Track running subprocesses for cleanup
_running: list[subprocess.Popen] = []


def _sig_handler(signum: int, frame: object) -> None:
    print("\nInterrupted — killing running jobs...")
    for proc in _running:
        try:
            proc.terminate()
        except Exception:
            pass
    # Give them a moment, then force-kill
    time.sleep(0.5)
    for proc in _running:
        try:
            proc.kill()
        except Exception:
            pass
    print("Cleaned up.")
    sys.exit(130)


signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)


def model_safe(name: str) -> str:
    return name.replace("/", "--")


def db_path(model: str) -> Path:
    return DATA_DIR / f"pf2e_{model_safe(model)}.db"


def embed_one(model: str) -> bool:
    """Run pf2e-codex index for one model. Returns True on success."""
    log_path = Path(f"/tmp/pf2e-embed-{model_safe(model)}.log")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            ["pf2e-codex", "index", "-m", model, "--data-dir", str(DATA_DIR)],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        _running.append(proc)
        try:
            rc = proc.wait()
        finally:
            _running.remove(proc)
    if rc == 0:
        log_path.unlink(missing_ok=True)  # clean up success logs
        return True
    else:
        print(f"  Log: {log_path}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all PF2E embedding DBs")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Data directory")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY, help="Max parallel jobs")
    parser.add_argument("models", nargs="*", help="Specific models (default: all)")
    args = parser.parse_args()

    global DATA_DIR, CONCURRENCY
    DATA_DIR = Path(args.data_dir)
    CONCURRENCY = args.concurrency
    models = args.models or ALL_MODELS

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Skip already-indexed models
    pending = []
    for model in models:
        if db_path(model).exists():
            print(f"[skip] {model}")
        else:
            pending.append(model)

    if not pending:
        print("All models already indexed.")
        return

    print(f"Embedding {len(pending)} model(s) with concurrency={CONCURRENCY}")
    print(f"Data dir: {DATA_DIR}")
    print()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[str, bool] = {}
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {pool.submit(embed_one, m): m for m in pending}
        for future in as_completed(futures):
            model = futures[future]
            try:
                ok = future.result()
            except Exception as e:
                print(f"[FAIL] {model} — {e}")
                ok = False
            results[model] = ok
            status = "[done]" if ok else "[FAIL]"
            print(f"{status}  {model}")

    print()
    print("=== Results ===")
    failed = 0
    for model in pending:
        if results.get(model):
            print(f"  OK:   {model}")
        else:
            print(f"  FAIL: {model}")
            failed += 1

    if failed:
        print(f"\n{failed} model(s) failed.")
        sys.exit(1)

    print("\nAll models embedded successfully.")


if __name__ == "__main__":
    main()
