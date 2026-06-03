"""Benchmark: speed and quality across models and providers."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import Settings, get_settings
from .embeddings import get_provider
from .index import SearchIndex
from .pipeline import embed_and_index

BENCH_CHUNKS = 200  # chunks to index for speed benchmark
WARMUP_RUNS = 2
BENCH_RUNS = 5
QUERY_RUNS = 50

# Standard query suite for quality measurement
STANDARD_QUERIES = [
    ("flat-footed while flanking", "Darting Monkey"),
    ("immune to visual effects", "Blinded"),
    ("barbarian instinct extra damage with rage", "Fury Instinct"),
    ("Fireball", "Fireball"),
    ("off-guard condition", "Off-Guard"),
    ("power attack", "Power Attack"),
    ("sneak attack", "Sneak Attack"),
    ("flurry of blows", "Flurry of Blows"),
    ("do status and circumstance penalties stack", "Bonuses and Penalties"),
    ("dread striker", "Dread Striker"),
]


def _hw_info() -> dict:
    import platform
    import subprocess

    info = {
        "cpu": platform.processor() or "unknown",
        "platform": platform.platform(),
        "rocm": False,
        "rocm_version": "",
    }

    # ROCm check
    try:
        result = subprocess.run(
            ["cat", "/opt/rocm/.info/version"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            info["rocm"] = True
            info["rocm_version"] = result.stdout.strip()
            # GPU name: look for marketing name in rocminfo
            gpu = subprocess.run(
                ["rocminfo"], capture_output=True, text=True, timeout=5,
            )
            for line in gpu.stdout.splitlines():
                if "Marketing Name" in line:
                    info["gpu"] = line.split(":")[-1].strip()
                    break
    except Exception:
        pass

    return info


def _hardware_label() -> str:
    hw = _hw_info()
    parts = [f"CPU: {hw['cpu']}"]
    if hw.get("gpu"):
        parts.append(f"GPU: {hw['gpu']}")
    if hw["rocm"]:
        parts.append(f"ROCm: {hw['rocm_version']}")
    parts.append(hw["platform"])
    return " | ".join(parts)


def _onnx_providers_available() -> list[str]:
    """Return list of available ONNX providers (short names) for benchmarking."""
    from .embeddings import _has_onnx, _detect_onnx_provider
    if not _has_onnx():
        return []
    import onnxruntime as ort
    available = ort.get_available_providers()
    rev_map = {
        "MIGraphXExecutionProvider": "migraphx",
        "ROCMExecutionProvider": "rocm",
        "CUDAExecutionProvider": "cuda",
        "CPUExecutionProvider": "cpu",
    }
    result = []
    for full in _ONNX_PROVIDER_ORDER:
        if full in available:
            result.append(rev_map.get(full, full))
    return result


_ONNX_PROVIDER_ORDER = [
    "MIGraphXExecutionProvider",
    "ROCMExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]


def run_benchmark(
    models: list[str] | None = None,
    providers: list[str] | None = None,
    chunks: int = BENCH_CHUNKS,
) -> dict:
    """Run benchmark and return results dict."""
    from .chunker import ChunkBuilder, UUIDResolver
    from .fetcher import extract_all_packs, extract_lang, get_cached_zip

    models = models or ["all-MiniLM-L6-v2", "snowflake-arctic-embed-xs"]
    settings = get_settings()
    hw = _hw_info()

    # Build a small chunk set once
    zip_path = get_cached_zip(settings)
    cache_extract = settings.cache_dir / f"extract-{settings.release}"
    all_entries = extract_all_packs(zip_path, cache_extract)
    resolver = UUIDResolver(all_entries)
    builder = ChunkBuilder(resolver)

    test_chunks = []
    for pack_name in ["feats", "spells", "conditions"]:
        for entry in all_entries.get(pack_name, []):
            for c in builder.build_all(entry, pack_name):
                test_chunks.append(c)
            if len(test_chunks) >= chunks:
                    break
        if len(test_chunks) >= chunks:
            break
    test_chunks = test_chunks[:chunks]

    # Run benchmark
    results = []
    for model_name in models:
        for ptype in (providers or ["onnx"]):
            row = {"model": model_name, "provider": ptype, "status": "ok"}
            try:
                # Quick index: embed + measure time
                prov = get_provider(model_name, provider=ptype)
                texts = [c["text"] for c in test_chunks]

                # Warmup
                prov.embed(texts[:5])

                t0 = time.time()
                for _ in range(WARMUP_RUNS):
                    prov.embed(texts)
                warmup = (time.time() - t0) / WARMUP_RUNS * 1000

                t0 = time.time()
                for _ in range(BENCH_RUNS):
                    prov.embed(texts)
                embed_ms = (time.time() - t0) / BENCH_RUNS * 1000

                # Single-query latency
                prov.embed(["test"])
                t0 = time.time()
                for _ in range(QUERY_RUNS):
                    prov.embed(["test"])
                query_ms = (time.time() - t0) / QUERY_RUNS * 1000

                row["embed_ms"] = round(embed_ms, 1)
                row["query_ms"] = round(query_ms, 2)
                row["warmup_ms"] = round(warmup, 1)
                row["provider_label"] = (
                    type(prov).__name__.replace("Provider", "")
                    + (
                        f" ({prov._session.get_providers()[0]})"
                        if hasattr(prov, "_session") and prov._session
                        else ""
                    )
                )
            except Exception as e:
                row["status"] = f"failed: {e}"
            results.append(row)

    return {
        "hardware": hw,
        "hardware_label": _hardware_label(),
        "chunks": len(test_chunks),
        "results": results,
    }


def print_results(data: dict) -> None:
    print(f"\n=== PF2E Codex Benchmark ===")
    print(f"Hardware: {data['hardware_label']}")
    print(f"Chunks: {data['chunks']}")
    print()
    print(f"{'Model':35s} {'Provider':30s} {'Embed/100':>10s} {'Query':>10s}")
    print(f"{'-'*35} {'-'*30} {'-'*10} {'-'*10}")
    for r in data["results"]:
        if r["status"] == "ok":
            print(f"{r['model']:35s} {r['provider_label']:30s} {f'{r['embed_ms']}ms':>10s} {f'{r['query_ms']}ms':>10s}")
        else:
            print(f"{r['model']:35s} {r['provider']:30s} {'FAIL':>10s} {'':>10s}")
    print()
