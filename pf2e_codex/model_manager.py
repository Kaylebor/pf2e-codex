"""Centralized model lifecycle manager.

Single process-wide owner of all ONNX sessions — one embedding model, one
reranker.  Eagerly initializes both models on start(), serializing MIGraphX
compilation (which is not thread-safe).  All inference flows through here;
no other module creates ONNX providers or sessions.
"""

from __future__ import annotations

import sys as _sys
import threading as _threading
import time as _time
from pathlib import Path
from typing import Any

from .embeddings import ONNXProvider, get_provider
from .reranker import Reranker


class ModelManager:
    """Process-wide singleton that owns embedding + reranker ONNX sessions."""

    def __init__(self, model_name: str, reranker_model: str = "", provider: str = "auto", onnx_provider: str | None = None):
        self._model_name = model_name
        self._reranker_model = reranker_model
        self._provider_type = provider
        self._onnx_provider = onnx_provider

        self._embedding: ONNXProvider | None = None
        self._reranker: Reranker | None = None
        self._ready = _threading.Event()

        # Serialize ALL MIGraphX access — one operation at a time.
        # MIGraphX global state is not thread-safe.
        self._lock = _threading.Lock()

    # ── lifecycle ──────────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        """Both models loaded and ready for inference."""
        return self._ready.is_set()

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Block until both models are ready."""
        return self._ready.wait(timeout=timeout)

    def start(self) -> None:
        """Eagerly load both models (blocks until done)."""
        t0 = _time.monotonic()
        _sys.stderr.write(f"[manager] Starting ({self._model_name}, reranker={self._reranker_model})\n")
        _sys.stderr.flush()
        try:
            with self._lock:
                _sys.stderr.write("[manager] Loading embedding model...\n")
                _sys.stderr.flush()
                self._embedding = get_provider(
                    self._model_name,
                    provider=self._provider_type,
                    onnx_provider=self._onnx_provider,
                )
                # Run one warmup inference to trigger MIGraphX compile.
                _ = self._embedding.embed_query("warmup")
                _sys.stderr.write(f"[manager] Embedding ready ({_time.monotonic() - t0:.0f}s)\n")
                _sys.stderr.flush()

                if self._reranker_model:
                    _sys.stderr.write(f"[manager] Loading reranker ({self._reranker_model})...\n")
                    _sys.stderr.flush()
                    self._reranker = Reranker(model_repo=self._reranker_model, force_provider=self._onnx_provider)
                    # Warmup inference.
                    self._reranker.rerank("warmup", [{"text": "warmup", "id": "_w"}], top_k=1)
                    _sys.stderr.write(f"[manager] Reranker ready ({_time.monotonic() - t0:.0f}s)\n")
                    _sys.stderr.flush()
        except Exception as e:
            _sys.stderr.write(f"[manager] FAILED: {e}\n")
            _sys.stderr.flush()
            raise

        self._ready.set()
        _sys.stderr.write(f"[manager] Done ({_time.monotonic() - t0:.0f}s)\n")
        _sys.stderr.flush()

    # ── inference API ──────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts (thread-safe)."""
        if not self._ready.is_set():
            raise RuntimeError("ModelManager not ready — call start() first")
        with self._lock:
            return self._embedding.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text (thread-safe)."""
        if not self._ready.is_set():
            raise RuntimeError("ModelManager not ready — call start() first")
        _sys.stderr.write(f"[manager] embed_query({repr(text[:30])})\n")
        _sys.stderr.flush()
        with self._lock:
            return self._embedding.embed_query(text)

    def rerank(
        self, query: str, documents: list[dict], top_k: int = 5
    ) -> list[dict]:
        """Rerank candidate documents with cross-encoder (thread-safe)."""
        if not self._ready.is_set():
            raise RuntimeError("ModelManager not ready — call start() first")
        if self._reranker is None:
            raise RuntimeError("Reranker model not configured")
        _sys.stderr.write(f"[manager] rerank({repr(query[:30])}, {len(documents)} docs)\n")
        _sys.stderr.flush()
        with self._lock:
            return self._reranker.rerank(query, documents, top_k=top_k)

    @property
    def embedding_dim(self) -> int:
        if self._embedding is None:
            raise RuntimeError("ModelManager not ready")
        return self._embedding.dim

    @property
    def model_name(self) -> str:
        return self._model_name
