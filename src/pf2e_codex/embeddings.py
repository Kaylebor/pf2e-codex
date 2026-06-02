"""Pluggable embedding providers (sentence-transformers, ONNX, remote)."""

from __future__ import annotations

import os
import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from .models import get_model_info


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embed([text])[0]


# ── Sentence Transformers ────────────────────────────────────────────────

class SentenceTransformersProvider(EmbeddingProvider):
    """Local sentence-transformers provider."""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        info = get_model_info(model_name)
        self._query_prefix = info.query_prefix if info else ""
        self._doc_prefix = info.doc_prefix if info else ""

    @property
    def dim(self) -> int:
        return self._model.get_embedding_dimension()  # type: ignore[no-any-return]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._doc_prefix:
            texts = [f"{self._doc_prefix}{t}" for t in texts]
        embs = self._model.encode(texts, batch_size=32, show_progress_bar=False)
        return embs.tolist()  # type: ignore[no-any-return]

    def embed_query(self, text: str) -> list[float]:
        if self._query_prefix:
            text = f"{self._query_prefix}{text}"
        emb = self._model.encode([text])[0]
        return emb.tolist()  # type: ignore[no-any-return]


# ── ONNX ────────────────────────────────────────────────────────────────

# Provider priority: native ROCm first, then CUDA, then CPU.
# ZLUDA self-identifies as CUDA; we prefer native ROCm over emulated CUDA.
_ONNX_EXEC_PROVIDERS = [
    "ROCMExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]

_ONNX_CACHE = Path.home() / ".cache" / "pf2e-codex" / "onnx"


def _onnx_cache_dir(model_name: str) -> Path:
    safe = model_name.replace("/", "--")
    return _ONNX_CACHE / safe


def _has_onnx() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def _detect_onnx_provider() -> str | None:
    """Return the best available ONNX execution provider, or None."""
    if not _has_onnx():
        return None
    import onnxruntime as ort
    available = set(ort.get_available_providers())
    for preferred in _ONNX_EXEC_PROVIDERS:
        if preferred in available:
            return preferred
    return None


class ONNXProvider(EmbeddingProvider):
    """ONNX Runtime provider with automatic model export."""

    def __init__(self, model_name: str):
        import onnxruntime as ort
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self._cache_dir = _onnx_cache_dir(model_name)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Export or reuse cached ONNX model
        self._export_if_needed()

        # Create session with best provider
        provider = _detect_onnx_provider()
        if not provider:
            raise RuntimeError("No ONNX execution provider available")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        model_path = self._cache_dir / "model.onnx"
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options,
            providers=[provider],
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        info = get_model_info(model_name)
        self._query_prefix = info.query_prefix if info else ""
        self._doc_prefix = info.doc_prefix if info else ""

        # Determine embedding dimension from first inference
        test_emb = self.embed(["test"])
        self._dim = len(test_emb[0])

    def _export_if_needed(self) -> None:
        model_path = self._cache_dir / "model.onnx"
        if model_path.exists():
            return

        print(f"Exporting {self.model_name} to ONNX (one-time)...")
        start = time.time()
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            model = ORTModelForFeatureExtraction.from_pretrained(
                self.model_name, export=True
            )
            model.save_pretrained(self._cache_dir)
            print(f"Exported in {time.time() - start:.1f}s -> {self._cache_dir}")
        except Exception as e:
            # Clean up partial export
            for f in self._cache_dir.glob("*"):
                f.unlink()
            raise RuntimeError(f"ONNX export failed: {e}")

    @property
    def dim(self) -> int:
        return self._dim

    def _tokenize(self, texts: list[str]) -> dict:
        return self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._doc_prefix:
            texts = [f"{self._doc_prefix}{t}" for t in texts]

        inputs = self._tokenize(texts)
        outputs = self._session.run(None, dict(inputs))[0]
        # Mean pooling
        import numpy as np
        attention_mask = inputs["attention_mask"]
        mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        sum_embeddings = np.sum(outputs * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_embeddings / sum_mask
        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        if self._query_prefix:
            text = f"{self._query_prefix}{text}"
        return self.embed([text])[0]


# ── Registry ────────────────────────────────────────────────────────────

class ProviderRegistry:
    """Registry for embedding providers."""

    def __init__(self):
        self._providers: dict[str, type[EmbeddingProvider]] = {}

    def register(self, name: str, provider_cls: type[EmbeddingProvider]) -> None:
        self._providers[name] = provider_cls

    def create(self, config: str | dict[str, Any]) -> EmbeddingProvider:
        if isinstance(config, str):
            return self._from_string(config)
        provider_type = config.get("type", "sentence_transformers")
        cls = self._providers.get(provider_type, SentenceTransformersProvider)
        return cls(**{k: v for k, v in config.items() if k != "type"})

    def _from_string(self, model_name: str) -> EmbeddingProvider:
        if model_name.startswith("openai:") or model_name.startswith("synthetic:"):
            raise NotImplementedError("Remote embedding providers not yet implemented")
        return SentenceTransformersProvider(model_name)


# Global registry
registry = ProviderRegistry()
registry.register("sentence_transformers", SentenceTransformersProvider)
registry.register("onnx", ONNXProvider)


def get_provider(model_name: str, provider: str = "auto") -> EmbeddingProvider:
    """Create the best available provider for the given model.

    Args:
        model_name: HuggingFace model name or identifier.
        provider: "auto" (try ONNX, fall back to sentence-transformers),
                  "onnx" (force ONNX, fail if unavailable),
                  "sentence_transformers" (skip ONNX).

    Returns:
        An EmbeddingProvider instance.
    """
    provider = os.environ.get("PF2E_PROVIDER", provider).lower()

    if provider == "sentence_transformers":
        return SentenceTransformersProvider(model_name)

    if provider in ("auto", "onnx"):
        if _has_onnx() and _detect_onnx_provider():
            try:
                prov = ONNXProvider(model_name)
                print(f"Using ONNX with {_detect_onnx_provider()}")
                return prov
            except Exception as e:
                if provider == "onnx":
                    raise
                warnings.warn(f"ONNX unavailable ({e}), falling back to sentence-transformers")
        elif provider == "onnx":
            raise RuntimeError("ONNX requested but onnxruntime not installed")

    return SentenceTransformersProvider(model_name)
