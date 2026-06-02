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

# Provider priority: MIGraphX (ROCm 7+ replacement), ROCm (ROCm 6), CUDA, then CPU.
# ZLUDA self-identifies as CUDA; prefer native AMD backends over emulated CUDA.
_ONNX_EXEC_PROVIDERS = [
    "MIGraphXExecutionProvider",
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
    """Return the best candidate from available ONNX execution providers.

    'ort.get_available_providers()' lists what the binary *can* do,
    but may include providers whose system libraries aren't loadable
    (e.g. ROCm 7.x with onnxruntime-rocm 1.22.2). Actual verification
    happens at session creation in ONNXProvider.__init__.
    """
    if not _has_onnx():
        return None
    import onnxruntime as ort
    available = ort.get_available_providers()
    for preferred in _ONNX_EXEC_PROVIDERS:
        if preferred in available:
            return preferred
    return None


class ONNXProvider(EmbeddingProvider):
    """ONNX Runtime provider with automatic model export."""

    def __init__(self, model_name: str, force_provider: str | None = None):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.model_name = model_name
        self._cache_dir = _onnx_cache_dir(model_name)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Resolve model to a local HF cache path (avoids Hub auth)
        local_path = self._resolve_cache_path()

        # Export or reuse cached ONNX model
        self._export_if_needed(local_path)

        model_path = self._cache_dir / "model.onnx"
        if not model_path.exists():
            raise RuntimeError("Model not exported yet")

        def _make_session(providers: list[str]) -> ort.InferenceSession:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            return ort.InferenceSession(
                str(model_path), opts, providers=providers
            )

        # Determine session providers
        if force_provider and force_provider not in ("auto", ""):
            # Map short names to ONNX provider names
            provider_map = {
                "migraphx": "MIGraphXExecutionProvider",
                "rocm": "ROCMExecutionProvider",
                "cuda": "CUDAExecutionProvider",
                "cpu": "CPUExecutionProvider",
            }
            mapped = provider_map.get(force_provider, force_provider)
            try:
                self._session = _make_session([mapped])
            except Exception as e:
                raise RuntimeError(f"ONNX provider '{force_provider}' unavailable: {e}")
        else:
            # Auto-detect best provider
            provider = _detect_onnx_provider()
            if not provider:
                raise RuntimeError("No ONNX execution provider available")
            try:
                self._session = _make_session([provider])
            except Exception:
                print(f"{provider} unavailable, falling back to CPU")
                self._session = _make_session(["CPUExecutionProvider"])

        # Load tokenizer from local path (avoids Hub auth for short names)
        tokenizer_path = local_path or model_name
        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        info = get_model_info(model_name)
        self._query_prefix = info.query_prefix if info else ""
        self._doc_prefix = info.doc_prefix if info else ""

        # Determine embedding dimension from first inference
        test_emb = self.embed(["test"])
        self._dim = len(test_emb[0])

    def _export_if_needed(self, local_path: str | None = None) -> None:
        model_path = self._cache_dir / "model.onnx"
        if model_path.exists():
            return

        export_path = local_path or self.model_name
        print(f"Exporting {self.model_name} to ONNX (one-time)...")
        start = time.time()
        try:
            # Export in offline mode if we have a local cache path
            kwargs = {"export": True}
            if local_path:
                kwargs["local_files_only"] = True
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            model = ORTModelForFeatureExtraction.from_pretrained(
                export_path, **kwargs
            )
            model.save_pretrained(self._cache_dir)
            print(f"Exported in {time.time() - start:.1f}s -> {self._cache_dir}")
        except Exception as e:
            for f in self._cache_dir.glob("*"):
                f.unlink()
            raise RuntimeError(f"ONNX export failed: {e}")

    def _resolve_cache_path(self) -> str | None:
        """Find local HF cache path for the model without hitting Hub."""
        from pathlib import Path as PPath

        cache_dir = PPath.home() / ".cache" / "huggingface" / "hub"
        if not cache_dir.is_dir():
            return None

        # Build candidate directory names from model name variants
        safe_name = self.model_name.replace("/", "--")
        candidates = [
            f"models--{safe_name}",
            f"models--sentence-transformers--{safe_name.split('--')[-1]}",
        ]

        for d in cache_dir.iterdir():
            if not d.name.startswith("models--"):
                continue
            # Match by suffix (e.g. all-MiniLM-L6-v2 matches models--sentence-transformers--all-MiniLM-L6-v2)
            if any(cand in d.name for cand in candidates):
                snap_dir = d / "snapshots"
                if not snap_dir.is_dir():
                    continue
                snapshots = list(snap_dir.iterdir())
                if snapshots:
                    return str(snapshots[0])
        return None

    @property
    def dim(self) -> int:
        return self._dim

    def _tokenize(self, texts: list[str]) -> dict:
        return self._tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="np",
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._doc_prefix:
            texts = [f"{self._doc_prefix}{t}" for t in texts]

        import numpy as np
        # Batch to avoid MIGraphX compiling for enormous shapes
        batch_size = 32
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenize(batch)
            outputs = self._session.run(None, dict(inputs))[0]
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
            sum_embeddings = np.sum(outputs * mask_expanded, axis=1)
            sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            batch_embeddings = sum_embeddings / sum_mask
            all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings)
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        embeddings = embeddings / norms
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


def get_provider(
    model_name: str,
    provider: str = "auto",
    onnx_provider: str | None = None,
) -> EmbeddingProvider:
    """Create the best available provider for the given model.

    Args:
        model_name: HuggingFace model name or identifier.
        provider: "auto" (try ONNX, fall back to sentence-transformers),
                  "onnx" (force ONNX, fail if unavailable),
                  "sentence_transformers" (skip ONNX).
        onnx_provider: Override ONNX execution provider. "auto" (default),
                       "migraphx", "rocm", "cuda", "cpu", or "none" (skip ONNX).
                       Falls back to os.environ["PF2E_ONNX_PROVIDER"].

    Returns:
        An EmbeddingProvider instance.
    """
    provider = os.environ.get("PF2E_PROVIDER", provider).lower()

    if provider == "sentence_transformers":
        return SentenceTransformersProvider(model_name)

    # Check onnx_provider override
    onnx_provider = (
        onnx_provider
        or os.environ.get("PF2E_ONNX_PROVIDER", "")
        or "auto"
    ).lower()

    if onnx_provider == "none" or onnx_provider == "skip":
        return SentenceTransformersProvider(model_name)

    if provider in ("auto", "onnx"):
        if _has_onnx():
            try:
                prov = ONNXProvider(model_name, force_provider=onnx_provider if onnx_provider != "auto" else None)
                detected = prov._session.get_providers()[0] if hasattr(prov, '_session') else _detect_onnx_provider()
                print(f"Using ONNX with {detected}")
                return prov
            except Exception as e:
                if provider == "onnx":
                    raise
                warnings.warn(f"ONNX unavailable ({e}), falling back to sentence-transformers")
        elif provider == "onnx":
            raise RuntimeError("ONNX requested but onnxruntime not installed")

    return SentenceTransformersProvider(model_name)
