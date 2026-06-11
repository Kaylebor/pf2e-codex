"""Pluggable embedding providers (ONNX, remote)."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

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


# ── ONNX ────────────────────────────────────────────────────────────────

_ONNX_EXEC_PROVIDERS = [
    "MIGraphXExecutionProvider",
    "ROCMExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]

_ONNX_CACHE = Path.home() / ".cache" / "pf2e-codex" / "onnx"
_MIGRAPHX_USER_CACHE = Path.home() / ".cache" / "pf2e-codex" / "onnx" / "migraphx_cache"


def _migraphx_cache_dir() -> Path:
    """Return active MIGraphX cache dir (user cache only)."""
    import os as _os
    env_override = _os.environ.get("PF2E_MIGRAPHX_CACHE_DIR")
    if env_override:
        p = Path(env_override)
        p.mkdir(parents=True, exist_ok=True)
        return p
    _MIGRAPHX_USER_CACHE.mkdir(parents=True, exist_ok=True)
    return _MIGRAPHX_USER_CACHE


def _onnx_cache_dir(model_name: str) -> Path:
    base = os.environ.get("PF2E_ONNX_CACHE_DIR", str(_ONNX_CACHE))
    safe = model_name.replace("/", "--")
    return Path(base) / safe


def _has_onnx() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def _detect_onnx_provider() -> str | None:
    """Return the best candidate from available ONNX execution providers."""
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

        local_path = self._resolve_cache_path()
        self._export_if_needed(local_path)

        model_path = self._cache_dir / "model.onnx"
        if not model_path.exists():
            raise RuntimeError("Model not exported yet")

        def _provider_opts(provider_name: str) -> list[dict[str, str]]:
            """Return provider options for the given provider."""
            if "MIGraphX" in provider_name:
                return [{"migraphx_model_cache_dir": str(_migraphx_cache_dir())}]
            return [{}]

        def _make_session(providers: list[str], p_opts: list[dict] | None = None) -> ort.InferenceSession:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            warmup_threads = os.environ.get("PF2E_WARMUP_THREADS")
            if warmup_threads:
                opts.intra_op_num_threads = int(warmup_threads)
            return ort.InferenceSession(str(model_path), opts, providers=providers, provider_options=p_opts or [])

        if force_provider and force_provider not in ("auto", ""):
            provider_map = {
                "migraphx": "MIGraphXExecutionProvider",
                "rocm": "ROCMExecutionProvider",
                "cuda": "CUDAExecutionProvider",
                "cpu": "CPUExecutionProvider",
            }
            mapped = provider_map.get(force_provider, force_provider)
            try:
                self._session = _make_session([mapped], _provider_opts(mapped))
            except Exception as e:
                raise RuntimeError(f"ONNX provider '{force_provider}' unavailable: {e}")
        else:
            provider = _detect_onnx_provider()
            if not provider:
                raise RuntimeError("No ONNX execution provider available")
            try:
                self._session = _make_session([provider], _provider_opts(provider))
            except Exception:
                print(f"{provider} unavailable, falling back to CPU")
                self._session = _make_session(["CPUExecutionProvider"])

        tokenizer_path = local_path or model_name
        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        info = get_model_info(model_name)
        self._query_prefix = info.query_prefix if info else ""
        self._doc_prefix = info.doc_prefix if info else ""

        test_emb = self.embed(["test"])
        self._dim = len(test_emb[0])

    def _export_if_needed(self, local_path: str | None = None) -> None:
        model_path = self._cache_dir / "model.onnx"
        if model_path.exists():
            return

        export_path = local_path or self.model_name
        print(f"Exporting {self.model_name} to ONNX (one-time, ~30s)...")
        start = time.time()
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
        except ImportError as e:
            for f in self._cache_dir.glob("*"):
                f.unlink()
            raise RuntimeError(
                f"ONNX export needs optimum + torch. Rebuild package or run:\n"
                f"  pip install optimum[onnxruntime]"
            )
        try:
            import transformers.utils.logging
            transformers.utils.logging.set_verbosity_error()
            kwargs = {"export": True, "trust_remote_code": True}
            try:
                model = ORTModelForFeatureExtraction.from_pretrained(export_path, **kwargs)
            except (OSError, EnvironmentError, ConnectionError):
                # Remote download failed (no network, partial cache, HF down)
                # Retry with local-only — works for fully cached models
                print("Remote failed, trying local cache...")
                kwargs["local_files_only"] = True
                model = ORTModelForFeatureExtraction.from_pretrained(export_path, **kwargs)
            transformers.utils.logging.set_verbosity_warning()
            model.save_pretrained(self._cache_dir)
            print(f"Exported in {time.time() - start:.1f}s -> {self._cache_dir}")
        except Exception as e:
            for f in self._cache_dir.glob("*"):
                f.unlink()
            raise RuntimeError(f"ONNX export failed: {e}")

    def _resolve_cache_path(self) -> str | None:
        from pathlib import Path as PPath
        cache_dir = PPath.home() / ".cache" / "huggingface" / "hub"
        if not cache_dir.is_dir():
            return None
        safe_name = self.model_name.replace("/", "--")
        candidates = [
            f"models--{safe_name}",
            f"models--sentence-transformers--{safe_name.split('--')[-1]}",
        ]
        for d in cache_dir.iterdir():
            if not d.name.startswith("models--"):
                continue
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
        result = self._tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        # Some ONNX exports (e5-small, XLM-RoBERTa) expect token_type_ids
        if "token_type_ids" not in result:
            result["token_type_ids"] = result["input_ids"] * 0
        return result

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._doc_prefix:
            texts = [f"{self._doc_prefix}{t}" for t in texts]
        import numpy as np
        batch_size = 32
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenize(batch)
            outputs = self._session.run(None, dict(inputs))[0]
            attention_mask = inputs["attention_mask"]
            mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
            sum_embeddings = np.sum(outputs * mask_expanded, axis=1)
            sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            batch_embeddings = sum_embeddings / sum_mask
            all_embeddings.append(batch_embeddings)
        embeddings = np.vstack(all_embeddings)
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
        provider_type = config.get("type", "onnx")
        cls = self._providers.get(provider_type, ONNXProvider)
        return cls(**{k: v for k, v in config.items() if k != "type"})

    def _from_string(self, model_name: str) -> EmbeddingProvider:
        if model_name.startswith("openai:") or model_name.startswith("synthetic:"):
            raise NotImplementedError("Remote embedding providers not yet implemented")
        return ONNXProvider(model_name)


# Global registry
registry = ProviderRegistry()
registry.register("onnx", ONNXProvider)


def _install_hint() -> str:
    """Return distro-appropriate install instructions for onnxruntime."""
    if Path("/etc/pacman.conf").exists() or Path("/usr/bin/pacman").exists():
        return ("\n  Rebuild package with: makepkg -Cf\n"
                "  The PKGBUILD auto-detects GPU and installs the right variant.")
    return ("\n  pip install onnxruntime       (CPU)\n"
            "  pip install onnxruntime-gpu    (NVIDIA GPU)\n"
            '  pip install onnxruntime-migraphx -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/  (AMD GPU)')


def get_provider(
    model_name: str,
    provider: str = "auto",
    onnx_provider: str | None = None,
) -> EmbeddingProvider:
    """Create the ONNX provider for the given model.

    Args:
        model_name: HuggingFace model name or identifier.
        provider: "auto" or "onnx" (default, uses ONNX). Ignored for now.
        onnx_provider: Override ONNX execution provider. "auto" (default),
                       "migraphx", "rocm", "cuda", "cpu", or "none".
                       Falls back to os.environ["PF2E_ONNX_PROVIDER"].

    Returns:
        An ONNXProvider instance.
    """
    onnx_provider = (
        onnx_provider
        or os.environ.get("PF2E_ONNX_PROVIDER", "")
        or "auto"
    ).lower()

    if onnx_provider == "none" or onnx_provider == "skip":
        raise RuntimeError(
            "ONNX provider disabled. Remove PF2E_ONNX_PROVIDER=none to enable."
        )

    if not _has_onnx():
        hint = _install_hint()
        raise RuntimeError(
            f"onnxruntime not installed. Install it with: {hint}"
        )

    force = onnx_provider if onnx_provider != "auto" else None
    prov = ONNXProvider(model_name, force_provider=force)
    detected = prov._session.get_providers()[0] if hasattr(prov, '_session') else _detect_onnx_provider()
    print(f"Using ONNX with {detected} (model={model_name})")
    return prov
