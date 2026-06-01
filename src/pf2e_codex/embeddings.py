"""Pluggable embedding providers (sentence-transformers, ONNX, remote)."""

from __future__ import annotations

from abc import ABC, abstractmethod
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


def get_provider(model_name: str) -> EmbeddingProvider:
    return registry.create(model_name)
