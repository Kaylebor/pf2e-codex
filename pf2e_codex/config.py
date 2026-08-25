"""Configuration via Pydantic Settings + TOML config file.

DB paths are derived from model name:
- clean: {data_dir}/pf2e_{model_safe}.db
- private local-full: {data_dir}/pf2e_{model_safe}.local.db
Resolution order (highest priority wins):
1. Keyword arguments (e.g. `Settings(data_dir="~/pf2e")`)
2. Environment variables (`PF2E_DATA_DIR`, `PF2E_MODEL`, etc.)
3. Config file (`~/.config/pf2e-codex/config.toml` or `./pf2e-codex.toml`)
4. Class defaults

Example `~/.config/pf2e-codex/config.toml`:
    model = "snowflake-arctic-embed-s"
    data_dir = "~/pf2e"
    release = "pf2e-8.4.1"
"""

from __future__ import annotations

import json
import os
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pf2e-codex"
DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / "pf2e-codex"
DEFAULT_MODEL = "Snowflake/snowflake-arctic-embed-xs"
DEFAULT_RELEASE = "pf2e-8.4.1"
GITHUB_RELEASE_URL = (
    "https://github.com/foundryvtt/pf2e/releases/download/{version}/json-assets.zip"
)


class CorpusScope(StrEnum):
    """Whether a seed may include private, user-owned PDF derivatives."""

    REDISTRIBUTABLE = "redistributable"
    LOCAL_FULL = "local-full"


class DatabaseScope(StrEnum):
    """Which physically separate database slot a process should use."""

    AUTO = "auto"
    CLEAN = "clean"
    LOCAL = "local"

_CONFIG_PATHS = [
    Path.home() / ".config" / "pf2e-codex" / "config.toml",
    Path.cwd() / "pf2e-codex.toml",
    Path.cwd() / ".pf2e-codex.toml",
]


def _model_safe_name(model: str) -> str:
    return model.replace("/", "--")


def _default_db_path(model: str, data_dir: Path) -> Path:
    safe = _model_safe_name(model)
    return data_dir / f"pf2e_{safe}.db"


def _local_db_path(model: str, data_dir: Path) -> Path:
    safe = _model_safe_name(model)
    return data_dir / f"pf2e_{safe}.local.db"


def _load_toml() -> dict[str, Any]:
    for path in _CONFIG_PATHS:
        if path.exists():
            return tomllib.loads(path.read_text())
    return {}


def _load_env() -> dict[str, Any]:
    raw: dict[str, str] = {}
    for key, val in os.environ.items():
        if key.startswith("PF2E_"):
            raw[key[5:].lower()] = val
    typed: dict[str, Any] = {}
    for k, v in raw.items():
        if k in ("cache_dir", "data_dir", "corpus_dir"):
            typed[k] = Path(v)
        elif k in ("languages", "corpus_include", "corpus_exclude", "corpus_prefer"):
            # get_settings merges sources manually to preserve the documented
            # file < env < kwargs priority. Decode structured values here just
            # as pydantic-settings would when reading the environment itself.
            try:
                typed[k] = json.loads(v)
            except json.JSONDecodeError:
                typed[k] = v
        else:
            typed[k] = v
    return typed


class Settings(BaseSettings):
    """PF2E Codex settings."""

    model_config = SettingsConfigDict(
        env_prefix="PF2E_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    cache_dir: Path = Field(default=DEFAULT_CACHE_DIR, description="Cache directory for downloads")
    data_dir: Path = Field(default=DEFAULT_DATA_DIR, description="Directory for databases and data")
    model: str = Field(default=DEFAULT_MODEL, description="Embedding model name or path")
    provider: str = Field(default="auto", description="Embedding provider: auto, onnx")
    onnx_provider: str = Field(default="auto", description="ONNX provider for batch embedding (pipeline). Prefer GPU for bulk indexing.")
    query_provider: str = Field(
        default="auto",
        description=(
            "ONNX provider for daemon queries (search/rerank). Auto prefers a GPU; "
            "set cpu explicitly only when CPU fallback is intended."
        ),
    )
    release: str = Field(default=DEFAULT_RELEASE, description="PF2E system release version")
    reranker_model: str = Field(default="Kaylebor/pf2e-codex-reranker-quantized", description="Fine-tuned reranker model on HuggingFace. Leave at default for best results (int8 quantized, 0.55GB, multilingual).")
    warmup_threads: int = Field(default=2, description="ONNX intra-op threads during 'pf2e-codex warmup'. Lower = less GPU power draw.")
    ref_weight: float = Field(default=0.0, description="Weight for ref-count boosting in search (0 = off, 0.3-0.5 = moderate). Entries with more incoming references get boosted.")
    rerank_candidates: int = Field(default=50, description="Candidates to feed reranker (more = better quality, slower). Default 50.")
    languages: list[str] = Field(default=["en"], description="Languages to index (e.g. ['en', 'es'])")
    transport: str = Field(default="stdio", description="MCP transport: stdio, sse, or streamable-http")
    corpus_dir: Path | None = Field(
        default=None,
        description=(
            "Local user-owned corpus root. When unset, ./.local-corpus is used "
            "only if that directory exists."
        ),
    )
    corpus_auto_discover: bool = Field(
        default=True,
        description="Automatically discover supported PZO sources under corpus_dir/sources",
    )
    corpus_scope: CorpusScope = Field(
        default=CorpusScope.REDISTRIBUTABLE,
        description=(
            "Seed policy: redistributable excludes purchased-PDF corpus; "
            "local-full explicitly includes it and marks the DB non-publishable"
        ),
    )
    database_scope: DatabaseScope = Field(
        default=DatabaseScope.AUTO,
        description=(
            "Database slot: auto prefers the private local-full DB when it exists; "
            "clean and local force one physical slot"
        ),
    )
    corpus_include: list[str] = Field(
        default_factory=list,
        description="Optional PZO product-code allowlist for local corpus discovery",
    )
    corpus_exclude: list[str] = Field(
        default_factory=list,
        description="Optional PZO product-code denylist for local corpus discovery",
    )
    corpus_prefer: dict[str, str] = Field(
        default_factory=dict,
        description="Optional product-code to preferred relative source path overrides",
    )

    @property
    def db(self) -> Path:
        """Selected physical database path for this process."""
        return self.local_db if self.resolved_database_scope is DatabaseScope.LOCAL else self.clean_db

    @property
    def clean_db(self) -> Path:
        """Redistribution-candidate database slot; never contains PDF rows."""
        return _default_db_path(self.model, self.data_dir)

    @property
    def local_db(self) -> Path:
        """Private complete database slot containing local purchased sources."""
        return _local_db_path(self.model, self.data_dir)

    @property
    def resolved_database_scope(self) -> DatabaseScope:
        """Resolve auto selection, preferring the more complete private slot."""
        if self.database_scope is not DatabaseScope.AUTO:
            return self.database_scope
        if self.corpus_scope is CorpusScope.LOCAL_FULL:
            return DatabaseScope.LOCAL
        return DatabaseScope.LOCAL if self.local_db.is_file() else DatabaseScope.CLEAN

    @property
    def github_release_url(self) -> str:
        return GITHUB_RELEASE_URL.format(version=self.release)

    @property
    def effective_corpus_dir(self) -> Path | None:
        """Return the explicitly configured or locally discovered corpus root."""
        if self.corpus_dir is not None:
            return self.corpus_dir.expanduser().resolve()
        local = Path.cwd() / ".local-corpus"
        return local.resolve() if local.is_dir() else None


def get_settings(**overrides: Any) -> Settings:
    """Load settings with correct priority: file < env < kwargs."""
    file_values = _load_toml()
    env_values = _load_env()
    merged = {**file_values, **env_values, **overrides}
    return Settings(**merged)
