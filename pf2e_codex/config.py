"""Configuration via Pydantic Settings + TOML config file.

DB path is derived from model name: {data_dir}/pf2e_{model_safe}.db
Resolution order (highest priority wins):
1. Keyword arguments (e.g. `Settings(data_dir="~/pf2e")`)
2. Environment variables (`PF2E_DATA_DIR`, `PF2E_MODEL`, etc.)
3. Config file (`~/.config/pf2e-codex/config.toml` or `./pf2e-codex.toml`)
4. Class defaults

Example `~/.config/pf2e-codex/config.toml`:
    model = "snowflake-arctic-embed-s"
    data_dir = "~/pf2e"
    release = "pf2e-8.1.2"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import tomllib
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pf2e-codex"
DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / "pf2e-codex"
SYSTEM_DATA_DIR = Path("/usr/share/pf2e-codex/db")
DEFAULT_MODEL = "Snowflake/snowflake-arctic-embed-xs"
DEFAULT_RELEASE = "pf2e-8.2.0"
GITHUB_RELEASE_URL = (
    "https://github.com/foundryvtt/pf2e/releases/download/{version}/json-assets.zip"
)

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
        if k in ("cache_dir", "data_dir"):
            typed[k] = Path(v)
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
    onnx_provider: str = Field(default="auto", description="ONNX execution provider: auto, migraphx, rocm, cuda, cpu, none")
    release: str = Field(default=DEFAULT_RELEASE, description="PF2E system release version")
    reranker_model: str = Field(default="Kaylebor/pf2e-codex-reranker", description="Fine-tuned reranker model on HuggingFace. Empty = use built-in bge-reranker-v2-m3.")
    warmup_threads: int = Field(default=2, description="ONNX intra-op threads during 'pf2e-codex warmup'. Lower = less GPU power draw.")
    transport: str = Field(default="stdio", description="MCP transport: stdio, sse, or streamable-http")

    @property
    def db(self) -> Path:
        """Derived database path from model name and data directory.

        Checks user data dir first, falls back to system dir (PKGBUILD).
        """
        user_db = _default_db_path(self.model, self.data_dir)
        if user_db.exists():
            return user_db
        system_db = _default_db_path(self.model, SYSTEM_DATA_DIR)
        if system_db.exists():
            return system_db
        return user_db  # doesn't exist yet; index command will create in user dir

    @property
    def github_release_url(self) -> str:
        return GITHUB_RELEASE_URL.format(version=self.release)


def get_settings(**overrides: Any) -> Settings:
    """Load settings with correct priority: file < env < kwargs."""
    file_values = _load_toml()
    env_values = _load_env()
    merged = {**file_values, **env_values, **overrides}
    return Settings(**merged)
