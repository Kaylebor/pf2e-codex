"""Configuration via Pydantic Settings + TOML config file.

Resolution order (highest priority wins):
1. Keyword arguments (e.g. `Settings(db="x.db")`)
2. Environment variables (`PF2E_DB`, `PF2E_MODEL`, etc.)
3. Config file (`~/.config/pf2e-codex/config.toml` or `./pf2e-codex.toml`)
4. Class defaults

Example `~/.config/pf2e-codex/config.toml`:
    model = "snowflake-arctic-embed-s"
    db = "~/pf2e/pf2e_v2.db"
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
DEFAULT_DB = Path("pf2e_v2.db")
DEFAULT_MODEL = "snowflake-arctic-embed-xs"
DEFAULT_RELEASE = "pf2e-8.1.2"
GITHUB_RELEASE_URL = (
    "https://github.com/foundryvtt/pf2e/releases/download/{version}/json-assets.zip"
)

_CONFIG_PATHS = [
    Path.home() / ".config" / "pf2e-codex" / "config.toml",
    Path.cwd() / "pf2e-codex.toml",
    Path.cwd() / ".pf2e-codex.toml",
]


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
        if k in ("cache_dir", "db"):
            typed[k] = Path(v)
        else:
            typed[k] = v
    return typed


class Settings(BaseSettings):
    """PF2E MCP settings."""

    model_config = SettingsConfigDict(
        env_prefix="PF2E_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    cache_dir: Path = Field(default=DEFAULT_CACHE_DIR, description="Cache directory for downloads")
    db: Path = Field(default=DEFAULT_DB, description="Path to sqlite-vec database")
    model: str = Field(default=DEFAULT_MODEL, description="Embedding model name or path")
    release: str = Field(default=DEFAULT_RELEASE, description="PF2E system release version")
    transport: str = Field(default="stdio", description="MCP transport: stdio or sse")

    @property
    def github_release_url(self) -> str:
        return GITHUB_RELEASE_URL.format(version=self.release)


def get_settings(**overrides: Any) -> Settings:
    """Load settings with correct priority: file < env < kwargs."""
    file_values = _load_toml()
    env_values = _load_env()
    merged = {**file_values, **env_values, **overrides}
    return Settings(**merged)
