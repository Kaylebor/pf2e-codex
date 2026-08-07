"""Daemon-proxy routing tests that do not start an MCP server or models."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pf2e_codex import cli, daemon_proxy


def _settings(data_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=data_dir,
        db=data_dir / "pf2e_test.db",
        rerank_candidates=50,
        ref_weight=0.0,
    )


def test_read_endpoint_uses_settings_data_dir(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "custom")
    settings.data_dir.mkdir()
    (settings.data_dir / "server.json").write_text(json.dumps({
        "endpoint": "http://127.0.0.1:8123/mcp",
        "db_path": str(settings.db.resolve()),
    }))

    assert daemon_proxy._read_endpoint(settings) == "http://127.0.0.1:8123/mcp"


def test_read_endpoint_rejects_registered_database_mismatch(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    (tmp_path / "server.json").write_text(json.dumps({
        "endpoint": "http://127.0.0.1:8123/mcp",
        "db_path": str((tmp_path / "other.db").resolve()),
    }))

    assert daemon_proxy._read_endpoint(settings) is None


@pytest.mark.parametrize(
    ("command", "arguments", "proxy_name", "proxy_result"),
    [
        ("search", ("fireball",), "proxy_search", {"results": []}),
        (
            "get",
            ("fireball",),
            "proxy_get_entry",
            {"id": "spells:fireball", "type": "spell", "name": "Fireball", "pack": "spells", "text": ""},
        ),
        ("related", ("fireball",), "proxy_related", {"results": {}}),
        ("status", (), "proxy_status", {"chunks": 1}),
        ("catalog", (), "proxy_catalog", {"packs": []}),
        ("rules_explain", ("cover",), "proxy_rules_explain", {"results": []}),
    ],
)
def test_cli_queries_pass_resolved_settings_to_daemon_proxy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
    arguments: tuple[str, ...],
    proxy_name: str,
    proxy_result: dict,
) -> None:
    settings = _settings(tmp_path / "custom")
    observed: dict[str, object] = {}

    def fake_proxy(*_args: object, **kwargs: object) -> dict:
        observed.update(kwargs)
        return proxy_result

    monkeypatch.setattr(cli, "_settings", lambda **_kwargs: settings)
    monkeypatch.setattr(cli, "_local_index", lambda _settings: pytest.fail("local index opened"))
    monkeypatch.setattr(daemon_proxy, proxy_name, fake_proxy)
    monkeypatch.setattr(cli, "print_search_results", lambda *_args: None)
    monkeypatch.setattr(cli, "print_status", lambda *_args: None)
    monkeypatch.setattr(cli, "print_catalog", lambda *_args: None)

    getattr(cli, command)(*arguments, data_dir=str(settings.data_dir))

    assert observed["settings"] is settings
