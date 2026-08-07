"""Focused MCP 2 server wiring tests that avoid loading real models."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.server import MCPServer

from pf2e_codex import mcp_server


class _FakeManager:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def start(self) -> None:
        pass


class _FakeSearchIndex:
    def __init__(
        self,
        db: Path,
        manager: _FakeManager,
        *,
        expected_scope: str | None = None,
        clean_db_path: Path | None = None,
    ) -> None:
        self.db = db
        self.manager = manager
        self.expected_scope = expected_scope
        self.clean_db_path = clean_db_path


class _FakeApp:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def run(self, *args: object, **kwargs: object) -> None:
        self.calls.append((args, kwargs))


def _settings(transport: str) -> SimpleNamespace:
    return SimpleNamespace(
        db=Path("test.db"),
        data_dir=Path("test-data"),
        model="test-model",
        reranker_model="test-reranker",
        provider="onnx",
        query_provider="cpu",
        transport=transport,
    )


def test_create_mcp_app_registers_tools_without_loading_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pf2e_codex.model_manager as model_manager

    monkeypatch.setattr(model_manager, "ModelManager", _FakeManager)
    monkeypatch.setattr(mcp_server, "SearchIndex", _FakeSearchIndex)

    app = mcp_server.create_mcp_app(_settings("stdio"))

    assert isinstance(app, MCPServer)
    tools = {tool.name for tool in asyncio.run(app.list_tools())}
    assert tools == {
        "pf2e_search",
        "pf2e_flag_result",
        "pf2e_get_entry",
        "pf2e_rules_explain",
        "pf2e_related",
        "pf2e_catalog",
        "pf2e_index_status",
        "pf2e_query_db",
    }


def test_flagged_result_stays_in_served_data_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import pf2e_codex.model_manager as model_manager

    settings = _settings("stdio")
    settings.data_dir = tmp_path / "custom-data"
    monkeypatch.setattr(model_manager, "ModelManager", _FakeManager)
    monkeypatch.setattr(mcp_server, "SearchIndex", _FakeSearchIndex)
    mcp_server._RECENT.clear()
    mcp_server._RECENT.append(
        {
            "query": "private rule",
            "results": [{"name": "Private Result", "text": "private text"}],
            "params": {},
            "ts": "2026-08-07T00:00:00+00:00",
        }
    )

    app = mcp_server.create_mcp_app(settings)
    asyncio.run(app.call_tool("pf2e_flag_result", {"result_index": 1}))

    flagged = settings.data_dir / "flagged_results.jsonl"
    assert flagged.is_file()
    assert "private text" in flagged.read_text()


@pytest.mark.parametrize(
    ("transport", "expected_kwargs"),
    [
        ("stdio", {"transport": "stdio"}),
        (
            "streamable-http",
            {"transport": "streamable-http", "host": "127.0.0.1", "port": 8123},
        ),
        ("sse", {"transport": "sse", "host": "127.0.0.1", "port": 8123}),
    ],
)
def test_serve_passes_mcp2_transport_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    transport: str,
    expected_kwargs: dict[str, object],
) -> None:
    app = _FakeApp()
    settings = _settings(transport)
    monkeypatch.setattr(mcp_server, "create_mcp_app", lambda _settings: app)

    if transport != "stdio":
        monkeypatch.setattr(mcp_server, "_pick_port", lambda _host, _port: 8123)
        monkeypatch.setattr(
            mcp_server,
            "_write_server_json",
            lambda _settings, _host, _port, _transport: tmp_path / "server.json",
        )
        monkeypatch.setattr(mcp_server, "_register_cleanup", lambda: None)

    mcp_server.serve(settings, host="127.0.0.1", port=8123)

    assert app.calls == [((), expected_kwargs)]


def test_write_server_json_uses_served_settings_data_dir(tmp_path: Path) -> None:
    settings = _settings("streamable-http")
    settings.data_dir = tmp_path / "custom-data"

    path = mcp_server._write_server_json(settings, "127.0.0.1", 8123, "streamable-http")

    assert path == settings.data_dir / "server.json"
    registration = json.loads(path.read_text())
    assert registration["endpoint"] == "http://127.0.0.1:8123/mcp"
    assert registration["db_path"] == str(settings.db.resolve())
    assert registration["pid"] == os.getpid()
    assert isinstance(registration["registration_id"], str)
    mcp_server._SERVER_JSON = path
    mcp_server._cleanup_server_json()


def test_write_server_json_refuses_live_registration(tmp_path: Path) -> None:
    settings = _settings("streamable-http")
    settings.data_dir = tmp_path
    path = tmp_path / "server.json"
    original = {"pid": os.getpid(), "registration_id": "other", "endpoint": "http://live"}
    path.write_text(json.dumps(original))

    with pytest.raises(RuntimeError, match="already live"):
        mcp_server._write_server_json(settings, "127.0.0.1", 8123, "streamable-http")

    assert json.loads(path.read_text()) == original


def test_write_server_json_replaces_verified_stale_registration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    settings = _settings("streamable-http")
    settings.data_dir = tmp_path
    path = tmp_path / "server.json"
    path.write_text(json.dumps({"pid": 987654, "registration_id": "stale"}))
    monkeypatch.setattr(mcp_server, "_pid_is_alive", lambda _pid: False)

    assert mcp_server._write_server_json(settings, "127.0.0.1", 8123, "streamable-http") == path
    registration = json.loads(path.read_text())
    assert registration["pid"] == os.getpid()
    assert registration["registration_id"] != "stale"
    mcp_server._SERVER_JSON = path
    mcp_server._cleanup_server_json()


def test_cleanup_does_not_remove_another_daemon_registration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    path = tmp_path / "server.json"
    path.write_text(json.dumps({"pid": os.getpid(), "registration_id": "new-owner"}))
    monkeypatch.setattr(mcp_server, "_SERVER_JSON", path)
    monkeypatch.setattr(mcp_server, "_SERVER_REGISTRATION_ID", "old-owner")

    mcp_server._cleanup_server_json()

    assert path.exists()
