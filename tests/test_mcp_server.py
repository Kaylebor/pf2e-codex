"""Focused MCP 2 server wiring tests that avoid loading real models."""

from __future__ import annotations

import asyncio
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
    def __init__(self, db: Path, manager: _FakeManager) -> None:
        self.db = db
        self.manager = manager


class _FakeApp:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def run(self, *args: object, **kwargs: object) -> None:
        self.calls.append((args, kwargs))


def _settings(transport: str) -> SimpleNamespace:
    return SimpleNamespace(
        db=Path("test.db"),
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
            lambda _host, _port, _transport: tmp_path / "server.json",
        )
        monkeypatch.setattr(mcp_server, "_register_cleanup", lambda: None)

    mcp_server.serve(settings, host="127.0.0.1", port=8123)

    assert app.calls == [((), expected_kwargs)]
