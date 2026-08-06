"""Regression tests for search and incremental-index fixes."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("sqlite_vec")
pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

from pf2e_codex import index as index_module
from pf2e_codex import pipeline


class _Cursor:
    def __init__(self, rows: list[tuple]):
        self._rows = rows

    def fetchall(self) -> list[tuple]:
        return self._rows


class _Connection:
    def __init__(self, rows: list[tuple]):
        self.rows = rows
        self.calls: list[tuple[str, object]] = []

    def execute(self, sql: str, params: object = ()) -> _Cursor:
        self.calls.append((sql, params))
        return _Cursor(self.rows)


class _EnrichmentConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def execute(self, sql: str, params: object = ()) -> _Cursor:
        self.calls.append((sql, params))
        if "target_uuid IN" in sql:
            return _Cursor([("a", "source-a"), ("b", "source-b")])
        if "SELECT id FROM chunks" in sql:
            return _Cursor([("pack:a",), ("pack:b",)])
        return _Cursor([])


def test_rules_explain_uses_a_defined_candidate_count() -> None:
    fake = index_module.SearchIndex.__new__(index_module.SearchIndex)
    fake._ensure_loaded = lambda: None
    fake._encode = lambda _topic: [0.0]
    fake._conn_ro = _Connection([
        ("conditions:flanking", "Flanking", "condition", "conditions", "text", "ORC", 1, 0.1),
    ])

    results = fake.rules_explain("flanking", top_k=1)

    assert results[0]["name"] == "Flanking"
    assert fake._conn_ro.calls[0][1][1] == 50


def test_reference_weight_changes_final_order() -> None:
    results = [
        {"id": "a", "rrf_score": 0.9, "incoming_refs": []},
        {"id": "b", "rrf_score": 0.8, "incoming_refs": [{"id": "x"}]},
    ]

    weighted = index_module._apply_ref_weight(results, 0.75, top_k=1)

    assert weighted[0]["id"] == "b"
    assert weighted[0]["ref_score"] == 1.0


def test_reference_weight_keeps_candidate_pool_without_reranking() -> None:
    class SearchConnection:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def execute(self, sql: str, params: object = ()) -> _Cursor:
            self.calls.append((sql, params))
            if "FROM vec_chunks" in sql:
                return _Cursor([
                    ("pack:entry", "Entry", "feat", "pack", "text", "ORC", 1, 0.1),
                ])
            return _Cursor([])

    fake = index_module.SearchIndex.__new__(index_module.SearchIndex)
    fake._ensure_loaded = lambda: None
    fake._encode = lambda _query: [0.0]
    fake._enrich_results = lambda _results: None
    fake._conn_ro = SearchConnection()

    results = fake.search(
        "entry", top_k=1, hybrid=False, rerank=False,
        rerank_candidates=50, ref_weight=0.5,
    )

    assert results[0]["id"] == "pack:entry"
    assert fake._conn_ro.calls[0][1][1] == 50


def test_incoming_refs_use_stable_ids_when_names_duplicate() -> None:
    fake = index_module.SearchIndex.__new__(index_module.SearchIndex)
    fake._conn_ro = _EnrichmentConnection()
    results = [
        {"id": "pack:a", "name": "Duplicate", "rrf_score": 0.1},
        {"id": "pack:b", "name": "Duplicate", "rrf_score": 0.1},
    ]

    fake._enrich_results(results)

    assert [ref["id"] for ref in results[0]["incoming_refs"]] == ["source-a"]
    assert [ref["id"] for ref in results[1]["incoming_refs"]] == ["source-b"]
    incoming_sql = next(sql for sql, _ in fake._conn_ro.calls if "target_uuid IN" in sql)
    assert "target_name IN" not in incoming_sql


def test_ambiguous_legacy_bare_refs_do_not_boost_duplicate_pack_ids() -> None:
    class AmbiguousConnection(_EnrichmentConnection):
        def execute(self, sql: str, params: object = ()) -> _Cursor:
            self.calls.append((sql, params))
            if "SELECT id FROM chunks" in sql:
                return _Cursor([("pack-a:same-id",), ("pack-b:same-id",)])
            if "target_uuid IN" in sql:
                return _Cursor([
                    ("same-id", "legacy-source"),
                    ("pack-a:same-id", "qualified-source"),
                ])
            return _Cursor([])

    fake = index_module.SearchIndex.__new__(index_module.SearchIndex)
    fake._conn_ro = AmbiguousConnection()
    results = [
        {"id": "pack-a:same-id", "name": "Same", "rrf_score": 0.1},
        {"id": "pack-b:same-id", "name": "Same", "rrf_score": 0.1},
    ]

    fake._enrich_results(results)

    assert [ref["id"] for ref in results[0]["incoming_refs"]] == ["qualified-source"]
    assert results[1]["incoming_refs"] == []


def test_entry_row_deletion_escapes_page_prefix_wildcards() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE vec_chunks (id TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE refs (source_id TEXT)")
    ids = [
        "journal:entry",
        "journal:entry_page_1",
        "journal:entryXpage_2",
        "journal:entry_page_extra",
    ]
    for table in ("chunks", "vec_chunks"):
        conn.executemany(f"INSERT INTO {table}(id) VALUES (?)", [(value,) for value in ids])
    conn.executemany("INSERT INTO refs(source_id) VALUES (?)", [(value,) for value in ids])

    pipeline._delete_entry_rows(conn, "journal:entry")

    assert conn.execute("SELECT id FROM chunks ORDER BY id").fetchall() == [
        ("journal:entryXpage_2",),
    ]
    assert conn.execute("SELECT id FROM vec_chunks ORDER BY id").fetchall() == [
        ("journal:entryXpage_2",),
    ]
    assert conn.execute("SELECT source_id FROM refs ORDER BY source_id").fetchall() == [
        ("journal:entryXpage_2",),
    ]
    conn.close()


def test_cli_get_does_not_start_local_index_when_daemon_is_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pf2e_codex import cli

    monkeypatch.setattr(cli, "_daemon_registered", lambda: True)
    monkeypatch.setattr(
        cli, "_local_index", lambda _settings: pytest.fail("local inference started"),
    )
    monkeypatch.setattr("pf2e_codex.daemon_proxy.proxy_get_entry", lambda _entry_id: None)

    with pytest.raises(cli.typer.Exit):
        cli.get("missing", data_dir=None, model=None)


def test_mcp_sql_boundary_denies_mutations_and_sets_execution_budget() -> None:
    from pf2e_codex import mcp_server

    assert mcp_server._read_only_sql_authorizer(
        sqlite3.SQLITE_UPDATE, "chunks", "text", None, None,
    ) == sqlite3.SQLITE_DENY
    assert mcp_server._read_only_sql_authorizer(
        sqlite3.SQLITE_FUNCTION, "load_extension", None, None, None,
    ) == sqlite3.SQLITE_DENY
    assert mcp_server._SQL_VM_STEP_BUDGET > 0
    assert mcp_server._SQL_PROGRESS_INTERVAL > 0
    assert mcp_server._SQL_DEADLINE_SECONDS > 0


def test_mcp_sql_connection_is_disposable_and_read_only(tmp_path: Path) -> None:
    from pf2e_codex import mcp_server

    db_path = tmp_path / "readonly.db"
    seed = sqlite3.connect(str(db_path))
    seed.execute("CREATE TABLE values_table (value TEXT)")
    seed.execute("INSERT INTO values_table VALUES ('ok')")
    seed.commit()
    seed.close()

    conn = mcp_server._open_readonly_sql_connection(db_path)
    assert conn.execute("SELECT value FROM values_table").fetchone() == ("ok",)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO values_table VALUES ('blocked')")
    conn.close()
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def _chunk(
    chunk_id: str,
    text: str,
    source_hash: str,
) -> tuple:
    return (
        chunk_id,
        "Journal entry",
        "journal_page",
        "journal",
        "",
        None,
        "[]",
        text,
        0,
        source_hash,
        "ORC",
        1,
        None,
    )


def test_incremental_update_preserves_pages_deletes_orphans_and_rebuilds_fts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "index.db"
    original_connect = sqlite3.connect
    initialized = False

    def connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        conn = original_connect(*args, **kwargs)
        conn.create_function("vec_f32", 1, lambda value: value)
        return conn

    def init_db(path: Path, _dim: int) -> None:
        nonlocal initialized
        conn = connect(str(path))
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS _meta (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY, name TEXT, type TEXT, pack TEXT, slug TEXT,
                level INTEGER, traits TEXT, text TEXT, raw_rules_count INTEGER,
                source_hash TEXT, license TEXT, remaster INTEGER, translations TEXT
            );
            CREATE TABLE IF NOT EXISTS vec_chunks (id TEXT PRIMARY KEY, embedding BLOB);
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                name, text, content='chunks', content_rowid='rowid'
            );
            CREATE TABLE IF NOT EXISTS refs (
                source_id TEXT, target_uuid TEXT, target_name TEXT, context TEXT
            );
            """
        )
        if initialized:
            conn.close()
            return
        initialized = True
        conn.executemany(
            "INSERT INTO _meta(key, value) VALUES (?, ?)",
            [("pf2e_release", "old"), ("total_chunks", "3")],
        )
        old_chunks = [
            _chunk("journal:entry_page_1", "old page one", "old"),
            _chunk("journal:entry_page_2", "old page two", "old"),
            _chunk("journal:gone", "removed entry", "old"),
        ]
        conn.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            old_chunks,
        )
        conn.executemany(
            "INSERT INTO vec_chunks(id, embedding) VALUES (?, ?)",
            [(row[0], b"old") for row in old_chunks],
        )
        conn.execute("INSERT INTO refs VALUES (?, ?, ?, ?)",
                     ("journal:other", "gone", "Removed entry", "old ref"))
        conn.execute("INSERT INTO fts_chunks(fts_chunks) VALUES ('rebuild')")
        conn.commit()
        conn.close()

    class Provider:
        dim = 3

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 2.0, 3.0] for _ in texts]

    class Builder:
        def __init__(self, _resolver: object):
            pass

        def build_all(self, _entry: dict, _pack: str) -> list[dict]:
            base = {
                "name": "Journal entry", "type": "journal_page", "pack": "journal",
                "slug": "entry", "level": None, "traits": [], "raw_rules_count": 0,
                "source_hash": "new", "license": "ORC", "remaster": True,
                "translations": None, "refs": [],
            }
            return [
                {**base, "id": "journal:entry_page_1", "text": "new page one"},
                {**base, "id": "journal:entry_page_2", "text": "new page two"},
            ]

    init_db(db_path, 3)
    monkeypatch.setattr(sqlite3, "connect", connect)
    monkeypatch.setattr(index_module, "init_db", init_db)
    monkeypatch.setattr(index_module, "load_vec_extension", lambda _conn: None)
    monkeypatch.setattr(pipeline, "ChunkBuilder", Builder)
    monkeypatch.setattr(pipeline, "UUIDResolver", lambda _entries: object())
    monkeypatch.setattr(pipeline, "entry_hash", lambda _entry: "new")

    import pf2e_codex.fetcher as fetcher
    monkeypatch.setattr(fetcher, "get_cached_zip", lambda _settings: "zip")
    monkeypatch.setattr(
        fetcher, "extract_all_packs", lambda *_args: {"journal": [{"_id": "entry"}]},
    )

    settings = SimpleNamespace(
        db=db_path,
        model="test-model",
        provider="auto",
        onnx_provider="cpu",
        release="new",
        cache_dir=tmp_path,
    )
    pipeline.update_index(settings, _provider=Provider())

    conn = connect(str(db_path))
    ids = [row[0] for row in conn.execute("SELECT id FROM chunks ORDER BY id")]
    texts = [row[0] for row in conn.execute("SELECT text FROM chunks ORDER BY id")]
    fts_count = conn.execute(
        "SELECT COUNT(*) FROM fts_chunks WHERE fts_chunks MATCH 'new'"
    ).fetchone()[0]
    release = conn.execute(
        "SELECT value FROM _meta WHERE key = 'pf2e_release'"
    ).fetchone()[0]
    stale_refs = conn.execute(
        "SELECT COUNT(*) FROM refs WHERE target_uuid = 'gone'"
    ).fetchone()[0]
    conn.close()

    assert ids == ["journal:entry_page_1", "journal:entry_page_2"]
    assert texts == ["new page one", "new page two"]
    assert fts_count == 2
    assert release == "new"
    # The orphan's bare target UUID is ambiguous across packs, so the stale
    # incoming row is retained rather than risking deletion of another target.
    assert stale_refs == 1


def test_incremental_update_removes_changed_entry_with_zero_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "empty-entry.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY, name TEXT, type TEXT, pack TEXT, slug TEXT,
            level INTEGER, traits TEXT, text TEXT, raw_rules_count INTEGER,
            source_hash TEXT, license TEXT, remaster INTEGER, translations TEXT
        );
        CREATE TABLE vec_chunks (id TEXT PRIMARY KEY, embedding BLOB);
        CREATE VIRTUAL TABLE fts_chunks USING fts5(
            name, text, content='chunks', content_rowid='rowid'
        );
        CREATE TABLE refs (
            source_id TEXT, target_uuid TEXT, target_name TEXT, context TEXT
        );
        """
    )
    conn.execute("INSERT INTO _meta VALUES ('pf2e_release', 'old')")
    conn.execute("INSERT INTO _meta VALUES ('total_chunks', '1')")
    conn.execute(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        _chunk("journal:entry_page_1", "stale page", "old"),
    )
    conn.execute("INSERT INTO vec_chunks VALUES (?, ?)", ("journal:entry_page_1", b"old"))
    conn.execute("INSERT INTO refs VALUES (?, ?, ?, ?)",
                 ("journal:other", "entry", "Entry", "stale target"))
    conn.execute("INSERT INTO fts_chunks(fts_chunks) VALUES ('rebuild')")
    conn.commit()
    conn.close()


    class Provider:
        dim = 3

        def embed(self, _texts: list[str]) -> list[list[float]]:
            pytest.fail("empty replacement must not be embedded")

    class EmptyBuilder:
        def __init__(self, _resolver: object):
            pass

        def build_all(self, _entry: dict, _pack: str) -> list[dict]:
            return []

    monkeypatch.setattr(pipeline, "init_db", lambda _path, _dim: None)
    monkeypatch.setattr(pipeline, "load_vec_extension", lambda _conn: None)
    monkeypatch.setattr(pipeline, "ChunkBuilder", EmptyBuilder)
    monkeypatch.setattr(pipeline, "UUIDResolver", lambda _entries: object())
    monkeypatch.setattr(pipeline, "entry_hash", lambda _entry: "new")
    import pf2e_codex.fetcher as fetcher
    monkeypatch.setattr(fetcher, "get_cached_zip", lambda _settings: "zip")
    monkeypatch.setattr(
        fetcher, "extract_all_packs", lambda *_args: {"journal": [{"_id": "entry"}]},
    )

    settings = SimpleNamespace(
        db=db_path,
        model="test-model",
        provider="auto",
        onnx_provider="cpu",
        release="new",
        cache_dir=tmp_path,
    )
    pipeline.update_index(settings, _provider=Provider())

    conn = sqlite3.connect(str(db_path))
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0] == 0
    # Legacy bare-target refs are retained conservatively; the missing target
    # cannot appear in search results or receive a reference boost.
    assert conn.execute("SELECT COUNT(*) FROM refs WHERE target_uuid = 'entry'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM fts_chunks").fetchone()[0] == 0
    assert conn.execute("SELECT value FROM _meta WHERE key = 'pf2e_release'").fetchone()[0] == "new"
    conn.close()


def test_ambiguous_ref_tombstone_survives_duplicate_orphan_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "duplicate-transition.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY, name TEXT, type TEXT, pack TEXT, slug TEXT,
            level INTEGER, traits TEXT, text TEXT, raw_rules_count INTEGER,
            source_hash TEXT, license TEXT, remaster INTEGER, translations TEXT
        );
        CREATE TABLE vec_chunks (id TEXT PRIMARY KEY, embedding BLOB);
        CREATE VIRTUAL TABLE fts_chunks USING fts5(
            name, text, content='chunks', content_rowid='rowid'
        );
        CREATE TABLE refs (
            source_id TEXT, target_uuid TEXT, target_name TEXT, context TEXT
        );
        """
    )
    conn.executemany(
        "INSERT INTO _meta VALUES (?, ?)",
        [("pf2e_release", "old"), ("total_chunks", "2")],
    )
    for chunk_id, pack in (("pack-a:same-id", "pack-a"), ("pack-b:same-id", "pack-b")):
        conn.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (chunk_id, "Same", "feat", pack, "same", None, "[]", "text", 0,
             "old", "ORC", 1, None),
        )
        conn.execute("INSERT INTO vec_chunks VALUES (?, ?)", (chunk_id, b"old"))
    conn.execute(
        "INSERT INTO refs VALUES (?, ?, ?, ?)",
        ("other:source", "same-id", "Same", "legacy bare target"),
    )
    conn.execute("INSERT INTO fts_chunks(fts_chunks) VALUES ('rebuild')")
    conn.commit()
    conn.close()

    class Provider:
        dim = 3

        def embed(self, _texts: list[str]) -> list[list[float]]:
            pytest.fail("unchanged survivor must not be embedded")

    monkeypatch.setattr(pipeline, "init_db", lambda _path, _dim: None)
    monkeypatch.setattr(pipeline, "load_vec_extension", lambda _conn: None)
    monkeypatch.setattr(pipeline, "UUIDResolver", lambda _entries: object())
    monkeypatch.setattr(pipeline, "ChunkBuilder", lambda _resolver: object())
    monkeypatch.setattr(pipeline, "entry_hash", lambda _entry: "old")
    import pf2e_codex.fetcher as fetcher
    monkeypatch.setattr(fetcher, "get_cached_zip", lambda _settings: "zip")
    monkeypatch.setattr(
        fetcher, "extract_all_packs", lambda *_args: {"pack-a": [{"_id": "same-id"}]},
    )

    settings = SimpleNamespace(
        db=db_path,
        model="test-model",
        provider="auto",
        onnx_provider="cpu",
        release="new",
        cache_dir=tmp_path,
    )
    pipeline.update_index(settings, _provider=Provider())

    conn = sqlite3.connect(str(db_path))
    assert conn.execute("SELECT id FROM chunks").fetchall() == [("pack-a:same-id",)]
    assert conn.execute(
        "SELECT bare_id FROM ambiguous_ref_targets",
    ).fetchall() == [("same-id",)]
    assert conn.execute(
        "SELECT target_uuid FROM refs WHERE source_id = 'other:source'",
    ).fetchall() == [("same-id",)]
    conn.close()

    search = index_module.SearchIndex.__new__(index_module.SearchIndex)
    search._conn_ro = sqlite3.connect(str(db_path))
    result = {"id": "pack-a:same-id", "name": "Same", "rrf_score": 0.1}
    search._enrich_results([result])
    assert result["incoming_refs"] == []
    search._conn_ro.close()
