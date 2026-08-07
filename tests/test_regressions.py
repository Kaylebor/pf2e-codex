"""Regression tests for search and incremental-index fixes."""

from __future__ import annotations

import json
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


def test_natural_language_fts_terms_use_or_and_drop_question_filler() -> None:
    terms = index_module._search_terms(
        "Do I roll a spell attack for Fireball, or does the target make a saving throw?",
    )

    assert "fireball" in terms
    assert "saving" in terms
    assert "throw" in terms
    assert "dc" in index_module._search_terms("How do I set a DC?")
    assert "does" not in terms
    assert "the" not in terms


def test_explicit_name_matching_uses_complete_terms() -> None:
    query = "Does Fireball hurt my allies?"

    assert index_module._query_contains_name(query, "Fireball")
    assert not index_module._query_contains_name(query, "Fire")
    assert not index_module._query_contains_name(query, "Fireball Rune")
    assert not index_module._query_contains_name("How do I set a DC?", "Set")


def test_remaster_preference_only_swaps_confirmed_exact_name_overlap() -> None:
    results = [
        {"id": "legacy", "name": "Dying and Recovery", "remaster": False},
        {"id": "unrelated", "name": "Recovery Checks", "remaster": False},
        {"id": "remaster", "name": "Dying and Recovery", "remaster": True},
        {"id": "legacy-only", "name": "Legacy Only Rule", "remaster": False},
    ]

    preferred = index_module._prefer_remaster_overlaps(results)

    assert [item["id"] for item in preferred] == [
        "remaster", "unrelated", "legacy", "legacy-only",
    ]


def test_rrf_keeps_a_top_lexical_only_candidate() -> None:
    semantic = [
        (f"semantic:{i}", {"id": f"semantic:{i}"})
        for i in range(1, 6)
    ]
    lexical = [("spells:fireball", {"id": "spells:fireball"})]

    results = index_module._rrf_fuse(semantic, lexical, top_k=5)

    assert "spells:fireball" in {result["id"] for result in results}


class _HybridSearchConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def execute(self, sql: str, params: object = ()) -> _Cursor:
        self.calls.append((sql, params))
        if "FROM vec_chunks" in sql:
            return _Cursor([
                (
                    f"spells:unrelated-{i}", f"Unrelated {i}", "spell", "spells",
                    "Unrelated spell text", "ORC", 1, i / 10,
                )
                for i in range(1, 6)
            ])
        if "bm25(fts_chunks, 10.0, 0.0)" in sql:
            return _Cursor([
                (
                    "spells:fireball", "Fireball", "spell", "spells",
                    "Save: reflex (basic)", "ORC", 1, -10.0,
                ),
            ])
        if "FROM fts_chunks" in sql:
            return _Cursor([
                (
                    "spells:fireball", "Fireball", "spell", "spells",
                    "Save: reflex (basic)", "ORC", 1, -10.0,
                ),
            ])
        return _Cursor([])


def _hybrid_search_index() -> index_module.SearchIndex:
    fake = index_module.SearchIndex.__new__(index_module.SearchIndex)
    fake._ensure_loaded = lambda: None
    fake._ensure_fts = lambda: None
    fake._encode = lambda _query: [0.0]
    fake._enrich_results = lambda _results: None
    fake._conn_ro = _HybridSearchConnection()
    return fake


def test_named_entity_is_promoted_for_a_novice_question() -> None:
    fake = _hybrid_search_index()

    results = fake.search(
        "Do I roll a spell attack for Fireball, or does the target make a saving throw?",
        top_k=3,
        hybrid=True,
        rerank=False,
    )

    assert results[0]["id"] == "spells:fireball"
    lexical_call = next(
        (sql, params) for sql, params in fake._conn_ro.calls
        if "bm25(fts_chunks, 10.0, 1.0)" in sql
    )
    fts_query = lexical_call[1][0]
    assert " OR " in fts_query
    assert '"fireball"' in fts_query
    assert '"does"' not in fts_query


def test_reranker_cannot_drop_an_explicitly_named_entity() -> None:
    class DroppingReranker:
        def rerank(
            self, _query: str, documents: list[dict], top_k: int,
        ) -> list[dict]:
            return [
                document for document in documents
                if document["id"] != "spells:fireball"
            ][:top_k]

    fake = _hybrid_search_index()
    fake._manager = DroppingReranker()

    results = fake.search(
        "Does Fireball hurt my allies?",
        top_k=3,
        hybrid=True,
        rerank=True,
    )

    assert results[0]["id"] == "spells:fireball"
    assert len(results) == 3


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

    monkeypatch.setattr(cli, "_daemon_registered", lambda _settings=None: True)
    monkeypatch.setattr(
        cli, "_local_index", lambda _settings: pytest.fail("local inference started"),
    )
    monkeypatch.setattr(
        "pf2e_codex.daemon_proxy.proxy_get_entry",
        lambda _entry_id, *, settings=None: None,
    )

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


def _source_chunk(chunk_id: str, *, origin: str = "foundry") -> dict:
    chunk = {
        "id": chunk_id,
        "name": "Source chunk",
        "type": "journal_page",
        "pack": "journal",
        "slug": "source-chunk",
        "level": None,
        "traits": [],
        "text": f"text for {chunk_id}",
        "raw_rules_count": 0,
        "source_hash": "stable",
        "license": "ORC",
        "remaster": None,
        "refs": [],
        "origin": origin,
    }
    if origin != "foundry":
        chunk.update({
            "source_id": "pzo2101e-4th",
            "source": {
                "source_id": "pzo2101e-4th",
                "source": "paizo-pdf",
                "product": "Pathfinder Core Rulebook",
                "revision": "4th printing",
                "parser": "native-pages-v1",
                "license": "OGL",
                "era": "legacy",
                "provenance": {"artifact": "native-pages.json"},
            },
            "source_page_start": 42,
            "source_page_end": 43,
            "printed_page": "40",
            "section_hash": "section-hash",
        })
    return chunk


class _SourceProvider:
    dim = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 2.0] for _ in texts]


def test_full_seed_persists_source_metadata_and_corpus_provenance(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    settings = SimpleNamespace(
        db=db_path,
        data_dir=tmp_path,
        model="test-model",
        provider="auto",
        onnx_provider="cpu",
        release="foundry-release",
        corpus_scope="local-full",
    )
    pipeline.embed_and_index(
        [_source_chunk("journal:foundry"), _source_chunk("corpus:rulebook", origin="corpus")],
        settings,
        rebuild=True,
        provider=_SourceProvider(),
    )

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT id, origin, source_id, source_page_start, source_page_end, printed_page, section_hash "
        "FROM chunks ORDER BY id"
    ).fetchall()
    source = conn.execute(
        "SELECT source, product, revision, parser, license, era, provenance "
        "FROM sources WHERE source_id = 'pzo2101e-4th'"
    ).fetchone()
    search = index_module.SearchIndex.__new__(index_module.SearchIndex)
    search._ensure_loaded = lambda: None
    search._conn_ro = conn
    fetched = search.fetch_by_id("corpus:rulebook")
    conn.close()

    assert rows == [
        ("corpus:rulebook", "corpus", "pzo2101e-4th", 42, 43, "40", "section-hash"),
        ("journal:foundry", "foundry", "foundry:foundry-release", None, None, None, None),
    ]
    assert source[:6] == (
        "paizo-pdf", "Pathfinder Core Rulebook", "4th printing", "native-pages-v1", "OGL", "legacy",
    )
    assert json.loads(source[6]) == {"artifact": "native-pages.json"}
    assert fetched["provenance"]["origin"] == "corpus"
    assert fetched["provenance"]["product"] == "Pathfinder Core Rulebook"


def test_rebuild_refuses_registered_daemon_before_provider_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "server.json").write_text("{}")
    settings = SimpleNamespace(
        db=tmp_path / "index.db",
        data_dir=tmp_path,
        model="test-model",
        provider="auto",
        onnx_provider="cpu",
        release="new",
    )
    monkeypatch.setattr(
        pipeline, "get_provider", lambda *_args, **_kwargs: pytest.fail("provider was created"),
    )

    with pytest.raises(RuntimeError, match="registered daemon"):
        pipeline.embed_and_index([_source_chunk("journal:foundry")], settings, rebuild=True)


def test_incremental_update_does_not_treat_corpus_rows_as_foundry_orphans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "ownership.db"
    settings = SimpleNamespace(
        db=db_path,
        model="test-model",
        provider="auto",
        onnx_provider="cpu",
        release="new",
        cache_dir=tmp_path,
    )
    index_module.init_db(db_path, 2)
    conn = sqlite3.connect(str(db_path))
    index_module.load_vec_extension(conn)
    pipeline._insert_chunk(conn, _source_chunk("journal:foundry"), [1.0, 2.0], settings)
    pipeline._insert_chunk(
        conn, _source_chunk("corpus:rulebook", origin="corpus"), [1.0, 2.0], settings,
    )
    conn.execute("INSERT INTO _meta(key, value) VALUES ('pf2e_release', 'old')")
    conn.commit()
    conn.close()

    class Provider:
        dim = 2

        def embed(self, _texts: list[str]) -> list[list[float]]:
            pytest.fail("unchanged Foundry rows should not be embedded")

    monkeypatch.setattr(pipeline, "UUIDResolver", lambda _entries: object())
    monkeypatch.setattr(pipeline, "ChunkBuilder", lambda _resolver: object())
    monkeypatch.setattr(pipeline, "entry_hash", lambda _entry: "stable")
    import pf2e_codex.fetcher as fetcher
    monkeypatch.setattr(fetcher, "get_cached_zip", lambda _settings: "zip")
    monkeypatch.setattr(
        fetcher, "extract_all_packs", lambda *_args: {"journal": [{"_id": "foundry"}]},
    )

    pipeline.update_index(settings, _provider=Provider())

    conn = sqlite3.connect(str(db_path))
    assert conn.execute("SELECT id FROM chunks ORDER BY id").fetchall() == [
        ("corpus:rulebook",), ("journal:foundry",),
    ]
    conn.close()


def test_remaster_false_excludes_unknown_rows() -> None:
    class SearchConnection:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def execute(self, sql: str, params: object = ()) -> _Cursor:
            self.calls.append((sql, params))
            return _Cursor([])

    fake = index_module.SearchIndex.__new__(index_module.SearchIndex)
    fake._ensure_loaded = lambda: None
    fake._encode = lambda _query: [0.0]
    fake._enrich_results = lambda _results: None
    fake._conn_ro = SearchConnection()

    fake.search("legacy", hybrid=False, rerank=False, remaster=False)

    sql = fake._conn_ro.calls[0][0]
    assert "chunks.remaster = 0" in sql
    assert "remaster IS NULL" not in sql


def test_failed_staged_rebuild_preserves_live_database(tmp_path: Path) -> None:
    db_path = tmp_path / "live.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE sentinel (value TEXT)")
    conn.execute("INSERT INTO sentinel VALUES ('live')")
    conn.commit()
    conn.close()

    class FailingProvider:
        dim = 2

        def embed(self, _texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embedding failed")

    settings = SimpleNamespace(
        db=db_path,
        data_dir=tmp_path,
        model="test-model",
        provider="auto",
        onnx_provider="cpu",
        release="new",
    )
    with pytest.raises(RuntimeError, match="embedding failed"):
        pipeline.embed_and_index(
            [_source_chunk("journal:foundry")], settings, rebuild=True, provider=FailingProvider(),
        )

    conn = sqlite3.connect(str(db_path))
    assert conn.execute("SELECT value FROM sentinel").fetchone() == ("live",)
    conn.close()
    assert not list(tmp_path.glob(".live.db.staging-*"))


def test_rejected_auto_download_never_reaches_canonical_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    private = tmp_path / "private-source.db"
    conn = sqlite3.connect(private)
    conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, origin TEXT)")
    conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO chunks VALUES ('corpus:one', 'corpus')")
    conn.execute("INSERT INTO _meta VALUES ('distribution_scope', 'local-full')")
    conn.commit()
    conn.close()

    destination = tmp_path / "pf2e_test-model.db"
    downloads: list[Path] = []

    def fake_download(_url: str, target: str | Path):
        target = Path(target)
        target.write_bytes(private.read_bytes())
        downloads.append(target)
        return str(target), None

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_download)
    manager = SimpleNamespace(model_name="test-model")

    for _ in range(2):
        search = index_module.SearchIndex(
            destination,
            manager,
            expected_scope="clean",
        )
        with pytest.raises(FileNotFoundError, match="Auto-download failed"):
            search._ensure_loaded()
        assert not destination.exists()
        assert not list(tmp_path.glob(".pf2e_test-model.db.download-*"))

    assert len(downloads) == 2


@pytest.mark.parametrize(
    ("release", "model"),
    [("pf2e-old", "test-model"), ("pf2e-8.4.0", "other-model")],
)
def test_auto_download_rejects_wrong_release_or_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release: str,
    model: str,
) -> None:
    source = tmp_path / "source.db"
    conn = sqlite3.connect(source)
    conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, origin TEXT)")
    conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO chunks VALUES ('foundry:one', 'foundry')")
    conn.executemany(
        "INSERT INTO _meta VALUES (?, ?)",
        [
            ("distribution_scope", "redistributable"),
            ("pf2e_release", release),
            ("embedding_model", model),
        ],
    )
    conn.commit()
    conn.close()

    destination = tmp_path / "pf2e_test-model.db"

    def fake_download(_url: str, target: str | Path):
        Path(target).write_bytes(source.read_bytes())
        return str(target), None

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_download)
    search = index_module.SearchIndex(
        destination,
        SimpleNamespace(model_name="test-model"),
        expected_scope="clean",
    )

    with pytest.raises(FileNotFoundError, match="Auto-download failed"):
        search._ensure_loaded()
    assert not destination.exists()
