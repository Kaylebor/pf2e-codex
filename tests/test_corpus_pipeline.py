"""Corpus ownership and source-scoped synchronization tests."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("sqlite_vec")

from pf2e_codex import corpus, pipeline
from pf2e_codex.index import load_vec_extension
from pf2e_codex.licensed_core import LicensedCoreBundle


class Provider:
    dim = 2

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(index + 1), 0.5] for index, _text in enumerate(texts)]


def _settings(tmp_path: Path):
    return SimpleNamespace(
        db=tmp_path / "index.db",
        data_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        model="test-model",
        provider="auto",
        onnx_provider="cpu",
        release="pf2e-test",
    )


def _chunk(
    chunk_id: str,
    text: str,
    *,
    origin: str,
    page: int = 10,
    printed_page: str | None = None,
) -> dict:
    chunk = {
        "id": chunk_id,
        "name": chunk_id,
        "type": "rulebook_section" if origin == "corpus" else "feat",
        "pack": "corpus-player-core" if origin == "corpus" else "feats",
        "slug": chunk_id.replace(":", "-"),
        "level": None,
        "traits": [],
        "text": text,
        "raw_rules_count": 0,
        "source_hash": text,
        "license": "ORC",
        "remaster": True,
        "refs": [],
        "origin": origin,
    }
    if origin == "foundry":
        chunk["publication_title"] = "Pathfinder Player Core"
    if origin == "corpus":
        chunk.update(
            {
                "source_id": "paizo:PZO12001:player-core",
                "source": {
                    "source_id": "paizo:PZO12001:player-core",
                    "source": "paizo-pdf",
                    "product": "Pathfinder Player Core",
                    "revision": "normalized-revision",
                    "parser": "paizo-native-v1",
                    "license": "ORC",
                    "era": "remaster",
                    "provenance": {"content_fingerprint": "normalized-revision"},
                },
                "source_page_start": page,
                "source_page_end": page,
                "printed_page": printed_page,
                "section_hash": text,
            }
        )
    return chunk


def _licensed_chunk() -> dict:
    fingerprint = "a" * 64
    text = "A reviewed public rule."
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    source_section_hash = hashlib.sha256(b"source section").hexdigest()
    return {
        **_chunk("licensed:one", text, origin="licensed-core"),
        "source_id": f"licensed:PZO12001:{fingerprint[:16]}",
        "source": {
            "source_id": f"licensed:PZO12001:{fingerprint[:16]}",
            "source": "licensed-core",
            "product": "PZO12001",
            "revision": fingerprint,
            "parser": "paizo-native-v1",
            "license": "ORC",
            "era": "remaster",
            "provenance": {
                "content_fingerprint": fingerprint,
                "public_schema_version": 1,
            },
        },
        "section_hash": content_hash,
        "source_page_start": 10,
        "source_page_end": 10,
        "printed_page": "8",
        "licensed_provenance": {
            "product_code": "PZO12001",
            "content_fingerprint": fingerprint,
            "source_section_id": "pzo12001:player-core:p10:h0123456789abcdef:i0",
            "source_section_hash": source_section_hash,
            "content_hash": content_hash,
            "license": "ORC",
            "era": "remaster",
            "extraction_method": "reviewed-v1",
            "policy_version": "mechanics-v1",
            "parser_version": "paizo-native-v1",
            "source_schema_version": "1",
            "notice_key": "ORC",
        },
        "licensed_notice": {
            "notice_key": "ORC",
            "license": "ORC",
            "text": "Complete ORC notice.",
        },
    }


def _rows(db: Path, sql: str) -> list[tuple]:
    conn = sqlite3.connect(str(db))
    load_vec_extension(conn)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def test_build_corpus_chunks_keeps_local_full_on_frozen_v1(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    settings.corpus_scope = "local-full"
    settings.corpus_auto_discover = True
    settings.effective_corpus_dir = tmp_path / "corpus"
    settings.corpus_dir = None
    settings.corpus_include = ()
    settings.corpus_exclude = ()
    settings.corpus_prefer = {}
    (settings.effective_corpus_dir / "sources").mkdir(parents=True)
    observed: list[str] = []

    monkeypatch.setattr(corpus, "discover_sources", lambda *_args, **_kwargs: [object()])
    monkeypatch.setattr(corpus, "select_revisions", lambda *_args, **_kwargs: [object()])
    monkeypatch.setattr(corpus, "prepare_exports", lambda *_args, **_kwargs: [object()])
    monkeypatch.setattr(
        corpus,
        "parse_exports",
        lambda _prepared, *, parser_version: observed.append(parser_version) or [],
    )

    assert pipeline.build_corpus_chunks(settings) == []
    assert observed == [corpus.PAIZO_NATIVE_PARSER_V1]


def test_corpus_sync_reembeds_only_changed_sections_and_preserves_foundry(tmp_path: Path):
    settings = _settings(tmp_path)
    seed_provider = Provider()
    pipeline.embed_and_index(
        [_chunk("foundry:feat", "foundry text", origin="foundry")],
        settings,
        provider=seed_provider,
    )

    first_provider = Provider()
    first = pipeline.sync_corpus_index(
        settings,
        [
            _chunk("corpus:one", "one", origin="corpus"),
            _chunk("corpus:two", "two", origin="corpus"),
        ],
        provider=first_provider,
    )
    assert first == {"active": 2, "changed": 2, "removed": 0, "unchanged": 0}

    second_provider = Provider()
    second = pipeline.sync_corpus_index(
        settings,
        [_chunk("corpus:one", "one corrected", origin="corpus")],
        provider=second_provider,
    )

    assert second == {"active": 1, "changed": 1, "removed": 1, "unchanged": 0}
    assert second_provider.calls == [["one corrected"]]
    assert _rows(
        settings.db, "SELECT id, origin FROM chunks ORDER BY id"
    ) == [("corpus:one", "corpus"), ("foundry:feat", "foundry")]
    assert _rows(
        settings.db, "SELECT source_id FROM sources ORDER BY source_id"
    ) == [
        ("foundry:pf2e-test",),
        ("paizo:PZO12001:player-core",),
    ]
    assert _rows(
        settings.db,
        "SELECT value FROM _meta WHERE key = 'distribution_scope'",
    ) == [("local-full",)]

    metadata_provider = Provider()
    metadata_only = pipeline.sync_corpus_index(
        settings,
        [
            _chunk(
                "corpus:one", "one corrected", origin="corpus", page=11,
                printed_page="9",
            )
        ],
        provider=metadata_provider,
    )
    assert metadata_only == {
        "active": 1, "changed": 0, "removed": 0, "unchanged": 1,
    }
    assert metadata_provider.calls == []
    assert _rows(
        settings.db,
        "SELECT source_page_start, printed_page FROM chunks WHERE id = 'corpus:one'",
    ) == [(11, "9")]


def test_corpus_sync_embedding_failure_leaves_database_untouched(tmp_path: Path):
    settings = _settings(tmp_path)
    pipeline.embed_and_index(
        [_chunk("foundry:feat", "foundry text", origin="foundry")],
        settings,
        provider=Provider(),
    )
    pipeline.sync_corpus_index(
        settings,
        [_chunk("corpus:one", "one", origin="corpus")],
        provider=Provider(),
    )

    class FailingProvider(Provider):
        def embed(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embedding failed")

    with pytest.raises(RuntimeError, match="embedding failed"):
        pipeline.sync_corpus_index(
            settings,
            [_chunk("corpus:one", "changed", origin="corpus")],
            provider=FailingProvider(),
        )

    assert _rows(
        settings.db, "SELECT id, text FROM chunks WHERE origin = 'corpus'"
    ) == [("corpus:one", "one")]


def test_corpus_sync_empty_snapshot_removes_last_private_source(tmp_path: Path):
    settings = _settings(tmp_path)
    pipeline.embed_and_index(
        [_chunk("foundry:feat", "foundry text", origin="foundry")],
        settings,
        provider=Provider(),
    )
    pipeline.sync_corpus_index(
        settings,
        [_chunk("corpus:one", "private text", origin="corpus")],
        provider=Provider(),
    )

    summary = pipeline.sync_corpus_index(settings, [], provider=Provider())

    assert summary == {"active": 0, "changed": 0, "removed": 1, "unchanged": 0}
    assert _rows(settings.db, "SELECT id, origin FROM chunks") == [
        ("foundry:feat", "foundry")
    ]
    assert _rows(settings.db, "SELECT source FROM sources") == [("foundry",)]
    assert _rows(
        settings.db,
        "SELECT value FROM _meta WHERE key = 'distribution_scope'",
    ) == [("local-full",)]


def test_corpus_sync_preserves_independently_owned_source_metadata(tmp_path: Path):
    settings = _settings(tmp_path)
    pipeline.embed_and_index(
        [_chunk("foundry:feat", "foundry text", origin="foundry")],
        settings,
        provider=Provider(),
    )
    pipeline.sync_corpus_index(
        settings,
        [_chunk("corpus:one", "private text", origin="corpus")],
        provider=Provider(),
    )

    conn = sqlite3.connect(settings.db)
    try:
        conn.execute(
            """INSERT INTO sources
               (source_id, source, product, revision, parser, license, era, provenance)
               VALUES ('other:reviewed:test', 'community-reviewed', 'PZO12001',
                       'test', 'reviewed-v1', 'ORC', 'remaster', '{}')"""
        )
        conn.commit()
    finally:
        conn.close()

    pipeline.sync_corpus_index(settings, [], provider=Provider())

    assert _rows(
        settings.db, "SELECT source_id, source FROM sources ORDER BY source_id"
    ) == [
        ("foundry:pf2e-test", "foundry"),
        ("other:reviewed:test", "community-reviewed"),
    ]


def test_corpus_sync_suppresses_and_restores_bundled_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    settings = _settings(tmp_path)
    licensed = _licensed_chunk()
    pipeline.embed_and_index(
        [
            _chunk("foundry:feat", "foundry text", origin="foundry"),
            licensed,
        ],
        settings,
        provider=Provider(),
    )

    from pf2e_codex import licensed_core

    def bundle(*, exclude_products=frozenset(), **_kwargs):
        chunks = () if "PZO12001" in exclude_products else (licensed,)
        return LicensedCoreBundle(chunks, (), ())

    monkeypatch.setattr(licensed_core, "load_licensed_core", bundle)

    pipeline.sync_corpus_index(
        settings,
        [_chunk("corpus:one", "private text", origin="corpus")],
        provider=Provider(),
    )
    assert _rows(
        settings.db, "SELECT id, origin FROM chunks ORDER BY id"
    ) == [
        ("corpus:one", "corpus"),
        ("foundry:feat", "foundry"),
    ]

    pipeline.sync_corpus_index(settings, [], provider=Provider())
    assert _rows(
        settings.db, "SELECT id, origin FROM chunks ORDER BY id"
    ) == [
        ("foundry:feat", "foundry"),
        ("licensed:one", "licensed-core"),
    ]
    assert _rows(settings.db, "SELECT public_id FROM licensed_sections") == [
        ("licensed:one",)
    ]


def test_empty_corpus_sync_does_not_initialize_embedding_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    settings = _settings(tmp_path)
    pipeline.embed_and_index(
        [_chunk("foundry:feat", "foundry text", origin="foundry")],
        settings,
        provider=Provider(),
    )
    pipeline.sync_corpus_index(
        settings,
        [_chunk("corpus:one", "private text", origin="corpus")],
        provider=Provider(),
    )

    def fail_provider(*_args, **_kwargs):
        raise AssertionError("deletion-only sync must not initialize a provider")

    monkeypatch.setattr(pipeline, "get_provider", fail_provider)
    summary = pipeline.sync_corpus_index(settings, [])

    assert summary == {"active": 0, "changed": 0, "removed": 1, "unchanged": 0}


def test_corpus_sync_refuses_registered_daemon(tmp_path: Path):
    settings = _settings(tmp_path)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "server.json").write_text("{}")

    with pytest.raises(RuntimeError, match="registered daemon"):
        pipeline.sync_corpus_index(
            settings,
            [_chunk("corpus:one", "one", origin="corpus")],
            provider=Provider(),
        )


def test_validated_sibling_can_be_compared_before_explicit_activation(tmp_path: Path):
    settings = _settings(tmp_path)
    pipeline.embed_and_index(
        [_chunk("foundry:feat", "old live text", origin="foundry")],
        settings,
        provider=Provider(),
    )

    staged = pipeline.embed_and_index(
        [_chunk("foundry:feat", "new staged text", origin="foundry")],
        settings,
        rebuild=True,
        provider=Provider(),
        activate=False,
    )

    assert staged.is_file()
    assert _rows(settings.db, "SELECT text FROM chunks") == [("old live text",)]
    assert _rows(staged, "SELECT text FROM chunks") == [("new staged text",)]

    marker = settings.data_dir / "server.json"
    marker.write_text("{}")
    with pytest.raises(RuntimeError, match="registered daemon"):
        pipeline.activate_staged_index(staged, settings)
    marker.unlink()

    activated = pipeline.activate_staged_index(staged, settings)
    assert activated == settings.db.resolve()
    assert not staged.exists()
    assert _rows(settings.db, "SELECT text FROM chunks") == [("new staged text",)]


def test_default_full_index_replaces_snapshot_and_removes_stale_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = _settings(tmp_path)
    pipeline.embed_and_index(
        [
            _chunk("foundry:keep", "old text", origin="foundry"),
            _chunk("foundry:stale", "obsolete", origin="foundry"),
        ],
        settings,
        provider=Provider(),
    )
    monkeypatch.setattr(
        pipeline,
        "build_chunks",
        lambda _settings: [_chunk("foundry:keep", "new text", origin="foundry")],
    )
    monkeypatch.setattr(pipeline, "get_provider", lambda *_args, **_kwargs: Provider())

    pipeline.index_all(settings)

    assert _rows(settings.db, "SELECT id, text FROM chunks") == [
        ("foundry:keep", "new text")
    ]
    assert _rows(
        settings.db,
        "SELECT value FROM _meta WHERE key = 'distribution_scope'",
    ) == [("redistributable",)]


def test_incremental_update_migrates_legacy_clean_db_to_core_publication_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    settings = _settings(tmp_path)
    allowed = _chunk("feats:one", "old allowed", origin="foundry")
    disallowed = _chunk("feats:other", "old adventure", origin="foundry")
    disallowed["publication_title"] = "Pathfinder Lost Omens: Example"
    pipeline.embed_and_index([allowed, disallowed], settings, provider=Provider())
    conn = sqlite3.connect(settings.db)
    conn.execute("DELETE FROM _meta WHERE key='foundry_scope'")
    conn.commit()
    conn.close()

    entry = {
        "_id": "one",
        "name": "Allowed Rule",
        "type": "feat",
        "system": {
            "description": {"value": "Updated allowed rule."},
            "publication": {
                "title": "Pathfinder Player Core",
                "license": "ORC",
                "remaster": True,
            },
            "rules": [],
            "traits": {"value": []},
        },
    }
    import pf2e_codex.fetcher as fetcher

    monkeypatch.setattr(fetcher, "get_cached_zip", lambda _settings: "zip")
    monkeypatch.setattr(
        fetcher,
        "extract_all_packs",
        lambda *_args: {"feats": [entry]},
    )

    pipeline.update_index(settings, _provider=Provider())

    assert _rows(
        settings.db,
        "SELECT id, publication_title FROM chunks WHERE origin='foundry' ORDER BY id",
    ) == [("feats:one", "Pathfinder Player Core")]
    assert _rows(
        settings.db,
        "SELECT value FROM _meta WHERE key='foundry_scope'",
    ) == [("core-publications-v1",)]


def test_default_full_index_late_failure_preserves_live_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = _settings(tmp_path)
    pipeline.embed_and_index(
        [_chunk("foundry:keep", "old text", origin="foundry")],
        settings,
        provider=Provider(),
    )
    monkeypatch.setattr(
        pipeline,
        "build_chunks",
        lambda _settings: [_chunk("foundry:keep", "new text", origin="foundry")],
    )
    monkeypatch.setattr(pipeline, "get_provider", lambda *_args, **_kwargs: Provider())

    def fail_late(_conn):
        raise RuntimeError("late failure")

    monkeypatch.setattr(pipeline, "rebuild_fts", fail_late)

    with pytest.raises(RuntimeError, match="late failure"):
        pipeline.index_all(settings)

    assert _rows(settings.db, "SELECT id, text FROM chunks") == [
        ("foundry:keep", "old text")
    ]
    assert not list(tmp_path.glob(".index.db.staging-*"))


def test_default_full_index_refuses_registered_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = _settings(tmp_path)
    (settings.data_dir / "server.json").write_text("{}")
    monkeypatch.setattr(pipeline, "build_chunks", lambda _settings: [])

    with pytest.raises(RuntimeError, match="registered daemon"):
        pipeline.index_all(settings)
