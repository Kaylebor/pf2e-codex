"""Distribution-scope gates for local and published databases."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from pf2e_codex.distribution import audit_database_slot, audit_redistributable_database


def _database(
    path: Path,
    *,
    scope: str | None,
    corpus_chunks: int = 0,
    legacy: bool = False,
    pf2e_release: str | None = None,
    embedding_model: str | None = None,
    extra_origins: tuple[str | None, ...] = (),
) -> Path:
    conn = sqlite3.connect(path)
    if legacy:
        conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO chunks VALUES ('foundry:one')")
    else:
        conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, origin TEXT)")
        conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO chunks VALUES ('foundry:one', 'foundry')")
        for index in range(corpus_chunks):
            conn.execute(
                "INSERT INTO chunks VALUES (?, 'corpus')",
                (f"corpus:{index}",),
            )
        for index, origin in enumerate(extra_origins):
            conn.execute(
                "INSERT INTO chunks VALUES (?, ?)",
                (f"extra:{index}", origin),
            )
        if scope is not None:
            conn.execute(
                "INSERT INTO _meta VALUES ('distribution_scope', ?)",
                (scope,),
            )
        if pf2e_release is not None:
            conn.execute("INSERT INTO _meta VALUES ('pf2e_release', ?)", (pf2e_release,))
        if embedding_model is not None:
            conn.execute(
                "INSERT INTO _meta VALUES ('embedding_model', ?)",
                (embedding_model,),
            )
    conn.commit()
    conn.close()
    return path


def test_explicit_redistributable_database_passes_strict_audit(tmp_path: Path):
    path = _database(
        tmp_path / "public.db",
        scope="redistributable",
    )

    audit = audit_redistributable_database(path, require_explicit_marker=True)

    assert audit.explicit_marker
    assert audit.corpus_chunks == 0


def test_strict_audit_requires_explicit_redistributable_marker(tmp_path: Path):
    path = _database(tmp_path / "unmarked.db", scope=None)

    with pytest.raises(RuntimeError, match="explicit redistributable"):
        audit_redistributable_database(path, require_explicit_marker=True)


@pytest.mark.parametrize(
    ("scope", "corpus_chunks"),
    [("local-full", 0), ("local-full", 1), ("redistributable", 1)],
)
def test_private_scope_or_corpus_rows_are_rejected(
    tmp_path: Path, scope: str, corpus_chunks: int
):
    path = _database(
        tmp_path / f"private-{scope}-{corpus_chunks}.db",
        scope=scope,
        corpus_chunks=corpus_chunks,
    )

    with pytest.raises(RuntimeError, match="must not be published"):
        audit_redistributable_database(path)


@pytest.mark.parametrize("origin", ["private-import", "Corpus", "", None])
def test_clean_audit_fails_closed_on_unknown_or_unowned_rows(
    tmp_path: Path, origin: str | None
):
    path = _database(
        tmp_path / "unknown-origin.db",
        scope="redistributable",
        extra_origins=(origin,),
    )

    with pytest.raises(RuntimeError, match="private, unowned, or local-full"):
        audit_redistributable_database(path)
    with pytest.raises(RuntimeError, match="non-Foundry or unowned"):
        audit_database_slot(path, "clean")


def test_local_slot_rejects_unknown_origins(tmp_path: Path):
    path = _database(
        tmp_path / "local.db",
        scope="local-full",
        extra_origins=("private-import",),
    )

    with pytest.raises(RuntimeError, match="unknown ownership"):
        audit_database_slot(path, "local")


def test_legacy_foundry_only_database_is_pull_compatible_but_not_publishable(tmp_path: Path):
    path = _database(tmp_path / "legacy.db", scope=None, legacy=True)

    audit = audit_redistributable_database(path)
    assert not audit.explicit_marker

    with pytest.raises(RuntimeError, match="explicit redistributable"):
        audit_redistributable_database(path, require_explicit_marker=True)


def test_physical_database_slots_reject_cross_scope_content(tmp_path: Path):
    clean = _database(tmp_path / "clean.db", scope="redistributable")
    local = _database(
        tmp_path / "local.db",
        scope="local-full",
        corpus_chunks=1,
    )

    assert audit_database_slot(clean, "clean").scope == "redistributable"
    assert audit_database_slot(local, "local").scope == "local-full"

    with pytest.raises(RuntimeError, match="clean database slot"):
        audit_database_slot(local, "clean")
    with pytest.raises(RuntimeError, match="local database slot"):
        audit_database_slot(clean, "local")


def test_release_audit_checks_requested_release_and_model(tmp_path: Path):
    path = _database(
        tmp_path / "public.db",
        scope="redistributable",
        pf2e_release="pf2e-8.4.0",
        embedding_model="expected-model",
    )

    audit_redistributable_database(
        path,
        require_explicit_marker=True,
        expected_release="pf2e-8.4.0",
        expected_model="expected-model",
    )
    with pytest.raises(RuntimeError, match="database release"):
        audit_redistributable_database(path, expected_release="pf2e-9.0.0")
    with pytest.raises(RuntimeError, match="embedding model"):
        audit_redistributable_database(path, expected_model="other-model")
