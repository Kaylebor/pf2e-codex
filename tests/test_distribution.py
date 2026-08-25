"""Distribution-scope gates for local and published databases."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from pf2e_codex.distribution import (
    _model_licensed_core_digest,
    audit_database_slot,
    audit_downloadable_database,
    audit_redistributable_database,
)
from pf2e_codex.licensed_core import licensed_core_contract_digest
from pf2e_codex.licensed_policy import licensed_policy_digest


def _database(
    path: Path,
    *,
    scope: str | None,
    corpus_chunks: int = 0,
    licensed_core_chunks: int = 0,
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
        conn.execute(
            """CREATE TABLE chunks (
                id TEXT PRIMARY KEY, origin TEXT, text TEXT, source_id TEXT,
                license TEXT, publication_title TEXT
            )"""
        )
        conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(
            """INSERT INTO chunks VALUES
            ('foundry:one', 'foundry', 'foundry', NULL, 'ORC', 'Pathfinder Player Core')"""
        )
        for index in range(corpus_chunks):
            conn.execute(
                "INSERT INTO chunks VALUES (?, 'corpus', 'private', NULL, 'ORC', NULL)",
                (f"corpus:{index}",),
            )
        if licensed_core_chunks:
            text = "Reviewed public rule."
            content_hash = hashlib.sha256(text.encode()).hexdigest()
            notice_text = "Complete ORC notice."
            notice_hash = hashlib.sha256(notice_text.encode()).hexdigest()
            fingerprint = "a" * 64
            conn.executescript(
                """
                CREATE TABLE sources (
                    source_id TEXT PRIMARY KEY, source TEXT, product TEXT, revision TEXT,
                    parser TEXT, license TEXT, era TEXT, provenance TEXT
                );
                CREATE TABLE license_notices (
                    notice_key TEXT PRIMARY KEY, license TEXT, text TEXT, content_hash TEXT
                );
                CREATE TABLE licensed_revisions (
                    product_code TEXT, content_fingerprint TEXT, license TEXT, era TEXT,
                    parser_version TEXT, source_schema_version TEXT, policy_versions TEXT
                );
                CREATE TABLE licensed_sections (
                    public_id TEXT PRIMARY KEY, product_code TEXT, content_fingerprint TEXT,
                    source_section_id TEXT, source_section_hash TEXT, page_start INTEGER,
                    page_end INTEGER, printed_page TEXT, heading TEXT, content_hash TEXT,
                    license TEXT, era TEXT, extraction_method TEXT, policy_version TEXT,
                    parser_version TEXT, notice_key TEXT
                );
                """
            )
            source_id = f"licensed:PZO12001:{fingerprint[:16]}"
            conn.execute(
                "INSERT INTO sources VALUES (?, 'licensed-core', 'PZO12001', ?, ?, ?, ?, ?)",
                (
                    source_id, fingerprint, "paizo-native-v1", "ORC", "remaster",
                    json.dumps(
                        {
                            "content_fingerprint": fingerprint,
                            "public_schema_version": 1,
                        }
                    ),
                ),
            )
            conn.execute(
                "INSERT INTO license_notices VALUES ('ORC', 'ORC', ?, ?)",
                (notice_text, notice_hash),
            )
            conn.execute(
                "INSERT INTO licensed_revisions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "PZO12001", fingerprint, "ORC", "remaster", "paizo-native-v1",
                    "1", '["mechanics-v1"]',
                ),
            )
            for index in range(licensed_core_chunks):
                public_id = f"licensed:{index}"
                conn.execute(
                    "INSERT INTO chunks VALUES (?, 'licensed-core', ?, ?, ?, NULL)",
                    (public_id, text, source_id, "ORC"),
                )
                conn.execute(
                    """INSERT INTO licensed_sections VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        public_id, "PZO12001", fingerprint,
                        f"pzo12001:player-core:p1:h0123456789abcdef:i{index}",
                        hashlib.sha256(f"source:{index}".encode()).hexdigest(),
                        1, 1, None, "Reviewed Rule",
                        content_hash, "ORC", "remaster",
                        "reviewed-v1", "mechanics-v1", "paizo-native-v1", "ORC",
                    ),
                )
            digest = licensed_core_contract_digest(
                schema_version=1,
                source_revisions=[
                    {
                        "product_code": "PZO12001",
                        "content_fingerprint": fingerprint,
                        "license": "ORC",
                        "era": "remaster",
                        "parser_version": "paizo-native-v1",
                        "source_schema_version": "1",
                        "policy_versions": ["mechanics-v1"],
                    }
                ],
                notices=[
                    {"notice_key": "ORC", "license": "ORC", "content_hash": notice_hash}
                ],
                sections=[
                    {
                        "id": f"licensed:{index}",
                        "heading": "Reviewed Rule",
                        "page_start": 1,
                        "page_end": 1,
                        "printed_page": None,
                        "content_hash": content_hash,
                        "provenance": {
                            "product_code": "PZO12001",
                            "content_fingerprint": fingerprint,
                            "source_section_id": (
                                "pzo12001:player-core:p1:"
                                f"h0123456789abcdef:i{index}"
                            ),
                            "source_section_hash": hashlib.sha256(
                                f"source:{index}".encode()
                            ).hexdigest(),
                            "content_hash": content_hash,
                            "license": "ORC",
                            "era": "remaster",
                            "extraction_method": "reviewed-v1",
                            "policy_version": "mechanics-v1",
                            "parser_version": "paizo-native-v1",
                            "source_schema_version": "1",
                            "notice_key": "ORC",
                        },
                    }
                    for index in range(licensed_core_chunks)
                ],
            )
            conn.execute(
                "INSERT INTO _meta VALUES ('licensed_core_digest', ?)", (digest,)
            )
        for index, origin in enumerate(extra_origins):
            conn.execute(
                "INSERT INTO chunks VALUES (?, ?, 'extra', NULL, 'ORC', NULL)",
                (f"extra:{index}", origin),
            )
        if scope is not None:
            conn.execute(
                "INSERT INTO _meta VALUES ('distribution_scope', ?)",
                (scope,),
            )
            if scope == "redistributable":
                conn.execute(
                    "INSERT INTO _meta VALUES ('foundry_scope', 'core-publications-v1')"
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


def _trusted_projection(path: Path) -> Path:
    text = "Reviewed public rule."
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    source_section_hash = hashlib.sha256(b"source:0").hexdigest()
    fingerprint = "a" * 64
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE source_revisions (
            product_code TEXT, content_fingerprint TEXT, license TEXT, era TEXT,
            parser_version TEXT, source_schema_version TEXT
        );
        CREATE TABLE notices (notice_key TEXT PRIMARY KEY, license TEXT, text TEXT);
        CREATE TABLE licensed_sections (
            public_id TEXT PRIMARY KEY, product_code TEXT, content_fingerprint TEXT,
            source_section_id TEXT, source_section_hash TEXT, page_start INTEGER,
            page_end INTEGER, printed_page TEXT, heading TEXT, text TEXT,
            content_hash TEXT, license TEXT, era TEXT, extraction_method TEXT,
            policy_version TEXT, parser_version TEXT, notice_key TEXT
        );
        """
    )
    conn.executemany(
        "INSERT INTO metadata VALUES (?, ?)",
        [
            ("public_schema_version", "1"),
            ("content_scope", "licensed-core-reviewed"),
            ("policy_version", "mechanics-v1"),
            ("policy_digest", licensed_policy_digest()),
        ],
    )
    conn.execute(
        "INSERT INTO source_revisions VALUES (?, ?, 'ORC', 'remaster', 'paizo-native-v1', '1')",
        ("PZO12001", fingerprint),
    )
    conn.execute("INSERT INTO notices VALUES ('ORC', 'ORC', 'Complete ORC notice.')")
    conn.execute(
        """INSERT INTO licensed_sections VALUES
        ('licensed:0', 'PZO12001', ?,
         'pzo12001:player-core:p1:h0123456789abcdef:i0', ?, 1, 1, NULL,
         'Reviewed Rule', ?, ?, 'ORC', 'remaster', 'reviewed-v1', 'mechanics-v1',
         'paizo-native-v1', 'ORC')""",
        (fingerprint, source_section_hash, text, content_hash),
    )
    conn.commit()
    conn.close()
    return path


def test_explicit_redistributable_database_passes_strict_audit(tmp_path: Path):
    path = _database(
        tmp_path / "public.db",
        scope="redistributable",
        licensed_core_chunks=1,
    )

    audit = audit_redistributable_database(
        path,
        require_explicit_marker=True,
        trusted_projection=_trusted_projection(tmp_path / "trusted.sqlite3"),
    )

    assert audit.explicit_marker
    assert audit.corpus_chunks == 0
    assert audit.licensed_core_chunks == 1


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
    with pytest.raises(RuntimeError, match="private or unowned"):
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
        licensed_core_chunks=1,
    )

    audit_redistributable_database(
        path,
        require_explicit_marker=True,
        expected_release="pf2e-8.4.0",
        expected_model="expected-model",
        trusted_projection=_trusted_projection(tmp_path / "trusted.sqlite3"),
    )
    with pytest.raises(RuntimeError, match="database release"):
        audit_redistributable_database(path, expected_release="pf2e-9.0.0")
    with pytest.raises(RuntimeError, match="embedding model"):
        audit_redistributable_database(path, expected_model="other-model")


def test_strict_audit_rejects_foundry_only_new_release(tmp_path: Path):
    path = _database(tmp_path / "public.db", scope="redistributable")

    with pytest.raises(RuntimeError, match="licensed-core content is missing"):
        audit_redistributable_database(path, require_explicit_marker=True)


def test_strict_audit_rejects_foundry_rows_outside_core_allowlist(tmp_path: Path):
    path = _database(tmp_path / "public.db", scope="redistributable")
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE chunks SET publication_title='Pathfinder Lost Omens: Example' WHERE origin='foundry'"
    )
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="approved core publications"):
        audit_redistributable_database(path, require_explicit_marker=True)


def test_audit_rejects_tampered_licensed_core_text(tmp_path: Path):
    path = _database(
        tmp_path / "public.db", scope="redistributable", licensed_core_chunks=1
    )
    conn = sqlite3.connect(path)
    conn.execute("UPDATE chunks SET text='tampered' WHERE origin='licensed-core'")
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="content hash mismatch"):
        audit_redistributable_database(path)


def test_marked_download_rejects_joint_licensed_hash_and_digest_tampering(tmp_path: Path):
    path = _database(
        tmp_path / "public.db", scope="redistributable", licensed_core_chunks=1
    )
    malicious = "Changed text with internally consistent hashes."
    malicious_hash = hashlib.sha256(malicious.encode()).hexdigest()
    conn = sqlite3.connect(path)
    conn.execute("UPDATE chunks SET text=? WHERE origin='licensed-core'", (malicious,))
    conn.execute(
        "UPDATE licensed_sections SET content_hash=?", (malicious_hash,)
    )
    forged_digest = _model_licensed_core_digest(conn)
    conn.execute(
        "UPDATE _meta SET value=? WHERE key='licensed_core_digest'", (forged_digest,)
    )
    conn.commit()
    conn.close()

    assert audit_redistributable_database(path).licensed_core_digest == forged_digest
    with pytest.raises(RuntimeError, match="trusted licensed-core projection"):
        audit_downloadable_database(
            path,
            trusted_projection=_trusted_projection(tmp_path / "trusted.sqlite3"),
        )


def test_markerless_foundry_only_download_remains_compatible(tmp_path: Path):
    path = _database(
        tmp_path / "legacy-foundry.db",
        scope=None,
        pf2e_release="pf2e-8.4.0",
        embedding_model="test-model",
    )

    audit = audit_downloadable_database(
        path,
        expected_release="pf2e-8.4.0",
        expected_model="test-model",
    )

    assert audit.scope is None


def test_markerless_download_with_licensed_content_is_rejected(tmp_path: Path):
    path = _database(
        tmp_path / "markerless-licensed.db",
        scope=None,
        licensed_core_chunks=1,
    )

    with pytest.raises(RuntimeError, match="markerless legacy downloads must be Foundry-only"):
        audit_downloadable_database(path)
