"""Validation tests for the bundled licensed-core projection loader."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from pf2e_codex.licensed_core import load_licensed_core
from pf2e_codex.licensed_policy import licensed_policy_digest


def _projection(path: Path, *, text: str = "A creature can Step 5 feet.") -> Path:
    fingerprint = "a" * 64
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    source_section_hash = hashlib.sha256(b"source section").hexdigest()
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE source_revisions (
            product_code TEXT, content_fingerprint TEXT, license TEXT, era TEXT,
            parser_version TEXT, source_schema_version TEXT, printing_revision TEXT,
            PRIMARY KEY (product_code, content_fingerprint)
        );
        CREATE TABLE notices (notice_key TEXT PRIMARY KEY, license TEXT, text TEXT);
        CREATE TABLE licensed_rules (
            public_id TEXT PRIMARY KEY, heading TEXT, text TEXT, content_hash TEXT,
            license TEXT, era TEXT, extraction_method TEXT, policy_version TEXT,
            notice_key TEXT
        );
        CREATE TABLE licensed_rule_sources (
            public_id TEXT, source_ordinal INTEGER, product_code TEXT,
            content_fingerprint TEXT, source_section_id TEXT, source_section_hash TEXT,
            page_start INTEGER, page_end INTEGER, printed_page TEXT,
            parser_version TEXT, printing_revision TEXT, notice_key TEXT,
            PRIMARY KEY(public_id, source_ordinal)
        );
        CREATE TABLE required_foundry_rows (
            foundry_id TEXT PRIMARY KEY, source_hash TEXT, normalized_hash TEXT,
            publication_title TEXT, license TEXT, era TEXT
        );
        """
    )
    conn.executemany(
        "INSERT INTO metadata VALUES (?, ?)",
        [
            ("public_schema_version", "3"),
            ("content_scope", "licensed-core-reviewed"),
            ("policy_version", "mechanics-v1"),
            ("policy_digest", licensed_policy_digest()),
            ("review_scope_version", "semantic-products-v1"),
            ("covered_products", '["PZO12001"]'),
            (
                "review_scope_digest",
                hashlib.sha256(
                    (
                        "semantic-products-v1\n"
                        + json.dumps(
                            [{"product_code": "PZO12001", "state": "enabled"}],
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    ).encode()
                ).hexdigest(),
            ),
        ],
    )
    conn.execute(
        "INSERT INTO source_revisions VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "PZO12001", fingerprint, "ORC", "remaster", "paizo-native-v1", "1",
            "printing-1",
        ),
    )
    conn.execute("INSERT INTO notices VALUES ('ORC', 'ORC', 'Complete ORC notice text')")
    conn.execute(
        "INSERT INTO licensed_rules VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "licensed:one", "Step", text, content_hash, "ORC", "remaster",
            "reviewed-extraction-v1", "mechanics-v1", "ORC",
        ),
    )
    conn.execute(
        "INSERT INTO licensed_rule_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "licensed:one", 0, "PZO12001", fingerprint,
            "pzo12001:combined:p10:h0123456789abcdef:i0", source_section_hash,
            10, 10, "8", "paizo-native-v1", "printing-1", "ORC",
        ),
    )
    conn.commit()
    conn.close()
    return path


def test_loader_builds_sanitized_index_chunks(tmp_path: Path):
    bundle = load_licensed_core(_projection(tmp_path / "licensed.sqlite3"))

    assert len(bundle.chunks) == 1
    chunk = bundle.chunks[0]
    assert chunk["id"] == "licensed:one"
    assert chunk["origin"] == "licensed-core"
    assert chunk["license"] == "ORC"
    assert chunk["remaster"] is True
    assert chunk["source"]["provenance"] == {
        "content_fingerprint": "a" * 64,
        "public_schema_version": 3,
        "printing_revision": "printing-1",
    }
    assert "path" not in str(chunk).lower()
    assert bundle.notices[0]["notice_key"] == "ORC"
    assert bundle.covered_products == ("PZO12001",)


def test_loader_excludes_products_represented_by_private_pdf(tmp_path: Path):
    bundle = load_licensed_core(
        _projection(tmp_path / "licensed.sqlite3"), exclude_products={"PZO12001"}
    )

    assert bundle.chunks == ()
    assert bundle.source_revisions == ()
    assert bundle.notices[0]["notice_key"] == "ORC"


def test_loader_rejects_content_or_provenance_tampering(tmp_path: Path):
    path = _projection(tmp_path / "licensed.sqlite3")
    conn = sqlite3.connect(path)
    conn.execute("UPDATE licensed_rules SET text = 'tampered'")
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="content hash mismatch"):
        load_licensed_core(path)


def test_loader_rejects_product_scope_tampering(tmp_path: Path):
    path = _projection(tmp_path / "licensed.sqlite3")
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE metadata SET value='[\"PZO12002\"]' WHERE key='covered_products'"
    )
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="product-scope digest"):
        load_licensed_core(path)


def test_loader_rejects_product_scope_version_tampering(tmp_path: Path):
    path = _projection(tmp_path / "licensed.sqlite3")
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE metadata SET value='semantic-products-v2' "
        "WHERE key='review_scope_version'"
    )
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="product-scope version"):
        load_licensed_core(path)


def test_loader_rejects_nonstructural_source_section_id(tmp_path: Path):
    path = _projection(tmp_path / "licensed.sqlite3")
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE licensed_rule_sources SET source_section_id = ?",
        ("private/path@example.invalid",),
    )
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="source provenance is invalid"):
        load_licensed_core(path)


@pytest.mark.parametrize("column", ["text", "heading"])
def test_loader_rejects_private_text_in_sections(tmp_path: Path, column: str):
    path = _projection(tmp_path / "licensed.sqlite3")
    conn = sqlite3.connect(path)
    conn.execute(
        f"UPDATE licensed_rules SET {column} = ?",
        ("reader@example.invalid",),
    )
    if column == "text":
        conn.execute(
            "UPDATE licensed_rules SET content_hash = ?",
            (hashlib.sha256(b"reader@example.invalid").hexdigest(),),
        )
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="private or unsafe text"):
        load_licensed_core(path)


def test_loader_rejects_private_text_in_notice(tmp_path: Path):
    path = _projection(tmp_path / "licensed.sqlite3")
    conn = sqlite3.connect(path)
    conn.execute("UPDATE notices SET text = 'reader@example.invalid'")
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="private or unsafe text"):
        load_licensed_core(path)


def test_loader_rejects_private_text_in_notice_key(tmp_path: Path):
    path = _projection(tmp_path / "licensed.sqlite3")
    conn = sqlite3.connect(path)
    conn.execute("UPDATE notices SET notice_key = 'reader@example.invalid'")
    conn.execute(
        "UPDATE licensed_rules SET notice_key = 'reader@example.invalid'"
    )
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="private or unsafe text"):
        load_licensed_core(path)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("public_id", "licensed:reader@example.invalid"),
        ("extraction_method", "file:///home/reader"),
        ("printed_page", "C:\\Users\\reader"),
    ],
)
def test_loader_rejects_private_text_in_public_scalars(
    tmp_path: Path, column: str, value: str
):
    path = _projection(tmp_path / "licensed.sqlite3")
    conn = sqlite3.connect(path)
    table = "licensed_rule_sources" if column == "printed_page" else "licensed_rules"
    conn.execute(f"UPDATE {table} SET {column} = ?", (value,))
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="private or unsafe text"):
        load_licensed_core(path)


def test_loader_binds_source_section_id_to_product(tmp_path: Path):
    path = _projection(tmp_path / "licensed.sqlite3")
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE licensed_rule_sources SET source_section_id = "
        "'pzo12002:combined:p10:h0123456789abcdef:i0'"
    )
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="source provenance is invalid"):
        load_licensed_core(path)


@pytest.mark.parametrize(
    "updates",
    [
        {"page_start": "/mnt/data/private.pdf"},
        {"page_end": "reader-name"},
        {"page_start": 11},
        {"page_end": 9},
    ],
)
def test_loader_rejects_invalid_page_provenance(
    tmp_path: Path, updates: dict[str, object]
):
    path = _projection(tmp_path / "licensed.sqlite3")
    conn = sqlite3.connect(path)
    for column, value in updates.items():
        conn.execute(f"UPDATE licensed_rule_sources SET {column} = ?", (value,))
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="page provenance is invalid"):
        load_licensed_core(path)


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("source_revisions", "content_fingerprint"),
        ("licensed_rule_sources", "source_section_hash"),
    ],
)
def test_loader_rejects_non_sha256_provenance(
    tmp_path: Path, table: str, column: str
):
    path = _projection(tmp_path / "licensed.sqlite3")
    conn = sqlite3.connect(path)
    conn.execute(f"UPDATE {table} SET {column} = ?", ("g" * 64,))
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="invalid|no source revision"):
        load_licensed_core(path)


def test_loader_rejects_untrusted_policy_metadata(tmp_path: Path):
    path = _projection(tmp_path / "licensed.sqlite3")
    conn = sqlite3.connect(path)
    conn.execute("UPDATE metadata SET value='untrusted' WHERE key='policy_digest'")
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="invalid review policy"):
        load_licensed_core(path)


def test_loader_rejects_section_policy_mismatch(tmp_path: Path):
    path = _projection(tmp_path / "licensed.sqlite3")
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE licensed_rules SET policy_version='tampered-policy'"
    )
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="provenance mismatch"):
        load_licensed_core(path)
