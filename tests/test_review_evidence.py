"""Read-only worker evidence boundary tests."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from pf2e_codex.review_evidence import execute, foundry, load_context, section


def _review_db(tmp_path: Path) -> Path:
    path = tmp_path / "review.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE source_revisions (
            product_code TEXT, content_fingerprint TEXT, license TEXT, era TEXT
        );
        CREATE TABLE parser_runs (
            parser_run_id TEXT PRIMARY KEY, product_code TEXT,
            state TEXT, review_enabled INTEGER
        );
        CREATE TABLE review_product_scope (
            product_code TEXT PRIMARY KEY, enabled INTEGER, reason TEXT, updated_at INTEGER
        );
        CREATE TABLE source_sections (
            section_key TEXT PRIMARY KEY, parser_run_id TEXT, product_code TEXT,
            content_fingerprint TEXT, source_section_id TEXT, heading TEXT,
            source_text TEXT, page_start INTEGER, page_end INTEGER,
            printed_page TEXT, layout_flags TEXT
        );
        CREATE TABLE stitch_candidates (
            candidate_id TEXT, parser_run_id TEXT, section_keys TEXT, evidence_json TEXT
        );
        CREATE TABLE aon_cache (
            query_digest TEXT, normalized_query TEXT, status TEXT,
            results_json TEXT, checked_at INTEGER
        );
        """
    )
    conn.execute("INSERT INTO source_revisions VALUES ('PZO12001','f','ORC','remaster')")
    conn.execute("INSERT INTO parser_runs VALUES ('run','PZO12001','active',1)")
    conn.execute("INSERT INTO review_product_scope VALUES ('PZO12001',1,'enabled',1)")
    for index in range(3):
        conn.execute(
            "INSERT INTO source_sections VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                f"section-{index}", "run", "PZO12001", "f", f"pzo12001:test:{index}",
                "Fireball" if index == 1 else f"Neighbor {index}", f"Private rules {index}",
                index + 1, index + 1, str(index + 1), "[]",
            ),
        )
    conn.commit()
    conn.close()
    return path


def _foundry_db(tmp_path: Path, *, scope: str = "redistributable", origin: str = "foundry") -> Path:
    path = tmp_path / f"foundry-{scope}-{origin}.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY, name TEXT, type TEXT, pack TEXT, text TEXT,
            license TEXT, remaster INTEGER, publication_title TEXT, origin TEXT
        );
        """
    )
    conn.execute("INSERT INTO _meta VALUES ('distribution_scope', ?)", (scope,))
    conn.execute(
        "INSERT INTO chunks VALUES ('spell:fireball','Fireball','spell','spells','Foundry mechanics','ORC',1,'Player Core',?)",
        (origin,),
    )
    conn.commit()
    conn.close()
    return path


def _context(tmp_path: Path, *, foundry_db: Path | None = None) -> dict[str, object]:
    return {
        "version": 1,
        "workspace": str(_review_db(tmp_path)),
        "foundry_database": str(foundry_db) if foundry_db else None,
        "allowed_ids": ["section-1"],
        "neighbor_ids": ["section-0", "section-2"],
    }


def test_evidence_section_exposes_rules_era_but_not_local_paths(tmp_path: Path):
    context = _context(tmp_path)
    value = section(context, "section-1")

    assert value["rules_era"] == "remaster"
    assert value["text"] == "Private rules 1"
    encoded = json.dumps(value).casefold()
    assert str(tmp_path).casefold() not in encoded
    assert "workspace" not in value


def test_evidence_rejects_claimed_id_after_product_is_held(tmp_path: Path):
    context = _context(tmp_path)
    conn = sqlite3.connect(str(context["workspace"]))
    conn.execute("UPDATE review_product_scope SET enabled=0, reason='legacy-study'")
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="unknown active section"):
        section(context, "section-1")


def test_evidence_rejects_unclaimed_ids(tmp_path: Path):
    context = _context(tmp_path)
    with pytest.raises(PermissionError, match="outside"):
        section(context, "section-0")


def test_neighbors_are_pre_authorized_and_bounded(tmp_path: Path):
    context = _context(tmp_path)
    value = execute(context, "neighbors", "section-1")

    assert [item["id"] for item in value["neighbors"]] == ["section-0", "section-2"]


def test_foundry_queries_only_validated_clean_foundry_rows(tmp_path: Path):
    clean = _foundry_db(tmp_path)
    context = _context(tmp_path, foundry_db=clean)
    value = foundry(context, "section-1")
    assert [item["id"] for item in value["results"]] == ["spell:fireball"]

    dirty = _foundry_db(tmp_path, origin="corpus")
    dirty_context = {**context, "foundry_database": str(dirty)}
    with pytest.raises(ValueError, match="private or unknown"):
        foundry(dirty_context, "section-1")

    private = _foundry_db(tmp_path, scope="local-full")
    private_context = {**context, "foundry_database": str(private)}
    with pytest.raises(ValueError, match="redistributable"):
        foundry(private_context, "section-1")


def test_context_loader_requires_bounded_id_sets(tmp_path: Path):
    path = tmp_path / "claim.json"
    path.write_text(json.dumps({"version": 1, "workspace": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_context(path)
