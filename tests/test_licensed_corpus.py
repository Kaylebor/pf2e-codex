"""Private licensing-review workspace and public projection tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import ANY, patch

import pytest

from pf2e_codex.corpus import (
    PAIZO_NATIVE_PARSER_V4,
    TrustedBlock,
    TrustedParseBundle,
    TrustedQuarantine,
    TrustedSection,
    _trusted_bundle_seal,
    _trusted_parser_output_digest,
    repair_trusted_bundle,
)
from pf2e_codex.licensed_core import load_licensed_core
from pf2e_codex.licensed_corpus import (
    REVIEW_SCHEMA_VERSION,
    _anchor_digest,
    _candidate_commitment,
    _native_word_coverage_digest,
    _parser_output_digest,
    activate_parser_run,
    build_public_corpus,
    claim_draft_screening_batch,
    claim_review,
    claim_shard,
    draft_screening_status,
    initialize_workspace,
    invalidate_reviews,
    next_draft_screening_record,
    read_claimed_review,
    read_claimed_shard,
    read_draft_screening_record,
    release_draft_screening_batch,
    reopen_draft_screening,
    stage_trusted_native_pdf,
    step_draft_screening,
    submit_candidate,
    submit_draft_screening_decision,
    submit_review,
    workspace_status,
)
from pf2e_codex.licensed_corpus import _bind_source_asset_inventory as bind_source_asset_inventory
from pf2e_codex.licensed_corpus import _stage_parser_run as stage_parser_run
from pf2e_codex.pdf_export import NativeWordInventory, native_word_inventory

PRIVATE_TEXT = "Private source prose must never reach the public database."
PUBLIC_TEXT = "A creature can use this action once each round."


def test_native_word_coverage_digest_is_anchor_order_independent():
    first = hashlib.sha256(b"first anchor").hexdigest()
    second = hashlib.sha256(b"second anchor").hexdigest()
    base = {
        "stable_identity": hashlib.sha256(b"section").hexdigest(),
        "native_word_count": 2,
        "native_word_digest": _anchor_digest(
            "section-native-word-anchors-v1", [first, second]
        ),
    }

    assert _native_word_coverage_digest(
        [{**base, "native_word_anchors": [first, second]}]
    ) == _native_word_coverage_digest(
        [{**base, "native_word_anchors": [second, first]}]
    )


def test_future_review_schema_is_rejected_without_downgrade(tmp_path: Path):
    workspace = _workspace(tmp_path)
    future = REVIEW_SCHEMA_VERSION + 1
    conn = sqlite3.connect(workspace)
    try:
        conn.execute(
            "UPDATE metadata SET value=? WHERE key='review_schema_version'",
            (str(future),),
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(ValueError, match=f"unsupported future schema {future}"):
        workspace_status(workspace)

    conn = sqlite3.connect(workspace)
    try:
        assert conn.execute(
            "SELECT value FROM metadata WHERE key='review_schema_version'"
        ).fetchone()[0] == str(future)
    finally:
        conn.close()


def _licensed_corpus_cli_module():
    script = Path(__file__).parents[1] / "scripts" / "licensed-corpus.py"
    spec = importlib.util.spec_from_file_location("licensed_corpus_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_db(tmp_path: Path, *, section_count: int = 1) -> Path:
    path = tmp_path / "local.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE sources (
            source_id TEXT PRIMARY KEY, source TEXT, product TEXT, revision TEXT,
            parser TEXT, license TEXT, era TEXT, provenance TEXT
        );
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY, section_hash TEXT, text TEXT, name TEXT,
            source_page_start INTEGER, source_page_end INTEGER, printed_page TEXT,
            license TEXT, source_id TEXT, origin TEXT
        );
        """
    )
    conn.execute("INSERT INTO _meta VALUES ('distribution_scope', 'local-full')")
    conn.execute(
        "INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "paizo:PZO12001:player-core",
            "paizo-pdf",
            "PZO12001",
            "a" * 64,
            "paizo-native-v1",
            "ORC",
            "remaster",
            json.dumps({"content_fingerprint": "a" * 64, "export_schema_version": 1}),
        ),
    )
    for index in range(section_count):
        text = f"{PRIVATE_TEXT} section {index}"
        conn.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'corpus')",
            (
                f"corpus:PZO12001:player-core:p{index}:rules:0",
                f"hash-{index}",
                text,
                f"Rules {index}",
                index + 1,
                index + 1,
                str(index + 1),
                "ORC",
                "paizo:PZO12001:player-core",
            ),
        )
    conn.commit()
    conn.close()
    return path


def _workspace(tmp_path: Path, *, section_count: int = 1, shard_size: int = 10) -> Path:
    path = tmp_path / "review.sqlite3"
    initialize_workspace(path, _source_db(tmp_path, section_count=section_count), shard_size=shard_size)
    return path


def _trusted_workspace(tmp_path: Path) -> Path:
    """Promote the sealed direct-PDF fixture; legacy imports remain nonpublishable."""
    workspace = _workspace(tmp_path)
    with patch("pf2e_codex.licensed_corpus.load_and_parse_verified_pdf", return_value=_trusted_bundle()):
        staged = stage_trusted_native_pdf(
            workspace, tmp_path / "owned.pdf", product_code="PZO12001", parser_version="paizo-native-v1",
        )
    activate_parser_run(workspace, str(staged["parser_run_id"]))
    return workspace


def _claim_and_read(workspace: Path, worker: str = "worker-a") -> tuple[dict[str, object], dict[str, object]]:
    claim = claim_shard(workspace, worker)
    assert claim is not None
    section = read_claimed_shard(workspace, int(claim["shard_id"]), worker)[0]
    return claim, section


def _submit_public(workspace: Path, section: dict[str, object], worker: str = "worker-a") -> str:
    result = submit_candidate(
        workspace,
        {
            "section_key": section["section_key"],
            "source_section_id": section["source_section_id"],
            "source_section_hash": section["source_section_hash"],
            "decision": "MIXED_NEEDS_EXTRACTION",
            "candidate_text": PUBLIC_TEXT,
            "public_heading": "Reviewed rules",
            "extraction_method": "human-reconstruction-v1",
            "reason_tags": ["layout-reviewed", "rules-text"],
            "confidence": 0.9,
            "worker": worker,
            "prompt_version": "pilot-v1",
        },
    )
    return str(result["candidate_id"])


def _submit_exclusion(workspace: Path, section: dict[str, object], worker: str = "worker-a") -> str:
    result = submit_candidate(
        workspace,
        {
            "section_key": section["section_key"],
            "source_section_id": section["source_section_id"],
            "source_section_hash": section["source_section_hash"],
            "decision": "EXCLUDE",
            "reason_tags": ["no-mechanics"],
            "worker": worker,
            "prompt_version": "pilot-v1",
        },
    )
    return str(result["candidate_id"])


def _notices(tmp_path: Path) -> Path:
    path = tmp_path / "notices.json"
    path.write_text(json.dumps({"ORC": {"license": "ORC", "text": "Supplied ORC notice."}}))
    return path


def _native_export(*, watermark: str = "buyer@example.invalid", complete: bool = True) -> dict[str, object]:
    """Two-page fixture with repeated furniture and one cross-page section."""
    def word(text: str, x0: float, top: float, size: float = 9.0, font: str = "Body") -> dict[str, object]:
        return {"text": text, "x0": x0, "top": top, "x1": x0 + 30, "bottom": top + size,
                "size": size, "font": font, "upright": True, "direction": "ltr"}
    pages = [
        {"number": 1, "width": 600, "height": 800, "images": [], "words": [
            word("Player", 30, 12, 8), word("Core", 70, 12, 8), word("1", 290, 770, 8),
            word(watermark, 50, 745, 7), word("Rules", 40, 100, 16, "Heading-Bold"),
            word("Use", 40, 140), word("this", 75, 140), word("action", 110, 140),
        ]},
        {"number": 2, "width": 600, "height": 800, "images": [], "words": [
            word("Player", 30, 12, 8), word("Core", 70, 12, 8), word("2", 290, 770, 8),
            word("once", 40, 120), word("each", 80, 120), word("round", 120, 120),
        ]},
    ]
    return {
        "schema_version": 1,
        "extractor": {"name": "synthetic", "profile_version": 1, "ocr": False},
        "source": {"filename": "PZO12001E.pdf", "sha256": "f" * 64, "page_count": 2},
        "selection": {"first_page": 1, "last_page": 2 if complete else 1},
        "pages": pages if complete else pages[:1],
    }


def _approve(workspace: Path, candidate_id: str, reviewer: str = "reviewer-b") -> None:
    claim = claim_review(workspace, reviewer)
    assert claim is not None
    assert claim["candidate_id"] == candidate_id
    review_input = read_claimed_review(workspace, candidate_id, reviewer)
    assert isinstance(review_input["candidate_text"], str) and review_input["candidate_text"].strip()
    assert str(review_input["source_text"]).strip()
    submit_review(
        workspace,
        {
            "candidate_id": candidate_id,
            "reviewer": reviewer,
            "verdict": "APPROVE",
            "policy_version": "mechanics-v1",
            "reason_tags": ["functional-rules"],
            "evidence": [
                {
                    "evidence_kind": "aon",
                    "status": "match",
                    "url": "https://2e.aonprd.example/rules",
                    "checked_at": 1,
                    "note": "matching heading only",
                }
            ],
        },
    )


def _duplicate_revision_source_db(tmp_path: Path) -> Path:
    """A synthetic historical DB: one section ID exists in two printings."""
    path = tmp_path / "two-revisions.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE sources (
            source_id TEXT PRIMARY KEY, source TEXT, product TEXT, revision TEXT,
            parser TEXT, license TEXT, era TEXT, provenance TEXT
        );
        CREATE TABLE chunks (
            id TEXT, section_hash TEXT, text TEXT, name TEXT,
            source_page_start INTEGER, source_page_end INTEGER, printed_page TEXT,
            license TEXT, source_id TEXT, origin TEXT
        );
        """
    )
    conn.execute("INSERT INTO _meta VALUES ('distribution_scope', 'local-full')")
    for suffix in ("first", "second"):
        fingerprint = f"revision-{suffix}"
        source_id = f"paizo:PZO12001:player-core:{suffix}"
        conn.execute(
            "INSERT INTO sources VALUES (?, 'paizo-pdf', 'PZO12001', ?, 'paizo-native-v1', 'ORC', 'remaster', ?)",
            (source_id, fingerprint, json.dumps({"content_fingerprint": fingerprint})),
        )
        conn.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, 'Rules', 1, 1, '1', 'ORC', ?, 'corpus')",
            (
                "corpus:PZO12001:player-core:p1:rules:0",
                f"hash-{suffix}",
                f"{PRIVATE_TEXT} {suffix}",
                source_id,
            ),
        )
    conn.commit()
    conn.close()
    return path


def test_workspace_status_and_claim_do_not_expose_private_text(tmp_path: Path):
    workspace = _workspace(tmp_path)

    status = workspace_status(workspace)
    claim = claim_shard(workspace, "worker-a")

    encoded = json.dumps({"status": status, "claim": claim})
    assert PRIVATE_TEXT not in encoded
    assert "source_text" not in encoded
    _, section = _claim_and_read(workspace, "worker-a")
    assert PRIVATE_TEXT in str(section["source_text"])


def test_cli_parses_long_inline_json_before_treating_it_as_a_path():
    module = _licensed_corpus_cli_module()
    payload = {"candidate_id": "candidate:" + ("a" * 512), "verdict": "REJECT"}

    assert module._json_inputs(json.dumps(payload)) == [payload]


def test_cli_accepts_only_an_exact_nonempty_review_id_selection():
    module = _licensed_corpus_cli_module()
    review_ids = ["review:one", "review:two"]

    assert module._review_id_inputs(json.dumps(review_ids)) == review_ids
    with pytest.raises(ValueError, match="non-empty JSON array"):
        module._review_id_inputs("[]")
    with pytest.raises(ValueError, match="duplicates"):
        module._review_id_inputs(json.dumps(["review:one", "review:one"]))


def test_cli_can_limit_private_shard_output_to_one_record():
    module = _licensed_corpus_cli_module()
    records = [{"section": 0}, {"section": 1}]

    assert module._select_private_record(records, None) == records
    assert module._select_private_record(records, 1) == {"section": 1}
    with pytest.raises(ValueError, match="outside"):
        module._select_private_record(records, 2)


def test_public_candidate_rejects_private_provenance_text(tmp_path: Path):
    workspace = _workspace(tmp_path)
    _, section = _claim_and_read(workspace)

    with pytest.raises(ValueError, match="private or unsafe provenance"):
        submit_candidate(
            workspace,
            {
                "section_key": section["section_key"],
                "source_section_id": section["source_section_id"],
                "source_section_hash": section["source_section_hash"],
                "decision": "MIXED_NEEDS_EXTRACTION",
                "candidate_text": "Rule text for buyer@example.invalid",
                "public_heading": "Reviewed rules",
                "extraction_method": "human-reconstruction-v1",
                "reason_tags": ["functional-rules", "layout-reviewed"],
                "worker": "worker-a",
                "prompt_version": "pilot-v1",
            },
        )


def test_concurrent_claims_are_atomic_and_non_overlapping(tmp_path: Path):
    workspace = _workspace(tmp_path, section_count=4, shard_size=1)
    barrier = threading.Barrier(4)
    claims: list[dict[str, object] | None] = []
    lock = threading.Lock()

    def claim(worker: str) -> None:
        barrier.wait()
        result = claim_shard(workspace, worker)
        with lock:
            claims.append(result)

    threads = [threading.Thread(target=claim, args=(f"worker-{index}",)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(result is not None for result in claims)
    assert len({int(result["shard_id"]) for result in claims if result}) == 4


def test_worker_can_atomically_target_a_specific_available_shard(tmp_path: Path):
    workspace = _workspace(tmp_path, section_count=3, shard_size=1)
    conn = sqlite3.connect(workspace)
    target = int(conn.execute("SELECT MAX(shard_id) FROM review_shards").fetchone()[0])
    conn.close()

    claim = claim_shard(workspace, "targeted-worker", preferred_shard_id=target)

    assert claim is not None
    assert claim["shard_id"] == target
    with pytest.raises(ValueError, match="already holds pending shard"):
        claim_shard(workspace, "targeted-worker", preferred_shard_id=target - 1)


def test_completed_shard_is_released_and_worker_advances(tmp_path: Path):
    workspace = _workspace(tmp_path, section_count=2, shard_size=1)
    first_claim, first = _claim_and_read(workspace)
    submit_candidate(
        workspace,
        {
            "section_key": first["section_key"],
            "source_section_id": first["source_section_id"],
            "source_section_hash": first["source_section_hash"],
            "decision": "EXCLUDE",
            "reason_tags": ["no-mechanics"],
            "worker": "worker-a",
            "prompt_version": "pilot-v1",
        },
    )

    next_claim = claim_shard(workspace, "worker-a")
    assert next_claim is not None
    assert next_claim["shard_id"] != first_claim["shard_id"]
    status = workspace_status(workspace)
    assert status["products"][0]["available_shards"] == 0


def test_expired_partially_submitted_shard_is_not_reassigned_as_new_work(tmp_path: Path):
    workspace = _workspace(tmp_path, section_count=3, shard_size=2)
    first_claim = claim_shard(workspace, "worker-a")
    assert first_claim is not None
    first_sections = read_claimed_shard(workspace, int(first_claim["shard_id"]), "worker-a")
    first = first_sections[0]
    submit_candidate(
        workspace,
        {
            "section_key": first["section_key"],
            "source_section_id": first["source_section_id"],
            "source_section_hash": first["source_section_hash"],
            "decision": "EXCLUDE",
            "reason_tags": ["no-mechanics"],
            "worker": "worker-a",
            "prompt_version": "pilot-v1",
        },
    )

    # Simulate an abandoned lease after one of two sections was decided.  The
    # shard must not be handed to another worker as if both sections were new.
    conn = sqlite3.connect(workspace)
    conn.execute("UPDATE review_shards SET lease_expires_at=0 WHERE shard_id=?", (first_claim["shard_id"],))
    conn.commit()
    conn.close()

    next_claim = claim_shard(workspace, "worker-b")
    assert next_claim is not None
    assert next_claim["shard_id"] != first_claim["shard_id"]
    with pytest.raises(ValueError, match="unavailable or complete"):
        claim_shard(workspace, "worker-c", preferred_shard_id=int(first_claim["shard_id"]))

    conn = sqlite3.connect(workspace)
    ordinal = conn.execute(
        "SELECT MAX(candidate_ordinal) FROM candidates WHERE section_key=?", (first["section_key"],)
    ).fetchone()[0]
    conn.close()
    assert ordinal == 1
    assert workspace_status(workspace)["products"][0]["available_shards"] == 0


def test_live_lease_holder_can_resume_a_partial_ordinary_shard(tmp_path: Path):
    workspace = _workspace(tmp_path, section_count=2, shard_size=2)
    first_claim = claim_shard(workspace, "worker-a")
    assert first_claim is not None
    assert first_claim["claim_mode"] == "ordinary"
    sections = read_claimed_shard(workspace, int(first_claim["shard_id"]), "worker-a")
    first, second = sections

    submit_candidate(
        workspace,
        {
            "section_key": first["section_key"],
            "source_section_id": first["source_section_id"],
            "source_section_hash": first["source_section_hash"],
            "decision": "EXCLUDE",
            "reason_tags": ["no-mechanics"],
            "worker": "worker-a",
            "prompt_version": "pilot-v1",
        },
    )

    resumed = claim_shard(workspace, "worker-a")
    assert resumed is not None
    assert resumed["shard_id"] == first_claim["shard_id"]
    assert resumed["claim_mode"] == "ordinary"
    submit_candidate(
        workspace,
        {
            "section_key": second["section_key"],
            "source_section_id": second["source_section_id"],
            "source_section_hash": second["source_section_hash"],
            "decision": "EXCLUDE",
            "reason_tags": ["no-mechanics"],
            "worker": "worker-a",
            "prompt_version": "pilot-v1",
        },
    )


def test_ordinary_claim_mode_survives_an_early_completed_review(tmp_path: Path):
    workspace = _workspace(tmp_path, section_count=2, shard_size=2)
    claim = claim_shard(workspace, "worker-a")
    assert claim is not None and claim["claim_mode"] == "ordinary"
    first, second = read_claimed_shard(workspace, int(claim["shard_id"]), "worker-a")
    first_candidate = _submit_exclusion(workspace, first)

    review_claim = claim_review(workspace, "reviewer-b")
    assert review_claim is not None and review_claim["candidate_id"] == first_candidate
    submit_review(
        workspace,
        {
            "candidate_id": first_candidate,
            "reviewer": "reviewer-b",
            "verdict": "REJECT",
            "policy_version": "mechanics-v1",
        },
    )

    with pytest.raises(PermissionError, match="ordinary assignments accept only"):
        _submit_exclusion(workspace, first)
    assert _submit_exclusion(workspace, second)


def test_pre_v2_workspace_migrates_claim_mode_without_losing_sections(tmp_path: Path):
    workspace = _workspace(tmp_path, section_count=2, shard_size=2)
    conn = sqlite3.connect(workspace)
    try:
        assert conn.execute("SELECT COUNT(*) FROM source_sections").fetchone()[0] == 2
        conn.execute("ALTER TABLE review_shards DROP COLUMN claim_mode")
        conn.execute("UPDATE metadata SET value='1' WHERE key='review_schema_version'")
        conn.commit()
    finally:
        conn.close()
    claim = claim_shard(workspace, "worker-a")
    assert claim is not None and claim["claim_mode"] == "ordinary"
    conn = sqlite3.connect(workspace)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(review_shards)")}
        assert "claim_mode" in columns
        assert conn.execute(
            "SELECT value FROM metadata WHERE key='review_schema_version'"
        ).fetchone()[0] == str(REVIEW_SCHEMA_VERSION)
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='review_invalidations'"
        ).fetchone() is not None
        assert conn.execute("SELECT COUNT(*) FROM source_sections").fetchone()[0] == 2
    finally:
        conn.close()


def test_concurrent_v1_workspace_migration_allows_both_writer_claims(tmp_path: Path):
    workspace = _workspace(tmp_path, section_count=2, shard_size=1)
    conn = sqlite3.connect(workspace)
    try:
        conn.execute("ALTER TABLE review_shards DROP COLUMN claim_mode")
        conn.execute("UPDATE metadata SET value='1' WHERE key='review_schema_version'")
        conn.commit()
    finally:
        conn.close()

    barrier = threading.Barrier(2)
    claims: list[dict[str, object] | None] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def migrate_and_claim(worker: str) -> None:
        try:
            barrier.wait()
            result = claim_shard(workspace, worker)
            with lock:
                claims.append(result)
        except BaseException as exc:  # pragma: no cover - asserted below
            with lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=migrate_and_claim, args=(worker,))
        for worker in ("worker-a", "worker-b")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert all(claim is not None for claim in claims)
    assert len({claim["shard_id"] for claim in claims if claim}) == 2
    assert {claim["claim_mode"] for claim in claims if claim} == {"ordinary"}
    conn = sqlite3.connect(workspace)
    try:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(review_shards)")]
        assert columns.count("claim_mode") == 1
        assert conn.execute("SELECT COUNT(*) FROM source_sections").fetchone()[0] == 2
        assert conn.execute(
            "SELECT COUNT(*) FROM review_shards WHERE claim_mode='ordinary'"
        ).fetchone()[0] == 2
    finally:
        conn.close()


def test_concurrent_v2_to_v3_migration_preserves_claim_modes(tmp_path: Path):
    workspace = _workspace(tmp_path, section_count=2, shard_size=1)
    first_claim = claim_shard(workspace, "worker-a")
    assert first_claim is not None and first_claim["claim_mode"] == "ordinary"

    conn = sqlite3.connect(workspace)
    try:
        before = conn.execute(
            "SELECT shard_id, claimant, claim_mode FROM review_shards ORDER BY shard_id"
        ).fetchall()
        conn.execute("DROP TABLE review_invalidations")
        conn.execute("UPDATE metadata SET value='2' WHERE key='review_schema_version'")
        conn.commit()
    finally:
        conn.close()

    barrier = threading.Barrier(2)
    claims: list[dict[str, object] | None] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def migrate_and_claim(worker: str) -> None:
        try:
            barrier.wait()
            result = claim_shard(workspace, worker)
            with lock:
                claims.append(result)
        except BaseException as exc:  # pragma: no cover - asserted below
            with lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=migrate_and_claim, args=(worker,))
        for worker in ("worker-a", "worker-b")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert all(claim is not None for claim in claims)
    assert {claim["shard_id"] for claim in claims if claim} == {1, 2}
    assert {claim["claim_mode"] for claim in claims if claim} == {"ordinary"}

    conn = sqlite3.connect(workspace)
    try:
        assert conn.execute(
            "SELECT value FROM metadata WHERE key='review_schema_version'"
        ).fetchone()[0] == str(REVIEW_SCHEMA_VERSION)
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name='review_invalidations'"
        ).fetchone()[0] == 1
        after = conn.execute(
            "SELECT shard_id, claimant, claim_mode FROM review_shards ORDER BY shard_id"
        ).fetchall()
        assert after[0] == before[0]
        assert after[1] == (2, "worker-b", "ordinary")
    finally:
        conn.close()


def test_awaiting_review_shard_is_not_available_for_candidate_claims(tmp_path: Path):
    workspace = _workspace(tmp_path)
    claim, section = _claim_and_read(workspace)
    submit_candidate(
        workspace,
        {
            "section_key": section["section_key"],
            "source_section_id": section["source_section_id"],
            "source_section_hash": section["source_section_hash"],
            "decision": "EXCLUDE",
            "reason_tags": ["no-mechanics"],
            "worker": "worker-a",
            "prompt_version": "pilot-v1",
        },
    )

    assert workspace_status(workspace)["products"][0]["available_shards"] == 0
    assert claim_shard(workspace, "worker-b") is None
    with pytest.raises(ValueError, match="unavailable or complete"):
        claim_shard(workspace, "worker-b", preferred_shard_id=int(claim["shard_id"]))


def test_revise_reopens_a_shard_until_a_replacement_candidate_is_submitted(tmp_path: Path):
    workspace = _workspace(tmp_path, section_count=2, shard_size=1)
    first_claim, first = _claim_and_read(workspace)
    original = _submit_public(workspace, first)
    review_claim = claim_review(workspace, "reviewer-b")
    assert review_claim is not None and review_claim["candidate_id"] == original
    submit_review(
        workspace,
        {
            "candidate_id": original,
            "reviewer": "reviewer-b",
            "verdict": "REVISE",
            "policy_version": "mechanics-v1",
        },
    )

    reopened = claim_shard(workspace, "worker-a")
    assert reopened is not None and reopened["shard_id"] == first_claim["shard_id"]
    revised = read_claimed_shard(workspace, int(reopened["shard_id"]), "worker-a")[0]
    _submit_public(workspace, revised)

    next_claim = claim_shard(workspace, "worker-a")
    assert next_claim is not None
    assert next_claim["shard_id"] != first_claim["shard_id"]


def test_rework_shard_accepts_only_explicitly_revised_sections(tmp_path: Path):
    workspace = _workspace(tmp_path, section_count=2, shard_size=2)
    claim = claim_shard(workspace, "worker-a")
    assert claim is not None
    first, second = read_claimed_shard(workspace, int(claim["shard_id"]), "worker-a")
    first_candidate = _submit_public(workspace, first)
    second_candidate = _submit_exclusion(workspace, second)
    # `claim_review` is deliberately deterministic by submission timestamp.
    # Avoid depending on opaque content-hash ordering when both test submits
    # happen in the same second.
    conn = sqlite3.connect(workspace)
    conn.execute("UPDATE candidates SET submitted_at=1 WHERE candidate_id=?", (first_candidate,))
    conn.execute("UPDATE candidates SET submitted_at=2 WHERE candidate_id=?", (second_candidate,))
    conn.commit()
    conn.close()

    first_review = claim_review(workspace, "reviewer-a")
    assert first_review is not None
    revised_candidate = str(first_review["candidate_id"])
    submit_review(
        workspace,
        {
            "candidate_id": revised_candidate,
            "reviewer": "reviewer-a",
            "verdict": "REVISE",
            "policy_version": "mechanics-v1",
        },
    )
    second_review = claim_review(workspace, "reviewer-b")
    assert second_review is not None
    rejected_candidate = str(second_review["candidate_id"])
    assert {revised_candidate, rejected_candidate} == {first_candidate, second_candidate}
    submit_review(
        workspace,
        {
            "candidate_id": rejected_candidate,
            "reviewer": "reviewer-b",
            "verdict": "REJECT",
            "policy_version": "mechanics-v1",
        },
    )

    rework = claim_shard(workspace, "worker-c")
    assert rework is not None and rework["shard_id"] == claim["shard_id"]
    assert rework["claim_mode"] == "rework"
    assert revised_candidate == first_candidate
    assert rejected_candidate == second_candidate
    revised_section = first
    rejected_section = second
    with pytest.raises(PermissionError, match="completed REVISE review"):
        _submit_public(workspace, rejected_section, worker="worker-c")
    replacement = _submit_public(workspace, revised_section, worker="worker-c")
    assert replacement

    conn = sqlite3.connect(workspace)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM candidates WHERE section_key=?", (rejected_section["section_key"],)
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_newly_eligible_rework_claim_is_atomic(tmp_path: Path):
    workspace = _workspace(tmp_path)
    _, section = _claim_and_read(workspace)
    candidate = _submit_public(workspace, section)
    reviewer_claim = claim_review(workspace, "reviewer-a")
    assert reviewer_claim is not None and reviewer_claim["candidate_id"] == candidate
    submit_review(
        workspace,
        {
            "candidate_id": candidate,
            "reviewer": "reviewer-a",
            "verdict": "REVISE",
            "policy_version": "mechanics-v1",
        },
    )

    barrier = threading.Barrier(2)
    claims: list[dict[str, object] | None] = []
    lock = threading.Lock()

    def claim_rework(worker: str) -> None:
        barrier.wait()
        result = claim_shard(workspace, worker)
        with lock:
            claims.append(result)

    threads = [threading.Thread(target=claim_rework, args=(worker,)) for worker in ("worker-b", "worker-c")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(result is not None for result in claims) == 1
    assert next(result for result in claims if result)["shard_id"] == 1


def test_build_uses_a_later_approved_replacement_after_revise(tmp_path: Path):
    workspace = _trusted_workspace(tmp_path)
    claim, section = _claim_and_read(workspace)
    original = _submit_public(workspace, section)
    review_claim = claim_review(workspace, "reviewer-b")
    assert review_claim is not None and review_claim["candidate_id"] == original
    submit_review(
        workspace,
        {
            "candidate_id": original,
            "reviewer": "reviewer-b",
            "verdict": "REVISE",
            "policy_version": "mechanics-v1",
        },
    )
    reopened = claim_shard(workspace, "worker-a")
    assert reopened is not None and reopened["shard_id"] == claim["shard_id"]
    replacement = _submit_public(
        workspace,
        read_claimed_shard(workspace, int(reopened["shard_id"]), "worker-a")[0],
    )
    _approve(workspace, replacement)

    assert build_public_corpus(workspace, tmp_path / "public.sqlite3", _notices(tmp_path))["sections"] == 1


def test_stale_hash_and_self_review_are_rejected(tmp_path: Path):
    workspace = _workspace(tmp_path)
    _, section = _claim_and_read(workspace)
    with pytest.raises(ValueError, match="stale source section hash"):
        submit_candidate(
            workspace,
            {
                "section_key": section["section_key"],
                "source_section_id": section["source_section_id"],
                "source_section_hash": "old-hash",
                "decision": "EXCLUDE",
                "worker": "worker-a",
                "prompt_version": "pilot-v1",
            },
        )
    candidate = _submit_public(workspace, section)
    with pytest.raises(PermissionError, match="cannot independently review"):
        submit_review(
            workspace,
            {
                "candidate_id": candidate,
                "reviewer": "worker-a",
                "verdict": "APPROVE",
                "policy_version": "mechanics-v1",
            },
        )


def test_clear_review_alias_confirms_only_nonpublic_decisions(tmp_path: Path):
    workspace = _workspace(tmp_path)
    _, section = _claim_and_read(workspace)
    result = submit_candidate(
        workspace,
        {
            "section_key": section["section_key"],
            "source_section_id": section["source_section_id"],
            "source_section_hash": section["source_section_hash"],
            "decision": "EXCLUDE",
            "reason_tags": ["no-mechanics"],
            "worker": "worker-a",
            "prompt_version": "pilot-v1",
        },
    )
    claim = claim_review(workspace, "reviewer-b")
    assert claim is not None

    submitted = submit_review(
        workspace,
        {
            "candidate_id": result["candidate_id"],
            "reviewer": "reviewer-b",
            "verdict": "CONFIRM_EXCLUSION",
            "policy_version": "mechanics-v1",
        },
    )

    assert submitted["review_id"].startswith("review:")


def test_versioned_section_keys_disambiguate_reused_source_ids(tmp_path: Path):
    workspace = tmp_path / "review.sqlite3"
    initialize_workspace(workspace, _duplicate_revision_source_db(tmp_path), shard_size=1)
    first_claim = claim_shard(workspace, "worker-first")
    assert first_claim is not None
    first = read_claimed_shard(workspace, int(first_claim["shard_id"]), "worker-first")[0]
    submit_candidate(
        workspace,
        {
            "section_key": first["section_key"],
            "source_section_id": first["source_section_id"],
            "source_section_hash": first["source_section_hash"],
            "decision": "EXCLUDE",
            "worker": "worker-first",
            "prompt_version": "pilot-v1",
        },
    )
    # V3 did not record which printing should remain the review target.  The
    # migration preserves both section identities but exposes exactly one.
    assert claim_shard(workspace, "worker-second") is None
    conn = sqlite3.connect(workspace)
    try:
        assert conn.execute("SELECT COUNT(*) FROM parser_runs WHERE state='retired'").fetchone()[0] == 1
    finally:
        conn.close()


def test_build_fails_closed_until_public_candidate_is_independently_approved(tmp_path: Path):
    workspace = _trusted_workspace(tmp_path)
    _, section = _claim_and_read(workspace)
    _submit_public(workspace, section)

    with pytest.raises(ValueError, match="unreviewed candidate"):
        build_public_corpus(workspace, tmp_path / "public.sqlite3", _notices(tmp_path))


def test_exact_review_invalidation_reopens_only_the_selected_candidate(tmp_path: Path):
    workspace = _workspace(tmp_path, section_count=2, shard_size=1)
    _, first = _claim_and_read(workspace)
    first_candidate = _submit_exclusion(workspace, first)
    _, second = _claim_and_read(workspace)
    second_candidate = _submit_exclusion(workspace, second)

    first_claim = claim_review(workspace, "reviewer-bad")
    assert first_claim is not None
    first_review = submit_review(
        workspace,
        {
            "candidate_id": first_claim["candidate_id"],
            "reviewer": "reviewer-bad",
            "verdict": "REJECT",
            "policy_version": "mechanics-v1",
        },
    )["review_id"]
    second_claim = claim_review(workspace, "reviewer-bad")
    assert second_claim is not None
    second_review = submit_review(
        workspace,
        {
            "candidate_id": second_claim["candidate_id"],
            "reviewer": "reviewer-bad",
            "verdict": "REJECT",
            "policy_version": "mechanics-v1",
        },
    )["review_id"]
    assert {first_claim["candidate_id"], second_claim["candidate_id"]} == {
        first_candidate,
        second_candidate,
    }

    with pytest.raises(ValueError, match="every requested review_id"):
        invalidate_reviews(
            workspace,
            "reviewer-bad",
            [second_review, "review:does-not-belong"],
            invalidated_by="audit-owner",
            reason="insufficient source review",
        )

    result = invalidate_reviews(
        workspace,
        "reviewer-bad",
        [first_review],
        invalidated_by="audit-owner",
        reason="insufficient source review",
    )
    assert result["reviewer"] == "reviewer-bad"
    assert result["invalidated"] == 1
    assert result["batch_id"].startswith("review-invalidation:")
    assert PRIVATE_TEXT not in json.dumps(result)

    status = workspace_status(workspace)["products"][0]
    assert status["reviews"] == 1
    assert status["invalidated_reviews"] == 1
    assert status["available_reviews"] == 1

    # The reviewer whose decision was invalidated cannot immediately validate
    # their own old work; a different independent reviewer receives it.
    assert claim_review(workspace, "reviewer-bad") is None
    recheck = claim_review(workspace, "reviewer-good")
    assert recheck is not None and recheck["candidate_id"] == first_candidate
    submit_review(
        workspace,
        {
            "candidate_id": first_candidate,
            "reviewer": "reviewer-good",
            "verdict": "REJECT",
            "policy_version": "mechanics-v1",
        },
    )

    status = workspace_status(workspace)["products"][0]
    assert status["reviews"] == 2
    assert status["invalidated_reviews"] == 1
    assert status["available_reviews"] == 0
    conn = sqlite3.connect(workspace)
    try:
        assert conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM review_invalidations").fetchone()[0] == 1
    finally:
        conn.close()


def test_invalidated_public_approval_cannot_build_until_independently_rechecked(tmp_path: Path):
    workspace = _trusted_workspace(tmp_path)
    _, section = _claim_and_read(workspace)
    candidate = _submit_public(workspace, section)
    _approve(workspace, candidate, "reviewer-bad")
    conn = sqlite3.connect(workspace)
    try:
        review_id = conn.execute(
            "SELECT review_id FROM reviews WHERE candidate_id=?", (candidate,)
        ).fetchone()[0]
    finally:
        conn.close()

    invalidate_reviews(
        workspace,
        "reviewer-bad",
        [review_id],
        invalidated_by="audit-owner",
        reason="batch did not inspect source text",
    )
    with pytest.raises(ValueError, match="unreviewed candidate"):
        build_public_corpus(workspace, tmp_path / "public.sqlite3", _notices(tmp_path))

    recheck = claim_review(workspace, "reviewer-good")
    assert recheck is not None and recheck["candidate_id"] == candidate
    submit_review(
        workspace,
        {
            "candidate_id": candidate,
            "reviewer": "reviewer-good",
            "verdict": "APPROVE",
            "policy_version": "mechanics-v1",
            "reason_tags": ["functional-rules"],
        },
    )
    assert build_public_corpus(
        workspace, tmp_path / "public.sqlite3", _notices(tmp_path)
    ) == {
            "sections": 1, "sources": 1, "foundry_requirements": 0,
            "revisions": 1, "notices": 1, "covered_products": 1,
            "review_scope_digest": ANY,
    }


def test_invalidated_revise_requires_active_rereview_before_rework_and_build(tmp_path: Path):
    workspace = _trusted_workspace(tmp_path)
    first_claim, section = _claim_and_read(workspace)
    original = _submit_public(workspace, section)
    review_claim = claim_review(workspace, "reviewer-bad")
    assert review_claim is not None and review_claim["candidate_id"] == original
    bad_review = submit_review(
        workspace,
        {
            "candidate_id": original,
            "reviewer": "reviewer-bad",
            "verdict": "REVISE",
            "policy_version": "mechanics-v1",
        },
    )["review_id"]

    invalidate_reviews(
        workspace,
        "reviewer-bad",
        [bad_review],
        invalidated_by="audit-owner",
        reason="revision advice did not inspect source text",
    )
    assert claim_shard(workspace, "worker-replacement") is None
    with pytest.raises(ValueError, match="unreviewed candidate"):
        build_public_corpus(workspace, tmp_path / "public.sqlite3", _notices(tmp_path))

    rereview = claim_review(workspace, "reviewer-good")
    assert rereview is not None and rereview["candidate_id"] == original
    submit_review(
        workspace,
        {
            "candidate_id": original,
            "reviewer": "reviewer-good",
            "verdict": "REVISE",
            "policy_version": "mechanics-v1",
        },
    )
    with pytest.raises(ValueError, match="rework remains pending"):
        build_public_corpus(workspace, tmp_path / "public.sqlite3", _notices(tmp_path))

    rework = claim_shard(workspace, "worker-replacement")
    assert rework is not None
    assert rework["shard_id"] == first_claim["shard_id"]
    assert rework["claim_mode"] == "rework"
    replacement = _submit_public(
        workspace,
        read_claimed_shard(workspace, int(rework["shard_id"]), "worker-replacement")[0],
        worker="worker-replacement",
    )
    _approve(workspace, replacement, "reviewer-final")

    assert build_public_corpus(
        workspace, tmp_path / "public.sqlite3", _notices(tmp_path)
    ) == {
            "sections": 1, "sources": 1, "foundry_requirements": 0,
            "revisions": 1, "notices": 1, "covered_products": 1,
            "review_scope_digest": ANY,
    }


def test_public_build_is_logically_deterministic_and_excludes_private_fields(tmp_path: Path):
    workspace = _trusted_workspace(tmp_path)
    _, section = _claim_and_read(workspace)
    candidate = _submit_public(workspace, section)
    _approve(workspace, candidate)
    first = tmp_path / "first.sqlite3"
    second = tmp_path / "second.sqlite3"
    notices = _notices(tmp_path)

    expected = {
        "sections": 1, "sources": 1, "foundry_requirements": 0,
        "revisions": 1, "notices": 1, "covered_products": 1,
        "review_scope_digest": ANY,
    }
    assert build_public_corpus(workspace, first, notices) == expected
    assert build_public_corpus(workspace, second, notices) == expected

    def logical_rows(path: Path) -> dict[str, list[tuple]]:
        conn = sqlite3.connect(path)
        try:
            return {
                table: conn.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
                for table in (
                    "metadata", "source_revisions", "notices", "licensed_rules",
                    "licensed_rule_sources", "required_foundry_rows",
                )
            }
        finally:
            conn.close()

    assert logical_rows(first) == logical_rows(second)
    conn = sqlite3.connect(first)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(licensed_rules)")}
        assert "source_text" not in columns
        assert "claimant" not in columns
        assert "worker" not in columns
        assert "url" not in columns
        values = " ".join(
            str(value)
            for table in ("licensed_rules", "licensed_rule_sources")
            for row in conn.execute(f"SELECT * FROM {table}")
            for value in row
        )
        assert PRIVATE_TEXT not in values
        assert PUBLIC_TEXT in values
        public_tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "evidence" not in public_tables
    finally:
        conn.close()


def test_public_build_deduplicates_converged_rules_and_keeps_all_sources(tmp_path: Path):
    workspace = _workspace(tmp_path)
    bundle = _trusted_multi_section_bundle(("Private source A.", "Private source B."))
    with patch("pf2e_codex.licensed_corpus.load_and_parse_verified_pdf", return_value=bundle):
        staged = stage_trusted_native_pdf(
            workspace, tmp_path / "owned.pdf", product_code="PZO12001",
            parser_version="paizo-native-v1", shard_size=1,
        )
    activate_parser_run(workspace, str(staged["parser_run_id"]))
    for producer, reviewer in (("worker-a", "reviewer-a"), ("worker-b", "reviewer-b")):
        _, section = _claim_and_read(workspace, producer)
        candidate = _submit_public(workspace, section, producer)
        _approve(workspace, candidate, reviewer)

    output = tmp_path / "deduplicated.sqlite3"
    result = build_public_corpus(workspace, output, _notices(tmp_path))
    assert result["sections"] == 1
    assert result["sources"] == 2
    with sqlite3.connect(output) as conn:
        assert conn.execute("SELECT COUNT(*) FROM licensed_rules").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM licensed_rule_sources").fetchone()[0] == 2


def test_build_rejects_multiple_approved_candidates_for_one_source(tmp_path: Path):
    workspace = _trusted_workspace(tmp_path)
    _, section = _claim_and_read(workspace)
    first = _submit_public(workspace, section)
    _approve(workspace, first, "reviewer-one")
    # The normal lifecycle releases a completed shard, so construct the
    # malformed historical state directly to prove the final builder fails
    # closed if a corrupted/legacy workspace has two approvals.
    second = "candidate:synthetic-second-approval"
    second_hash = _candidate_commitment(
        source_section_hash=section["source_section_hash"], decision="PUBLIC_AS_IS",
        candidate_text=str(section["source_text"]), public_heading="Reviewed rules",
        extraction_method="human-reconstruction-v1", reason_tags=["layout-reviewed"],
    )
    conn = sqlite3.connect(workspace)
    conn.execute(
        """INSERT INTO candidates
        (candidate_id, section_key, source_section_hash, candidate_ordinal, decision, candidate_text,
         public_heading, candidate_hash, extraction_method, reason_tags, confidence, worker,
         prompt_version, submitted_at)
        VALUES (?, ?, ?, 2, 'PUBLIC_AS_IS', ?, 'Reviewed rules', ?,
        'human-reconstruction-v1', '["layout-reviewed"]', 0.9, 'worker-a', 'pilot-v1', 2)""",
        (second, section["section_key"], section["source_section_hash"], str(section["source_text"]), second_hash),
    )
    conn.commit()
    conn.close()
    _approve(workspace, second, "reviewer-two")

    with pytest.raises(ValueError, match="ambiguous multiple approvals"):
        build_public_corpus(workspace, tmp_path / "public.sqlite3", _notices(tmp_path))


def _staged_run_input(
    section: dict[str, object], *, text: str | None = None, complete: bool = True
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Use stored V3-migration anchors as the parser contract for seam tests."""
    source_text = text if text is not None else str(section["source_text"])
    records = [{
        "source_section_id": section["source_section_id"],
        "source_section_hash": "new-" + str(section["source_section_hash"]),
        "source_text": source_text,
        "heading": section["heading"],
        "page_start": section["page_start"],
        "page_end": section["page_end"],
        "printed_page": section["printed_page"],
        "stable_identity": section["stable_identity"],
        "provenance_hash": section["provenance_hash"],
        "text_hash": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "native_word_count": 1,
        "native_word_anchors": [hashlib.sha256(b"native-word-anchor-0").hexdigest()],
    }]
    records[0]["native_word_digest"] = _anchor_digest(
        "section-native-word-anchors-v1", records[0]["native_word_anchors"]
    )
    output_digest = _parser_output_digest(records)
    run = {
        "product_code": section["product_code"],
        "source_fingerprint": section["content_fingerprint"],
        "parser_version": "paizo-native-v2",
        "parser_output_digest": output_digest,
        "license": "ORC",
        "era": "remaster",
    }
    if complete:
        run["complete_manifest"] = {
            "version": "native-coverage-v1",
            "declared_section_count": len(records),
            "output_digest": output_digest,
            "native_word_coverage_digest": _native_word_coverage_digest(records),
            "removed_stable_identities": [],
            "ignored_anchor_policy": "none-v1",
            "ignored_anchors": [],
        }
    return run, records


def _bind_inventory(workspace: Path, section: dict[str, object], records: list[dict[str, object]]) -> None:
    anchors = [str(anchor) for record in records for anchor in record["native_word_anchors"]]
    bind_source_asset_inventory(
        workspace,
        str(section["product_code"]),
        str(section["content_fingerprint"]),
        {
            "inventory_profile": "native-words-v1",
            "version": "1",
            "ignored_anchor_policy": "none-v1",
            "native_word_anchors": anchors,
            "ignored_anchors": [],
        },
    )


def test_stage_parser_run_exact_reuses_completed_resolution_and_activation_scopes_claims(tmp_path: Path):
    workspace = _trusted_workspace(tmp_path)
    _, section = _claim_and_read(workspace)
    candidate = _submit_exclusion(workspace, section)
    review_claim = claim_review(workspace, "reviewer-a")
    assert review_claim is not None and review_claim["candidate_id"] == candidate
    submit_review(workspace, {
        "candidate_id": candidate, "reviewer": "reviewer-a", "verdict": "REJECT", "policy_version": "mechanics-v1",
    })
    staged = _stage_bundle(workspace, tmp_path, _trusted_bundle(source_id="pzo12001:player-core:p1:h0123456789abcdef:i2"))
    assert staged["state"] == "staged"
    assert staged["unchanged"] == staged["reused"] == 1
    assert claim_shard(workspace, "worker-b") is None
    activated = activate_parser_run(workspace, str(staged["parser_run_id"]))
    assert activated["state"] == "active"
    # Reused final exclusions need no additional candidate claim and old run
    # rows cannot be selected after the atomic switch.
    assert claim_shard(workspace, "worker-b") is None
    assert build_public_corpus(workspace, tmp_path / "public.sqlite3", _notices(tmp_path))["sections"] == 0


def test_stage_parser_run_changes_and_removals_queue_without_fuzzy_reuse(tmp_path: Path):
    workspace = _trusted_workspace(tmp_path)
    staged = _stage_bundle(workspace, tmp_path, _trusted_bundle(
        source_id="pzo12001:player-core:p1:h0123456789abcdef:i2", text="Private source rule text changed.",
    ))
    assert staged["changed"] == 1
    assert staged["removed"] == 0
    assert staged["reused"] == 0
    activate_parser_run(workspace, str(staged["parser_run_id"]))


def test_parser_run_activation_rejects_live_product_claim(tmp_path: Path):
    workspace = _trusted_workspace(tmp_path)
    staged = _stage_bundle(workspace, tmp_path, _trusted_bundle(source_id="pzo12001:player-core:p1:h0123456789abcdef:i2"))
    assert claim_shard(workspace, "worker-a") is not None
    with pytest.raises(PermissionError, match="live"):
        activate_parser_run(workspace, str(staged["parser_run_id"]))


def test_parser_run_activation_rejects_live_binary_screen_claim(tmp_path: Path):
    workspace = _trusted_workspace(tmp_path)
    staged = _stage_bundle(
        workspace,
        tmp_path,
        _trusted_bundle(source_id="pzo12001:player-core:p1:h0123456789abcdef:i2"),
    )
    claim = claim_draft_screening_batch(workspace, "screen-a")
    assert claim is not None
    with pytest.raises(PermissionError, match="live"):
        activate_parser_run(workspace, str(staged["parser_run_id"]))

    assert release_draft_screening_batch(
        workspace, int(claim["shard_id"]), "screen-a"
    )["released"] is True
    assert activate_parser_run(workspace, str(staged["parser_run_id"]))["state"] == "active"


def test_exact_parser_reuse_keeps_canonical_public_fingerprint_and_id(tmp_path: Path):
    workspace = _trusted_workspace(tmp_path)
    _, section = _claim_and_read(workspace)
    candidate = _submit_public(workspace, section)
    _approve(workspace, candidate)
    before = tmp_path / "before.sqlite3"
    assert build_public_corpus(workspace, before, _notices(tmp_path))["sections"] == 1
    before_bundle = load_licensed_core(before)
    before_chunk = before_bundle.chunks[0]
    staged = _stage_bundle(workspace, tmp_path, _trusted_bundle(source_id="pzo12001:player-core:p1:h0123456789abcdef:i2"))
    assert staged["reused"] == 1
    activate_parser_run(workspace, str(staged["parser_run_id"]))
    after = tmp_path / "after.sqlite3"
    assert build_public_corpus(workspace, after, _notices(tmp_path))["sections"] == 1
    after_chunk = load_licensed_core(after).chunks[0]
    assert after_chunk["id"] == before_chunk["id"]
    assert after_chunk["licensed_provenance"]["content_fingerprint"] == before_chunk["licensed_provenance"]["content_fingerprint"]


def test_incomplete_manifest_activation_leaves_old_target_active(tmp_path: Path):
    workspace = _trusted_workspace(tmp_path)
    # The lower-level historical seam is no longer capable of declaring a
    # run complete: only a direct PDF bundle receives the private capability.
    with pytest.raises(ValueError, match="direct-PDF bundle"):
        stage_parser_run(workspace, {"product_code": "PZO12001"}, [])
    conn = sqlite3.connect(workspace)
    try:
        assert conn.execute("SELECT state FROM parser_runs WHERE state='active'").fetchone()[0] == "active"
    finally:
        conn.close()


def test_stage_fails_closed_when_target_state_is_corrupt(tmp_path: Path):
    workspace = _trusted_workspace(tmp_path)
    conn = sqlite3.connect(workspace)
    try:
        conn.execute("DROP INDEX parser_runs_one_active_target")
        row = conn.execute("SELECT * FROM parser_runs WHERE state='active'").fetchone()
        conn.execute(
            """INSERT INTO parser_runs
            (parser_run_id, asset_id, product_code, source_fingerprint, parser_version,
             parser_output_digest, state, review_enabled, complete, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'active', 1, 1, 1)""",
            ("corrupt-run", row[1], row[2], row[3], "bad", "b" * 64),
        )
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(ValueError, match="ambiguous"):
        _stage_bundle(workspace, tmp_path, _trusted_bundle(source_id="pzo12001:player-core:p1:h0123456789abcdef:i2"))


def test_v3_workspace_migrates_then_accepts_a_staged_run(tmp_path: Path):
    workspace = _workspace(tmp_path)
    _, section = _claim_and_read(workspace)
    conn = sqlite3.connect(workspace)
    try:
        conn.execute("DROP TABLE parser_run_sections")
        conn.execute("DROP TABLE parser_runs")
        conn.execute("DROP TABLE source_assets")
        conn.execute("ALTER TABLE source_sections DROP COLUMN parser_run_id")
        conn.execute("ALTER TABLE source_sections DROP COLUMN stable_identity")
        conn.execute("ALTER TABLE source_sections DROP COLUMN provenance_hash")
        conn.execute("ALTER TABLE source_sections DROP COLUMN text_hash")
        conn.execute("ALTER TABLE review_shards DROP COLUMN parser_run_id")
        conn.execute("UPDATE metadata SET value='3' WHERE key='review_schema_version'")
        conn.commit()
    finally:
        conn.close()
    staged = _stage_bundle(workspace, tmp_path, _trusted_bundle())
    assert staged["state"] == "staged"


def test_complete_stage_rejects_truncated_self_consistent_parser_inventory(tmp_path: Path):
    workspace = _workspace(tmp_path)
    bundle = _trusted_bundle()
    # A bundle with a self-consistent subset cannot pass because the direct
    # bridge checks the immutable whole-asset inventory before staging.
    bad_inventory = NativeWordInventory(bundle.semantic_fingerprint, bundle.inventory.anchors + (hashlib.sha256(b"missing").hexdigest(),), bundle.inventory.ignored_anchors, {}, {})
    malformed = TrustedParseBundle(
        bundle.product_code, bundle.parser_version, bundle.exporter_profile_version, bundle.semantic_fingerprint,
        bundle.artifact_attestation, bad_inventory, bundle.sections, bundle.parser_output_digest,
        bundle.sealed_digest, artifact_attestation_digest=bundle.artifact_attestation_digest,
    )
    with pytest.raises(ValueError, match="exactly match"):
        _stage_bundle(workspace, tmp_path, malformed)
    conn = sqlite3.connect(workspace)
    try:
        assert conn.execute("SELECT COUNT(*) FROM parser_runs WHERE state='active'").fetchone()[0] == 1
    finally:
        conn.close()


def test_native_anchor_validation_ignored_reasons_and_inventory_immutability(tmp_path: Path):
    workspace = _trusted_workspace(tmp_path)
    first = _stage_bundle(workspace, tmp_path, _trusted_bundle(source_id="pzo12001:player-core:p1:h0123456789abcdef:i2"))
    assert first["state"] == "staged"
    # Same canonical asset but a different inventory cannot replace the
    # immutable asset contract.
    bundle = _trusted_bundle(source_id="pzo12001:player-core:p1:h0123456789abcdef:i3")
    changed_inventory = NativeWordInventory(
        bundle.semantic_fingerprint,
        bundle.inventory.anchors + (hashlib.sha256(b"changed-anchor").hexdigest(),),
        bundle.inventory.ignored_anchors, {}, {},
    )
    changed = TrustedParseBundle(
        bundle.product_code, bundle.parser_version, bundle.exporter_profile_version,
        bundle.semantic_fingerprint, bundle.artifact_attestation, changed_inventory,
        bundle.sections, bundle.parser_output_digest, bundle.sealed_digest,
        artifact_attestation_digest=bundle.artifact_attestation_digest,
    )
    with pytest.raises(ValueError, match="provenance conflicts|immutable"):
        _stage_bundle(workspace, tmp_path, changed)


def test_native_inventory_is_watermark_independent_and_only_ignores_constrained_furniture():
    first = native_word_inventory(_native_export(watermark="alice@example.invalid"), "PZO12001")
    second = native_word_inventory(_native_export(watermark="bob@example.invalid"), "PZO12001")
    moved_watermark = _native_export(watermark="carol@example.invalid")
    watermark_word = moved_watermark["pages"][0]["words"].pop(3)
    moved_watermark["pages"][0]["words"].insert(0, watermark_word)
    third = native_word_inventory(moved_watermark, "PZO12001")
    assert first.content_fingerprint == second.content_fingerprint
    assert first.anchors == second.anchors
    assert first.content_fingerprint == third.content_fingerprint
    assert first.anchors == third.anchors
    assert {item["reason"] for item in first.ignored_anchors} >= {
        "printed-page-number-v1", "repeated-margin-furniture-v1",
    }
    assert all("@" not in anchor for anchor in first.anchors)


def test_trusted_native_pdf_staging_never_accepts_cached_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import pf2e_codex.licensed_corpus as review_workspace

    workspace = _workspace(tmp_path)
    export = tmp_path / "native.json"
    export.write_text(json.dumps(_native_export()), encoding="utf-8")
    monkeypatch.setattr(
        review_workspace, "load_and_parse_verified_pdf",
        lambda path, **_kwargs: (_ for _ in ()).throw(ValueError(f"direct PDF bridge received {path}")),
    )
    with pytest.raises(ValueError, match="direct PDF bridge"):
        stage_trusted_native_pdf(workspace, export, product_code="PZO12001", parser_version="paizo-native-v1")


def _trusted_bundle(
    *,
    source_id: str = "pzo12001:player-core:p1:h0123456789abcdef:i0",
    text: str = "Private source rule text.",
    flags: tuple[str, ...] = (),
    section_anchors: tuple[str, ...] | None = None,
) -> TrustedParseBundle:
    section_anchors = section_anchors or (hashlib.sha256(b"section-anchor").hexdigest(),)
    ignored_anchor = hashlib.sha256(b"ignored-anchor").hexdigest()
    fingerprint = hashlib.sha256(b"source-fingerprint").hexdigest()
    section = TrustedSection(
        id="private:section:1", source_section_id=source_id, heading="Source heading", text=text,
        text_hash=hashlib.sha256(text.encode()).hexdigest(), physical_pages=(1,), printed_page="1",
        stable_section_identity=hashlib.sha256(b"stable-section").hexdigest(), layout_flags=flags,
        coverage_anchors=section_anchors,
    )
    inventory = NativeWordInventory(
        fingerprint, (*section_anchors, ignored_anchor),
        ({"anchor_hash": ignored_anchor, "reason": "watermark-email-span-v1"},), {}, {},
    )
    attestation = {"product_verified": True, "page_count": 1, "title_marker_verified": True, "matched_product_count": 1, "conflict_product_count": 0}
    attestation_digest = hashlib.sha256(b"attestation").hexdigest()
    parser_digest = _trusted_parser_output_digest((section,))
    seal = _trusted_bundle_seal(
        product_code="PZO12001", parser_version="paizo-native-v1", exporter_profile_version=1,
        semantic_fingerprint=fingerprint, artifact_attestation=attestation,
        artifact_attestation_digest=attestation_digest, inventory=inventory, sections=(section,),
        parser_output_digest=parser_digest,
    )
    return TrustedParseBundle(
        "PZO12001", "paizo-native-v1", 1, fingerprint, attestation, inventory,
        (section,), parser_digest, seal, artifact_attestation_digest=attestation_digest,
    )


def _stage_bundle(workspace: Path, tmp_path: Path, bundle: TrustedParseBundle) -> dict[str, object]:
    with patch("pf2e_codex.licensed_corpus.load_and_parse_verified_pdf", return_value=bundle):
        return stage_trusted_native_pdf(
            workspace, tmp_path / "owned.pdf", product_code="PZO12001", parser_version="paizo-native-v1",
        )


def test_v4_stage_persists_structural_blocks_and_quarantine(tmp_path: Path):
    workspace = _workspace(tmp_path)
    heading_anchor = hashlib.sha256(b"v4-heading").hexdigest()
    body_anchor = hashlib.sha256(b"v4-body").hexdigest()
    quarantine_anchor = hashlib.sha256(b"v4-quarantine").hexdigest()
    ignored_anchor = hashlib.sha256(b"v4-ignored").hexdigest()
    fingerprint = hashlib.sha256(b"v4-source").hexdigest()
    heading = "Rule Heading"
    body = "A creature gains a circumstance bonus."
    text = f"{heading} {body}"
    blocks = (
        TrustedBlock(
            "heading", 1, 0, heading, hashlib.sha256(heading.encode()).hexdigest(),
            (heading_anchor,),
        ),
        TrustedBlock(
            "body", 1, 1, body, hashlib.sha256(body.encode()).hexdigest(),
            (body_anchor,),
        ),
    )
    section = TrustedSection(
        id="private:v4:1",
        source_section_id="pzo12001:player-core:p1:h0123456789abcdef:i0",
        heading=heading,
        text=text,
        text_hash=hashlib.sha256(text.encode()).hexdigest(),
        physical_pages=(1,),
        printed_page="1",
        stable_section_identity=hashlib.sha256(b"v4-stable").hexdigest(),
        layout_flags=(),
        coverage_anchors=(heading_anchor, body_anchor),
        blocks=blocks,
    )
    quarantine = TrustedQuarantine(
        "pzo12001:player-core:p1:q0:0123456789abcdef",
        "unbound-layout",
        1,
        "Private unresolved fragment.",
        hashlib.sha256(b"Private unresolved fragment.").hexdigest(),
        (quarantine_anchor,),
    )
    inventory = NativeWordInventory(
        fingerprint,
        (heading_anchor, body_anchor, quarantine_anchor, ignored_anchor),
        ({"anchor_hash": ignored_anchor, "reason": "watermark-email-span-v1"},),
        {},
        {ignored_anchor: "watermark-email-span-v1"},
    )
    attestation = {
        "product_verified": True, "page_count": 1, "title_marker_verified": True,
        "matched_product_count": 1, "conflict_product_count": 0,
    }
    attestation_digest = hashlib.sha256(b"v4-attestation").hexdigest()
    parser_digest = _trusted_parser_output_digest((section,), (quarantine,))
    layout_digest = hashlib.sha256(b"v4-layout").hexdigest()
    seal = _trusted_bundle_seal(
        product_code="PZO12001", parser_version=PAIZO_NATIVE_PARSER_V4,
        exporter_profile_version=1, semantic_fingerprint=fingerprint,
        artifact_attestation=attestation, artifact_attestation_digest=attestation_digest,
        inventory=inventory, sections=(section,), quarantine=(quarantine,),
        parser_output_digest=parser_digest, layout_binding_digest=layout_digest,
    )
    bundle = TrustedParseBundle(
        "PZO12001", PAIZO_NATIVE_PARSER_V4, 1, fingerprint, attestation, inventory,
        (section,), parser_digest, seal, artifact_attestation_digest=attestation_digest,
        layout_binding_digest=layout_digest, quarantine=(quarantine,),
    )
    with patch("pf2e_codex.licensed_corpus.load_and_parse_verified_pdf", return_value=bundle):
        staged = stage_trusted_native_pdf(
            workspace, tmp_path / "PZO12001E.pdf", product_code="PZO12001",
            parser_version=PAIZO_NATIVE_PARSER_V4, layout_artifact={},
        )
    activate_parser_run(workspace, str(staged["parser_run_id"]))

    with sqlite3.connect(workspace) as conn:
        assert conn.execute("SELECT COUNT(*) FROM parser_section_blocks").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM parser_section_block_anchors").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM parser_quarantine").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM parser_quarantine_anchors").fetchone()[0] == 1


def test_quarantine_schema_upgrade_preserves_records_and_anchors(tmp_path: Path):
    workspace = _workspace(tmp_path)
    with sqlite3.connect(workspace) as conn:
        parser_run_id = conn.execute(
            "SELECT parser_run_id FROM parser_runs ORDER BY parser_run_id LIMIT 1"
        ).fetchone()[0]
        conn.executescript(
            """
            DROP TABLE parser_quarantine_anchors;
            DROP INDEX parser_quarantine_by_product;
            DROP TABLE parser_quarantine;
            CREATE TABLE parser_quarantine (
                parser_run_id TEXT NOT NULL,
                quarantine_id TEXT NOT NULL,
                product_code TEXT NOT NULL,
                reason TEXT NOT NULL CHECK(reason IN ('unbound-layout')),
                physical_page INTEGER NOT NULL,
                source_text TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                anchor_count INTEGER NOT NULL,
                anchor_digest TEXT NOT NULL,
                PRIMARY KEY(parser_run_id, quarantine_id)
            );
            CREATE TABLE parser_quarantine_anchors (
                parser_run_id TEXT NOT NULL,
                quarantine_id TEXT NOT NULL,
                anchor_hash TEXT NOT NULL,
                PRIMARY KEY(parser_run_id, anchor_hash)
            );
            """
        )
        conn.execute(
            "INSERT INTO parser_quarantine VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (parser_run_id, "q1", "PZO12001", "unbound-layout", 1,
             "private fragment", hashlib.sha256(b"private fragment").hexdigest(),
             1, hashlib.sha256(b"anchor").hexdigest()),
        )
        conn.execute(
            "INSERT INTO parser_quarantine_anchors VALUES (?, ?, ?)",
            (parser_run_id, "q1", hashlib.sha256(b"native anchor").hexdigest()),
        )

    workspace_status(workspace)

    with sqlite3.connect(workspace) as conn:
        assert conn.execute("SELECT reason FROM parser_quarantine").fetchone()[0] == "unbound-layout"
        assert conn.execute("SELECT COUNT(*) FROM parser_quarantine_anchors").fetchone()[0] == 1
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='parser_quarantine'"
        ).fetchone()[0]
        assert "layout-order-conflict" in sql
        assert "oversize-block" in sql


def _trusted_multi_section_bundle(
    texts: tuple[str, ...], *, headings: tuple[str, ...] | None = None
) -> TrustedParseBundle:
    """Build a sealed direct-PDF fixture with deterministic, distinct sections."""
    if headings is not None and len(headings) != len(texts):
        raise ValueError("fixture headings must match fixture texts")
    fingerprint = hashlib.sha256(b"multi-section-source-fingerprint").hexdigest()
    sections: list[TrustedSection] = []
    anchors: list[str] = []
    for index, text in enumerate(texts):
        anchor = hashlib.sha256(f"section-anchor-{index}".encode()).hexdigest()
        anchors.append(anchor)
        sections.append(
            TrustedSection(
                id=f"private:section:{index}",
                source_section_id=f"pzo12001:player-core:p{index + 1}:h{index:016x}:i0",
                heading=headings[index] if headings is not None else f"Source heading {index}",
                text=text,
                text_hash=hashlib.sha256(text.encode()).hexdigest(),
                physical_pages=(index + 1,),
                printed_page=str(index + 1),
                stable_section_identity=hashlib.sha256(
                    f"stable-section-{index}".encode()
                ).hexdigest(),
                layout_flags=(),
                coverage_anchors=(anchor,),
            )
        )
    ignored_anchor = hashlib.sha256(b"multi-ignored-anchor").hexdigest()
    inventory = NativeWordInventory(
        fingerprint,
        (*anchors, ignored_anchor),
        ({"anchor_hash": ignored_anchor, "reason": "watermark-email-span-v1"},),
        {},
        {},
    )
    attestation = {
        "product_verified": True,
        "page_count": len(sections),
        "title_marker_verified": True,
        "matched_product_count": 1,
        "conflict_product_count": 0,
    }
    attestation_digest = hashlib.sha256(b"multi-attestation").hexdigest()
    section_tuple = tuple(sections)
    parser_digest = _trusted_parser_output_digest(section_tuple)
    seal = _trusted_bundle_seal(
        product_code="PZO12001",
        parser_version="paizo-native-v1",
        exporter_profile_version=1,
        semantic_fingerprint=fingerprint,
        artifact_attestation=attestation,
        artifact_attestation_digest=attestation_digest,
        inventory=inventory,
        sections=section_tuple,
        parser_output_digest=parser_digest,
    )
    return TrustedParseBundle(
        "PZO12001",
        "paizo-native-v1",
        1,
        fingerprint,
        attestation,
        inventory,
        section_tuple,
        parser_digest,
        seal,
        artifact_attestation_digest=attestation_digest,
    )


def test_trusted_stitch_unions_only_adjacent_full_sections_and_preserves_anchors():
    bundle = _trusted_multi_section_bundle(("Rule fragment", "continues here.", "Other rule."))
    first, second, third = bundle.sections

    repaired = repair_trusted_bundle(
        bundle, [[first.source_section_id, second.source_section_id]]
    )

    assert len(repaired.sections) == 2
    assert repaired.sections[0].text == "Rule fragment\n\ncontinues here."
    assert repaired.sections[0].coverage_anchors == (
        *first.coverage_anchors, *second.coverage_anchors,
    )
    assert repaired.sections[1] == third
    repaired.verify_seal()

    with pytest.raises(ValueError, match="consecutive"):
        repair_trusted_bundle(
            bundle, [[first.source_section_id, third.source_section_id]]
        )


def _activate_bundle(
    workspace: Path,
    tmp_path: Path,
    bundle: TrustedParseBundle,
    *,
    shard_size: int = 100,
) -> dict[str, object]:
    with patch("pf2e_codex.licensed_corpus.load_and_parse_verified_pdf", return_value=bundle):
        staged = stage_trusted_native_pdf(
            workspace,
            tmp_path / "owned.pdf",
            product_code="PZO12001",
            parser_version="paizo-native-v1",
            shard_size=shard_size,
        )
    activate_parser_run(workspace, str(staged["parser_run_id"]))
    return staged


def test_binary_screen_is_private_minimal_and_idempotent(tmp_path: Path):
    workspace = _workspace(tmp_path)
    _activate_bundle(workspace, tmp_path, _trusted_bundle())

    status = draft_screening_status(workspace)
    assert status == {
        "products": [
                {
                    "product_code": "PZO12001",
                    "sections": 1,
                    "unprocessed": 1,
                    "accepted": 0,
                    "rejected": 0,
                    "deferred": 0,
                    "duplicate_rejected": 0,
                    "unprocessed_batches": 1,
                    "deferred_batches": 0,
                    "live_claims": 0,
            }
        ],
        "workspace_scope": "private-draft-screen",
    }
    assert PRIVATE_TEXT not in json.dumps(status)

    claim = claim_draft_screening_batch(workspace, "screen-a")
    assert claim is not None
    assert "source_text" not in claim
    shard_id = int(claim["shard_id"])
    with pytest.raises(PermissionError):
        read_draft_screening_record(workspace, shard_id, "screen-b", 0)
    record = read_draft_screening_record(workspace, shard_id, "screen-a", 0)
    assert record["state"] == "pending"
    assert record["source_text"] == "Private source rule text."

    inserted = submit_draft_screening_decision(
        workspace, shard_id, "screen-a", 0, "add"
    )
    assert inserted == {
        "index": 0,
        "state": "inserted",
        "decision": "add",
        "duplicate_rejected": False,
        "batch_complete": True,
    }
    assert submit_draft_screening_decision(
        workspace, shard_id, "screen-a", 0, "add"
    ) == {**inserted, "state": "unchanged"}
    with pytest.raises(ValueError, match="conflicting"):
        submit_draft_screening_decision(
            workspace, shard_id, "screen-a", 0, "reject"
        )
    with pytest.raises(PermissionError, match="another worker"):
        submit_draft_screening_decision(
            workspace, shard_id, "screen-b", 0, "add"
        )

    final = draft_screening_status(workspace)["products"][0]
    assert final["accepted"] == 1
    assert final["unprocessed"] == 0
    assert final["live_claims"] == 0


def test_binary_screen_does_not_guess_unprepared_duplicate_groups(tmp_path: Path):
    workspace = _workspace(tmp_path)
    _activate_bundle(
        workspace,
        tmp_path,
        _trusted_multi_section_bundle(
            ("Same private rule.", "Same private rule."),
            headings=("Same heading", "Same heading"),
        ),
    )
    claim = claim_draft_screening_batch(workspace, "screen-a")
    assert claim is not None
    shard_id = int(claim["shard_id"])

    duplicate = submit_draft_screening_decision(
        workspace, shard_id, "screen-a", 1, "add"
    )
    canonical = submit_draft_screening_decision(
        workspace, shard_id, "screen-a", 0, "add"
    )
    assert duplicate["decision"] == "add"
    assert duplicate["duplicate_rejected"] is False
    assert canonical["decision"] == "add"
    status = draft_screening_status(workspace)["products"][0]
    assert status == {
            "product_code": "PZO12001",
            "sections": 2,
            "unprocessed": 0,
            "accepted": 2,
            "rejected": 0,
            "deferred": 0,
            "duplicate_rejected": 0,
            "unprocessed_batches": 0,
            "deferred_batches": 0,
            "live_claims": 0,
    }


def test_binary_screen_step_returns_only_the_next_pending_record(tmp_path: Path):
    workspace = _workspace(tmp_path)
    _activate_bundle(
        workspace,
        tmp_path,
        _trusted_multi_section_bundle(("General rule one.", "General rule two.")),
    )
    claim = claim_draft_screening_batch(workspace, "screen-a")
    assert claim is not None
    shard_id = int(claim["shard_id"])

    first = step_draft_screening(workspace, shard_id, "screen-a", 0, "reject")
    assert first["result"]["decision"] == "reject"
    assert first["next_record"]["index"] == 1
    assert first["next_record"]["source_text"] == "General rule two."
    final = step_draft_screening(workspace, shard_id, "screen-a", 1, "add")
    assert final["result"]["batch_complete"] is True
    assert final["next_record"] is None


def test_binary_screen_next_skips_existing_decisions_on_resumed_batch(tmp_path: Path):
    workspace = _workspace(tmp_path)
    _activate_bundle(
        workspace,
        tmp_path,
        _trusted_multi_section_bundle(("Rule zero.", "Rule one.", "Rule two.")),
    )
    claim = claim_draft_screening_batch(workspace, "screen-a")
    assert claim is not None
    shard_id = int(claim["shard_id"])
    submit_draft_screening_decision(workspace, shard_id, "screen-a", 0, "reject")
    submit_draft_screening_decision(workspace, shard_id, "screen-a", 1, "reject")
    release_draft_screening_batch(workspace, shard_id, "screen-a")
    assert claim_draft_screening_batch(
        workspace, "screen-b", preferred_shard_id=shard_id
    ) is not None

    pending = next_draft_screening_record(workspace, shard_id, "screen-b")
    assert pending is not None
    assert pending["index"] == 2
    assert pending["source_text"] == "Rule two."


def test_binary_screen_defer_uses_a_separate_escalation_queue(tmp_path: Path):
    workspace = _workspace(tmp_path)
    _activate_bundle(
        workspace,
        tmp_path,
        _trusted_multi_section_bundle(("Difficult general rule.", "Obvious furniture.")),
    )
    claim = claim_draft_screening_batch(workspace, "luna-a")
    assert claim is not None
    assert claim["queue"] == "unprocessed"
    shard_id = int(claim["shard_id"])

    deferred = step_draft_screening(
        workspace,
        shard_id,
        "luna-a",
        0,
        "defer",
        defer_reason="complex-rule",
    )
    assert deferred["result"]["decision"] == "defer"
    assert deferred["next_record"]["index"] == 1
    assert claim_draft_screening_batch(
        workspace, "terra-early", queue="deferred"
    ) is None
    with pytest.raises(PermissionError, match="escalation"):
        submit_draft_screening_decision(
            workspace, shard_id, "luna-a", 0, "add"
        )
    assert submit_draft_screening_decision(
        workspace, shard_id, "luna-a", 1, "reject"
    )["batch_complete"] is True

    status = draft_screening_status(workspace)["products"][0]
    assert status["unprocessed"] == 0
    assert status["accepted"] == 0
    assert status["rejected"] == 1
    assert status["deferred"] == 1
    assert status["unprocessed_batches"] == 0
    assert status["deferred_batches"] == 1
    assert claim_draft_screening_batch(workspace, "luna-b") is None

    escalation = claim_draft_screening_batch(
        workspace, "terra-a", queue="deferred"
    )
    assert escalation is not None
    assert escalation["queue"] == "deferred"
    assert escalation["eligible_count"] == 1
    record = next_draft_screening_record(workspace, shard_id, "terra-a")
    assert record is not None
    assert record["state"] == "deferred"
    assert record["defer_reason"] == "complex-rule"
    assert record["source_text"] == "Difficult general rule."

    resolved = step_draft_screening(workspace, shard_id, "terra-a", 0, "add")
    assert resolved["result"] == {
        "index": 0,
        "state": "resolved",
        "decision": "add",
        "duplicate_rejected": False,
        "batch_complete": True,
    }
    assert resolved["next_record"] is None
    final = draft_screening_status(workspace)["products"][0]
    assert final["accepted"] == 1
    assert final["rejected"] == 1
    assert final["deferred"] == 0
    assert final["live_claims"] == 0

    conn = sqlite3.connect(workspace)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """SELECT requested_decision, decision, defer_reason, deferred_by,
                      deferred_at, worker
               FROM draft_screening_current WHERE decision='ADD'"""
        ).fetchone()
        assert row is not None
        assert row["defer_reason"] == "complex-rule"
        assert row["deferred_by"] == "luna-a"
        assert row["deferred_at"] is not None
        assert row["worker"] == "terra-a"
    finally:
        conn.close()

    with pytest.raises(ValueError, match="conflicting"):
        submit_draft_screening_decision(
            workspace,
            shard_id,
            "luna-a",
            0,
            "defer",
            defer_reason="complex-rule",
        )


def test_binary_screen_defer_requires_a_bounded_reason(tmp_path: Path):
    workspace = _workspace(tmp_path)
    _activate_bundle(workspace, tmp_path, _trusted_bundle())
    claim = claim_draft_screening_batch(workspace, "luna-a")
    assert claim is not None
    shard_id = int(claim["shard_id"])
    with pytest.raises(ValueError, match="bounded reason"):
        submit_draft_screening_decision(
            workspace, shard_id, "luna-a", 0, "defer"
        )
    with pytest.raises(ValueError, match="bounded reason"):
        submit_draft_screening_decision(
            workspace,
            shard_id,
            "luna-a",
            0,
            "defer",
            defer_reason="too-hard",
        )


def test_screening_reopen_is_append_only_and_requeues_latest_state(tmp_path: Path):
    workspace = _workspace(tmp_path)
    _activate_bundle(workspace, tmp_path, _trusted_bundle())
    claim = claim_draft_screening_batch(workspace, "screen-a")
    assert claim is not None
    submit_draft_screening_decision(
        workspace, int(claim["shard_id"]), "screen-a", 0, "reject"
    )
    with sqlite3.connect(workspace) as conn:
        section_key = str(conn.execute(
            "SELECT section_key FROM draft_screening_current"
        ).fetchone()[0])

    reopened = reopen_draft_screening(
        workspace, section_key, maintainer="maintainer-a", reason="scope-correction"
    )
    assert reopened["state"] == "reopened"
    status = draft_screening_status(workspace)["products"][0]
    assert status["unprocessed"] == 1
    assert status["rejected"] == 0
    with sqlite3.connect(workspace) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM draft_screening_events WHERE section_key=?",
            (section_key,),
        ).fetchone()[0] == 2
        assert conn.execute(
            "SELECT COUNT(*) FROM draft_screening_current WHERE section_key=?",
            (section_key,),
        ).fetchone()[0] == 0


def test_binary_screen_claims_split_work_and_release_keeps_decisions(tmp_path: Path):
    workspace = _workspace(tmp_path)
    _activate_bundle(
        workspace,
        tmp_path,
        _trusted_multi_section_bundle(("Rule one.", "Rule two.")),
        shard_size=1,
    )
    first = claim_draft_screening_batch(workspace, "screen-a")
    second = claim_draft_screening_batch(workspace, "screen-b")
    assert first is not None and second is not None
    assert first["shard_id"] != second["shard_id"]
    submit_draft_screening_decision(
        workspace, int(first["shard_id"]), "screen-a", 0, "reject"
    )
    assert release_draft_screening_batch(
        workspace, int(second["shard_id"]), "screen-b"
    )["released"] is True
    reclaimed = claim_draft_screening_batch(workspace, "screen-c")
    assert reclaimed is not None
    assert reclaimed["shard_id"] == second["shard_id"]
    status = draft_screening_status(workspace)["products"][0]
    assert status["rejected"] == 1
    assert status["unprocessed"] == 1


def test_binary_screen_does_not_authorize_public_projection(tmp_path: Path):
    workspace = _workspace(tmp_path)
    _activate_bundle(workspace, tmp_path, _trusted_bundle())
    claim = claim_draft_screening_batch(workspace, "screen-a")
    assert claim is not None
    submit_draft_screening_decision(
        workspace, int(claim["shard_id"]), "screen-a", 0, "add"
    )

    output = tmp_path / "screen-only.sqlite3"
    with pytest.raises(ValueError, match="unreviewed source section"):
        build_public_corpus(workspace, output, _notices(tmp_path))
    assert not output.exists()


def test_binary_screen_decisions_are_scoped_to_parser_run(tmp_path: Path):
    workspace = _workspace(tmp_path)
    first = _activate_bundle(workspace, tmp_path, _trusted_bundle(text="First parse."))
    claim = claim_draft_screening_batch(workspace, "screen-a")
    assert claim is not None
    submit_draft_screening_decision(
        workspace, int(claim["shard_id"]), "screen-a", 0, "add"
    )

    second = _activate_bundle(workspace, tmp_path, _trusted_bundle(text="Reparsed text."))
    assert second["parser_run_id"] != first["parser_run_id"]
    status = draft_screening_status(workspace)["products"][0]
    assert status["sections"] == 1
    assert status["accepted"] == 0
    assert status["unprocessed"] == 1
    conn = sqlite3.connect(workspace)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM draft_screening_events WHERE parser_run_id=? AND decision<>'REOPEN'",
            (first["parser_run_id"],),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM draft_screening_events WHERE parser_run_id=? AND decision<>'REOPEN'",
            (second["parser_run_id"],),
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_direct_pdf_stage_and_activation_reject_post_stage_tampering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import pf2e_codex.licensed_corpus as review_workspace

    workspace = _workspace(tmp_path)
    monkeypatch.setattr(review_workspace, "load_and_parse_verified_pdf", lambda *args, **kwargs: _trusted_bundle())
    staged = stage_trusted_native_pdf(
        workspace, tmp_path / "owned.pdf", product_code="PZO12001", parser_version="paizo-native-v1",
    )
    conn = sqlite3.connect(workspace)
    try:
        conn.execute(
            "UPDATE source_sections SET source_text='tampered' WHERE parser_run_id=?",
            (staged["parser_run_id"],),
        )
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(ValueError, match="source text no longer matches"):
        activate_parser_run(workspace, str(staged["parser_run_id"]))
    conn = sqlite3.connect(workspace)
    try:
        assert conn.execute("SELECT state FROM parser_runs WHERE parser_run_id=?", (staged["parser_run_id"],)).fetchone()[0] == "staged"
        assert conn.execute("SELECT COUNT(*) FROM parser_runs WHERE state='active'").fetchone()[0] == 1
    finally:
        conn.close()


@pytest.mark.parametrize(
    "column",
    ["parser_output_digest", "manifest_digest", "native_word_coverage_digest"],
)
def test_activation_recomputes_staged_run_metadata(tmp_path: Path, column: str):
    workspace = _workspace(tmp_path)
    staged = _stage_bundle(workspace, tmp_path, _trusted_bundle())
    conn = sqlite3.connect(workspace)
    try:
        conn.execute(
            f"UPDATE parser_runs SET {column}=? WHERE parser_run_id=?",
            ("f" * 64, staged["parser_run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(ValueError, match="metadata no longer matches"):
        activate_parser_run(workspace, str(staged["parser_run_id"]))


def test_v8_migration_repairs_only_staged_v7_anchor_order_digest(tmp_path: Path):
    """A v7 staged direct-PDF run can recover without altering its trust state."""
    first = hashlib.sha256(b"first section anchor").hexdigest()
    second = hashlib.sha256(b"second section anchor").hexdigest()
    anchors = (second, first)
    workspace = _workspace(tmp_path)
    staged = _stage_bundle(workspace, tmp_path, _trusted_bundle(section_anchors=anchors))
    conn = sqlite3.connect(workspace)
    conn.row_factory = sqlite3.Row
    try:
        run = conn.execute(
            "SELECT * FROM parser_runs WHERE parser_run_id=?", (staged["parser_run_id"],)
        ).fetchone()
        assert run is not None
        canonical_digest = str(run["native_word_coverage_digest"])
        legacy_record = {
            "stable_identity": hashlib.sha256(b"stable-section").hexdigest(),
            "native_word_count": len(anchors),
            "native_word_digest": _anchor_digest("section-native-word-anchors-v1", anchors),
            "native_word_anchors": list(anchors),
        }
        legacy_digest = hashlib.sha256(
            (
                "native-word-coverage-v1\n"
                + json.dumps([legacy_record], sort_keys=True, separators=(",", ":"))
            ).encode("utf-8")
        ).hexdigest()
        assert legacy_digest != canonical_digest
        conn.execute(
            "UPDATE parser_runs SET native_word_coverage_digest=? WHERE parser_run_id=?",
            (legacy_digest, staged["parser_run_id"]),
        )
        conn.execute("UPDATE metadata SET value='7' WHERE key='review_schema_version'")
        conn.commit()
        before = dict(conn.execute(
            "SELECT * FROM parser_runs WHERE parser_run_id=?", (staged["parser_run_id"],)
        ).fetchone())
    finally:
        conn.close()

    # BEGIN IMMEDIATE serializes concurrent migration attempts; both callers
    # observe the same repaired staged run.
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []
    lock = threading.Lock()

    def migrate() -> None:
        try:
            barrier.wait()
            workspace_status(workspace)
        except BaseException as exc:  # pragma: no cover - asserted below
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=migrate) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []

    conn = sqlite3.connect(workspace)
    conn.row_factory = sqlite3.Row
    try:
        after = dict(conn.execute(
            "SELECT * FROM parser_runs WHERE parser_run_id=?", (staged["parser_run_id"],)
        ).fetchone())
        assert after["native_word_coverage_digest"] == canonical_digest
        assert {
            key: value for key, value in after.items() if key != "native_word_coverage_digest"
        } == {
            key: value for key, value in before.items() if key != "native_word_coverage_digest"
        }
        assert conn.execute(
            "SELECT value FROM metadata WHERE key='review_schema_version'"
        ).fetchone()[0] == str(REVIEW_SCHEMA_VERSION)
    finally:
        conn.close()

    assert activate_parser_run(workspace, str(staged["parser_run_id"]))["state"] == "active"


@pytest.mark.parametrize(
    ("statement", "error"),
    [
        ("UPDATE parser_run_sections SET section_commitment='0' WHERE parser_run_id=?", "section commitment"),
        ("UPDATE source_assets SET native_word_anchor_digest='0' WHERE asset_id=?", "source inventory"),
    ],
)
def test_v8_migration_rejects_malformed_staged_run_and_rolls_back(
    tmp_path: Path, statement: str, error: str
):
    workspace = _workspace(tmp_path)
    staged = _stage_bundle(workspace, tmp_path, _trusted_bundle())
    conn = sqlite3.connect(workspace)
    conn.row_factory = sqlite3.Row
    try:
        run = conn.execute(
            "SELECT * FROM parser_runs WHERE parser_run_id=?", (staged["parser_run_id"],)
        ).fetchone()
        assert run is not None
        parameter = (
            staged["parser_run_id"]
            if "parser_run_sections" in statement
            else run["asset_id"]
        )
        conn.execute(statement, (parameter,))
        conn.execute(
            "UPDATE parser_runs SET native_word_coverage_digest=? WHERE parser_run_id=?",
            ("f" * 64, staged["parser_run_id"]),
        )
        conn.execute("UPDATE metadata SET value='7' WHERE key='review_schema_version'")
        conn.commit()
        before = dict(conn.execute(
            "SELECT * FROM parser_runs WHERE parser_run_id=?", (staged["parser_run_id"],)
        ).fetchone())
    finally:
        conn.close()

    with pytest.raises(ValueError, match=error):
        workspace_status(workspace)

    conn = sqlite3.connect(workspace)
    conn.row_factory = sqlite3.Row
    try:
        assert dict(conn.execute(
            "SELECT * FROM parser_runs WHERE parser_run_id=?", (staged["parser_run_id"],)
        ).fetchone()) == before
        assert conn.execute(
            "SELECT value FROM metadata WHERE key='review_schema_version'"
        ).fetchone()[0] == "7"
    finally:
        conn.close()


def test_direct_pdf_staging_rejects_a_mutated_bundle_seal(tmp_path: Path):
    workspace = _workspace(tmp_path)
    bundle = replace(_trusted_bundle(), sealed_digest="0" * 64)
    with pytest.raises(ValueError, match="bundle seal"):
        _stage_bundle(workspace, tmp_path, bundle)


def test_complex_layout_and_table_cells_are_visible_to_reviewer_and_gate_public_candidates(tmp_path: Path):
    workspace = _workspace(tmp_path)
    staged = _stage_bundle(
        workspace,
        tmp_path,
        _trusted_bundle(
            flags=(
                "complex-layout", "native-layout-fallback",
                "table-ambiguous", "table-cell",
            )
        ),
    )
    activate_parser_run(workspace, str(staged["parser_run_id"]))
    _, section = _claim_and_read(workspace)
    common = {
        "section_key": section["section_key"], "source_section_id": section["source_section_id"],
        "source_section_hash": section["source_section_hash"], "public_heading": "Reviewed rules",
        "extraction_method": "human-reconstruction-v1", "worker": "worker-a", "prompt_version": "test-v1",
    }
    with pytest.raises(ValueError, match="forbidden"):
        submit_candidate(workspace, {**common, "decision": "PUBLIC_AS_IS", "candidate_text": section["source_text"], "reason_tags": ["rules"]})
    with pytest.raises(ValueError, match="layout-reviewed"):
        submit_candidate(workspace, {**common, "decision": "MIXED_NEEDS_EXTRACTION", "candidate_text": PUBLIC_TEXT, "reason_tags": ["rules"]})
    candidate_id = submit_candidate(workspace, {**common, "decision": "MIXED_NEEDS_EXTRACTION", "candidate_text": PUBLIC_TEXT, "reason_tags": ["rules", "layout-reviewed"]})["candidate_id"]
    claim = claim_review(workspace, "reviewer-b")
    assert claim is not None and claim["candidate_id"] == candidate_id
    assert json.loads(str(read_claimed_review(workspace, str(candidate_id), "reviewer-b")["layout_flags"])) == [
        "complex-layout", "native-layout-fallback", "table-ambiguous", "table-cell",
    ]


def test_direct_pdf_stage_rolls_back_asset_and_run_on_database_failure(tmp_path: Path):
    workspace = _workspace(tmp_path)
    conn = sqlite3.connect(workspace)
    try:
        before = conn.execute("SELECT COUNT(*) FROM source_assets").fetchone()[0]
        conn.execute(
            """CREATE TRIGGER fail_trusted_run BEFORE INSERT ON parser_runs
            WHEN NEW.origin='trusted-direct-pdf-v1' BEGIN SELECT RAISE(ABORT, 'injected stage failure'); END"""
        )
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(sqlite3.IntegrityError, match="injected stage failure"):
        _stage_bundle(workspace, tmp_path, _trusted_bundle())
    conn = sqlite3.connect(workspace)
    try:
        assert conn.execute("SELECT COUNT(*) FROM source_assets").fetchone()[0] == before
        assert conn.execute("SELECT COUNT(*) FROM parser_runs WHERE origin='trusted-direct-pdf-v1'").fetchone()[0] == 0
    finally:
        conn.close()


def test_legacy_workspace_cannot_build_public_projection(tmp_path: Path):
    workspace = _workspace(tmp_path)
    with pytest.raises(ValueError, match="complete direct-PDF"):
        build_public_corpus(workspace, tmp_path / "public.sqlite3", _notices(tmp_path))


def test_candidate_commitment_tampering_is_not_reviewable_or_exportable(tmp_path: Path):
    workspace = _trusted_workspace(tmp_path)
    _, section = _claim_and_read(workspace)
    candidate_id = _submit_public(workspace, section)
    conn = sqlite3.connect(workspace)
    try:
        conn.execute("UPDATE candidates SET extraction_method='mutated-method' WHERE candidate_id=?", (candidate_id,))
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(ValueError, match="candidate commitment"):
        claim_review(workspace, "reviewer-b")
    with pytest.raises(ValueError, match="candidate commitment"):
        build_public_corpus(workspace, tmp_path / "public.sqlite3", _notices(tmp_path))


@pytest.mark.parametrize(
    ("column", "value", "error"),
    [
        ("verdict", "REJECT", "commitment"),
        ("reviewer", "worker-a", "review ID|independent"),
        ("policy_version", "mechanics-v0", "current supported policy|commitment"),
        ("reason_tags", '["mutated"]', "commitment"),
    ],
)
def test_review_commitment_tampering_is_not_reusable_or_exportable(
    tmp_path: Path, column: str, value: str, error: str
):
    workspace = _trusted_workspace(tmp_path)
    _, section = _claim_and_read(workspace)
    candidate_id = _submit_public(workspace, section)
    _approve(workspace, candidate_id)
    conn = sqlite3.connect(workspace)
    try:
        conn.execute(f"UPDATE reviews SET {column}=? WHERE candidate_id=?", (value, candidate_id))
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(ValueError, match=error):
        build_public_corpus(workspace, tmp_path / "public.sqlite3", _notices(tmp_path))


def test_uncommitted_legacy_review_is_nonpublishable(tmp_path: Path):
    workspace = _trusted_workspace(tmp_path)
    _, section = _claim_and_read(workspace)
    candidate_id = _submit_public(workspace, section)
    _approve(workspace, candidate_id)
    conn = sqlite3.connect(workspace)
    try:
        conn.execute("ALTER TABLE reviews DROP COLUMN review_commitment")
        conn.execute("ALTER TABLE reviews DROP COLUMN review_lineage")
        conn.execute("UPDATE metadata SET value='5' WHERE key='review_schema_version'")
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(ValueError, match="unreviewed candidate"):
        build_public_corpus(workspace, tmp_path / "public.sqlite3", _notices(tmp_path))

    # Historical rows are preserved for audit, but a fresh independent review
    # can replace their decision without an out-of-band invalidation.
    claim = claim_review(workspace, "reviewer-c")
    assert claim is not None and claim["candidate_id"] == candidate_id
    submit_review(
        workspace,
        {
            "candidate_id": candidate_id,
            "reviewer": "reviewer-c",
            "verdict": "APPROVE",
            "policy_version": "mechanics-v1",
            "reason_tags": ["fresh-review"],
        },
    )
    assert build_public_corpus(workspace, tmp_path / "recovered.sqlite3", _notices(tmp_path))["sections"] == 1


def test_invalidating_predecessor_review_blocks_descendant_clone_until_rereview(tmp_path: Path):
    workspace = _trusted_workspace(tmp_path)
    _, section = _claim_and_read(workspace)
    original_candidate = _submit_public(workspace, section)
    _approve(workspace, original_candidate, "reviewer-b")
    conn = sqlite3.connect(workspace)
    try:
        original_review = conn.execute(
            "SELECT review_id FROM reviews WHERE candidate_id=?", (original_candidate,)
        ).fetchone()[0]
    finally:
        conn.close()

    staged = _stage_bundle(
        workspace, tmp_path, _trusted_bundle(source_id="pzo12001:player-core:p1:h0123456789abcdef:i2")
    )
    assert staged["reused"] == 1
    activate_parser_run(workspace, str(staged["parser_run_id"]))
    invalidate_reviews(
        workspace,
        "reviewer-b",
        [original_review],
        invalidated_by="audit",
        reason="original review withdrawn",
    )

    with pytest.raises(ValueError, match="unreviewed candidate"):
        build_public_corpus(workspace, tmp_path / "blocked.sqlite3", _notices(tmp_path))

    claim = claim_review(workspace, "reviewer-c")
    assert claim is not None
    submit_review(
        workspace,
        {
            "candidate_id": claim["candidate_id"],
            "reviewer": "reviewer-c",
            "verdict": "APPROVE",
            "policy_version": "mechanics-v1",
            "reason_tags": ["fresh-review"],
        },
    )
    assert build_public_corpus(workspace, tmp_path / "rereviewed.sqlite3", _notices(tmp_path))["sections"] == 1


def test_trusted_staging_requires_source_id_start_page_match(tmp_path: Path):
    workspace = _workspace(tmp_path)
    with pytest.raises(ValueError, match="page_start"):
        _stage_bundle(
            workspace,
            tmp_path,
            _trusted_bundle(source_id="pzo12001:player-core:p2:h0123456789abcdef:i0"),
        )


@pytest.mark.parametrize("page_start", ["C:\\\\Users\\\\buyer", 0])
def test_public_build_rejects_tampered_page_provenance(tmp_path: Path, page_start: object):
    workspace = _trusted_workspace(tmp_path)
    _, section = _claim_and_read(workspace)
    candidate_id = _submit_public(workspace, section)
    _approve(workspace, candidate_id)
    conn = sqlite3.connect(workspace)
    try:
        conn.execute(
            "UPDATE source_sections SET page_start=? WHERE section_key=?",
            (page_start, section["section_key"]),
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(ValueError, match="pages must|page_start"):
        build_public_corpus(workspace, tmp_path / "public.sqlite3", _notices(tmp_path))


@pytest.mark.parametrize("notices", [
    {"OGL": {"license": "OGL", "text": "Contact buyer@example.invalid"}},
    {"OGL": {"license": "OGL", "text": "C:\\Users\\buyer"}},
    {"OGL": {"license": "INVALID", "text": "Notice"}},
])
def test_notices_reject_private_or_unknown_public_scalars(tmp_path: Path, notices: dict[str, object]):
    workspace = _trusted_workspace(tmp_path)
    path = tmp_path / "notices-invalid.json"
    path.write_text(json.dumps(notices), encoding="utf-8")
    with pytest.raises(ValueError, match="private|unsupported"):
        build_public_corpus(workspace, tmp_path / "public.sqlite3", path)
