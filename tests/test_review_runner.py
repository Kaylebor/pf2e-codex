"""Deterministic runner scheduling, schema, and session tests."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pf2e_codex.corpus import (
    PRODUCT_CATALOG,
    TrustedBlock,
    TrustedParseBundle,
    TrustedSection,
    _trusted_bundle_seal,
    _trusted_parser_output_digest,
    apply_layout_evidence,
)
from pf2e_codex.licensed_core import load_licensed_core
from pf2e_codex.licensed_corpus import (
    REVIEW_SCHEMA_VERSION,
    activate_parser_run,
    claim_draft_screening_batch,
    initialize_trusted_workspace,
    prepare_deterministic_review,
    set_review_product_scope,
    stage_trusted_native_pdf,
    stage_trusted_native_pdf_with_approved_stitches,
    submit_draft_screening_decision,
)
from pf2e_codex.pdf_export import NativeWordInventory
from pf2e_codex.pdf_layout import BoundLayoutRegion, BoundNativeLayout
from pf2e_codex.review_runner import (
    EXPECTED_PRODUCTS,
    MAX_BATCH_BYTES,
    MAX_BATCH_RECORDS,
    CodexExecutor,
    CodexResult,
    _CodexProcessError,
    _prior_stitch_candidate,
    _reconcile_stitch_maintenance,
    _reuse_stitch_judgment,
    _stitch_records,
    build_base,
    generate_stitch_candidates,
    maintainer_item_evidence,
    pack_batches,
    prepare_review_data,
    prepare_workspace,
    preview_screen_batches,
    refresh_aon_cache,
    reopen_screening,
    resolve_maintainer_item,
    run_codex_batch,
    run_queues,
    runner_status,
    validate_exact_results,
    verify_workspace,
)


class FakeCodex:
    version = "codex-cli test"

    def __init__(self, payloads: list[dict[str, object]] | None = None):
        self.payloads = list(payloads or [])
        self.thread_inputs: list[str | None] = []

    def execute(self, *, thread_id: str | None, **kwargs: object) -> CodexResult:
        del kwargs
        self.thread_inputs.append(thread_id)
        payload = self.payloads.pop(0) if self.payloads else {
            "results": [
                {"id": "section-1", "decision": "add", "reason": None, "foundry_ids": []}
            ]
        }
        return CodexResult(payload, "thread-test", {"input_tokens": 10}, "f" * 64)


def _workspace(tmp_path: Path) -> Path:
    path = tmp_path / "review.sqlite3"
    initialize_trusted_workspace(path)
    return path


def _foundry_database(tmp_path: Path) -> Path:
    path = tmp_path / "foundry.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, text TEXT NOT NULL,
            source_hash TEXT NOT NULL, publication_title TEXT NOT NULL,
            license TEXT NOT NULL, remaster INTEGER, type TEXT NOT NULL,
            origin TEXT NOT NULL
        );
        """
    )
    conn.executemany(
        "INSERT INTO _meta VALUES (?, ?)",
        [
            ("distribution_scope", "redistributable"),
            ("foundry_scope", "core-publications-v1"),
            ("pf2e_release", "pf2e-test"),
        ],
    )
    for product_code in EXPECTED_PRODUCTS:
        product = PRODUCT_CATALOG[product_code]
        text = f"A distinct public Foundry rule for {product_code}."
        conn.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'foundry')",
            (
                f"rules:{product_code.casefold()}", f"Foundry {product_code}", text,
                hashlib.sha256(text.encode()).hexdigest(), product.title, product.license,
                1 if product.remaster else 0, "rule",
            ),
        )
    conn.commit()
    conn.close()
    return path


def test_empty_trusted_workspace_has_current_runner_schema(tmp_path: Path):
    workspace = _workspace(tmp_path)
    conn = sqlite3.connect(workspace)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    version = conn.execute(
        "SELECT value FROM metadata WHERE key='review_schema_version'"
    ).fetchone()[0]
    conn.close()

    assert version == str(REVIEW_SCHEMA_VERSION)
    assert {
        "runner_sessions",
        "runner_attempts",
        "runner_maintenance",
        "aon_cache",
        "stitch_candidates",
        "stitch_claims",
    } <= tables


def test_verify_cli_exits_nonzero_when_validation_fails(tmp_path: Path):
    workspace = _workspace(tmp_path)
    script = Path(__file__).parents[1] / "scripts" / "licensed-corpus-runner.py"

    completed = subprocess.run(
        [sys.executable, str(script), "verify", str(workspace)],
        capture_output=True, text=True, check=False,
    )

    assert completed.returncode == 1
    assert json.loads(completed.stdout)["ok"] is False


def test_rules_era_is_catalog_metadata_not_license_inference():
    assert PRODUCT_CATALOG["PZO2101"].rules_era == "legacy"
    assert PRODUCT_CATALOG["PZO2101"].license == "OGL"
    for code in ("PZO12001", "PZO12002", "PZO12003", "PZO12004"):
        assert PRODUCT_CATALOG[code].rules_era == "remaster"
        assert PRODUCT_CATALOG[code].license == "ORC"


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"results": [{"id": "a"}]}, "exactly match"),
        ({"results": [{"id": "a"}, {"id": "a"}]}, "duplicate"),
        ({"results": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}, "exactly match"),
        ({"results": [{"id": "a"}], "extra": True}, "only a results"),
    ],
)
def test_exact_id_validation_rejects_omissions_duplicates_and_extras(payload: object, message: str):
    with pytest.raises(ValueError, match=message):
        validate_exact_results(payload, ["a", "b"])


def test_batch_packing_is_deterministic_and_bounded():
    records = [{"id": f"id-{index}", "text": "x" * 500} for index in range(70)]
    first = pack_batches(records)
    second = pack_batches(records)

    assert first == second
    assert [len(batch) for batch in first] == [MAX_BATCH_RECORDS, MAX_BATCH_RECORDS, 6]
    assert all(len(json.dumps(batch).encode()) <= MAX_BATCH_BYTES for batch in first)


def test_schema_failure_retries_without_accepted_partial_state(tmp_path: Path):
    workspace = _workspace(tmp_path)
    fake = FakeCodex(
        [
            {"results": []},
            {
                "results": [
                    {"id": "section-1", "decision": "add", "reason": None, "foundry_ids": []}
                ]
            },
        ]
    )
    results = run_codex_batch(
        workspace,
        queue="screen",
        slot=0,
        records=[{"id": "section-1", "heading": "Rules", "text": "Private"}],
        foundry_db=None,
        executor=fake,
    )

    assert results[0]["decision"] == "add"
    conn = sqlite3.connect(workspace)
    states = [row[0] for row in conn.execute("SELECT status FROM runner_attempts ORDER BY attempt")]
    private_matches = conn.execute(
        "SELECT COUNT(*) FROM runner_attempts WHERE usage_json LIKE '%Private%'"
    ).fetchone()[0]
    conn.close()
    assert states == ["schema-failure", "accepted"]
    assert private_matches == 0


def test_session_rotates_after_four_completed_batches(tmp_path: Path):
    workspace = _workspace(tmp_path)
    fake = FakeCodex()
    for index in range(5):
        section_id = f"section-{index}"
        fake.payloads.append(
            {
                "results": [
                    {"id": section_id, "decision": "add", "reason": None, "foundry_ids": []}
                ]
            }
        )
        run_codex_batch(
            workspace,
            queue="screen",
            slot=0,
            records=[{"id": section_id, "heading": "Rules", "text": "Private"}],
            foundry_db=None,
            executor=fake,
        )

    assert fake.thread_inputs == [None, "thread-test", "thread-test", "thread-test", None]


def test_codex_process_is_read_only_config_ignored_and_schema_constrained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        del kwargs
        commands.append(command)
        output = Path(command[command.index("-o") + 1])
        output.write_text('{"results":[]}', encoding="utf-8")
        return SimpleNamespace(
            returncode=0,
            stdout='{"type":"thread.started","thread_id":"thread-new"}\n',
            stderr="",
        )

    monkeypatch.setattr("pf2e_codex.review_runner.subprocess.run", fake_run)
    executor = CodexExecutor()
    schema = {"type": "object"}
    executor.execute(
        model="gpt-5.6-luna", prompt="bounded", schema=schema,
        workdir=tmp_path, thread_id=None,
    )
    executor.execute(
        model="gpt-5.6-luna", prompt="bounded", schema=schema,
        workdir=tmp_path, thread_id="thread-new",
    )

    for command in commands:
        assert command[:3] == ["codex", "--sandbox", "read-only"]
        assert "--ignore-user-config" in command
        assert "--output-schema" in command
        assert "--json" in command
        assert "-o" in command
    assert commands[1][3:5] == ["exec", "resume"]


def test_codex_usage_limit_is_sanitized_and_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    workspace = _workspace(tmp_path)

    class QuotaExecutor:
        version = "codex-cli test"
        calls = 0

        def execute(self, **_kwargs: object) -> CodexResult:
            self.calls += 1
            raise _CodexProcessError("model-usage-limit", retryable=False)

    executor = QuotaExecutor()
    with pytest.raises(RuntimeError, match="blocked by model-usage-limit"):
        run_codex_batch(
            workspace,
            queue="screen",
            slot=0,
            records=[{"id": "section-1", "heading": "Rules", "text": "Private"}],
            foundry_db=None,
            executor=executor,
        )

    assert executor.calls == 1
    conn = sqlite3.connect(workspace)
    attempt = conn.execute(
        "SELECT status, error_kind FROM runner_attempts"
    ).fetchone()
    conn.close()
    assert attempt == ("transport-failure", "model-usage-limit")


def test_codex_executor_classifies_usage_limit_without_exposing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "pf2e_codex.review_runner.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="You've hit your usage limit for gpt-5.3-codex-spark",
            stderr="private diagnostic must not escape",
        ),
    )
    executor = CodexExecutor()

    with pytest.raises(_CodexProcessError) as failure:
        executor.execute(
            model="gpt-5.3-codex-spark", prompt="bounded",
            schema={"type": "object"}, workdir=tmp_path, thread_id=None,
        )

    assert str(failure.value) == "model-usage-limit"
    assert failure.value.retryable is False


class FakeResponse:
    def __init__(self, body: bytes, url: str = "https://2e.aonprd.com/Search.aspx?q=Fireball"):
        self.body = body
        self.url = url

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def geturl(self) -> str:
        return self.url

    def read(self, _limit: int) -> bytes:
        return self.body


def _aon_workspace(tmp_path: Path) -> Path:
    path = tmp_path / "aon.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE parser_runs (
            parser_run_id TEXT PRIMARY KEY, product_code TEXT,
            state TEXT, review_enabled INTEGER
        );
        CREATE TABLE review_product_scope (
            product_code TEXT PRIMARY KEY, enabled INTEGER, reason TEXT, updated_at INTEGER
        );
        CREATE TABLE source_sections (
            section_key TEXT PRIMARY KEY, parser_run_id TEXT, heading TEXT
        );
        CREATE TABLE draft_screening_events (
            event_id INTEGER PRIMARY KEY,
            parser_run_id TEXT, section_key TEXT, event_type TEXT,
            requested_decision TEXT, decision TEXT, duplicate_of_section_key TEXT,
            reject_reason TEXT, defer_reason TEXT, deferred_by TEXT, deferred_at INTEGER,
            worker TEXT, decided_at INTEGER, reopen_reason TEXT, supersedes_event_id INTEGER
        );
        CREATE VIEW draft_screening_current AS
            SELECT parser_run_id, section_key, requested_decision, decision,
                   duplicate_of_section_key, reject_reason, defer_reason,
                   deferred_by, deferred_at, worker, decided_at
            FROM draft_screening_events WHERE event_type='DECISION';
        CREATE TABLE aon_cache (
            query_digest TEXT PRIMARY KEY, normalized_query TEXT, status TEXT,
            results_json TEXT, checked_at INTEGER
        );
        INSERT INTO parser_runs VALUES ('run','PZO12001','active',1);
        INSERT INTO review_product_scope VALUES ('PZO12001',1,'enabled',1);
        INSERT INTO source_sections VALUES ('section','run','Fireball');
        INSERT INTO draft_screening_events VALUES
            (1,'run','section','DECISION','DEFER','DEFER',NULL,NULL,'scope',
             'screen',1,'screen',1,NULL,NULL);
        """
    )
    conn.close()
    return path


def test_aon_queue_caches_only_title_url_and_skips_cache_hits(tmp_path: Path):
    workspace = _aon_workspace(tmp_path)
    calls = 0

    def opener(*args: object, **kwargs: object) -> FakeResponse:
        nonlocal calls
        del args, kwargs
        calls += 1
        return FakeResponse(b'<a href="Spells.aspx?ID=1530">Fireball</a><p>body must not persist</p>')

    first = refresh_aon_cache(workspace, opener=opener, interval_seconds=0)
    second = refresh_aon_cache(workspace, opener=opener, interval_seconds=0)

    assert first["match"] == 1
    assert second["match"] == 0
    assert calls == 1
    conn = sqlite3.connect(workspace)
    stored = conn.execute("SELECT results_json FROM aon_cache").fetchone()[0]
    conn.close()
    assert json.loads(stored) == [
        {"title": "Fireball", "url": "https://2e.aonprd.com/Spells.aspx?ID=1530"}
    ]
    assert "body must not persist" not in stored


@pytest.mark.parametrize(
    "response, status",
    [
        (FakeResponse(b"<html>No results</html>"), "no-match"),
        (FakeResponse(b"<html></html>", "https://example.invalid/redirect"), "inconclusive"),
    ],
)
def test_aon_absence_and_invalid_redirect_are_nonterminal(
    tmp_path: Path, response: FakeResponse, status: str
):
    workspace = _aon_workspace(tmp_path)
    result = refresh_aon_cache(
        workspace, opener=lambda *_args, **_kwargs: response, interval_seconds=0
    )
    assert result[status] == 1


def _trusted_product_bundle(
    product_code: str,
    *,
    parser_version: str = "paizo-native-v3",
    mixed: bool = False,
    section_count: int = 1,
    complete: bool = True,
    layout_flags: tuple[str, ...] = (),
    same_heading: bool = False,
    changed_last: bool = False,
) -> TrustedParseBundle:
    product = PRODUCT_CATALOG[product_code]
    text = (
        "A creature gains a +1 circumstance bonus while the condition applies."
        if not mixed
        else "Example hero prose. A creature gains a +2 circumstance bonus to Armor Class."
    )
    if not complete:
        text = text.removesuffix(".")
    anchors = tuple(
        hashlib.sha256(f"anchor-{product_code}-{index}".encode()).hexdigest()
        if section_count > 1 else hashlib.sha256(f"anchor-{product_code}".encode()).hexdigest()
        for index in range(section_count)
    )
    ignored = hashlib.sha256(f"ignored-{product_code}".encode()).hexdigest()
    fingerprint = hashlib.sha256(f"source-{product_code}".encode()).hexdigest()
    section_texts = tuple(
        text + (" Updated mechanics." if changed_last and index == section_count - 1 else "")
        for index in range(section_count)
    )
    sections = tuple(
        TrustedSection(
            id=f"private:{product_code}:{index}",
            source_section_id=(
                f"{product_code.casefold()}:{product.component}:p{index + 1}:"
                f"h{index:016x}:i{index}"
            ),
            heading=(
                "Shared Rule"
                if same_heading
                else "Mixed Rule" if mixed else f"Rule {product_code} {index}"
            ),
            text=section_texts[index],
            text_hash=hashlib.sha256(section_texts[index].encode()).hexdigest(),
            physical_pages=(index + 1,),
            printed_page=str(index + 1),
            stable_section_identity=hashlib.sha256(
                f"stable-{product_code}-{index}".encode()
            ).hexdigest(),
            layout_flags=layout_flags,
            coverage_anchors=(anchors[index],),
            blocks=(
                TrustedBlock(
                    kind="body",
                    physical_page=index + 1,
                    ordinal=0,
                    text=section_texts[index],
                    text_hash=hashlib.sha256(section_texts[index].encode()).hexdigest(),
                    coverage_anchors=(anchors[index],),
                ),
            ) if parser_version in {"paizo-native-v4", "paizo-native-v5"} else (),
        )
        for index in range(section_count)
    )
    inventory = NativeWordInventory(
        fingerprint,
        (*anchors, ignored),
        ({"anchor_hash": ignored, "reason": "watermark-email-span-v1"},),
        {},
        {ignored: "watermark-email-span-v1"},
    )
    attestation = {
        "product_verified": True,
        "page_count": 1,
        "title_marker_verified": True,
        "matched_product_count": 1,
        "conflict_product_count": 0,
    }
    attestation_digest = hashlib.sha256(f"attestation-{product_code}".encode()).hexdigest()
    parser_digest = _trusted_parser_output_digest(sections)
    layout_binding_digest = (
        hashlib.sha256(f"layout-{product_code}".encode()).hexdigest()
        if parser_version in {"paizo-native-v4", "paizo-native-v5"}
        else None
    )
    seal = _trusted_bundle_seal(
        product_code=product_code,
        parser_version=parser_version,
        exporter_profile_version=1,
        semantic_fingerprint=fingerprint,
        artifact_attestation=attestation,
        artifact_attestation_digest=attestation_digest,
        inventory=inventory,
        sections=sections,
        parser_output_digest=parser_digest,
        layout_binding_digest=layout_binding_digest,
    )
    return TrustedParseBundle(
        product_code,
        parser_version,
        1,
        fingerprint,
        attestation,
        inventory,
        sections,
        parser_digest,
        seal,
        artifact_attestation_digest=attestation_digest,
        layout_binding_digest=layout_binding_digest,
    )


def _stage_expected_products(
    workspace: Path,
    tmp_path: Path,
    *,
    mixed_product: str | None = None,
) -> None:
    for product_code in EXPECTED_PRODUCTS:
        bundle = _trusted_product_bundle(
            product_code, mixed=product_code == mixed_product
        )
        with patch(
            "pf2e_codex.licensed_corpus.load_and_parse_verified_pdf",
            return_value=bundle,
        ):
            staged = stage_trusted_native_pdf(
                workspace,
                tmp_path / f"{product_code}E.pdf",
                product_code=product_code,
                parser_version="paizo-native-v3",
                shard_size=2,
            )
        activate_parser_run(workspace, str(staged["parser_run_id"]))


def _layout_binding_for(bundle: TrustedParseBundle) -> BoundNativeLayout:
    anchors = tuple(
        anchor for section in bundle.sections for anchor in section.coverage_anchors
    )
    return BoundNativeLayout(
        product_code=bundle.product_code,
        regions=(
            BoundLayoutRegion(
                page=1,
                label="text",
                score=0.99,
                order=0,
                box=(10.0, 10.0, 500.0, 700.0),
                native_word_anchors=anchors,
            ),
        ),
        unbound_native_anchors=(),
        selected_pages=(1,),
        binding_digest=hashlib.sha256(
            f"layout-{bundle.product_code}".encode()
        ).hexdigest(),
    )


def test_layout_evidence_only_annotates_native_sections() -> None:
    bundle = _trusted_product_bundle("PZO12001", section_count=2)
    adapted = apply_layout_evidence(bundle, _layout_binding_for(bundle))

    assert adapted.parser_version == "paizo-native-v3+pp-doclayout-v3-v1"
    assert adapted.layout_binding_digest is not None
    assert [section.text for section in adapted.sections] == [
        section.text for section in bundle.sections
    ]
    assert [section.coverage_anchors for section in adapted.sections] == [
        section.coverage_anchors for section in bundle.sections
    ]
    assert all("layout-region-split" in section.layout_flags for section in adapted.sections)
    shared_tokens = [
        {flag for flag in section.layout_flags if flag.startswith("layout-region-split:")}
        for section in adapted.sections
    ]
    assert shared_tokens[0] == shared_tokens[1]


def test_shared_layout_region_only_corroborates_suspicious_adjacent_boundary(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    bundle = _trusted_product_bundle(
        "PZO12001",
        section_count=2,
        same_heading=True,
    )
    layout_bundle = apply_layout_evidence(bundle, _layout_binding_for(bundle))

    with patch(
        "pf2e_codex.licensed_corpus.load_and_parse_verified_pdf",
        return_value=layout_bundle,
    ):
        staged = stage_trusted_native_pdf(
            workspace,
            tmp_path / "PZO12001E.pdf",
            product_code="PZO12001",
            parser_version="paizo-native-v3",
            layout_artifact=tmp_path / "layout.json",
        )
    activate_parser_run(workspace, str(staged["parser_run_id"]))

    result = generate_stitch_candidates(workspace)
    conn = sqlite3.connect(workspace)
    groups = [json.loads(row[0]) for row in conn.execute(
        "SELECT section_keys FROM stitch_candidates ORDER BY candidate_id"
    )]
    conn.close()

    assert result["created"] == 1
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_shared_layout_region_alone_is_not_a_stitch_signal(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    bundle = _trusted_product_bundle("PZO12001", section_count=2)
    layout_bundle = apply_layout_evidence(bundle, _layout_binding_for(bundle))

    with patch(
        "pf2e_codex.licensed_corpus.load_and_parse_verified_pdf",
        return_value=layout_bundle,
    ):
        staged = stage_trusted_native_pdf(
            workspace,
            tmp_path / "PZO12001E.pdf",
            product_code="PZO12001",
            parser_version="paizo-native-v3",
            layout_artifact=tmp_path / "layout.json",
        )
    activate_parser_run(workspace, str(staged["parser_run_id"]))

    assert generate_stitch_candidates(workspace)["created"] == 0


def test_stitch_candidates_are_minimal_and_nonoverlapping(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    bundle = _trusted_product_bundle(
        "PZO12001",
        section_count=3,
        layout_flags=("fragment",),
    )
    layout_bundle = apply_layout_evidence(bundle, _layout_binding_for(bundle))

    with patch(
        "pf2e_codex.licensed_corpus.load_and_parse_verified_pdf",
        return_value=layout_bundle,
    ):
        staged = stage_trusted_native_pdf(
            workspace,
            tmp_path / "PZO12001E.pdf",
            product_code="PZO12001",
            parser_version="paizo-native-v3",
            layout_artifact=tmp_path / "layout.json",
        )
    activate_parser_run(workspace, str(staged["parser_run_id"]))

    assert generate_stitch_candidates(workspace)["created"] == 1
    conn = sqlite3.connect(workspace)
    width, keys = conn.execute(
        "SELECT json_extract(evidence_json, '$.width'), section_keys FROM stitch_candidates"
    ).fetchone()
    conn.close()
    assert width == 2
    assert len(json.loads(keys)) == 2


def test_stitch_claims_are_disjoint_and_expired_leases_are_reclaimed(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    bundle = _trusted_product_bundle("PZO12001", section_count=4)
    with patch(
        "pf2e_codex.licensed_corpus.load_and_parse_verified_pdf",
        return_value=bundle,
    ):
        staged = stage_trusted_native_pdf(
            workspace,
            tmp_path / "PZO12001E.pdf",
            product_code="PZO12001",
            parser_version="paizo-native-v3",
        )
    activate_parser_run(workspace, str(staged["parser_run_id"]))
    conn = sqlite3.connect(workspace)
    keys = [
        row[0]
        for row in conn.execute(
            "SELECT section_key FROM source_sections ORDER BY source_section_id"
        )
    ]
    conn.executemany(
        "INSERT INTO stitch_candidates VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                "stitch:a",
                staged["parser_run_id"],
                "PZO12001",
                json.dumps(keys[:2]),
                "{}",
                1,
            ),
            (
                "stitch:b",
                staged["parser_run_id"],
                "PZO12001",
                json.dumps(keys[2:]),
                "{}",
                1,
            ),
        ],
    )
    conn.commit()
    conn.close()

    first = _stitch_records(workspace, "stitch-select", "selector:0", limit=1)
    second = _stitch_records(workspace, "stitch-select", "selector:1", limit=1)
    assert {first[0]["id"]}.isdisjoint({second[0]["id"]})
    assert verify_workspace(workspace)["live_claims"] == 2

    conn = sqlite3.connect(workspace)
    conn.execute(
        "UPDATE stitch_claims SET lease_expires_at=0 WHERE claimant='selector:0'"
    )
    conn.commit()
    conn.close()
    reclaimed = _stitch_records(workspace, "stitch-select", "selector:2", limit=1)
    assert reclaimed[0]["id"] == first[0]["id"]


class WorkflowCodex(FakeCodex):
    def execute(self, *, prompt: str, thread_id: str | None, **kwargs: object) -> CodexResult:
        del kwargs
        self.thread_inputs.append(thread_id)
        records = json.loads(prompt.split("Records:\n", 1)[1])
        results = []
        if "bounded screen worker" in prompt:
            results = [
                {"id": record["id"], "decision": "add", "reason": None, "foundry_ids": []}
                for record in records
            ]
        elif "bounded classify worker" in prompt:
            results = [
                {
                    "id": record["id"],
                    "decision": "MIXED_NEEDS_EXTRACTION" if record["heading"] == "Mixed Rule" else "PUBLIC_AS_IS",
                    "reason_tags": ["functional-mechanics"],
                    "confidence": 0.95,
                }
                for record in records
            ]
        elif "bounded extract worker" in prompt:
            results = [
                {
                    "id": record["id"],
                    "heading": "Armor Class bonus",
                    "text": "A creature gains a +2 circumstance bonus to Armor Class.",
                    "reason_tags": ["functional-mechanics"],
                    "confidence": 0.95,
                }
                for record in records
            ]
        elif "bounded review" in prompt:
            results = [
                {"id": record["id"], "verdict": "APPROVE", "reason_tags": ["independent-review"]}
                for record in records
            ]
        else:  # pragma: no cover - stitch queues have no synthetic candidates
            raise AssertionError(prompt[:120])
        payload = {"results": results}
        return CodexResult(payload, "thread-workflow", {"input_tokens": 10}, hashlib.sha256(json.dumps(payload).encode()).hexdigest())


class ComplexWorkflowCodex(WorkflowCodex):
    def execute(self, *, prompt: str, thread_id: str | None, **kwargs: object) -> CodexResult:
        records = json.loads(prompt.split("Records:\n", 1)[1])
        if any(
            marker in prompt
            for marker in (
                "bounded screen worker", "bounded screen-deferred worker", "bounded screen-terra worker",
            )
        ):
            self.thread_inputs.append(thread_id)
            payload = {
                "results": [
                    {
                        "id": record["id"], "decision": "defer",
                        "reason": "complex-rule", "foundry_ids": [],
                    }
                    for record in records
                ]
            }
            return CodexResult(
                payload, "thread-complex", {"input_tokens": 10},
                hashlib.sha256(json.dumps(payload).encode()).hexdigest(),
            )
        return super().execute(prompt=prompt, thread_id=thread_id, **kwargs)


def test_persistent_scope_holds_legacy_without_discarding_parser_work(tmp_path: Path):
    workspace = _workspace(tmp_path)
    _stage_expected_products(workspace, tmp_path)
    remaster_products = [
        product for product in EXPECTED_PRODUCTS if product != "PZO2101"
    ]

    scope = set_review_product_scope(
        workspace, remaster_products, held_reason="legacy-study"
    )

    assert scope["enabled_products"] == sorted(remaster_products)
    assert scope["held_products"] == ["PZO2101"]
    with sqlite3.connect(workspace) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM parser_runs WHERE state='active' AND review_enabled=1"
        ).fetchone()[0] == 5
    assert claim_draft_screening_batch(
        workspace, "held-worker", product_code="PZO2101"
    ) is None
    active_claim = claim_draft_screening_batch(
        workspace, "active-worker", product_code="PZO12001"
    )
    assert active_claim is not None
    with pytest.raises(ValueError, match="claims are live"):
        set_review_product_scope(workspace, remaster_products)


def test_prepare_and_preview_are_spark_free_idempotent_and_read_only(tmp_path: Path):
    workspace = _workspace(tmp_path)
    foundry = _foundry_database(tmp_path)
    _stage_expected_products(workspace, tmp_path)
    remaster_products = [
        product for product in EXPECTED_PRODUCTS if product != "PZO2101"
    ]
    set_review_product_scope(
        workspace, remaster_products, held_reason="legacy-study"
    )

    first = prepare_review_data(workspace, foundry)
    second = prepare_review_data(workspace, foundry)
    assert first["preparation_digest"] == second["preparation_digest"]
    assert {item["product_code"]: item["scope_state"] for item in second["products"]} == {
        **dict.fromkeys(remaster_products, "enabled"),
        "PZO2101": "held",
    }
    with sqlite3.connect(workspace) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    before = workspace.read_bytes()

    preview = preview_screen_batches(workspace, foundry)

    assert workspace.read_bytes() == before
    assert preview["ready"] is True
    assert preview["model"] == "gpt-5.3-codex-spark"
    assert preview["scope"]["held_products"] == ["PZO2101"]
    assert {item["product_code"] for item in preview["products"]} == set(remaster_products)
    assert preview["eligible_records"] == 4
    assert preview["batches"] == 4
    assert all(int(item["max_batch_bytes"]) <= MAX_BATCH_BYTES for item in preview["products"])
    with sqlite3.connect(workspace) as conn:
        assert conn.execute("SELECT COUNT(*) FROM runner_sessions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM runner_attempts").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM draft_screening_claims").fetchone()[0] == 0


def test_synthetic_five_product_workflow_builds_deterministic_base(tmp_path: Path):
    workspace = _workspace(tmp_path)
    foundry = _foundry_database(tmp_path)
    for product_code in EXPECTED_PRODUCTS:
        bundle = _trusted_product_bundle(product_code, mixed=product_code == "PZO12002")
        with patch("pf2e_codex.licensed_corpus.load_and_parse_verified_pdf", return_value=bundle):
            staged = stage_trusted_native_pdf(
                workspace,
                tmp_path / f"{product_code}E.pdf",
                product_code=product_code,
                parser_version="paizo-native-v3",
                shard_size=2,
            )
        activate_parser_run(workspace, str(staged["parser_run_id"]))

    result = run_queues(
        workspace, concurrency=1, foundry_database=foundry, executor=WorkflowCodex()
    )
    assert result["status"]["needs_maintainer"] == 0
    verification = verify_workspace(
        workspace, require_complete=True, foundry_database=foundry
    )
    assert verification["ok"] is True, verification

    notices = tmp_path / "notices.json"
    notices.write_text(
        json.dumps(
            {
                "OGL": {"license": "OGL", "text": "Synthetic complete OGL notice."},
                "ORC": {"license": "ORC", "text": "Synthetic complete ORC notice."},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "licensed_core.sqlite3"
    built = build_base(workspace, output, notices, foundry)
    bundle = load_licensed_core(output)
    assert built["sections"] == 5
    assert len(bundle.chunks) == 5
    assert {row["era"] for row in bundle.source_revisions} == {"legacy", "remaster"}


def test_scoped_remaster_workflow_builds_explicit_four_product_base(tmp_path: Path):
    workspace = _workspace(tmp_path)
    foundry = _foundry_database(tmp_path)
    _stage_expected_products(workspace, tmp_path, mixed_product="PZO12002")
    remaster_products = [
        product for product in EXPECTED_PRODUCTS if product != "PZO2101"
    ]
    set_review_product_scope(
        workspace, remaster_products, held_reason="legacy-study"
    )

    run_queues(
        workspace, concurrency=1, foundry_database=foundry,
        executor=WorkflowCodex(),
    )
    verification = verify_workspace(
        workspace, require_complete=True, foundry_database=foundry
    )
    assert verification["ok"] is True, verification
    assert verification["semantic_scope"]["held_products"] == ["PZO2101"]
    notices = tmp_path / "scoped-notices.json"
    notices.write_text(
        json.dumps(
            {
                "ORC": {"license": "ORC", "text": "Synthetic complete ORC notice."},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "licensed_core_scoped.sqlite3"
    build_base(workspace, output, notices, foundry)
    bundle = load_licensed_core(output)

    assert bundle.covered_products == tuple(sorted(remaster_products))
    assert len(bundle.chunks) == 4
    assert {row["product_code"] for row in bundle.source_revisions} == set(remaster_products)
    assert {row["era"] for row in bundle.source_revisions} == {"remaster"}


def test_layout_queue_stops_before_screening(tmp_path: Path):
    workspace = _workspace(tmp_path)
    bundle = _trusted_product_bundle("PZO12001")
    with patch("pf2e_codex.licensed_corpus.load_and_parse_verified_pdf", return_value=bundle):
        staged = stage_trusted_native_pdf(
            workspace, tmp_path / "PZO12001E.pdf", product_code="PZO12001",
            parser_version="paizo-native-v3", shard_size=1,
        )
    activate_parser_run(workspace, str(staged["parser_run_id"]))

    result = run_queues(workspace, queue="layout", concurrency=1, executor=WorkflowCodex())

    assert result["completed_batches"] == {
        "stitch-select": 0,
        "stitch-confirm": 0,
        "repaired_products": 0,
    }
    with sqlite3.connect(workspace) as conn:
        assert conn.execute("SELECT COUNT(*) FROM draft_screening_events").fetchone()[0] == 0


def test_complex_screen_gets_luna_then_terra_and_defaults_to_add(tmp_path: Path):
    workspace = _workspace(tmp_path)
    foundry = _foundry_database(tmp_path)
    for product_code in EXPECTED_PRODUCTS:
        bundle = _trusted_product_bundle(product_code)
        with patch("pf2e_codex.licensed_corpus.load_and_parse_verified_pdf", return_value=bundle):
            staged = stage_trusted_native_pdf(
                workspace, tmp_path / f"{product_code}E.pdf",
                product_code=product_code, parser_version="paizo-native-v3", shard_size=2,
            )
        activate_parser_run(workspace, str(staged["parser_run_id"]))

    result = run_queues(
        workspace, concurrency=1, foundry_database=foundry,
        executor=ComplexWorkflowCodex(),
    )
    products = result["status"]["screening"]["products"]
    assert sum(product["accepted"] for product in products) == 5
    assert sum(product["deferred"] for product in products) == 0
    conn = sqlite3.connect(workspace)
    escalations = conn.execute("SELECT COUNT(*) FROM runner_screen_escalations").fetchone()[0]
    terra_attempts = conn.execute(
        "SELECT COUNT(*) FROM runner_attempts WHERE queue_name='screen-terra' AND status='accepted'"
    ).fetchone()[0]
    conn.close()
    assert escalations == 5
    assert terra_attempts == 5


def test_screening_pilot_processes_one_batch_per_product_without_draining(tmp_path: Path):
    workspace = _workspace(tmp_path)
    foundry = _foundry_database(tmp_path)
    for product_code in EXPECTED_PRODUCTS:
        bundle = _trusted_product_bundle(product_code, section_count=2)
        with patch("pf2e_codex.licensed_corpus.load_and_parse_verified_pdf", return_value=bundle):
            staged = stage_trusted_native_pdf(
                workspace, tmp_path / f"{product_code}E.pdf",
                product_code=product_code, parser_version="paizo-native-v3", shard_size=1,
            )
        activate_parser_run(workspace, str(staged["parser_run_id"]))

    result = run_queues(
        workspace, queue="screen", concurrency=2, foundry_database=foundry,
        executor=WorkflowCodex(), pilot=True,
    )

    assert result["completed_products"] == sorted(EXPECTED_PRODUCTS)
    assert result["completed_batches"] == 5
    products = result["status"]["screening"]["products"]
    assert sum(product["accepted"] + product["rejected"] for product in products) == 5
    assert sum(product["unprocessed"] for product in products) == 5
    incomplete = verify_workspace(
        workspace, require_complete=True, foundry_database=foundry
    )
    assert incomplete["ok"] is False
    assert incomplete["unresolved"] == 10


def test_exact_pdf_duplicates_screen_only_canonical_and_reopen_as_one_group(tmp_path: Path):
    workspace = _workspace(tmp_path)
    foundry = _foundry_database(tmp_path)
    bundle = _trusted_product_bundle("PZO12001", section_count=2, same_heading=True)
    with patch("pf2e_codex.licensed_corpus.load_and_parse_verified_pdf", return_value=bundle):
        staged = stage_trusted_native_pdf(
            workspace, tmp_path / "PZO12001E.pdf", product_code="PZO12001",
            parser_version="paizo-native-v3", shard_size=2,
        )
    activate_parser_run(workspace, str(staged["parser_run_id"]))

    prepared = prepare_deterministic_review(workspace, foundry)
    assert prepared["canonical_sections"] == 1
    assert prepared["shadow_duplicates"] == 1
    with sqlite3.connect(workspace) as conn:
        canonical, shadow = conn.execute(
            """SELECT groups.canonical_section_key, member.section_key
                 FROM duplicate_groups AS groups
                 JOIN duplicate_group_members AS member USING(group_id)
                WHERE member.source_ordinal=1"""
        ).fetchone()
        assert conn.execute(
            "SELECT decision FROM draft_screening_current WHERE section_key=?", (shadow,)
        ).fetchone()[0] == "REJECT"

    reopened = reopen_screening(
        workspace, shadow, reason="maintainer-review", maintainer="maintainer"
    )
    assert reopened["requested_section_key"] == shadow
    assert reopened["section_key"] == canonical
    with sqlite3.connect(workspace) as conn:
        assert conn.execute(
            "SELECT 1 FROM draft_screening_current WHERE section_key=?", (canonical,)
        ).fetchone() is None


def test_changed_foundry_snapshot_reopens_coverage_rejection(tmp_path: Path):
    workspace = _workspace(tmp_path)
    foundry = _foundry_database(tmp_path)
    bundle = _trusted_product_bundle("PZO12001")
    source = bundle.sections[0]
    with sqlite3.connect(foundry) as conn:
        conn.execute(
            """UPDATE chunks SET name=?, text=?, source_hash=?
                WHERE id='rules:pzo12001'""",
            (source.heading, source.text, hashlib.sha256(source.text.encode()).hexdigest()),
        )
    with patch("pf2e_codex.licensed_corpus.load_and_parse_verified_pdf", return_value=bundle):
        staged = stage_trusted_native_pdf(
            workspace, tmp_path / "PZO12001E.pdf", product_code="PZO12001",
            parser_version="paizo-native-v3", shard_size=1,
        )
    activate_parser_run(workspace, str(staged["parser_run_id"]))
    prepared = prepare_deterministic_review(workspace, foundry)
    assert prepared["coverage_candidates"] == 1
    claim = claim_draft_screening_batch(workspace, "screen:0")
    assert claim is not None
    submit_draft_screening_decision(
        workspace, int(claim["shard_id"]), "screen:0", 0, "reject",
        reject_reason="duplicate", foundry_ids=["rules:pzo12001"],
    )

    changed = "A materially changed public rule with a different numeric value of 4d8."
    with sqlite3.connect(foundry) as conn:
        conn.execute(
            """UPDATE chunks SET text=?, source_hash=? WHERE id='rules:pzo12001'""",
            (changed, hashlib.sha256(changed.encode()).hexdigest()),
        )
    refreshed = prepare_deterministic_review(workspace, foundry)
    assert refreshed["stale_coverage_reopened"] == 1
    with sqlite3.connect(workspace) as conn:
        section_key = conn.execute(
            "SELECT section_key FROM source_sections WHERE parser_run_id=?",
            (staged["parser_run_id"],),
        ).fetchone()[0]
        assert conn.execute(
            "SELECT 1 FROM draft_screening_current WHERE section_key=?", (section_key,)
        ).fetchone() is None


def test_replacement_run_carries_only_exact_terminal_screening_work(tmp_path: Path):
    workspace = _workspace(tmp_path)
    foundry = _foundry_database(tmp_path)
    original = _trusted_product_bundle("PZO12001", section_count=2)
    with patch(
        "pf2e_codex.licensed_corpus.load_and_parse_verified_pdf",
        return_value=original,
    ):
        staged = stage_trusted_native_pdf(
            workspace,
            tmp_path / "PZO12001E.pdf",
            product_code="PZO12001",
            parser_version="paizo-native-v3",
            shard_size=2,
        )
    activate_parser_run(workspace, str(staged["parser_run_id"]))
    run_queues(
        workspace, queue="screen", concurrency=1, foundry_database=foundry,
        executor=WorkflowCodex(),
    )

    replacement = _trusted_product_bundle(
        "PZO12001",
        parser_version="paizo-native-v4",
        section_count=2,
        changed_last=True,
    )
    with patch(
        "pf2e_codex.licensed_corpus.load_and_parse_verified_pdf",
        return_value=replacement,
    ):
        restaged = stage_trusted_native_pdf(
            workspace,
            tmp_path / "PZO12001E.pdf",
            product_code="PZO12001",
            parser_version="paizo-native-v4",
            shard_size=2,
        )
    assert restaged["screening_reused"] == 1
    activate_parser_run(workspace, str(restaged["parser_run_id"]))

    product = runner_status(workspace)["screening"]["products"][0]
    assert product["accepted"] == 1
    assert product["unprocessed"] == 1
    with sqlite3.connect(workspace) as conn:
        carried = conn.execute(
            """SELECT COUNT(*) FROM draft_screening_events
               WHERE parser_run_id=? AND supersedes_event_id IS NOT NULL""",
            (restaged["parser_run_id"],),
        ).fetchone()[0]
    assert carried == 1


def test_stitch_pilot_targets_at_most_one_batch_per_product(tmp_path: Path):
    workspace = _workspace(tmp_path)
    products: list[str] = []
    enabled = [product for product in EXPECTED_PRODUCTS if product != "PZO2101"]

    def process(
        _workspace: Path,
        _queue: str,
        _slot: int,
        _executor: object,
        *,
        product_code: str | None = None,
        pilot: bool = False,
    ) -> bool:
        assert product_code is not None
        assert pilot is True
        products.append(product_code)
        return True

    with (
        patch("pf2e_codex.review_runner._process_stitches", side_effect=process),
        patch(
            "pf2e_codex.review_runner.review_product_scope",
            return_value={"enabled_products": enabled},
        ),
    ):
        result = run_queues(
            workspace, queue="stitch-select", concurrency=2,
            executor=WorkflowCodex(), pilot=True,
        )

    assert sorted(products) == sorted(enabled)
    assert result["completed_products"] == sorted(enabled)
    assert result["completed_batches"] == 4


def test_stitch_disagreement_immediately_becomes_maintainer_work(tmp_path: Path):
    workspace = _workspace(tmp_path)
    bundle = _trusted_product_bundle("PZO2101", section_count=2)
    with patch("pf2e_codex.licensed_corpus.load_and_parse_verified_pdf", return_value=bundle):
        staged = stage_trusted_native_pdf(
            workspace, tmp_path / "PZO2101E.pdf", product_code="PZO2101",
            parser_version="paizo-native-v3", shard_size=1,
        )
    activate_parser_run(workspace, str(staged["parser_run_id"]))
    conn = sqlite3.connect(workspace)
    section_keys = [row[0] for row in conn.execute(
        "SELECT section_key FROM source_sections WHERE parser_run_id=?",
        (staged["parser_run_id"],),
    )]
    conn.execute(
        "INSERT INTO stitch_candidates VALUES (?, ?, ?, ?, ?, ?)",
        ("stitch:test", staged["parser_run_id"], "PZO2101", json.dumps(section_keys), "{}", 1),
    )
    conn.executemany(
        "INSERT INTO stitch_votes VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("stitch:test", "selector", "luna", "merge", "possible fragment", 2),
            ("stitch:test", "confirmer", "terra", "no-merge", "separate rules", 3),
        ],
    )
    conn.commit()
    conn.close()

    result = run_queues(
        workspace, queue="stitch-confirm", concurrency=1,
        executor=WorkflowCodex(), pilot=True,
    )

    assert result["completed_batches"] == 0
    assert result["status"]["needs_maintainer"] == 1
    item = result["status"]["maintainer_items"][0]
    evidence = maintainer_item_evidence(workspace, item["maintenance_id"])
    assert evidence["candidate_id"] == "stitch:test"
    assert evidence["private_text_included"] is False
    assert "source_text" not in evidence["sections"][0]
    assert {vote["decision"] for vote in evidence["votes"]} == {"merge", "no-merge"}
    private_evidence = maintainer_item_evidence(
        workspace, item["maintenance_id"], include_text=True,
    )
    assert private_evidence["sections"][0]["source_text"]

    assert resolve_maintainer_item(
        workspace,
        item["maintenance_id"],
        "no-merge",
    )["decision"] == "no-merge"
    assert runner_status(workspace)["needs_maintainer"] == 0
    assert resolve_maintainer_item(
        workspace,
        item["maintenance_id"],
        "no-merge",
    )["decision"] == "no-merge"
    with sqlite3.connect(workspace) as conn:
        conn.row_factory = sqlite3.Row
        prior = conn.execute(
            "SELECT * FROM stitch_candidates WHERE candidate_id='stitch:test'"
        ).fetchone()
        reused_id = "stitch:test-reused"
        conn.execute(
            "INSERT INTO stitch_candidates VALUES (?, ?, ?, ?, ?, ?)",
            (
                reused_id,
                prior["parser_run_id"],
                prior["product_code"],
                json.dumps(list(reversed(json.loads(prior["section_keys"])))),
                json.dumps({"offset": 999}),
                4,
            ),
        )
        matched = _prior_stitch_candidate(
            conn,
            reused_id,
            str(prior["product_code"]),
            str(prior["section_keys"]),
        )
        assert matched["candidate_id"] == "stitch:test"
        _reuse_stitch_judgment(conn, reused_id, str(matched["candidate_id"]))
        carried = conn.execute(
            "SELECT resolution, resolved_at FROM runner_maintenance WHERE subject_id=?",
            (reused_id,),
        ).fetchone()
        assert carried["resolution"] == "no-merge"
        assert carried["resolved_at"] is not None
    with pytest.raises(ValueError, match="different resolution"):
        resolve_maintainer_item(workspace, item["maintenance_id"], "merge")


def test_held_product_maintainer_item_is_preserved_but_not_actionable(tmp_path: Path):
    workspace = _workspace(tmp_path)
    staged_runs: dict[str, dict[str, object]] = {}
    for product_code in ("PZO2101", "PZO12001"):
        bundle = _trusted_product_bundle(product_code, section_count=2)
        with patch(
            "pf2e_codex.licensed_corpus.load_and_parse_verified_pdf",
            return_value=bundle,
        ):
            staged = stage_trusted_native_pdf(
                workspace,
                tmp_path / f"{product_code}E.pdf",
                product_code=product_code,
                parser_version="paizo-native-v3",
                shard_size=1,
            )
        activate_parser_run(workspace, str(staged["parser_run_id"]))
        staged_runs[product_code] = staged

    legacy = staged_runs["PZO2101"]
    with sqlite3.connect(workspace) as conn:
        section_keys = [
            row[0]
            for row in conn.execute(
                "SELECT section_key FROM source_sections WHERE parser_run_id=?",
                (legacy["parser_run_id"],),
            )
        ]
        conn.execute(
            "INSERT INTO stitch_candidates VALUES (?, ?, ?, ?, ?, ?)",
            (
                "stitch:held",
                legacy["parser_run_id"],
                "PZO2101",
                json.dumps(section_keys),
                "{}",
                1,
            ),
        )
        conn.executemany(
            "INSERT INTO stitch_votes VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("stitch:held", "selector", "luna", "merge", "fragment", 2),
                ("stitch:held", "confirmer", "terra", "no-merge", "separate", 3),
            ],
        )

    assert _reconcile_stitch_maintenance(workspace) == 1
    with sqlite3.connect(workspace) as conn:
        maintenance_id = str(conn.execute(
            "SELECT item_id FROM runner_maintenance WHERE subject_id='stitch:held'"
        ).fetchone()[0])

    set_review_product_scope(workspace, ["PZO12001"])

    assert _reconcile_stitch_maintenance(workspace) == 0
    assert runner_status(workspace)["needs_maintainer"] == 0
    with pytest.raises(ValueError, match="outside the semantic product scope"):
        maintainer_item_evidence(workspace, maintenance_id)
    with pytest.raises(ValueError, match="outside the semantic product scope"):
        resolve_maintainer_item(workspace, maintenance_id, "no-merge")
    with sqlite3.connect(workspace) as conn:
        assert conn.execute(
            "SELECT resolved_at FROM runner_maintenance WHERE item_id=?",
            (maintenance_id,),
        ).fetchone()[0] is None


def test_maintainer_merge_resolution_counts_as_explicit_approval(tmp_path: Path):
    workspace = _workspace(tmp_path)
    foundry = _foundry_database(tmp_path)
    bundle = _trusted_product_bundle("PZO2101", section_count=2)
    with patch("pf2e_codex.licensed_corpus.load_and_parse_verified_pdf", return_value=bundle):
        staged = stage_trusted_native_pdf(
            workspace, tmp_path / "PZO2101E.pdf", product_code="PZO2101",
            parser_version="paizo-native-v3", shard_size=1,
        )
    activate_parser_run(workspace, str(staged["parser_run_id"]))
    with sqlite3.connect(workspace) as conn:
        section_keys = [row[0] for row in conn.execute(
            "SELECT section_key FROM source_sections WHERE parser_run_id=?",
            (staged["parser_run_id"],),
        )]
        conn.execute(
            "INSERT INTO stitch_candidates VALUES (?, ?, ?, ?, ?, ?)",
            ("stitch:test", staged["parser_run_id"], "PZO2101", json.dumps(section_keys), "{}", 1),
        )
        conn.executemany(
            "INSERT INTO stitch_votes VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("stitch:test", "selector", "luna", "merge", "possible fragment", 2),
                ("stitch:test", "confirmer", "terra", "no-merge", "separate rules", 3),
            ],
        )
    status = run_queues(
        workspace, queue="stitch-confirm", concurrency=1,
        executor=WorkflowCodex(), pilot=True,
    )["status"]
    item = status["maintainer_items"][0]

    resolve_maintainer_item(workspace, item["maintenance_id"], "merge")

    assert runner_status(workspace)["needs_maintainer"] == 0
    with patch(
        "pf2e_codex.licensed_corpus.load_and_parse_verified_pdf",
        return_value=bundle,
    ):
        repaired = stage_trusted_native_pdf_with_approved_stitches(
            workspace,
            tmp_path / "PZO2101E.pdf",
            product_code="PZO2101",
            parser_version="paizo-native-v3",
            shard_size=1,
        )
    with sqlite3.connect(workspace) as conn:
        assert conn.execute(
            "SELECT declared_section_count FROM parser_runs WHERE parser_run_id=?",
            (repaired["parser_run_id"],),
        ).fetchone()[0] == 1
    with pytest.raises(ValueError, match="confirmed stitches require --sources"):
        run_queues(
            workspace, concurrency=1, foundry_database=foundry,
            executor=WorkflowCodex(),
        )


def _source_directory(tmp_path: Path) -> Path:
    root = tmp_path / "sources"
    root.mkdir()
    for product_code in EXPECTED_PRODUCTS:
        (root / f"{product_code}E.pdf").write_bytes(f"synthetic-{product_code}".encode())
    return root


def test_prepare_creates_workspace_only_after_all_five_validate(tmp_path: Path):
    sources = _source_directory(tmp_path)
    target = tmp_path / "review.sqlite3"

    def bundle_for(_path: Path, *, product_code: str, **_kwargs: object) -> TrustedParseBundle:
        return _trusted_product_bundle(product_code, parser_version="paizo-native-v5")

    with (
        patch("pf2e_codex.licensed_corpus.load_and_parse_verified_pdf", side_effect=bundle_for),
        patch("pf2e_codex.review_runner.validate_quality", return_value={"passed": True}),
        patch(
            "pf2e_codex.corpus._candidate_content_fingerprint",
            side_effect=lambda source, _root: hashlib.sha256(source.product.code.encode()).hexdigest(),
        ),
    ):
        result = prepare_workspace(target, sources, shard_size=2)

    assert result["verification"]["ok"] is True
    assert verify_workspace(target)["ok"] is True
    assert not list(tmp_path.glob(".review.sqlite3.rebuild-*"))


def test_prepare_reuses_one_layout_session_for_all_five_products(tmp_path: Path):
    sources = _source_directory(tmp_path)
    target = tmp_path / "review.sqlite3"
    analyzer = object()
    exported_analyzers: list[object] = []

    def bundle_for(
        _path: Path,
        *,
        product_code: str,
        layout_artifact: Path | None = None,
        **_kwargs: object,
    ) -> TrustedParseBundle:
        del layout_artifact
        return _trusted_product_bundle(product_code, parser_version="paizo-native-v5")

    def export_layout(_source: Path, output: Path, *, analyzer: object, **_kwargs: object) -> None:
        exported_analyzers.append(analyzer)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}")

    with (
        patch("pf2e_codex.licensed_corpus.load_and_parse_verified_pdf", side_effect=bundle_for),
        patch("pf2e_codex.review_runner.validate_quality", return_value={"passed": True}),
        patch(
            "pf2e_codex.corpus._candidate_content_fingerprint",
            side_effect=lambda source, _root: hashlib.sha256(source.product.code.encode()).hexdigest(),
        ),
        patch("pf2e_codex.pdf_layout.LayoutAnalyzer", return_value=analyzer) as create_analyzer,
        patch("pf2e_codex.pdf_layout.export_pdf_layout", side_effect=export_layout),
    ):
        result = prepare_workspace(
            target,
            sources,
            shard_size=2,
            layout_model_dir=tmp_path / "layout-model",
        )

    assert result["verification"]["ok"] is True
    create_analyzer.assert_called_once()
    assert exported_analyzers == [analyzer] * len(EXPECTED_PRODUCTS)


def test_prepare_failure_preserves_valid_existing_workspace(tmp_path: Path):
    sources = _source_directory(tmp_path)
    target = tmp_path / "review.sqlite3"
    initialize_trusted_workspace(target)
    original = target.read_bytes()

    def fail_last(_path: Path, *, product_code: str, **_kwargs: object) -> TrustedParseBundle:
        if product_code == "PZO12004":
            raise ValueError("synthetic parser failure")
        return _trusted_product_bundle(product_code, parser_version="paizo-native-v5")

    with (
        patch("pf2e_codex.licensed_corpus.load_and_parse_verified_pdf", side_effect=fail_last),
        patch(
            "pf2e_codex.corpus._candidate_content_fingerprint",
            side_effect=lambda source, _root: hashlib.sha256(source.product.code.encode()).hexdigest(),
        ),
        pytest.raises(ValueError, match="synthetic parser failure"),
    ):
        prepare_workspace(target, sources, shard_size=2)

    assert target.read_bytes() == original
    assert not list(tmp_path.glob(".review.sqlite3.rebuild-*"))
