"""Focused tests for the read-only, content-free corpus quality audit."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

from pf2e_codex.corpus_quality import (
    ProductQuality,
    QualityReport,
    audit_workspace,
    compare_quality,
    validate_quality,
)


def _create_workspace(
    path: Path,
    *,
    with_quarantine: bool = True,
    reverse_insert: bool = False,
    source_text_override: str | None = None,
) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE source_assets (
            asset_id TEXT PRIMARY KEY,
            product_code TEXT,
            native_word_anchor_count INTEGER
        );
        CREATE TABLE parser_runs (
            parser_run_id TEXT PRIMARY KEY,
            asset_id TEXT,
            product_code TEXT,
            state TEXT,
            activated_at INTEGER,
            created_at INTEGER
        );
        CREATE TABLE source_sections (
            section_key TEXT PRIMARY KEY,
            product_code TEXT,
            parser_run_id TEXT,
            source_section_id TEXT,
            source_section_hash TEXT,
            page_start INTEGER,
            page_end INTEGER,
            heading TEXT,
            source_text TEXT,
            layout_flags TEXT
        );
        CREATE TABLE parser_run_sections (
            parser_run_id TEXT,
            section_key TEXT,
            membership_state TEXT
        );
        CREATE TABLE parser_section_anchors (
            parser_run_id TEXT,
            section_key TEXT,
            anchor_hash TEXT
        );
        CREATE TABLE parser_ignored_anchors (
            parser_run_id TEXT,
            anchor_hash TEXT
        );
        """
    )
    if with_quarantine:
        conn.execute(
            """CREATE TABLE parser_quarantine (
                parser_run_id TEXT,
                product_code TEXT,
                reason TEXT,
                anchor_hash TEXT,
                source_text TEXT
            )"""
        )

    conn.executemany(
        "INSERT INTO source_assets VALUES (?, ?, ?)",
        [("asset-1", "PZO12001", 5), ("asset-2", "PZO2101", 2)],
    )
    conn.executemany(
        "INSERT INTO parser_runs VALUES (?, ?, ?, 'active', ?, ?)",
        [("run-player", "asset-1", "PZO12001", 20, 20), ("run-legacy", "asset-2", "PZO2101", 10, 10)],
    )
    player_rows = [
        (
            "section-player-fireball",
            "PZO12001",
            "run-player",
            "fireball",
            "hash-fireball",
            10,
            10,
            "Fireball",
            source_text_override or "A coherent rules section with enough detail.",
            "[\"layout-order-conflict\"]",
        ),
        (
            "section-player-page",
            "PZO12001",
            "run-player",
            "page",
            "hash-page",
            11,
            11,
            "123",
            "tiny",
            "[\"native-layout-fallback\"]",
        ),
        (
            "section-player-unclassified",
            "PZO12001",
            "run-player",
            "unclassified",
            "hash-unclassified",
            12,
            12,
            "Unclassified native text",
            "Another short fragment.",
            "[\"unclassified-native-coverage\"]",
        ),
    ]
    legacy_rows = [
        (
            "section-legacy-rule",
            "PZO2101",
            "run-legacy",
            "legacy-rule",
            "hash-legacy",
            20,
            20,
            "Legacy Rule",
            "A legacy rule section.",
            "[]",
        )
    ]
    rows = player_rows + legacy_rows
    if reverse_insert:
        rows.reverse()
    conn.executemany("INSERT INTO source_sections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    memberships = [
        ("run-player", row[0], "present") for row in player_rows
    ] + [("run-legacy", row[0], "present") for row in legacy_rows]
    if reverse_insert:
        memberships.reverse()
    conn.executemany("INSERT INTO parser_run_sections VALUES (?, ?, ?)", memberships)
    conn.executemany(
        "INSERT INTO parser_section_anchors VALUES (?, ?, ?)",
        [
            ("run-player", "section-player-fireball", "anchor-1"),
            ("run-player", "section-player-page", "anchor-2"),
            ("run-player", "section-player-unclassified", "anchor-3"),
            ("run-legacy", "section-legacy-rule", "anchor-legacy"),
        ],
    )
    conn.executemany(
        "INSERT INTO parser_ignored_anchors VALUES (?, ?)",
        [("run-player", "anchor-4"), ("run-legacy", "anchor-legacy-ignored")],
    )
    if with_quarantine:
        conn.executemany(
            "INSERT INTO parser_quarantine VALUES (?, ?, ?, ?, ?)",
            [
                ("run-player", "PZO12001", "unresolved-table", "anchor-q1", "private@example.invalid"),
                ("run-player", "PZO12001", "unexpected-private-reason", "anchor-q2", "safe"),
            ],
        )
    conn.commit()
    conn.close()


def _clean_quality(
    *,
    product: str = "PZO12001",
    run: str = "run",
    conflicts: int = 0,
    short: int = 0,
    sentence: int = 0,
) -> ProductQuality:
    return ProductQuality(
        product_code=product,
        parser_run_id=run,
        section_count=10,
        quarantine_count=0,
        quarantined_anchor_count=0,
        quarantine_by_reason={},
        quarantine_anchor_ratio=0.0,
        expected_anchor_count=10,
        assigned_anchor_count=10,
        ignored_anchor_count=0,
        missing_anchor_count=0,
        extra_anchor_count=0,
        duplicate_anchor_count=0,
        anchor_coverage_ratio=1.0,
        empty_heading_count=0,
        numeric_only_heading_count=0,
        sentence_like_heading_count=sentence,
        heading_defect_count=sentence,
        layout_order_conflict_count=conflicts,
        layout_metadata_error_count=0,
        unclassified_count=0,
        unresolved_table_count=0,
        short_under_40_count=short,
        short_under_80_count=short,
        native_fallback_section_count=0,
        native_fallback_short_under_40_count=0,
        oversize_over_5000_count=0,
        oversize_over_10000_count=0,
        length_buckets={
            "under-40": short,
            "40-79": 0,
            "80-499": 10 - short,
            "500-1999": 0,
            "2000-4999": 0,
            "5000-9999": 0,
            "10000-plus": 0,
        },
        privacy_violation_count=0,
        digest=f"digest-{product}-{run}",
        probe_hits={
            name: {"matched": 1, "coherent": 1}
            for name in (
                "afflictions",
                "difficulty-classes",
                "dying-recovery",
                "encounter-building",
                "exploration",
                "fireball",
                "line-of-effect",
            )
        },
    )


def test_audit_reports_metrics_and_never_emits_private_values(tmp_path: Path):
    workspace = tmp_path / "review.sqlite3"
    _create_workspace(workspace, source_text_override="Private source text private@example.invalid")

    report = audit_workspace(workspace)
    payload = report.as_dict()
    player = next(item for item in payload["products"] if item["product_code"] == "PZO12001")

    assert payload["selected_runs"] == {"PZO12001": "run-player", "PZO2101": "run-legacy"}
    assert player["section_count"] == 3
    assert player["quarantine_count"] == 2
    assert player["quarantined_anchor_count"] == 2
    assert player["quarantine_anchor_ratio"] == 0.4
    assert player["quarantine_by_reason"] == {"other": 1, "unresolved-table": 1}
    assert player["expected_anchor_count"] == 5
    assert player["assigned_anchor_count"] == 3
    assert player["ignored_anchor_count"] == 1
    assert player["missing_anchor_count"] == 1
    assert player["layout_order_conflict_count"] == 1
    assert player["numeric_only_heading_count"] == 1
    assert player["unclassified_count"] == 1
    assert player["short_under_40_count"] == 1
    assert player["native_fallback_section_count"] == 1
    assert player["native_fallback_short_under_40_count"] == 1
    assert player["privacy_violation_count"] == 2
    assert player["probe_hits"]["fireball"] == {"matched": 1, "coherent": 0}

    serialized = json.dumps(payload, sort_keys=True)
    assert "private@example.invalid" not in serialized
    assert "Private source text" not in serialized
    assert "anchor-q1" not in serialized
    assert "/home/" not in serialized


def test_absent_quarantine_table_is_a_valid_v3_baseline(tmp_path: Path):
    workspace = tmp_path / "review-v3.sqlite3"
    _create_workspace(workspace, with_quarantine=False)

    report = audit_workspace(workspace)
    assert all(item.quarantine_count == 0 for item in report.products)
    assert all(item.quarantine_by_reason == {} for item in report.products)


def test_digest_is_stable_when_rows_are_inserted_in_a_different_order(tmp_path: Path):
    first = tmp_path / "first.sqlite3"
    second = tmp_path / "second.sqlite3"
    _create_workspace(first)
    _create_workspace(second, reverse_insert=True)

    first_report = audit_workspace(first)
    second_report = audit_workspace(second)
    assert first_report.digest == second_report.digest
    assert [item.digest for item in first_report.products] == [item.digest for item in second_report.products]


def test_compare_quality_applies_overall_and_per_product_gates():
    baseline_player = _clean_quality(conflicts=100, short=100, sentence=40)
    baseline_legacy = _clean_quality(product="PZO2101", run="legacy", conflicts=100, short=100, sentence=4)
    candidate_player = replace(
        baseline_player,
        parser_run_id="candidate",
        layout_order_conflict_count=10,
        short_under_40_count=20,
        short_under_80_count=20,
        sentence_like_heading_count=10,
    )
    candidate_legacy = replace(
        baseline_legacy,
        parser_run_id="candidate-legacy",
        layout_order_conflict_count=10,
        short_under_40_count=20,
        short_under_80_count=20,
        sentence_like_heading_count=1,
    )
    baseline = QualityReport("corpus-quality-v1", {"PZO12001": "run", "PZO2101": "legacy"}, (baseline_player, baseline_legacy), "base")
    candidate = QualityReport("corpus-quality-v1", {"PZO12001": "candidate", "PZO2101": "candidate-legacy"}, (candidate_player, candidate_legacy), "candidate")

    comparison = compare_quality(baseline, candidate)
    assert comparison.passed
    assert comparison.as_dict()["gates"]["layout_order_conflicts_overall"]["actual_reduction"] == 0.9


def test_compare_quality_rejects_per_product_regression_and_zero_baseline_increase():
    baseline_player = _clean_quality(conflicts=100, short=100, sentence=10)
    baseline_legacy = _clean_quality(product="PZO2101", run="legacy", conflicts=0, short=0, sentence=0)
    candidate_player = replace(
        baseline_player,
        parser_run_id="candidate",
        layout_order_conflict_count=0,
        short_under_40_count=0,
        short_under_80_count=0,
        sentence_like_heading_count=9,
    )
    candidate_legacy = replace(
        baseline_legacy,
        parser_run_id="candidate-legacy",
        layout_order_conflict_count=1,
        short_under_40_count=1,
        short_under_80_count=1,
        sentence_like_heading_count=0,
    )
    baseline = QualityReport("corpus-quality-v1", {"PZO12001": "run", "PZO2101": "legacy"}, (baseline_player, baseline_legacy), "base")
    candidate = QualityReport("corpus-quality-v1", {"PZO12001": "candidate", "PZO2101": "candidate-legacy"}, (candidate_player, candidate_legacy), "candidate")

    comparison = compare_quality(baseline, candidate)
    assert not comparison.passed
    assert not comparison.as_dict()["gates"]["layout_order_conflicts_per_product"]["PZO2101"]["passed"]
    assert not comparison.as_dict()["gates"]["short_fragments_per_product"]["PZO2101"]["passed"]


def test_compare_quality_requires_each_general_rule_probe():
    baseline = _clean_quality()
    candidate = replace(
        baseline,
        parser_run_id="candidate",
        probe_hits={
            **baseline.probe_hits,
            "line-of-effect": {"matched": 1, "coherent": 0},
        },
    )
    comparison = compare_quality(
        QualityReport("corpus-quality-v1", {"PZO12001": "run"}, (baseline,), "base"),
        QualityReport(
            "corpus-quality-v1",
            {"PZO12001": "candidate"},
            (candidate,),
            "candidate",
        ),
    )

    assert not comparison.passed
    assert not comparison.as_dict()["gates"]["hard_checks"]["candidate_quality_probes"]


def test_absolute_quality_rejects_excessive_quarantine_volume():
    acceptable = _clean_quality()
    excessive = replace(
        acceptable,
        quarantined_anchor_count=3,
        quarantine_anchor_ratio=0.3,
    )

    accepted = validate_quality(
        QualityReport("corpus-quality-v1", {"PZO12001": "run"}, (acceptable,), "ok")
    )
    rejected = validate_quality(
        QualityReport("corpus-quality-v1", {"PZO12001": "run"}, (excessive,), "bad")
    )

    assert accepted["passed"]
    assert not rejected["passed"]
    assert not rejected["checks"]["quarantine_bound"]
