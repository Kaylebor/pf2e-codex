"""Deterministic normalization and Foundry-candidate tests."""

from __future__ import annotations

from pf2e_codex.licensed_coverage import (
    FoundryMatcher,
    FoundryRow,
    FoundrySnapshot,
    comparison_metrics,
    duplicate_identity,
    normalize_heading,
    normalize_rule_text,
)


def _row(
    chunk_id: str,
    *,
    name: str = "Cast a Spell",
    text: str = "Spend ◆ to deal 2d6 fire damage within 30 feet.",
) -> FoundryRow:
    from pf2e_codex.licensed_coverage import normalized_hash

    return FoundryRow(
        chunk_id=chunk_id,
        name=name,
        text=text,
        source_hash=(chunk_id.encode().hex() + "0" * 64)[:64],
        normalized_hash=normalized_hash(text),
        heading_hash=normalized_hash(name, heading=True),
        publication_title="Pathfinder Player Core",
        license="ORC",
        era="remaster",
        type="action",
    )


def test_normalization_preserves_mechanical_action_and_number_differences():
    assert normalize_heading("  Cast—A SPELl  ") == "cast a spell"
    assert normalize_rule_text("Source: Spend ◆ to deal 2d6 damage.") == (
        "spend action to deal 2d6 damage"
    )
    assert normalize_rule_text("Spend ◆◆ to deal 2d6 damage.") != normalize_rule_text(
        "Spend ◆ to deal 2d6 damage."
    )
    assert normalize_rule_text("Deal 2d6 damage.") != normalize_rule_text(
        "Deal 3d6 damage."
    )
    assert normalize_rule_text("Deal 1d6+2 damage.") != normalize_rule_text(
        "Deal 1d6-2 damage."
    )
    assert normalize_rule_text("Gain a +2 bonus.") != normalize_rule_text(
        "Take a -2 penalty."
    )
    assert normalize_rule_text("Deal 2 × (1d6 + 1) damage.") != normalize_rule_text(
        "Deal 2 × 1d6 + 1 damage."
    )


def test_duplicate_identity_is_strict_about_heading_license_and_era():
    base = duplicate_identity(
        heading="Step", text="Move 5 feet.", license_name="ORC", era="remaster"
    )
    assert base == duplicate_identity(
        heading="STEP", text="Move 5 feet!", license_name="ORC", era="remaster"
    )
    assert base != duplicate_identity(
        heading="Stride", text="Move 5 feet.", license_name="ORC", era="remaster"
    )
    assert base != duplicate_identity(
        heading="Step", text="Move 5 feet.", license_name="OGL", era="remaster"
    )
    assert base != duplicate_identity(
        heading="Step", text="Move 5 feet.", license_name="ORC", era="legacy"
    )


def test_foundry_candidate_ranking_is_stable_and_scope_bounded():
    rows = (
        _row("rules:z", text="Spend ◆ to deal 3d6 fire damage within 30 feet."),
        _row("rules:a"),
        _row("rules:wrong-era"),
    )
    wrong_era = FoundryRow(**{**rows[2].__dict__, "era": "legacy"})
    section = {
        "heading": "Cast a Spell",
        "source_text": "Spend ◆ to deal 2d6 fire damage within 30 feet.",
        "publication_title": "Pathfinder Player Core",
        "license": "ORC",
        "era": "remaster",
    }
    first = FoundryMatcher(FoundrySnapshot("test", "a" * 64, (rows[0], rows[1], wrong_era)))
    second = FoundryMatcher(FoundrySnapshot("test", "b" * 64, (wrong_era, rows[1], rows[0])))

    assert [item["foundry_id"] for item in first.candidates(section)] == [
        item["foundry_id"] for item in second.candidates(section)
    ]
    assert first.candidates(section)[0]["foundry_id"] == "rules:a"
    assert all(item["era"] == "remaster" for item in first.candidates(section))
    assert len(first.candidates(section)) <= 3


def test_foundry_structured_wrapper_exposes_exact_description_identity():
    description = "Spend ◆ to deal 2d6 fire damage within 30 feet."
    wrapped = (
        "action: Cast a Spell (cast-a-spell)\n"
        "Action: 1 action\nTraits: concentrate\n\n"
        f"Description:\n{description}\n\n"
        "Source: Pathfinder Player Core (ORC)"
    )
    matcher = FoundryMatcher(
        FoundrySnapshot("test", "a" * 64, (_row("rules:wrapped", text=wrapped),))
    )
    section = {
        "heading": "Cast a Spell",
        "source_text": f"Cast a Spell\n{description}",
        "publication_title": "Pathfinder Player Core",
        "license": "ORC",
        "era": "remaster",
    }

    candidate = matcher.candidates(section)[0]

    assert candidate["metrics"]["exact_identity"] is True


def test_comparison_metrics_fail_closed_on_partial_numeric_coverage():
    metrics = comparison_metrics(
        "Deal 2d6 fire damage at 30 feet and 1 persistent damage.",
        ["Deal 2d6 fire damage at 30 feet."],
    )
    assert metrics["numeric_coverage"] < 1.0
    assert metrics["token_coverage"] < 1.0
