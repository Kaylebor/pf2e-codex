"""Chunker tests — verify rule element flattening, UUID resolution, expression simplification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def entry_offguard() -> dict:
    with open(FIXTURES / "condition_off-guard.json") as f:
        return json.load(f)


@pytest.fixture
def entry_fury() -> dict:
    with open(FIXTURES / "feat_fury_instinct.json") as f:
        return json.load(f)


@pytest.fixture
def entry_flatmodifier() -> dict:
    with open(FIXTURES / "rule_flatmodifier.json") as f:
        return json.load(f)


def _chunk(entry: dict, pack: str = "test") -> list[dict]:
    """Run ChunkBuilder on an entry and return chunks."""
    from pf2e_codex.chunker import ChunkBuilder, UUIDResolver
    resolver = UUIDResolver({pack: [entry]})
    builder = ChunkBuilder(resolver)
    return list(builder.build_all(entry, pack))


class TestChunkerBasics:
    """Basic chunk structure."""

    def test_offguard_has_expected_fields(self, entry_offguard):
        chunks = _chunk(entry_offguard)
        assert len(chunks) >= 1
        chunk = chunks[0]
        assert "Off-Guard" in chunk["name"]
        assert chunk["type"] == "condition"
        assert chunk["pack"] == "test"
        assert chunk.get("source_hash")

    def test_entry_hash_stable(self, entry_fury):
        from pf2e_codex.chunker import entry_hash
        h1 = entry_hash(entry_fury)
        h2 = entry_hash(entry_fury)
        assert h1 == h2


class TestRuleFlatteners:
    """Rule element flattening produces expected English text."""

    def test_flatmodifier_present(self, entry_flatmodifier):
        chunks = _chunk(entry_flatmodifier)
        text = chunks[0]["text"]
        # Should contain the flat modifier value
        assert any(str(v) in text for v in (2, -1, -2, 1, 4, 5, 6))

    def test_offguard_text_nonempty(self, entry_offguard):
        chunks = _chunk(entry_offguard)
        assert len(chunks[0]["text"]) > 50


class TestExpressionSimplification:
    """Ternaries and actor variables are simplified."""

    def test_ternary_simplified(self, entry_fury):
        chunks = _chunk(entry_fury)
        text = chunks[0]["text"]
        # Ternary expressions should not appear raw
        assert "ternary(" not in text


class TestUUIDResolution:
    """@UUID references are resolved to readable names."""

    def test_uuids_not_raw(self, entry_fury):
        chunks = _chunk(entry_fury)
        text = chunks[0]["text"]
        # Should not contain raw UUID brackets
        assert "@UUID[" not in text


class TestCrossPackChunking:
    """Same entry in different packs produces distinct chunk IDs."""

    def test_pack_prefixed_ids(self, entry_offguard):
        c1 = _chunk(entry_offguard, "pack_a")
        c2 = _chunk(entry_offguard, "pack_b")
        assert c1[0]["id"] != c2[0]["id"]
        assert "pack_a:" in c1[0]["id"]
        assert "pack_b:" in c2[0]["id"]
