"""Foundry publication ownership and clean-seed scope tests."""

from __future__ import annotations

from pf2e_codex.chunker import ChunkBuilder, UUIDResolver
from pf2e_codex.foundry_scope import (
    is_redistributable_foundry_entry,
    owning_publication,
)


def _entry(publication: dict, *, nested: dict | None = None) -> dict:
    system = {
        "description": {"value": "Rules text."},
        "publication": publication,
        "rules": [],
        "traits": {"value": []},
    }
    if nested is not None:
        system["spell"] = {"system": {"publication": nested}}
    return {"_id": "one", "name": "Rule", "type": "feat", "system": system}


def test_clean_scope_requires_allowlisted_owning_title_and_license():
    assert is_redistributable_foundry_entry(
        _entry({"title": "Pathfinder Player Core", "license": "ORC"})
    )
    assert not is_redistributable_foundry_entry(
        _entry({"title": "Pathfinder Lost Omens: Example", "license": "ORC"})
    )
    assert not is_redistributable_foundry_entry(
        _entry({"title": "Pathfinder Player Core", "license": "NONE"})
    )


def test_details_publication_is_the_only_fallback_and_nested_citations_are_ignored():
    fallback = _entry({})
    fallback["system"]["details"] = {
        "publication": {"title": "Pathfinder Monster Core", "license": "ORC"}
    }
    assert owning_publication(fallback)["title"] == "Pathfinder Monster Core"
    assert is_redistributable_foundry_entry(fallback)

    nested_only = _entry(
        {},
        nested={"title": "Pathfinder Player Core", "license": "ORC"},
    )
    assert owning_publication(nested_only) == {}
    assert not is_redistributable_foundry_entry(nested_only)


def test_chunker_persists_details_publication_for_auditing():
    entry = _entry({})
    entry["system"]["details"] = {
        "publication": {
            "title": "Pathfinder GM Core",
            "license": "ORC",
            "remaster": True,
        }
    }

    chunk = ChunkBuilder(UUIDResolver({})).build_all(entry, "feats")[0]

    assert chunk["publication_title"] == "Pathfinder GM Core"
    assert chunk["license"] == "ORC"
    assert chunk["remaster"] is True
