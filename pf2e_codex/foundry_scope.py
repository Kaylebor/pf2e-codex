"""Redistribution scope for Foundry PF2E entries."""

from __future__ import annotations

from typing import Any

REDISTRIBUTABLE_FOUNDRY_PUBLICATIONS = {
    "Pathfinder Core Rulebook": "PZO2101",
    "Pathfinder Player Core": "PZO12001",
    "Pathfinder GM Core": "PZO12002",
    "Pathfinder Monster Core": "PZO12003",
    "Pathfinder Player Core 2": "PZO12004",
}
REDISTRIBUTABLE_LICENSES = {"OGL", "ORC"}


def owning_publication(entry: dict[str, Any]) -> dict[str, Any]:
    """Return only the entry-level owning publication, never nested citations."""
    system = entry.get("system")
    if not isinstance(system, dict):
        return {}
    publication = system.get("publication")
    if isinstance(publication, dict) and publication.get("title"):
        return publication
    details = system.get("details")
    if isinstance(details, dict):
        publication = details.get("publication")
        if isinstance(publication, dict) and publication.get("title"):
            return publication
    return {}


def is_redistributable_foundry_entry(entry: dict[str, Any]) -> bool:
    """Fail closed unless owning title and declared license are both approved."""
    publication = owning_publication(entry)
    return (
        publication.get("title") in REDISTRIBUTABLE_FOUNDRY_PUBLICATIONS
        and publication.get("license") in REDISTRIBUTABLE_LICENSES
    )


__all__ = [
    "REDISTRIBUTABLE_FOUNDRY_PUBLICATIONS",
    "REDISTRIBUTABLE_LICENSES",
    "is_redistributable_foundry_entry",
    "owning_publication",
]
