"""Download and extract PF2E json-assets from GitHub releases."""

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from .config import Settings


def get_latest_release() -> str:
    """Detect the latest PF2E system release tag from GitHub API."""
    import json
    from urllib.request import urlopen, Request
    req = Request(
        "https://api.github.com/repos/foundryvtt/pf2e/releases/latest",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "pf2e-codex/0.1.0"},
    )
    with urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())  # type: ignore[no-untyped-call]
    return data["tag_name"]

CORE_PACKS = [
    "conditions",
    "actions",
    "classes",
    "feats",
    "spells",
    "ancestries",
    "heritages",
    "backgrounds",
    "class-features",
    "deities",
    "hazards",
    "vehicles",
    "familiar-abilities",
    "equipment-effects",
    "feat-effects",
    "spell-effects",
    "other-effects",
    "campaign-effects",
    "bestiary-ability-glossary-srd",
    "bestiary-family-ability-glossary",
    "rollable-tables",
    "journals",
    "pathfinder-dark-archive",
    "pathfinder-society-boons",
    "boons-and-curses",
    "ancestry-features",
    "kingmaker-features",
    "standalone-adventures",
    "adventure-specific-actions",
    "macros",
    "action-macros",
    "iconics",
    "npc-gallery",
    "pathfinder-bestiary",
    "pathfinder-bestiary-2",
    "pathfinder-bestiary-3",
    "pathfinder-monster-core",
    "pathfinder-monster-core-2",
    "pathfinder-npc-core",
    "blog-bestiary",
    "menace-under-otari-bestiary",
]


def get_cached_zip(settings: Settings) -> Path:
    """Download json-assets.zip or return cached copy."""
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = settings.cache_dir / f"json-assets-{settings.release}.zip"
    if zip_path.exists():
        return zip_path
    url = settings.github_release_url
    print(f"Downloading {url} …")
    with urlopen(url) as resp:  # noqa: S310
        data = resp.read()
    zip_path.write_bytes(data)
    print(f"Saved {len(data):,} bytes -> {zip_path}")
    return zip_path


def extract_all_packs(zip_path: Path, dest: Path) -> dict[str, list[dict[str, Any]]]:
    """Extract and load all pack JSONs. Returns pack_name -> entries."""
    dest.mkdir(parents=True, exist_ok=True)
    all_entries: dict[str, list[dict[str, Any]]] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        pack_files = [
            m
            for m in members
            if m.startswith("packs/") and m.endswith(".json") and "_folders" not in m
        ]
        for member in pack_files:
            pack_name = Path(member).stem
            zf.extract(member, dest.parent)
            extracted = dest.parent / member
            if extracted.exists():
                target = dest / f"{pack_name}.json"
                shutil.move(str(extracted), str(target))
                try:
                    all_entries[pack_name] = json.loads(target.read_text())
                except Exception as e:  # noqa: BLE001
                    print(f"  Failed to load {pack_name}: {e}")
    return all_entries


def extract_lang(zip_path: Path, dest: Path) -> dict[str, Any]:
    """Extract and load the English localization file."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract("lang/en.json", dest.parent)
        extracted = dest.parent / "lang" / "en.json"
        target = dest / "en.json"
        shutil.move(str(extracted), str(target))
        return json.loads(target.read_text())
