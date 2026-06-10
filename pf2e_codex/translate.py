"""Language translation merging for multilingual compendium support.

Fetches Babele-format translation modules from community projects
and merges translated name/text into English compendium entries.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

logger = logging.getLogger(__name__)

# Spanish translation module
ES_MODULE_URL = (
    "https://github.com/HonzoNebro/pf2e-esp-translation-ml/releases/download/v8.1.30/"
    "pf2e-esp-translation-ml.zip"
)
ES_MODULE_CACHE = "pf2e-esp-translation-ml.zip"


def fetch_translations(cache_dir: Path, lang: str = "es") -> dict[str, dict[str, Any]]:
    """Download and load a Babele-format translation module.

    Returns dict mapping pack_name -> Babele data {label, entries, mapping}.
    """
    if lang != "es":
        raise ValueError(f"Unsupported language: {lang}")

    zip_path = cache_dir / ES_MODULE_CACHE
    if not zip_path.exists():
        print(f"Downloading {lang} translations from {ES_MODULE_URL}")
        with urlopen(ES_MODULE_URL) as resp:  # noqa: S310
            data = resp.read()
        zip_path.write_bytes(data)
        print(f"  Saved {len(data):,} bytes -> {zip_path}")

    return _load_babele_translations(zip_path)


def _load_babele_translations(zip_path: Path) -> dict[str, dict[str, Any]]:
    """Load all Babele translation files from a module zip."""
    translations: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if "compendium/" in name and name.endswith(".json"):
                pack_name = name.split("/")[-1].replace(".json", "")
                try:
                    data = json.loads(z.read(name))
                    # Babele format: {label, entries: {name: {name, description, ...}}, mapping}
                    translations[pack_name] = data
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Failed to load {name}: {e}")
    return translations


def build_pack_map(translations: dict[str, dict], en_pack_names: set[str]) -> dict[str, str]:
    """Map English pack names -> Babele pack names.

    Handles pf2e.prefix, -srd suffixes, Animal Companions, and other
    naming differences between json-assets and Babele.
    """
    es_map: dict[str, str] = {}

    for es_name in translations:
        base = es_name[5:] if es_name.startswith("pf2e.") else es_name
        if base.startswith("pf2e-"):
            base = base[5:]

        candidates = [base]
        if base.endswith("-srd") and base[:-4] not in candidates:
            candidates.append(base[:-4])

        for c in candidates:
            if c in en_pack_names:
                es_map[c] = es_name

    # Fuzzy match remaining (normalized comparison)
    matched = set(es_map.values())
    for es_name in translations:
        if es_name in matched:
            continue
        base = es_name[5:] if es_name.startswith("pf2e.") else es_name
        if base.startswith("pf2e-"):
            base = base[5:]
        norm_es = base.replace("-", "").replace("_", "").lower()
        for en_name in en_pack_names:
            if en_name in es_map:
                continue
            norm_en = en_name.replace("-", "").replace("_", "").lower()
            if norm_es == norm_en:
                es_map[en_name] = es_name
                break
            if len(norm_es) > 8 and norm_es in norm_en:
                es_map[en_name] = es_name
                break

    return es_map


def merge_entries(
    en_entries: list[dict[str, Any]],
    babele_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Merge Spanish translations into English entries for one pack.

    Each EN entry gets a ``translations`` dict if a Spanish match is found:
        entry["translations"] = {"es": {"name": "...", "text": "..."}}

    Returns entries with translations added in-place.
    """
    es_by_name: dict[str, dict] = {}
    for entry_name, fields in babele_data.get("entries", {}).items():
        es_by_name[entry_name] = fields

    for entry in en_entries:
        en_name = entry.get("name", "")
        if en_name in es_by_name:
            es_fields = es_by_name[en_name]
            entry.setdefault("translations", {})["es"] = {
                "name": es_fields.get("name", en_name),
                "text": _strip_html(es_fields.get("description", "")),
            }
        else:
            entry.setdefault("translations", {})["es"] = None

    return en_entries


def _strip_html(text: str) -> str:
    """Remove HTML tags from translated description text."""
    import re as _re
    text = _re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&apos;", "'").replace("&quot;", '"')
    return text.strip()
