#!/usr/bin/env python3
"""Test merging Spanish Babele translations into English compendium entries.

Usage:
    uv run python scripts/test-es-merge.py
    uv run python scripts/test-es-merge.py --pack spells
"""

import argparse, json, sys, zipfile
from pathlib import Path
from collections import Counter


def load_english_packs(cache_dir: Path) -> dict[str, list[dict]]:
    packs = {}
    for f in sorted(cache_dir.glob("*.json")):
        if f.name.endswith(".json") and not f.name.startswith("_"):
            try:
                data = json.loads(f.read_text())
                if isinstance(data, list):
                    packs[f.stem] = data
            except (json.JSONDecodeError, OSError):
                pass
    return packs


def load_spanish_translations(zip_path: Path) -> dict[str, dict]:
    es = {}
    if not zip_path.exists():
        return es
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if "compendium/" in name and name.endswith(".json"):
                pack_name = name.split("/")[-1].replace(".json", "")
                try:
                    es[pack_name] = json.loads(z.read(name))
                except (json.JSONDecodeError, OSError):
                    pass
    return es


def build_pack_mapping(es_packs: dict, en_packs: dict) -> dict[str, str]:
    """Map English pack name -> Spanish Babele pack name."""
    es_name_map = {}

    for es_name in es_packs:
        base = es_name[5:] if es_name.startswith("pf2e.") else es_name
        if base.startswith("pf2e-"):
            base = base[5:]

        # Try base name
        candidates = [base]
        if base.endswith("-srd") and base[:-4] not in candidates:
            candidates.append(base[:-4])
        if base.endswith("-rd") and base[:-3] not in candidates:
            candidates.append(base[:-3])

        for c in candidates:
            if c in en_packs:
                es_name_map[c] = es_name

    # Fuzzy fallback for remaining unmatched packs
    matched_so_far = set(es_name_map.values())
    for es_name in es_packs:
        if es_name in matched_so_far:
            continue
        base = es_name[5:] if es_name.startswith("pf2e.") else es_name
        if base.startswith("pf2e-"):
            base = base[5:]

        norm_es = base.replace("-", "").replace("_", "").lower()
        for en_name in en_packs:
            if en_name in es_name_map:
                continue
            norm_en = en_name.replace("-", "").replace("_", "").lower()
            if norm_es == norm_en:
                es_name_map[en_name] = es_name
                break
            # Check substring: lost-omens prefix etc.
            if len(norm_es) > 8 and (norm_es in norm_en or norm_en in norm_es):
                es_name_map[en_name] = es_name
                break

    return es_name_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", help="Single pack to test")
    parser.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "pf2e-codex" / "extract-pf2e-8.2.0"))
    parser.add_argument("--es-zip", default="/tmp/pf2e-es.zip")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    es_zip = Path(args.es_zip)

    if not cache_dir.exists():
        alt_dirs = sorted(Path.home().glob(".cache/pf2e-codex/extract-*"))
        if alt_dirs:
            cache_dir = alt_dirs[-1]
            print(f"Using cache: {cache_dir}")
        else:
            print("No EN cache found. Run 'pf2e-codex fetch' first.")
            return

    print(f"Loading English packs from {cache_dir}...")
    en_packs = load_english_packs(cache_dir)
    print(f"  Found {len(en_packs)} packs")

    print(f"Loading Spanish translations from {es_zip}...")
    es_packs = load_spanish_translations(es_zip)
    print(f"  Found {len(es_packs)} packs")

    # Build mapping
    es_name_map = build_pack_mapping(es_packs, en_packs)
    print(f"  Matched {len(es_name_map)} packs\n")

    # Which packs to test
    packs_to_test = [args.pack] if args.pack else sorted(es_name_map.keys())

    total_merged = 0
    total_en_only = 0

    for en_name in packs_to_test:
        en_entries = en_packs.get(en_name, [])
        es_full = es_name_map.get(en_name)
        if not es_full:
            print(f"[SKIP] {en_name}: no ES mapping")
            continue

        es_data = es_packs.get(es_full, {})
        es_by_name = {k: v for k, v in es_data.get("entries", {}).items()}

        merged_count = 0
        en_only_count = 0
        for entry in en_entries:
            en_entry_name = entry.get("name", "")
            if en_entry_name in es_by_name:
                entry["name_es"] = es_by_name[en_entry_name].get("name", en_entry_name)
                entry["text_es"] = es_by_name[en_entry_name].get("description", "")
                merged_count += 1
            else:
                entry["name_es"] = None
                entry["text_es"] = None
                en_only_count += 1

        total_merged += merged_count
        total_en_only += en_only_count
        pct = merged_count / (merged_count + en_only_count) * 100 if (merged_count + en_only_count) > 0 else 0
        print(f"[{en_name:30s}] {merged_count:4d} ES + {en_only_count:4d} EN-only = {pct:5.1f}%")

    print(f"\n{'='*50}")
    total = total_merged + total_en_only
    print(f"TOTAL: {total_merged} ES + {total_en_only} EN-only across {len(packs_to_test)} packs")
    if total > 0:
        print(f"Overall coverage: {total_merged/total*100:.1f}%")

    # Show samples from first matched pack
    if packs_to_test:
        first = packs_to_test[0]
        en_entries = en_packs.get(first, [])
        es_full = es_name_map.get(first)
        es_data = es_packs.get(es_full, {})
        es_by_name = {k: v for k, v in es_data.get("entries", {}).items()}

        print(f"\nSamples from '{first}':")
        seen_en_only = 0
        seen_merged = 0
        for entry in en_entries:
            name = entry.get("name", "")
            if name in es_by_name and seen_merged < 3:
                es = es_by_name[name]
                en_name = entry.get("name", "")
                es_name = es.get("name", "")
                en_desc = (entry.get("system", {}).get("description", {}).get("value", "") or "")[:100]
                es_desc = (es.get("description", "") or "")[:100]
                print(f"\n  ES: {es_name:30s} EN: {en_name}")
                print(f"    EN desc: {en_desc}...")
                print(f"    ES desc: {es_desc}...")
                seen_merged += 1
            elif name not in es_by_name and seen_en_only < 2:
                en_desc = (entry.get("system", {}).get("description", {}).get("value", "") or "")[:100]
                print(f"\n  [EN-only] {name}")
                print(f"    desc: {en_desc}...")
                seen_en_only += 1
            if seen_merged >= 3 and seen_en_only >= 2:
                break


if __name__ == "__main__":
    main()
