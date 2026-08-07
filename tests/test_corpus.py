"""Synthetic tests for catalog discovery and native rulebook parsing."""

from __future__ import annotations

import calendar
import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest

from pf2e_codex import corpus


def _word(text: str, *, x: float, top: float, size: float = 9, font: str = "Body"):
    width = max(8.0, len(text) * 5.0)
    return {
        "text": text,
        "x0": x,
        "top": top,
        "x1": x + width,
        "bottom": top + size,
        "font": font,
        "size": size,
        "upright": True,
        "direction": "ltr",
    }


def _page(number: int, lines: list[tuple[str, float, float, str, float]]):
    words = []
    for text, x, top, font, size in lines:
        words.append(_word(text, x=x, top=top, font=font, size=size))
    return {"number": number, "width": 600.0, "height": 800.0, "words": words, "images": []}


def _export_payload(filename: str, text: str, *, email: str = ""):
    lines = [
        _word("Pathfinder Player Core", x=30, top=12, size=8),
        _word(str(1), x=290, top=770, size=8),
        _word("Chapter One", x=40, top=100, size=16, font="Heading-Bold"),
        _word(text, x=40, top=130, size=9),
    ]
    if email:
        lines.append(_word(email, x=40, top=740, size=7))
    return {
        "schema_version": 1,
        "extractor": {"name": "synthetic", "profile_version": 1},
        "source": {
            "filename": filename,
            "sha256": hashlib.sha256(filename.encode()).hexdigest(),
            "page_count": 1,
        },
        "pages": [
            {
                "number": 1,
                "width": 600,
                "height": 800,
                "words": lines,
                "images": [],
            }
        ],
    }


def test_discover_catalog_pdfs_and_zip_members(tmp_path):
    source_dir = tmp_path / "nested" / "sources"
    source_dir.mkdir(parents=True)
    (source_dir / "PZO2101E-4th Printing.pdf").write_bytes(b"combined")
    (source_dir / "PZO2101E-1.pdf").write_bytes(b"part one")
    archive = source_dir / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("books/PZO12001E_Part_1.pdf", b"zip part")
        bundle.writestr("books/not-a-catalog.pdf", b"ignore")

    found = corpus.discover_sources(tmp_path)

    assert {(item.product.code, item.part, item.combined, item.member) for item in found} == {
        ("PZO2101", None, True, None),
        ("PZO2101", "1", False, None),
        ("PZO12001", "1", False, "books/PZO12001E_Part_1.pdf"),
    }


def test_zip_member_directory_is_not_retained_in_source_or_provenance(
    tmp_path, monkeypatch
):
    archive = tmp_path / "buyer@example.invalid.zip"
    member = "buyer@example.invalid/private/PZO12001E.pdf"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(member, b"rules")

    found = corpus.discover_sources(tmp_path)
    source = found[0]
    assert source.member == member  # exact path remains available internally
    assert source.source_name == "PZO12001E.pdf"
    assert source.display_name == "PZO12001E.pdf"

    original_fingerprint = corpus._candidate_content_fingerprint
    monkeypatch.setattr(
        corpus,
        "_candidate_content_fingerprint",
        lambda candidate, _root: candidate.source_sha256,
    )
    corpus.select_revisions(found, state_root=tmp_path)
    state_text = (tmp_path / corpus.SELECTION_STATE_FILENAME).read_text()
    assert "buyer@example.invalid" not in state_text

    export_path = tmp_path / "native.json"
    export_path.write_text(
        json.dumps(_export_payload(source.source_name, "Rules text")), encoding="utf-8"
    )
    chunks = corpus.parse_rulebook_export(export_path, source=source)
    serialized = json.dumps(chunks)
    assert "buyer@example.invalid" not in serialized
    assert chunks[0]["provenance"]["source_path"] == "PZO12001E.pdf"

    monkeypatch.setattr(corpus, "_candidate_content_fingerprint", original_fingerprint)
    monkeypatch.setattr(corpus, "_artifact_current", lambda *_args: True)
    monkeypatch.setattr(corpus, "parse_rulebook_export", lambda *_args, **_kwargs: [])
    with pytest.raises(ValueError) as error:
        corpus._candidate_content_fingerprint(source, tmp_path / "cache")
    assert "buyer@example.invalid" not in str(error.value)


def test_zip_member_mtime_selects_newer_revision_within_one_archive(tmp_path):
    archive = tmp_path / "revisions.zip"
    old_name = "PZO12001E-a.pdf"
    new_name = "PZO12001E-z.pdf"
    with zipfile.ZipFile(archive, "w") as bundle:
        old_info = zipfile.ZipInfo(old_name, date_time=(2024, 1, 2, 3, 4, 4))
        new_info = zipfile.ZipInfo(new_name, date_time=(2025, 6, 7, 8, 9, 10))
        bundle.writestr(old_info, b"old rules")
        bundle.writestr(new_info, b"new rules")
    # Make the container mtime newer than either member.  Selection must use
    # each member's complete internal timestamp rather than flattening both to
    # the outer archive's mtime.
    os.utime(archive, ns=(2_000_000_000_000_000_000,) * 2)

    found = corpus.discover_sources(tmp_path)
    by_name = {source.source_name: source for source in found}
    assert by_name[old_name].mtime_ns == calendar.timegm((2024, 1, 2, 3, 4, 4, 0)) * 1_000_000_000
    assert by_name[new_name].mtime_ns == calendar.timegm((2025, 6, 7, 8, 9, 10, 0)) * 1_000_000_000

    selected = corpus.select_revisions(found)
    assert selected[0].sources[0].source_name == new_name


def test_discover_real_paizo_split_page_range_names(tmp_path):
    archive = tmp_path / "chapters-randomized-name.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("PZO12001 Player Core 001-027.pdf", b"player")
        bundle.writestr("PZO12002 GM Core 000-Cover.pdf", b"cover")
        bundle.writestr("PZO12003 Monster Core 001-017.pdf", b"monster")
        bundle.writestr("PZO12004 Player Core 2 001-019.pdf", b"player two")
        bundle.writestr("PZO2101 032-065 Ancestries-4th Printing.pdf", b"legacy")

    found = corpus.discover_sources(tmp_path)

    assert {(item.product.code, item.part, item.printing) for item in found} == {
        ("PZO12001", "001-027", None),
        ("PZO12002", "000-cover", None),
        ("PZO12003", "001-017", None),
        ("PZO12004", "001-019", None),
        ("PZO2101", "032-065", 4),
    }
    assert all(not item.combined for item in found)


def test_select_combined_authoritative_and_persists_state(tmp_path, monkeypatch):
    paths = [
        ("PZO2101E-1.pdf", b"old split", 1),
        ("PZO2101E-2.pdf", b"old split 2", 2),
        ("PZO2101E-2nd Printing.pdf", b"new combined", 3),
        ("PZO2101E-4th Printing.pdf", b"latest combined", 4),
    ]
    for name, value, _mtime in paths:
        path = tmp_path / name
        path.write_bytes(value)
    monkeypatch.setattr(
        corpus,
        "_candidate_content_fingerprint",
        lambda source, _root: source.source_sha256,
    )
    found = corpus.discover_sources(tmp_path)
    selected = corpus.select_revisions(found, state_root=tmp_path)

    assert len(selected) == 1
    assert selected[0].combined
    assert selected[0].sources[0].source_name == "PZO2101E-4th Printing.pdf"
    state = json.loads((tmp_path / corpus.SELECTION_STATE_FILENAME).read_text())
    assert state["selections"]["PZO2101"]["candidate_ids"] == [
        selected[0].sources[0].candidate_id
    ]


def test_persisted_equivalent_choice_survives_mtime_but_new_printing_wins(
    tmp_path, monkeypatch
):
    first = tmp_path / "PZO12001E-old.pdf"
    second = tmp_path / "PZO12001E-copy.pdf"
    first.write_bytes(b"same rules, watermark one")
    second.write_bytes(b"same rules, watermark two")
    os.utime(first, ns=(1_000, 1_000))
    os.utime(second, ns=(2_000, 2_000))
    monkeypatch.setattr(
        corpus,
        "_candidate_content_fingerprint",
        lambda source, _root: (
            "new-printing" if "2nd Printing" in source.source_name else "same-rules"
        ),
    )
    selected = corpus.select_revisions(corpus.discover_sources(tmp_path), state_root=tmp_path)
    chosen = selected[0].sources[0].source_name

    # A later copy/restore does not silently change an already persisted
    # choice when its normalized rules are equivalent.
    changed = first if chosen == second.name else second
    os.utime(changed, ns=(3_000, 3_000))
    selected_again = corpus.select_revisions(
        corpus.discover_sources(tmp_path), state_root=tmp_path
    )
    assert selected_again[0].sources[0].source_name == chosen

    newest = tmp_path / "PZO12001E-2nd Printing.pdf"
    newest.write_bytes(b"explicit second printing")
    selected_new_printing = corpus.select_revisions(
        corpus.discover_sources(tmp_path), state_root=tmp_path
    )
    assert selected_new_printing[0].sources[0].source_name == newest.name


def test_same_printing_errata_beats_persisted_old_content_but_equivalent_copy_does_not(
    tmp_path, monkeypatch
):
    fingerprints = {
        b"old rules": "old-content",
        b"corrected rules": "corrected-content",
        b"corrected rules, new watermark": "corrected-content",
    }
    monkeypatch.setattr(
        corpus,
        "_candidate_content_fingerprint",
        lambda source, _root: fingerprints[source.path.read_bytes()],
    )

    old = tmp_path / "PZO12001E-old.pdf"
    old.write_bytes(b"old rules")
    os.utime(old, ns=(1_000, 1_000))
    first = corpus.select_revisions(corpus.discover_sources(tmp_path), state_root=tmp_path)
    assert first[0].sources[0].source_name == old.name

    corrected = tmp_path / "PZO12001E-corrected.pdf"
    corrected.write_bytes(b"corrected rules")
    os.utime(corrected, ns=(2_000, 2_000))
    second = corpus.select_revisions(corpus.discover_sources(tmp_path), state_root=tmp_path)
    assert second[0].sources[0].source_name == corrected.name

    equivalent = tmp_path / "PZO12001E-redownload.pdf"
    equivalent.write_bytes(b"corrected rules, new watermark")
    os.utime(equivalent, ns=(3_000, 3_000))
    third = corpus.select_revisions(corpus.discover_sources(tmp_path), state_root=tmp_path)
    assert third[0].sources[0].source_name == corrected.name

    state = json.loads((tmp_path / corpus.SELECTION_STATE_FILENAME).read_text())
    assert state["selections"]["PZO12001"]["content_fingerprints"] == [
        "corrected-content"
    ]


def test_include_exclude_and_prefer_are_applied(tmp_path):
    (tmp_path / "PZO2101E.pdf").write_bytes(b"legacy")
    (tmp_path / "PZO12001E.pdf").write_bytes(b"player")
    found = corpus.discover_sources(tmp_path)
    selected = corpus.select_revisions(
        found,
        include="player-core",
        exclude="PZO2101",
        prefer="PZO12001E.pdf",
    )
    assert [item.product.code for item in selected] == ["PZO12001"]


def test_prepare_exports_handles_zip_and_stale_artifact(tmp_path, monkeypatch):
    archive = tmp_path / "sources.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("PZO12002E.pdf", b"gm bytes")
    found = corpus.discover_sources(tmp_path)
    selected = corpus.select_revisions(found)
    calls: list[tuple[Path, Path, bool]] = []

    def fake_export(source, output, *, overwrite=False, **_kwargs):
        calls.append((source, output, overwrite))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source": {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
                }
            )
        )

    monkeypatch.setattr(corpus, "export_pdf", fake_export)
    first = corpus.prepare_exports(tmp_path, selected)
    second = corpus.prepare_exports(tmp_path, selected)

    assert first[0].exported and not first[0].stale
    assert not second[0].exported
    assert len(calls) == 1
    assert calls[0][0].parent.parent.name == "PZO12002"
    assert calls[0][0].name == "PZO12002E.pdf"
    assert calls[0][0].read_bytes() == b"gm bytes"


def test_parse_strips_repeated_furniture_and_email_and_is_watermark_stable(tmp_path):
    first = _export_payload("PZO12001E.pdf", "Rules text", email="buyer@example.invalid")
    second = _export_payload("PZO12001E.pdf", "Rules text", email="different@example.invalid")
    first["pages"].append(_page(2, [("Pathfinder Player Core", 30, 12, "Body", 8), ("2", 290, 770, "Body", 8), ("More rules", 40, 100, "Body", 9)]))
    second["pages"].append(_page(2, [("Pathfinder Player Core", 30, 12, "Body", 8), ("2", 290, 770, "Body", 8), ("More rules", 40, 100, "Body", 9)]))
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first))
    second_path.write_text(json.dumps(second))

    one = corpus.parse_rulebook_export(first_path)
    two = corpus.parse_rulebook_export(second_path)

    assert len(one) == 1
    assert one[0]["type"] == "rulebook_section"
    assert "buyer@example.invalid" not in one[0]["text"]
    assert "Pathfinder Player Core" not in one[0]["text"]
    assert one[0]["pages"] == [1, 2]
    assert one[0]["id"] == two[0]["id"]
    assert one[0]["section_hash"] == two[0]["section_hash"]
    assert one[0]["source"]["revision"] == two[0]["source"]["revision"]
    assert one[0]["license"] == "ORC"


def test_parse_uses_geometry_for_two_column_reading_order(tmp_path):
    payload = {
        "schema_version": 1,
        "extractor": {},
        "source": {"filename": "PZO12002E.pdf", "sha256": "hash"},
        "pages": [
            {
                "number": 1,
                "width": 600,
                "height": 800,
                "words": [
                    _word("Section", x=30, top=20, size=16, font="Heading-Bold"),
                    _word("Left first", x=30, top=100),
                    _word("Left second", x=30, top=120),
                    _word("Left third", x=30, top=140),
                    _word("Left fourth", x=30, top=160),
                    _word("Right first", x=330, top=100),
                    _word("Right second", x=330, top=120),
                    _word("Right third", x=330, top=140),
                    _word("Right fourth", x=330, top=160),
                ],
            }
        ],
    }
    path = tmp_path / "columns.json"
    path.write_text(json.dumps(payload))
    chunk = corpus.parse_rulebook_export(path)[0]
    assert chunk["text"].index("Left first") < chunk["text"].index("Right first")
    assert chunk["page_start"] == chunk["page_end"] == 1


def test_errata_changes_hash_not_page_anchored_id_and_chunk_contract(tmp_path):
    before = _export_payload("PZO12001E.pdf", "The original rule text")
    after = _export_payload("PZO12001E.pdf", "The corrected errata text")
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    before_path.write_text(json.dumps(before))
    after_path.write_text(json.dumps(after))

    old = next(
        chunk for chunk in corpus.parse_rulebook_export(before_path)
        if chunk["name"] == "Chapter One"
    )
    new = next(
        chunk for chunk in corpus.parse_rulebook_export(after_path)
        if chunk["name"] == "Chapter One"
    )

    assert old["id"] == new["id"]
    assert old["section_hash"] != new["section_hash"]
    assert old["origin"] == "corpus"
    assert old["source_id"] == "paizo:PZO12001:player-core"
    assert old["pack"] == "corpus-player-core"
    assert old["raw_rules_count"] == 0
    assert old["source_page_start"] == old["source_page_end"] == 1
    assert old["source"]["parser"] == "paizo-native-v1"


@pytest.mark.parametrize(
    "fixture_name",
    ["core-rulebook.json", "player-core.json", "gm-core.json"],
)
def test_sanitized_paizo_layout_golden_fixtures(tmp_path, fixture_name):
    fixture_path = Path(__file__).parent / "fixtures" / "corpus" / fixture_name
    fixture = json.loads(fixture_path.read_text())
    pages = []
    for page in fixture["pages"]:
        lines = [tuple(line) for line in page["lines"]]
        lines.append((page["printed"], 30, 770, "GoodOT", 8))
        pages.append(_page(page["number"], lines))
    payload = {
        "schema_version": 1,
        "extractor": {"name": "sanitized-paizo-layout", "profile_version": 1},
        "source": {
            "filename": fixture["source_filename"],
            "sha256": "local-watermarked-hash",
            "page_count": len(pages),
        },
        "pages": pages,
    }
    export_path = tmp_path / fixture_name
    export_path.write_text(json.dumps(payload))

    chunks = corpus.parse_rulebook_export(export_path)

    assert [chunk["name"] for chunk in chunks] == fixture["expected_names"]
    combined_text = " ".join(chunk["text"] for chunk in chunks)
    assert "example.invalid" not in combined_text
    assert fixture["pages"][0]["lines"][0][0] not in combined_text
    assert all(chunk["printed_page"] for chunk in chunks)
    if fixture_name == "core-rulebook.json":
        assert chunks[0]["pages"] == [12, 13]
        assert chunks[0]["printed_page"] == "10-11"
        assert "[one-action]" in chunks[0]["text"]
    if fixture_name == "player-core.json":
        assert "Outcome" in chunks[0]["text"]
        assert chunks[0]["text"].index("Track the dying") < chunks[0]["text"].index("Outcome")
