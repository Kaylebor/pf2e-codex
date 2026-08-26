"""Synthetic tests for catalog discovery and native rulebook parsing."""

from __future__ import annotations

import calendar
import hashlib
import json
import os
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from pf2e_codex import corpus, pdf_export


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


def _trusted_payload(pages: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "extractor": {
            "name": "pf2e-codex-native-pdf", "profile_version": 1,
            "backend": "pdfplumber", "backend_version": "test", "ocr": False,
        },
        "source": {
            "filename": "PZO12001E.pdf", "sha256": "a" * 64,
            "size": 1, "page_count": len(pages),
        },
        "selection": {"first_page": 1, "last_page": len(pages)},
        "pages": pages,
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


def test_native_inventory_semantics_ignore_geometry_furniture_watermarks_and_extraction_order():
    first_pages = [
        _page(1, [
            ("Player Core", 30, 12, "Body", 8), ("1", 290, 770, "Body", 8),
            ("Ada", 40, 700, "Body", 7), ("@", 65, 700, "Body", 7),
            ("example", 75, 700, "Body", 7), (".", 110, 700, "Body", 7),
            ("invalid", 115, 700, "Body", 7), ("Rules", 40, 100, "Heading-Bold", 16),
            ("Use", 40, 140, "Body", 9), ("an", 70, 140, "Body", 9), ("action", 90, 140, "Body", 9),
        ]),
        _page(2, [
            ("Player Core", 30, 12, "Body", 8), ("2", 290, 770, "Body", 8),
            ("Ada Smith Order 12345", 40, 700, "Body", 7),
            ("once", 40, 120, "Body", 9), ("each", 80, 120, "Body", 9), ("round", 120, 120, "Body", 9),
        ]),
    ]
    first = _trusted_payload(first_pages)
    changed = json.loads(json.dumps(first))
    # Harmless geometry noise and a reordered content stream do not alter the
    # semantic revision; the opaque inventory remains deterministic too.
    changed["pages"][0]["words"][7]["x0"] += 0.01
    changed["pages"][0]["words"] = list(reversed(changed["pages"][0]["words"]))
    one = pdf_export.native_word_inventory(first, "PZO12001", strict=True)
    two = pdf_export.native_word_inventory(changed, "PZO12001", strict=True)
    assert one.content_fingerprint == two.content_fingerprint
    assert one.anchors == two.anchors
    assert len(one.anchors) == 17  # quarantined watermark spans remain accounted for
    assert any(item["reason"] == "watermark-email-span-v1" for item in one.ignored_anchors)

    errata = json.loads(json.dumps(first))
    errata["pages"][1]["words"][-1]["text"] = "turn"
    assert pdf_export.native_word_inventory(errata, "PZO12001", strict=True).content_fingerprint != one.content_fingerprint


def test_native_inventory_quarantines_split_margin_watermark_but_not_body_syntax():
    payload = _trusted_payload([
        _page(1, [
            ("buyer", 20, 700, "Body", 7), ("@", 20, 710, "Body", 7),
            ("example", 20, 720, "Body", 7), (".invalid", 20, 730, "Body", 7),
            ("table", 40, 100, "Body", 9), ("header", 90, 100, "Body", 9),
            ("use", 40, 130, "Body", 9), ("@actor.level", 70, 130, "Body", 9),
        ]),
        _page(2, [
            ("table", 40, 100, "Body", 9), ("header", 90, 100, "Body", 9),
            ("continue", 40, 130, "Body", 9),
        ]),
    ])
    inventory = pdf_export.native_word_inventory(payload, "PZO12001", strict=True)
    annotated = pdf_export.annotate_native_words(payload, inventory)
    reasons = [word.get("_native_ignore_reason") for word in annotated["pages"][0]["words"]]
    assert reasons[:4] == ["watermark-email-span-v1"] * 4
    assert reasons[-1] is None
    assert all(word.get("_native_ignore_reason") is None for word in annotated["pages"][1]["words"])


def test_semantic_fingerprint_preserves_visual_word_order_but_not_geometry_noise():
    payload = _trusted_payload([_page(1, [
        ("first", 40, 100, "Body", 9), ("second", 90, 100, "Body", 9),
    ])])
    jittered = json.loads(json.dumps(payload))
    jittered["pages"][0]["words"][0]["x0"] += 0.01
    swapped = json.loads(json.dumps(payload))
    swapped["pages"][0]["words"][0]["text"] = "second"
    swapped["pages"][0]["words"][1]["text"] = "first"
    original = pdf_export.native_word_inventory(payload, "PZO12001", strict=True).content_fingerprint
    assert pdf_export.native_word_inventory(jittered, "PZO12001", strict=True).content_fingerprint == original
    assert pdf_export.native_word_inventory(swapped, "PZO12001", strict=True).content_fingerprint != original


def test_trusted_parse_bundle_has_complete_anchor_coverage_and_no_private_repr(tmp_path):
    pages = [
        _page(1, [("Rules", 40, 100, "Heading-Bold", 16), ("Use", 40, 140, "Body", 9)]),
        _page(2, [("it", 40, 120, "Body", 9), ("once", 70, 120, "Body", 9)]),
    ]
    payload = _trusted_payload(pages)
    untrusted = pdf_export.VerifiedNativeExport(
        payload=payload, product_code="PZO12001", source_basename="PZO12001E.pdf",
        page_count=2, extractor_profile_version=1, pdf_verified=False,
    )
    with pytest.raises(ValueError, match="PDF-verified"):
        corpus.parse_verified_native_export(untrusted)
    artifact = pdf_export.VerifiedNativeExport(
        payload=payload, product_code="PZO12001", source_basename="PZO12001E.pdf",
        page_count=2, extractor_profile_version=1, pdf_verified=True,
        attestation_digest="a" * 64, _verification_token=pdf_export._TRUSTED_PDF_ORIGIN,
        _payload_digest=pdf_export.trusted_payload_digest(payload),
    )
    bundle = corpus.parse_verified_native_export(artifact)
    assigned = [anchor for section in bundle.sections for anchor in section.coverage_anchors]
    expected = set(bundle.inventory.anchors) - set(bundle.inventory.ignored_anchor_reasons)
    assert set(assigned) == expected and len(assigned) == len(set(assigned))
    assert all(section.stable_section_identity for section in bundle.sections)
    assert all(corpus._TRUSTED_SOURCE_SECTION_ID_RE.fullmatch(section.source_section_id) for section in bundle.sections)
    bundle.verify_seal()
    assert "source_sha256" not in repr(bundle)
    assert "source_sha256" not in repr(bundle)
    with pytest.raises((AttributeError, TypeError)):
        bundle.artifact_attestation["x"] = True
    with pytest.raises((AttributeError, TypeError)):
        bundle.sections[0].text = "mutated"
    with pytest.raises((AttributeError, TypeError)):
        artifact.payload["pages"][0]["words"][0]["text"] = "mutated"
    with pytest.raises((AttributeError, TypeError)):
        pdf_export.IgnoredAnchor("a" * 64, "printed-page-number-v1").reason = "mutated"
    with pytest.raises(ValueError, match="seal"):
        replace(bundle, parser_version="different-parser").verify_seal()
    jittered = json.loads(json.dumps(payload))
    jittered["pages"][0]["words"][0]["x0"] += 0.01
    jittered_artifact = pdf_export.VerifiedNativeExport(
        payload=jittered, product_code="PZO12001", source_basename="PZO12001E.pdf",
        page_count=2, extractor_profile_version=1, pdf_verified=True,
        attestation_digest="b" * 64, _verification_token=pdf_export._TRUSTED_PDF_ORIGIN,
        _payload_digest=pdf_export.trusted_payload_digest(jittered),
    )
    jittered_bundle = corpus.parse_verified_native_export(jittered_artifact)
    assert [section.source_section_id for section in jittered_bundle.sections] == [
        section.source_section_id for section in bundle.sections
    ]


def test_trusted_layout_requires_every_native_pdf_page(tmp_path, monkeypatch):
    payload = _trusted_payload(
        [
            _page(1, [("Rules", 40, 100, "Heading-Bold", 16)]),
            _page(2, [("Continue", 40, 100, "Body", 9)]),
        ]
    )
    artifact = pdf_export.VerifiedNativeExport(
        payload=payload,
        product_code="PZO12001",
        source_basename="PZO12001E.pdf",
        page_count=2,
        extractor_profile_version=1,
        pdf_verified=True,
        attestation_digest="a" * 64,
        _verification_token=pdf_export._TRUSTED_PDF_ORIGIN,
        _payload_digest=pdf_export.trusted_payload_digest(payload),
    )
    monkeypatch.setattr(corpus, "verified_native_export_from_pdf", lambda *_args, **_kwargs: artifact)
    monkeypatch.setattr(
        "pf2e_codex.pdf_layout.bind_layout_to_native_export",
        lambda *_args, **_kwargs: SimpleNamespace(selected_pages=(1,)),
    )

    with pytest.raises(ValueError, match="every exported native PDF page"):
        corpus.load_and_parse_verified_pdf(
            tmp_path / "PZO12001E.pdf",
            product_code="PZO12001",
            parser_version="paizo-native-v3",
            layout_artifact={},
        )


def test_v4_uses_layout_order_and_integrates_unbound_native_words():
    page = _page(1, [
        ("Left Rules", 30, 50, "Heading-Bold", 16),
        ("left body", 30, 90, "Body", 9),
        ("Right Rules", 330, 50, "Heading-Bold", 16),
        ("right body", 330, 90, "Body", 9),
        ("orphan", 280, 400, "Body", 9),
    ])
    payload = _trusted_payload([page])
    artifact = pdf_export.VerifiedNativeExport(
        payload=payload, product_code="PZO12001", source_basename="PZO12001E.pdf",
        page_count=1, extractor_profile_version=1, pdf_verified=True,
        attestation_digest="a" * 64,
        _verification_token=pdf_export._TRUSTED_PDF_ORIGIN,
        _payload_digest=pdf_export.trusted_payload_digest(payload),
    )
    layout = {
        "schema_version": 1,
        "extractor": {"name": "pf2e-codex-pdf-layout", "profile_version": 1},
        "source": {"sha256": "a" * 64, "page_count": 1},
        "pages": [{
            "number": 1, "width": 600.0, "height": 800.0,
            "regions": [
                {"label": "paragraph_title", "score": 0.99, "order": 0,
                 "box": [320, 35, 500, 75]},
                {"label": "text", "score": 0.99, "order": 1,
                 "box": [320, 75, 500, 120]},
                {"label": "paragraph_title", "score": 0.99, "order": 2,
                 "box": [20, 35, 200, 75]},
                {"label": "text", "score": 0.99, "order": 3,
                 "box": [20, 75, 200, 120]},
            ],
        }],
    }
    from pf2e_codex.pdf_layout import bind_layout_to_native_export

    binding = bind_layout_to_native_export(artifact, layout)
    bundle = corpus.parse_verified_native_export(
        artifact, parser_version=corpus.PAIZO_NATIVE_PARSER_V4,
        layout_binding=binding,
    )

    assert [section.heading for section in bundle.sections] == ["Right Rules", "Left Rules"]
    assert "right body" in bundle.sections[0].text
    assert "left body" in bundle.sections[1].text
    fallback_section = next(section for section in bundle.sections if "orphan" in section.text)
    assert "native-layout-fallback" in fallback_section.layout_flags
    assert not bundle.quarantine
    assigned = [
        anchor
        for group in (
            *(section.coverage_anchors for section in bundle.sections),
            *(item.coverage_anchors for item in bundle.quarantine),
        )
        for anchor in group
    ]
    expected = set(bundle.inventory.anchors) - set(bundle.inventory.ignored_anchor_reasons)
    assert set(assigned) == expected and len(assigned) == len(set(assigned))
    assert all(section.blocks for section in bundle.sections)
    bundle.verify_seal()


def test_v4_reconstructs_stable_native_table_and_quarantines_ambiguous_table():
    page = _page(1, [
        ("Rule Table", 30, 40, "Heading-Bold", 16),
        ("Level", 30, 100, "Body-Bold", 9),
        ("DC", 300, 100, "Body-Bold", 9),
        ("One", 30, 125, "Body", 9),
        ("15", 300, 125, "Body", 9),
        ("Loose", 30, 250, "Body", 9),
        ("Cells", 300, 250, "Body", 9),
    ])
    payload = _trusted_payload([page])
    artifact = pdf_export.VerifiedNativeExport(
        payload=payload, product_code="PZO12001", source_basename="PZO12001E.pdf",
        page_count=1, extractor_profile_version=1, pdf_verified=True,
        attestation_digest="a" * 64,
        _verification_token=pdf_export._TRUSTED_PDF_ORIGIN,
        _payload_digest=pdf_export.trusted_payload_digest(payload),
    )
    layout = {
        "schema_version": 1,
        "extractor": {"name": "pf2e-codex-pdf-layout", "profile_version": 1},
        "source": {"sha256": "a" * 64, "page_count": 1},
        "pages": [{
            "number": 1, "width": 600.0, "height": 800.0,
            "regions": [
                {"label": "paragraph_title", "score": 0.99, "order": 0,
                 "box": [20, 25, 200, 75]},
                {"label": "table", "score": 0.99, "order": 1,
                 "box": [20, 80, 420, 170]},
                {"label": "table", "score": 0.99, "order": 2,
                 "box": [20, 220, 420, 290]},
            ],
        }],
    }
    from pf2e_codex.pdf_layout import bind_layout_to_native_export

    binding = bind_layout_to_native_export(artifact, layout)
    bundle = corpus.parse_verified_native_export(
        artifact, parser_version=corpus.PAIZO_NATIVE_PARSER_V4,
        layout_binding=binding,
    )

    assert len(bundle.sections) == 1
    assert "structured-table" in bundle.sections[0].layout_flags
    assert any(block.kind == "table" for block in bundle.sections[0].blocks)
    assert {item.reason for item in bundle.quarantine} == {"unresolved-table"}
    bundle.verify_seal()


def test_v4_carries_consecutive_heading_chain_into_next_rule_section():
    page = _page(1, [
        ("Chapter Rules", 30, 40, "Heading-Bold", 18),
        ("Specific Rule", 30, 80, "Heading-Bold", 15),
        ("The complete mechanic follows this nested heading.", 30, 120, "Body", 9),
    ])
    payload = _trusted_payload([page])
    artifact = pdf_export.VerifiedNativeExport(
        payload=payload, product_code="PZO12001", source_basename="PZO12001E.pdf",
        page_count=1, extractor_profile_version=1, pdf_verified=True,
        attestation_digest="a" * 64,
        _verification_token=pdf_export._TRUSTED_PDF_ORIGIN,
        _payload_digest=pdf_export.trusted_payload_digest(payload),
    )
    layout = {
        "schema_version": 1,
        "extractor": {"name": "pf2e-codex-pdf-layout", "profile_version": 1},
        "source": {"sha256": "a" * 64, "page_count": 1},
        "pages": [{
            "number": 1, "width": 600.0, "height": 800.0,
            "regions": [
                {"label": "paragraph_title", "score": 0.99, "order": 0,
                 "box": [20, 25, 500, 70]},
                {"label": "paragraph_title", "score": 0.99, "order": 1,
                 "box": [20, 70, 500, 110]},
                {"label": "text", "score": 0.99, "order": 2,
                 "box": [20, 110, 500, 160]},
            ],
        }],
    }
    from pf2e_codex.pdf_layout import bind_layout_to_native_export

    bundle = corpus.parse_verified_native_export(
        artifact,
        parser_version=corpus.PAIZO_NATIVE_PARSER_V4,
        layout_binding=bind_layout_to_native_export(artifact, layout),
    )

    assert len(bundle.sections) == 1
    assert bundle.sections[0].heading == "Specific Rule"
    assert "Chapter Rules Specific Rule" in bundle.sections[0].text
    assert [block.kind for block in bundle.sections[0].blocks] == [
        "heading", "heading", "body",
    ]
    assert not bundle.quarantine
    bundle.verify_seal()


def test_v4_splits_oversize_sections_on_existing_native_blocks():
    pages = []
    for page_number in (1, 2):
        lines = []
        if page_number == 1:
            lines.append(("Long Rule", 30, 20, "Heading-Bold", 16))
        lines.extend(
            (
                f"mechanic {page_number}-{index} " + "x" * 64,
                30,
                55 + index * 12,
                "Body",
                9,
            )
            for index in range(60)
        )
        pages.append(_page(page_number, lines))
    payload = _trusted_payload(pages)
    artifact = pdf_export.VerifiedNativeExport(
        payload=payload, product_code="PZO12001", source_basename="PZO12001E.pdf",
        page_count=2, extractor_profile_version=1, pdf_verified=True,
        attestation_digest="a" * 64,
        _verification_token=pdf_export._TRUSTED_PDF_ORIGIN,
        _payload_digest=pdf_export.trusted_payload_digest(payload),
    )
    layout_pages = []
    for page_number in (1, 2):
        regions = []
        if page_number == 1:
            regions.append(
                {"label": "paragraph_title", "score": 0.99, "order": 0,
                 "box": [20, 10, 500, 50]}
            )
        regions.append(
            {"label": "text", "score": 0.99, "order": 1,
             "box": [20, 50, 520, 790]}
        )
        layout_pages.append({
            "number": page_number, "width": 600.0, "height": 800.0,
            "regions": regions,
        })
    layout = {
        "schema_version": 1,
        "extractor": {"name": "pf2e-codex-pdf-layout", "profile_version": 1},
        "source": {"sha256": "a" * 64, "page_count": 2},
        "pages": layout_pages,
    }
    from pf2e_codex.pdf_layout import bind_layout_to_native_export

    bundle = corpus.parse_verified_native_export(
        artifact,
        parser_version=corpus.PAIZO_NATIVE_PARSER_V4,
        layout_binding=bind_layout_to_native_export(artifact, layout),
    )

    assert len(bundle.sections) == 2
    assert all(len(section.text) < 10_000 for section in bundle.sections)
    assert all("oversize-split" in section.layout_flags for section in bundle.sections)
    assert not bundle.sections[1].text.startswith("Long Rule")
    assigned = [
        anchor
        for section in bundle.sections
        for anchor in section.coverage_anchors
    ]
    expected = set(bundle.inventory.anchors) - set(bundle.inventory.ignored_anchor_reasons)
    assert set(assigned) == expected and len(assigned) == len(set(assigned))
    assert not bundle.quarantine
    bundle.verify_seal()


def test_v4_quarantines_contradictory_same_band_layout_order():
    page = _page(1, [
        ("Later Region", 30, 200, "Heading-Bold", 16),
        ("Earlier region body", 30, 50, "Body", 9),
    ])
    payload = _trusted_payload([page])
    artifact = pdf_export.VerifiedNativeExport(
        payload=payload, product_code="PZO12001", source_basename="PZO12001E.pdf",
        page_count=1, extractor_profile_version=1, pdf_verified=True,
        attestation_digest="a" * 64,
        _verification_token=pdf_export._TRUSTED_PDF_ORIGIN,
        _payload_digest=pdf_export.trusted_payload_digest(payload),
    )
    layout = {
        "schema_version": 1,
        "extractor": {"name": "pf2e-codex-pdf-layout", "profile_version": 1},
        "source": {"sha256": "a" * 64, "page_count": 1},
        "pages": [{
            "number": 1, "width": 600.0, "height": 800.0,
            "regions": [
                {"label": "paragraph_title", "score": 0.99, "order": 0,
                 "box": [20, 180, 500, 240]},
                {"label": "text", "score": 0.99, "order": 1,
                 "box": [20, 30, 500, 90]},
            ],
        }],
    }
    from pf2e_codex.pdf_layout import bind_layout_to_native_export

    bundle = corpus.parse_verified_native_export(
        artifact,
        parser_version=corpus.PAIZO_NATIVE_PARSER_V4,
        layout_binding=bind_layout_to_native_export(artifact, layout),
    )

    assert not bundle.sections
    assert {item.reason for item in bundle.quarantine} == {"layout-order-conflict"}
    bundle.verify_seal()


def test_v4_sentence_like_model_title_cannot_bypass_heading_checks():
    sentence = "This is a complete sentence that the model mislabeled as a paragraph title."
    page = _page(1, [
        (sentence, 30, 40, "Body", 9),
        ("Actual Rule", 30, 100, "Heading-Bold", 16),
        ("The actual mechanic remains active and correctly headed.", 30, 140, "Body", 9),
    ])
    payload = _trusted_payload([page])
    artifact = pdf_export.VerifiedNativeExport(
        payload=payload, product_code="PZO12001", source_basename="PZO12001E.pdf",
        page_count=1, extractor_profile_version=1, pdf_verified=True,
        attestation_digest="a" * 64,
        _verification_token=pdf_export._TRUSTED_PDF_ORIGIN,
        _payload_digest=pdf_export.trusted_payload_digest(payload),
    )
    layout = {
        "schema_version": 1,
        "extractor": {"name": "pf2e-codex-pdf-layout", "profile_version": 1},
        "source": {"sha256": "a" * 64, "page_count": 1},
        "pages": [{
            "number": 1, "width": 600.0, "height": 800.0,
            "regions": [
                {"label": "paragraph_title", "score": 0.99, "order": 0,
                 "box": [20, 25, 500, 80]},
                {"label": "paragraph_title", "score": 0.99, "order": 1,
                 "box": [20, 85, 500, 130]},
                {"label": "text", "score": 0.99, "order": 2,
                 "box": [20, 130, 500, 180]},
            ],
        }],
    }
    from pf2e_codex.pdf_layout import bind_layout_to_native_export

    bundle = corpus.parse_verified_native_export(
        artifact,
        parser_version=corpus.PAIZO_NATIVE_PARSER_V4,
        layout_binding=bind_layout_to_native_export(artifact, layout),
    )

    assert [section.heading for section in bundle.sections] == ["Actual Rule"]
    assert sentence not in bundle.sections[0].heading
    assert {item.reason for item in bundle.quarantine} == {"unresolved-continuation"}
    bundle.verify_seal()


def test_v4_quarantines_one_indivisible_oversize_native_block():
    long_line = "x" * 8_100
    page = {
        "number": 1,
        "width": 50_000.0,
        "height": 800.0,
        "words": [
            _word("Large Native Block", x=30, top=30, size=16, font="Heading-Bold"),
            _word(long_line, x=30, top=100, size=9, font="Body"),
        ],
        "images": [],
    }
    payload = _trusted_payload([page])
    artifact = pdf_export.VerifiedNativeExport(
        payload=payload, product_code="PZO12001", source_basename="PZO12001E.pdf",
        page_count=1, extractor_profile_version=1, pdf_verified=True,
        attestation_digest="a" * 64,
        _verification_token=pdf_export._TRUSTED_PDF_ORIGIN,
        _payload_digest=pdf_export.trusted_payload_digest(payload),
    )
    layout = {
        "schema_version": 1,
        "extractor": {"name": "pf2e-codex-pdf-layout", "profile_version": 1},
        "source": {"sha256": "a" * 64, "page_count": 1},
        "pages": [{
            "number": 1, "width": 50_000.0, "height": 800.0,
            "regions": [
                {"label": "paragraph_title", "score": 0.99, "order": 0,
                 "box": [20, 15, 500, 70]},
                {"label": "text", "score": 0.99, "order": 1,
                 "box": [20, 80, 45_000, 160]},
            ],
        }],
    }
    from pf2e_codex.pdf_layout import bind_layout_to_native_export

    bundle = corpus.parse_verified_native_export(
        artifact,
        parser_version=corpus.PAIZO_NATIVE_PARSER_V4,
        layout_binding=bind_layout_to_native_export(artifact, layout),
    )

    assert not bundle.sections
    assert {item.reason for item in bundle.quarantine} == {"oversize-block"}
    bundle.verify_seal()


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
    assert old["source"]["parser"] == corpus.PAIZO_NATIVE_PARSER_VERSION


@pytest.mark.parametrize("gutter", [25, 55])
def test_parse_infers_narrow_and_wide_column_gutters(tmp_path, gutter):
    # The long left line ends at x=280, so this gives the page an actual
    # 25pt/55pt native-space gutter rather than merely changing indentation.
    right_x = 280 + gutter
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
                    _word("Adaptive Columns", x=30, top=50, size=16, font="Heading-Bold"),
                    _word("Left column carries a deliberately long first line", x=30, top=100),
                    _word("Left column second line", x=30, top=120),
                    _word("Right column first line", x=right_x, top=100),
                    _word("Right column second line", x=right_x, top=120),
                ],
            }
        ],
    }
    path = tmp_path / f"gutter-{gutter}.json"
    path.write_text(json.dumps(payload))

    v1 = corpus.parse_rulebook_export(path)[0]
    v2 = corpus.parse_rulebook_export(
        path, parser_version=corpus.PAIZO_NATIVE_PARSER_V2
    )[0]

    assert v1["source"]["parser"] == corpus.PAIZO_NATIVE_PARSER_V1
    assert v2["source"]["parser"] == corpus.PAIZO_NATIVE_PARSER_V2
    assert "layout_flags" not in v2["source"]["provenance"]
    assert v1["text"].index("Right column first") < v1["text"].index("Left column second")
    assert v2["text"].index("Left column second") < v2["text"].index("Right column first")


def test_parse_keeps_spanning_blocks_in_their_vertical_band(tmp_path):
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
                    _word("Banded Reading", x=30, top=50, size=16, font="Heading-Bold"),
                    _word("Left above", x=30, top=100),
                    _word("Left above second", x=30, top=120),
                    _word("Right above", x=330, top=100),
                    _word("Right above second", x=330, top=120),
                    _word("Spanning block remains between the two vertical column bands " * 2, x=30, top=145),
                    _word("Ragged spanning continuation", x=30, top=160),
                    _word("Left below", x=30, top=180),
                    _word("Left below second", x=30, top=200),
                    _word("Right below", x=330, top=180),
                    _word("Right below second", x=330, top=200),
                ],
            }
        ],
    }
    path = tmp_path / "vertical-band.json"
    path.write_text(json.dumps(payload))

    text = corpus.parse_rulebook_export(
        path, parser_version=corpus.PAIZO_NATIVE_PARSER_V2
    )[0]["text"]

    assert text.index("Left above second") < text.index("Right above")
    assert text.index("Right above second") < text.index("Spanning block")
    assert text.index("Spanning block") < text.index("Ragged spanning continuation")
    assert text.index("Ragged spanning continuation") < text.index("Left below")
    assert text.index("Left below second") < text.index("Right below")


def test_parse_does_not_promote_body_sized_condensed_caps_label(tmp_path):
    payload = _export_payload("PZO12001E.pdf", "Body text after the label")
    payload["pages"][0]["words"].insert(
        3, _word("TABLE LABEL", x=40, top=120, size=9, font="GoodOT-CondBold")
    )
    path = tmp_path / "caps-label.json"
    path.write_text(json.dumps(payload))

    chunks = corpus.parse_rulebook_export(
        path, parser_version=corpus.PAIZO_NATIVE_PARSER_V2
    )

    assert "TABLE LABEL" not in [chunk["name"] for chunk in chunks]
    chapter = next(chunk for chunk in chunks if chunk["name"] == "Chapter One")
    assert "TABLE LABEL" in chapter["text"]


def test_v3_marks_recurrent_two_cell_labels_without_losing_real_narrow_headings(tmp_path):
    """V3 changes only the evidenced cell-label false-positive path.

    The fixture is sanitized and deliberately combines two normal reading
    columns with a two-cell region.  V1/V2 snapshots retain their historic
    section boundaries, while V3 keeps the recurring condensed cell labels in
    their surrounding rule section and surfaces a review gate.
    """
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
                    _word("Combat Rules", x=30, top=50, size=16, font="Heading-Bold"),
                    _word("Left column opening", x=30, top=90),
                    _word("Right column opening", x=330, top=90),
                    _word("Dominant Section Heading", x=30, top=120, size=10, font="Body-CondBold"),
                    _word("Right column continuation", x=330, top=120),
                    _word("Cell companion one", x=30, top=160),
                    _word("CELL LABEL ONE", x=150, top=160, size=12, font="Body-CondBold"),
                    _word("Cell companion two", x=30, top=185),
                    _word("CELL LABEL TWO", x=150, top=185, size=12, font="Body-CondBold"),
                    _word("Indented Heading", x=55, top=230, size=10, font="Body-CondBold"),
                    _word("• narrow list entry", x=55, top=250),
                    _word("Right column closes", x=330, top=250),
                ],
            }
        ],
    }
    path = tmp_path / "sanitized-two-column-two-cell.json"
    path.write_text(json.dumps(payload))

    v1 = corpus.parse_rulebook_export(path, parser_version=corpus.PAIZO_NATIVE_PARSER_V1)
    v2 = corpus.parse_rulebook_export(path, parser_version=corpus.PAIZO_NATIVE_PARSER_V2)
    v3 = corpus.parse_rulebook_export(path, parser_version=corpus.PAIZO_NATIVE_PARSER_V3)

    # Regression snapshots for frozen profiles: V3 must not backport its
    # stricter geometry into ordinary local/full parsing.
    assert [chunk["name"] for chunk in v1] == [
        "Combat Rules", "Indented Heading", "Unclassified native text",
    ]
    assert [chunk["name"] for chunk in v2] == [
        "Combat Rules", "Dominant Section Heading", "CELL LABEL ONE", "Indented Heading",
        "Unclassified native text",
    ]

    assert [chunk["name"] for chunk in v3] == [
        "Combat Rules", "Dominant Section Heading", "Indented Heading",
    ]
    dominant = next(chunk for chunk in v3 if chunk["name"] == "Dominant Section Heading")
    assert "CELL LABEL ONE" in dominant["text"]
    assert "CELL LABEL TWO" in dominant["text"]
    assert dominant["source"]["provenance"]["layout_flags"] == ["table-cell"]
    assert any(chunk["name"] == "Indented Heading" for chunk in v3)
    indented = next(chunk for chunk in v3 if chunk["name"] == "Indented Heading")
    assert "narrow list entry" in indented["text"]


def test_v3_preserves_recurrent_indented_headings_aligned_with_other_column(tmp_path):
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
                    _word("Chapter", x=30, top=50, size=16, font="Heading-Bold"),
                    _word("Left opening", x=30, top=90),
                    _word("Right opening", x=330, top=90),
                    _word("Indented Heading One", x=55, top=120, size=10, font="Body-CondBold"),
                    _word("Right aligned one", x=330, top=120),
                    _word("First rule body", x=55, top=140),
                    _word("Left continuation", x=30, top=160),
                    _word("Right continuation", x=330, top=160),
                    _word("Indented Heading Two", x=55, top=190, size=10, font="Body-CondBold"),
                    _word("Right aligned two", x=330, top=190),
                    _word("Second rule body", x=55, top=210),
                ],
            }
        ],
    }
    path = tmp_path / "recurrent-indented-headings.json"
    path.write_text(json.dumps(payload))

    v2 = corpus.parse_rulebook_export(path, parser_version=corpus.PAIZO_NATIVE_PARSER_V2)
    v3 = corpus.parse_rulebook_export(path, parser_version=corpus.PAIZO_NATIVE_PARSER_V3)
    v2_names = [chunk["name"] for chunk in v2]
    v3_names = [chunk["name"] for chunk in v3]

    assert v3_names == v2_names
    assert "Indented Heading One" in v3_names
    assert "Indented Heading Two" in v3_names


def test_v3_preserves_display_sized_and_noncondensed_interior_headings(tmp_path):
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
                    _word("Chapter", x=30, top=50, size=16, font="Heading-Bold"),
                    _word("Body opening", x=30, top=90),
                    _word("Display companion", x=30, top=130),
                    _word("Display Heading", x=150, top=130, size=13, font="Body-CondBold"),
                    _word("Display rule body", x=150, top=150),
                    _word("Regular companion", x=30, top=180),
                    _word("Regular Bold Heading", x=150, top=180, size=12, font="Body-Bold"),
                    _word("Regular rule body", x=150, top=200),
                ],
            }
        ],
    }
    path = tmp_path / "interior-real-headings.json"
    path.write_text(json.dumps(payload))

    v3 = corpus.parse_rulebook_export(path, parser_version=corpus.PAIZO_NATIVE_PARSER_V3)
    names = [chunk["name"] for chunk in v3]

    assert "Display Heading" in names
    assert "Regular Bold Heading" in names


def test_parse_accepts_bounded_three_cell_table_row(tmp_path):
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
                    _word("first region", x=30, top=100),
                    _word("second region", x=220, top=100),
                    _word("third region", x=410, top=100),
                ],
            }
        ],
    }
    path = tmp_path / "three-regions.json"
    path.write_text(json.dumps(payload))

    text = corpus.parse_rulebook_export(
        path, parser_version=corpus.PAIZO_NATIVE_PARSER_V2
    )[0]["text"]
    assert text.index("first region") < text.index("second region") < text.index("third region")


def test_v2_keeps_aligned_multirow_three_cell_table_as_reviewable_block(tmp_path):
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
                    _word("TABLE VALUES", x=30, top=60, size=16, font="Heading-Bold"),
                    _word("row one left", x=30, top=100),
                    _word("row one middle", x=220, top=100),
                    _word("row one right", x=410, top=100),
                    _word("row two left", x=30, top=160),
                    _word("row two middle", x=220, top=160),
                    _word("row two right", x=410, top=160),
                    _word("row three left", x=30, top=220),
                    _word("row three middle", x=220, top=220),
                    _word("row three right", x=410, top=220),
                    _word("row four left", x=30, top=280),
                    _word("row four middle", x=220, top=280),
                    _word("row four right", x=410, top=280),
                ],
            }
        ],
    }
    path = tmp_path / "three-cell-grid.json"
    path.write_text(json.dumps(payload))

    chunk = corpus.parse_rulebook_export(
        path, parser_version=corpus.PAIZO_NATIVE_PARSER_V2
    )[0]

    assert chunk["text"].index("row one left") < chunk["text"].index("row four right")
    assert chunk["source"]["provenance"]["layout_flags"] == ["table-grid"]


def test_v2_retains_bounded_ambiguous_table_for_section_review(tmp_path):
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
                    _word("TABLE CASE", x=30, top=60, size=16, font="Heading-Bold"),
                    _word("first left", x=30, top=100),
                    _word("first middle", x=220, top=100),
                    _word("first right", x=410, top=100),
                    _word("second left", x=30, top=150),
                    _word("second middle", x=250, top=150),
                    _word("second right", x=410, top=150),
                    _word("third left", x=30, top=200),
                    _word("third middle", x=220, top=200),
                    _word("third right", x=410, top=200),
                ],
            }
        ],
    }
    path = tmp_path / "ambiguous-three-cell-table.json"
    path.write_text(json.dumps(payload))

    chunk = corpus.parse_rulebook_export(
        path, parser_version=corpus.PAIZO_NATIVE_PARSER_V2
    )[0]

    assert chunk["source"]["provenance"]["layout_flags"] == ["table-ambiguous"]


def test_v2_complex_layout_forces_visual_row_order(tmp_path):
    table_rows = [
        word
        for row_index in range(12)
        for word in (
            _word("table left", x=30, top=140 + row_index * 50),
            _word("table middle", x=220, top=140 + row_index * 50),
            _word("table right", x=410, top=140 + row_index * 50),
        )
    ]
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
                    _word("Left ordinary first", x=30, top=60),
                    _word("Right ordinary first", x=330, top=60),
                    _word("Left ordinary second", x=30, top=80),
                    _word("Right ordinary second", x=330, top=80),
                    _word("Left ordinary third", x=30, top=100),
                    _word("Right ordinary third", x=330, top=100),
                    *table_rows,
                ],
            }
        ],
    }
    path = tmp_path / "forced-visual-order.json"
    path.write_text(json.dumps(payload))

    chunk = corpus.parse_rulebook_export(
        path, parser_version=corpus.PAIZO_NATIVE_PARSER_V2
    )[0]

    assert chunk["text"].index("Right ordinary first") < chunk["text"].index(
        "Left ordinary second"
    )
    assert "complex-layout" in chunk["source"]["provenance"]["layout_flags"]


def test_parse_marks_three_persistent_regions_for_section_review(tmp_path):
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
                    _word("left one", x=30, top=100),
                    _word("middle one", x=220, top=120),
                    _word("right one", x=410, top=140),
                    _word("left two", x=30, top=300),
                    _word("middle two", x=220, top=320),
                    _word("right two", x=410, top=340),
                    _word("left three", x=30, top=500),
                    _word("middle three", x=220, top=520),
                    _word("right three", x=410, top=540),
                ],
            }
        ],
    }
    path = tmp_path / "persistent-three-regions.json"
    path.write_text(json.dumps(payload))

    chunks = corpus.parse_rulebook_export(
        path, parser_version=corpus.PAIZO_NATIVE_PARSER_V2
    )
    assert all(
        "complex-layout" in chunk["source"]["provenance"]["layout_flags"]
        for chunk in chunks
    )


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


@pytest.mark.parametrize(
    "fixture_name",
    ["core-rulebook.json", "player-core.json", "gm-core.json"],
)
def test_v2_sanitized_paizo_layout_smoke(tmp_path, fixture_name):
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
        "source": {"filename": fixture["source_filename"], "sha256": "hash"},
        "pages": pages,
    }
    export_path = tmp_path / f"v2-{fixture_name}"
    export_path.write_text(json.dumps(payload))

    chunks = corpus.parse_rulebook_export(
        export_path, parser_version=corpus.PAIZO_NATIVE_PARSER_V2
    )

    assert [chunk["name"] for chunk in chunks] == fixture["expected_names"]
    assert all(chunk["source"]["parser"] == corpus.PAIZO_NATIVE_PARSER_V2 for chunk in chunks)
    combined_text = " ".join(chunk["text"] for chunk in chunks)
    assert "example.invalid" not in combined_text
    assert fixture["pages"][0]["lines"][0][0] not in combined_text
    if fixture_name == "core-rulebook.json":
        assert chunks[0]["pages"] == [12, 13]
        assert chunks[0]["printed_page"] == "10-11"
