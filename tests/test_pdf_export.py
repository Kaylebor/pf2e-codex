"""Tests for the loss-minimal native PDF exporter."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from pf2e_codex import pdf_export


class _FakePage:
    width = 620.607
    height = 799.143
    images = [
        {"name": "Im0", "x0": 10, "top": 20, "x1": 30, "bottom": 40},
    ]

    def __init__(self, text: str) -> None:
        self.text = text
        self.extract_kwargs: dict | None = None

    def extract_words(self, **kwargs):
        self.extract_kwargs = kwargs
        return [
            {
                "text": self.text,
                "x0": 10.123456,
                "top": 20,
                "x1": 50,
                "bottom": 30,
                "fontname": "BodyFont",
                "size": 9,
                "upright": True,
                "direction": "ltr",
            }
        ]


class _FakePdf:
    def __init__(self, pages: list[_FakePage]) -> None:
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def _backend(pages: list[_FakePage]):
    return SimpleNamespace(__version__="0.11.9", open=lambda _path: _FakePdf(pages))


def test_export_pdf_writes_versioned_native_word_json(tmp_path, monkeypatch):
    source = tmp_path / "book.pdf"
    source.write_bytes(b"selectable PDF fixture")
    output = tmp_path / "book.json"
    pages = [_FakePage("cover"), _FakePage("fireball"), _FakePage("[one-action]")]
    monkeypatch.setattr(pdf_export, "_load_pdfplumber", lambda: _backend(pages))

    summary = pdf_export.export_pdf(source, output, first_page=2, last_page=3)
    data = json.loads(output.read_text())

    assert data["schema_version"] == 1
    assert data["extractor"] == {
        "name": "pf2e-codex-native-pdf",
        "profile_version": 1,
        "backend": "pdfplumber",
        "backend_version": "0.11.9",
        "ocr": False,
    }
    assert data["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert data["source"]["page_count"] == 3
    assert [page["number"] for page in data["pages"]] == [2, 3]
    assert data["pages"][0]["words"][0] == {
        "text": "fireball",
        "x0": 10.1235,
        "top": 20.0,
        "x1": 50.0,
        "bottom": 30.0,
        "font": "BodyFont",
        "size": 9.0,
        "upright": True,
        "direction": "ltr",
    }
    assert pages[1].extract_kwargs == {
        "use_text_flow": True,
        "keep_blank_chars": False,
        "extra_attrs": ["fontname", "size"],
        "expand_ligatures": True,
    }
    assert summary.exported_pages == 2
    assert summary.words == 2


@pytest.mark.parametrize(
    ("first_page", "last_page"),
    [(0, None), (3, 2), (1, 4)],
)
def test_export_pdf_rejects_invalid_page_ranges(
    tmp_path, monkeypatch, first_page, last_page
):
    source = tmp_path / "book.pdf"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(
        pdf_export,
        "_load_pdfplumber",
        lambda: _backend([_FakePage("one"), _FakePage("two")]),
    )

    with pytest.raises(ValueError):
        pdf_export.export_pdf(
            source,
            tmp_path / "book.json",
            first_page=first_page,
            last_page=last_page,
        )


def test_export_pdf_does_not_replace_existing_output(tmp_path, monkeypatch):
    source = tmp_path / "book.pdf"
    source.write_bytes(b"fixture")
    output = tmp_path / "book.json"
    output.write_text("keep me")
    monkeypatch.setattr(pdf_export, "_load_pdfplumber", lambda: _backend([_FakePage("one")]))

    with pytest.raises(FileExistsError):
        pdf_export.export_pdf(source, output)

    assert output.read_text() == "keep me"


def test_verified_native_export_comes_from_pdf_not_forged_cache(tmp_path, monkeypatch):
    source = tmp_path / "PZO12001E.pdf"
    source.write_bytes(b"fixture PDF bytes")
    output = tmp_path / "native.json"
    pages = [_FakePage("Pathfinder Player Core"), _FakePage("two")]
    monkeypatch.setattr(pdf_export, "_load_pdfplumber", lambda: _backend(pages))
    pdf_export.export_pdf(source, output)

    # A cached export can be forged, but direct verification never reads it.
    forged = json.loads(output.read_text())
    forged["pages"][0]["words"][0]["text"] = "forged"
    output.write_text(json.dumps(forged))
    verified = pdf_export.verified_native_export_from_pdf(
        source, product_code="PZO12001", expected_title_markers=("Pathfinder Player Core",)
    )
    assert verified.pdf_verified is True
    assert verified.page_count == 2
    assert "fixture PDF bytes" not in repr(verified)
    assert verified.payload["pages"][0]["words"][0]["text"] == "Pathfinder Player Core"

    untrusted = pdf_export.load_untrusted_native_export(output, product_code="PZO12001")
    assert untrusted.pdf_verified is False

    with pytest.raises(ValueError, match="title evidence"):
        pdf_export.verified_native_export_from_pdf(
            source, product_code="PZO12001", expected_title_markers=("GM Core",)
        )
    with pytest.raises(ValueError, match="PZO product"):
        pdf_export.verified_native_export_from_pdf(
            source, product_code="PZO12002", expected_title_markers=("GM Core",)
        )


def test_catalog_title_evidence_rejects_player_core_2_as_player_core():
    pages = [{
        "number": 1,
        "words": [{"text": "Pathfinder Player Core 2"}],
    }]
    catalog = {
        "PZO12001": ("Pathfinder Player Core", "Player Core"),
        "PZO12004": ("Pathfinder Player Core 2", "Player Core 2"),
        "PZO12002": ("Pathfinder GM Core", "GM Core"),
        "PZO12003": ("Pathfinder Monster Core", "Monster Core"),
        "PZO2101": ("Pathfinder Core Rulebook", "Core Rulebook"),
    }
    wrong = pdf_export._title_marker_evidence(
        pages, selected_product_code="PZO12001", selected_markers=catalog["PZO12001"],
        catalog_markers=catalog,
    )
    assert wrong["title_marker_verified"] is False
    assert wrong["conflict_product_codes"] == ("PZO12004",)
    right = pdf_export._title_marker_evidence(
        pages, selected_product_code="PZO12004", selected_markers=catalog["PZO12004"],
        catalog_markers=catalog,
    )
    assert right["title_marker_verified"] is True
    assert right["conflict_product_codes"] == ()


def test_catalog_title_evidence_accepts_unambiguous_short_cover_title():
    pages = [{"number": 1, "words": [{"text": "GM Core"}]}]
    catalog = {
        "PZO12001": ("Pathfinder Player Core", "Player Core"),
        "PZO12002": ("Pathfinder GM Core", "GM Core"),
    }

    evidence = pdf_export._title_marker_evidence(
        pages, selected_product_code="PZO12002", selected_markers=catalog["PZO12002"],
        catalog_markers=catalog,
    )

    assert evidence["title_marker_verified"] is True
    assert evidence["matched_product_codes"] == ("PZO12002",)
