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
