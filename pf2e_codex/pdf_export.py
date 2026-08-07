"""Loss-minimal native-text PDF export for locally owned rulebooks.

This module deliberately stops at PDF structure.  It records words, fonts,
coordinates, and image bounds in content-stream order; corpus-specific cleanup
and section construction belong to a separate parsing stage.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PDF_EXPORT_SCHEMA_VERSION = 1
EXTRACTOR_PROFILE_VERSION = 1


class PdfExportDependencyError(RuntimeError):
    """Raised when the optional native PDF dependency is unavailable."""


@dataclass(frozen=True)
class PdfExportSummary:
    """Summary returned after a successful export."""

    output_path: Path
    source_sha256: str
    source_pages: int
    exported_pages: int
    words: int


def _load_pdfplumber() -> Any:
    try:
        import pdfplumber
    except ImportError as exc:
        raise PdfExportDependencyError(
            "PDF corpus export requires the optional 'corpus' dependencies. "
            "Install with: pip install 'pf2e-codex[corpus]'"
        ) from exc
    return pdfplumber


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any) -> float:
    """Normalize backend numeric types and discard insignificant PDF noise."""
    return round(float(value), 4)


def _compact_word(word: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": word["text"],
        "x0": _number(word["x0"]),
        "top": _number(word["top"]),
        "x1": _number(word["x1"]),
        "bottom": _number(word["bottom"]),
        "font": word.get("fontname"),
        "size": _number(word["size"]),
        "upright": bool(word.get("upright", True)),
        "direction": word.get("direction", "ltr"),
    }


def _compact_image(image: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": image.get("name"),
        "x0": _number(image["x0"]),
        "top": _number(image["top"]),
        "x1": _number(image["x1"]),
        "bottom": _number(image["bottom"]),
    }


def _page_payload(page: Any, page_number: int) -> tuple[dict[str, Any], int]:
    # use_text_flow preserves the PDF content-stream order.  Geometry remains
    # available so the later parser can independently reconstruct columns and
    # suppress repeated navigation furniture.
    extracted = page.extract_words(
        use_text_flow=True,
        keep_blank_chars=False,
        extra_attrs=["fontname", "size"],
        expand_ligatures=True,
    )
    words = [_compact_word(word) for word in extracted]
    payload = {
        "number": page_number,
        "width": _number(page.width),
        "height": _number(page.height),
        "words": words,
        "images": [_compact_image(image) for image in page.images],
    }
    return payload, len(words)


def export_pdf(
    source_path: Path,
    output_path: Path,
    *,
    first_page: int = 1,
    last_page: int | None = None,
    overwrite: bool = False,
) -> PdfExportSummary:
    """Export native PDF words and geometry to a versioned JSON artifact.

    Page numbers are one-based and refer to physical PDF pages, not printed
    page labels.  The artifact is streamed page-by-page to a temporary file and
    atomically installed only after a complete successful export.
    """
    source_path = source_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path == output_path:
        raise ValueError("source and output paths must differ")
    if first_page < 1:
        raise ValueError("first_page must be at least 1")
    if last_page is not None and last_page < first_page:
        raise ValueError("last_page must be greater than or equal to first_page")
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)

    pdfplumber = _load_pdfplumber()
    source_hash = _sha256(source_path)
    source_size = source_path.stat().st_size
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with pdfplumber.open(source_path) as pdf:
            source_pages = len(pdf.pages)
            selected_last = source_pages if last_page is None else last_page
            if first_page > source_pages or selected_last > source_pages:
                raise ValueError(
                    f"page range {first_page}-{selected_last} exceeds "
                    f"the {source_pages}-page PDF"
                )

            header = {
                "schema_version": PDF_EXPORT_SCHEMA_VERSION,
                "extractor": {
                    "name": "pf2e-codex-native-pdf",
                    "profile_version": EXTRACTOR_PROFILE_VERSION,
                    "backend": "pdfplumber",
                    "backend_version": pdfplumber.__version__,
                    "ocr": False,
                },
                "source": {
                    "filename": source_path.name,
                    "sha256": source_hash,
                    "size": source_size,
                    "page_count": source_pages,
                },
                "selection": {
                    "first_page": first_page,
                    "last_page": selected_last,
                },
            }

            total_words = 0
            exported_pages = 0
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as target:
                temporary_path = Path(target.name)
                target.write("{")
                for index, (key, value) in enumerate(header.items()):
                    if index:
                        target.write(",")
                    target.write(json.dumps(key))
                    target.write(":")
                    json.dump(value, target, ensure_ascii=False, separators=(",", ":"))
                target.write(',"pages":[')

                for page_number in range(first_page, selected_last + 1):
                    if exported_pages:
                        target.write(",")
                    payload, word_count = _page_payload(pdf.pages[page_number - 1], page_number)
                    json.dump(payload, target, ensure_ascii=False, separators=(",", ":"))
                    total_words += word_count
                    exported_pages += 1

                target.write("]}")
                target.flush()
                os.fsync(target.fileno())

        os.replace(temporary_path, output_path)
        temporary_path = None
        return PdfExportSummary(
            output_path=output_path,
            source_sha256=source_hash,
            source_pages=source_pages,
            exported_pages=exported_pages,
            words=total_words,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
