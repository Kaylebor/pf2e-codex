"""Loss-minimal native-text PDF export for locally owned rulebooks.

This module deliberately stops at PDF structure.  It records words, fonts,
coordinates, and image bounds in content-stream order; corpus-specific cleanup
and section construction belong to a separate parsing stage.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

PDF_EXPORT_SCHEMA_VERSION = 1
EXTRACTOR_PROFILE_VERSION = 1
NATIVE_WORD_INVENTORY_VERSION = "native-words-v1"

_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_PAGE_NUMBER_RE = re.compile(r"^[-–—]?\s*\d{1,4}\s*[-–—]?$")
_SPACE_RE = re.compile(r"\s+")
_PZO_RE = re.compile(r"(?i)(PZO\d+)")
_ORDER_OR_ID_RE = re.compile(r"(?i)(?:order|customer|account|id|ref(?:erence)?)?\s*[-:#]?\s*\d{4,}")
_PERSON_NAME_RE = re.compile(r"\b[A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,}){1,3}\b")
_TRUSTED_PDF_ORIGIN = object()


class PdfExportDependencyError(RuntimeError):
    """Raised when the optional native PDF dependency is unavailable."""


def _freeze(value: object) -> object:
    """Recursively freeze private bridge data before it crosses a trust boundary."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: object) -> object:
    """Materialize immutable bridge data only for local parser work."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class IgnoredAnchor(Mapping[str, str]):
    """An immutable, private exclusion with a constrained reason."""

    anchor_hash: str
    reason: str

    def __getitem__(self, key: str) -> str:
        if key == "anchor_hash":
            return self.anchor_hash
        if key == "reason":
            return self.reason
        raise KeyError(key)

    def __iter__(self):
        return iter(("anchor_hash", "reason"))

    def __len__(self) -> int:
        return 2


@dataclass(frozen=True)
class PdfExportSummary:
    """Summary returned after a successful export."""

    output_path: Path
    source_sha256: str
    source_pages: int
    exported_pages: int
    words: int


@dataclass(frozen=True)
class NativeWordInventory:
    """Private, deterministic coverage contract for one native export.

    ``anchors`` and ``ignored_anchors`` deliberately never belong in a public
    corpus projection.  They are transient inputs to the private review
    workspace, where they prove that parsing has not silently skipped source
    words.
    """

    content_fingerprint: str
    anchors: tuple[str, ...]
    ignored_anchors: tuple[IgnoredAnchor, ...]
    word_anchors: Mapping[tuple[int, int], str]
    ignored_anchor_reasons: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchors", tuple(self.anchors))
        object.__setattr__(
            self, "ignored_anchors", tuple(
                item if isinstance(item, IgnoredAnchor) else IgnoredAnchor(item["anchor_hash"], item["reason"])
                for item in self.ignored_anchors
            ),
        )
        object.__setattr__(self, "word_anchors", _freeze(self.word_anchors))
        object.__setattr__(self, "ignored_anchor_reasons", _freeze(self.ignored_anchor_reasons))

    def __repr__(self) -> str:
        return (
            "NativeWordInventory(anchors="
            f"{len(self.anchors)}, ignored={len(self.ignored_anchors)})"
        )


@dataclass(frozen=True)
class VerifiedNativeExport:
    """Validated, in-memory exporter artifact safe to hand to a parser.

    It intentionally retains no filesystem path or personalized PDF hash in
    its representation.  Verification happens before any parser sees words.
    """

    payload: Mapping[str, Any] = field(repr=False)
    product_code: str
    source_basename: str
    page_count: int
    extractor_profile_version: int
    pdf_verified: bool
    attestation_digest: str = field(repr=False, default="")
    product_evidence: Mapping[str, object] = field(
        repr=False, default_factory=lambda: MappingProxyType({})
    )
    _verification_token: object | None = field(repr=False, default=None, compare=False)
    _payload_digest: str = field(repr=False, default="", compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze(self.payload))
        object.__setattr__(self, "product_evidence", _freeze(self.product_evidence))

    def __repr__(self) -> str:
        return (
            "VerifiedNativeExport(product_verified="
            f"{self.pdf_verified}, pages={self.page_count}, profile={self.extractor_profile_version})"
        )


def _normalize_word_text(value: object) -> str | None:
    """Return normalized native text without making a PII classification.

    Watermark words must remain in the private inventory: deleting them before
    anchors are assigned would let an incomplete parser appear complete.
    """
    if not isinstance(value, str):
        return None
    normalized = _SPACE_RE.sub(" ", value).strip()
    return normalized or None


def _quantize(value: object) -> int:
    """Use a fixed geometry grid so harmless PDF float noise is irrelevant."""
    try:
        return int(round(float(value) * 10))
    except (TypeError, ValueError):
        return 0


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"native PDF export has invalid {field} geometry")
    return float(value)


def _product_from_basename(value: object) -> str | None:
    if not isinstance(value, str) or value.replace("\\", "/").rsplit("/", 1)[-1] != value:
        return None
    match = _PZO_RE.search(value)
    return match.group(1).upper() if match else None


def _validate_native_export_payload(payload: Mapping[str, Any], *, product_code: str) -> tuple[str, int, int]:
    """Strictly validate exporter-owned structure without reading a second file."""
    if payload.get("schema_version") != PDF_EXPORT_SCHEMA_VERSION:
        raise ValueError("unsupported native PDF export schema")
    if not re.fullmatch(r"PZO\d+", product_code):
        raise ValueError("native export requires a PZO product code")
    source = payload.get("source")
    extractor = payload.get("extractor")
    pages = payload.get("pages")
    if not isinstance(source, Mapping) or not isinstance(extractor, Mapping) or not isinstance(pages, Sequence) or isinstance(pages, (str, bytes)):
        raise ValueError("native PDF export is missing source, extractor, or pages")
    if set(extractor) - {"name", "profile_version", "backend", "backend_version", "ocr"}:
        raise ValueError("native PDF export contains unsupported extractor provenance")
    if set(source) - {"filename", "sha256", "size", "page_count"}:
        raise ValueError("native PDF export contains unsupported source provenance")
    filename = source.get("filename")
    source_product = _product_from_basename(filename)
    if source_product != product_code:
        raise ValueError("native PDF export filename does not match the selected PZO product")
    if (
        extractor.get("name") != "pf2e-codex-native-pdf"
        or extractor.get("backend") != "pdfplumber"
        or extractor.get("ocr") is not False
        or isinstance(extractor.get("profile_version"), bool)
        or not isinstance(extractor.get("profile_version"), int)
    ):
        raise ValueError("native PDF export must be produced by pdfplumber with OCR disabled")
    page_count = source.get("page_count")
    if isinstance(page_count, bool) or not isinstance(page_count, int) or page_count < 1 or len(pages) != page_count:
        raise ValueError("native PDF export page count is inconsistent")
    if not isinstance(source.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", str(source["sha256"])):
        raise ValueError("native PDF export source SHA-256 is invalid")
    if isinstance(source.get("size"), bool) or not isinstance(source.get("size"), int) or int(source["size"]) < 1:
        raise ValueError("native PDF export source size is invalid")
    for expected_page, page in enumerate(pages, start=1):
        if not isinstance(page, Mapping) or page.get("number") != expected_page:
            raise ValueError("native PDF export pages must be unique and ordered from 1")
        if set(page) - {"number", "width", "height", "words", "images"}:
            raise ValueError("native PDF export page contains unsupported fields")
        width = _finite_number(page.get("width"), "page")
        height = _finite_number(page.get("height"), "page")
        if width <= 0 or height <= 0 or not isinstance(page.get("words"), Sequence) or isinstance(page.get("words"), (str, bytes)) or not isinstance(page.get("images"), Sequence) or isinstance(page.get("images"), (str, bytes)):
            raise ValueError("native PDF export page structure is invalid")
        for word in page["words"]:
            if not isinstance(word, Mapping) or not isinstance(word.get("text"), str):
                raise ValueError("native PDF export word is invalid")
            if set(word) - {"text", "x0", "top", "x1", "bottom", "font", "size", "upright", "direction"}:
                raise ValueError("native PDF export word contains unsupported fields")
            x0 = _finite_number(word.get("x0"), "word")
            top = _finite_number(word.get("top"), "word")
            x1 = _finite_number(word.get("x1"), "word")
            bottom = _finite_number(word.get("bottom"), "word")
            size = _finite_number(word.get("size"), "word")
            if x1 < x0 or bottom < top or size <= 0:
                raise ValueError("native PDF export word geometry is invalid")
        for image in page["images"]:
            if not isinstance(image, Mapping):
                raise ValueError("native PDF export image is invalid")
            if set(image) - {"name", "x0", "top", "x1", "bottom"}:
                raise ValueError("native PDF export image contains unsupported fields")
            x0 = _finite_number(image.get("x0"), "image")
            top = _finite_number(image.get("top"), "image")
            x1 = _finite_number(image.get("x1"), "image")
            bottom = _finite_number(image.get("bottom"), "image")
            if x1 < x0 or bottom < top:
                raise ValueError("native PDF export image geometry is invalid")
    return str(filename), page_count, int(extractor["profile_version"])


def load_untrusted_native_export(
    export_path: Path | str,
    *,
    product_code: str,
) -> VerifiedNativeExport:
    """Load a cached export for exploration only.

    A JSON export can be edited independently of its PDF.  It therefore never
    establishes the release/staging trust root, even when neighbouring bytes
    happen to match an on-disk PDF.
    """
    payload = json.loads(Path(export_path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("native PDF export root must be an object")
    basename, page_count, profile_version = _validate_native_export_payload(payload, product_code=product_code)
    return VerifiedNativeExport(
        payload=payload,
        product_code=product_code,
        source_basename=basename,
        page_count=page_count,
        extractor_profile_version=profile_version,
        pdf_verified=False,
    )


def _sha256_open_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    stream.seek(0)
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    stream.seek(0)
    return digest.hexdigest()


def trusted_payload_digest(payload: Mapping[str, Any]) -> str:
    """Private integrity digest for the in-memory extraction payload."""
    canonical = json.dumps(_plain(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalized_marker(value: str) -> str:
    return _SPACE_RE.sub(" ", value).casefold().strip()


def _marker_matches(text: str, marker: str) -> bool:
    """Match complete normalized title tokens, never a bare substring."""
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])", text))


def _title_marker_evidence(
    pages: Sequence[Mapping[str, Any]],
    *,
    selected_product_code: str,
    selected_markers: Sequence[str],
    catalog_markers: Mapping[str, Sequence[str]],
) -> Mapping[str, object]:
    """Return aggregate catalog evidence without retaining title text.

    The selected title must match one of its complete markers.  All catalog
    products are checked, so a Player Core 2 title cannot be accepted as Player
    Core merely because the shorter marker is a prefix.  Requiring the longest
    marker is too strict for real covers whose logo inserts or omits
    ``Pathfinder`` while still exposing the unambiguous book title.
    """
    normalized_catalog = {
        code: tuple(sorted({_normalized_marker(marker) for marker in markers if marker.strip()}, key=len, reverse=True))
        for code, markers in catalog_markers.items()
    }
    selected = tuple(sorted({_normalized_marker(marker) for marker in selected_markers if marker.strip()}, key=len, reverse=True))
    if not selected or selected_product_code not in normalized_catalog:
        raise ValueError("selected product lacks catalog title markers")
    first_words = " ".join(
        str(word.get("text", ""))
        for page in pages[:4]
        for word in page.get("words", [])
        if isinstance(word, Mapping)
    )
    normalized = _normalized_marker(first_words)
    matched = tuple(sorted(
        code for code, markers in normalized_catalog.items()
        if any(_marker_matches(normalized, marker) for marker in markers)
    ))
    strongest = next((marker for marker in selected if _marker_matches(normalized, marker)), selected[0])
    selected_exact = any(_marker_matches(normalized, marker) for marker in selected)
    # A shorter catalog title is expected to appear inside a selected longer
    # edition title (Player Core inside Player Core 2); it is not a conflict.
    conflicts = tuple(
        code for code in matched
        if code != selected_product_code
        and not any(strongest.startswith(marker + " ") for marker in normalized_catalog[code])
    )
    return MappingProxyType({
        "title_marker_verified": selected_exact and not conflicts,
        "matched_product_codes": matched,
        "conflict_product_codes": conflicts,
    })


def verified_native_export_from_pdf(
    source_pdf: Path | str,
    *,
    product_code: str,
    expected_title_markers: Sequence[str],
    catalog_title_markers: Mapping[str, Sequence[str]] | None = None,
) -> VerifiedNativeExport:
    """Extract and attest a PDF in one opened file-handle lifecycle.

    This is the sole trusted exporter entry point.  Cached JSON is deliberately
    not consulted, so a forged cache cannot affect a trusted parse bundle.
    """
    pdf_path = Path(source_pdf).expanduser().resolve()
    if not pdf_path.is_file() or _product_from_basename(pdf_path.name) != product_code:
        raise ValueError("source PDF does not match the selected PZO product")
    pdfplumber = _load_pdfplumber()
    with pdf_path.open("rb") as stream:
        source_size = os.fstat(stream.fileno()).st_size
        if source_size < 1:
            raise ValueError("source PDF is empty")
        source_hash = _sha256_open_stream(stream)
        with pdfplumber.open(stream) as pdf:
            page_count = len(pdf.pages)
            if page_count < 1:
                raise ValueError("source PDF has no pages")
            pages: list[dict[str, Any]] = []
            for page_number, page in enumerate(pdf.pages, start=1):
                page_payload, _word_count = _page_payload(page, page_number)
                pages.append(page_payload)
    catalog = catalog_title_markers or {product_code: expected_title_markers}
    evidence = _title_marker_evidence(
        pages, selected_product_code=product_code, selected_markers=expected_title_markers,
        catalog_markers=catalog,
    )
    if not evidence["title_marker_verified"]:
        raise ValueError("source PDF lacks unambiguous expected product title evidence")
    payload: dict[str, Any] = {
        "schema_version": PDF_EXPORT_SCHEMA_VERSION,
        "extractor": {
            "name": "pf2e-codex-native-pdf",
            "profile_version": EXTRACTOR_PROFILE_VERSION,
            "backend": "pdfplumber",
            "backend_version": getattr(pdfplumber, "__version__", "unknown"),
            "ocr": False,
        },
        "source": {
            "filename": pdf_path.name,
            "sha256": source_hash,
            "size": source_size,
            "page_count": page_count,
        },
        "selection": {"first_page": 1, "last_page": page_count},
        "pages": pages,
    }
    basename, verified_pages, profile_version = _validate_native_export_payload(payload, product_code=product_code)
    attestation = hashlib.sha256(
        (f"trusted-pdf-attestation-v1\n{product_code}\n{source_hash}\n{source_size}\n{verified_pages}\n{profile_version}").encode()
    ).hexdigest()
    return VerifiedNativeExport(
        payload=MappingProxyType(payload), product_code=product_code, source_basename=basename,
        page_count=verified_pages, extractor_profile_version=profile_version, pdf_verified=True,
        attestation_digest=attestation,
        product_evidence=evidence,
        _verification_token=_TRUSTED_PDF_ORIGIN,
        _payload_digest=trusted_payload_digest(payload),
    )


# Compatibility name for callers that only inspect cached artifacts.  It is
# intentionally untrusted and cannot be accepted by parse_verified_native_export.
load_verified_native_export = load_untrusted_native_export


def _native_word_record(page_number: int, ordinal: int, word: Mapping[str, Any]) -> dict[str, object] | None:
    text = _normalize_word_text(word.get("text"))
    if text is None:
        return None
    return {
        "page": page_number,
        "ordinal": ordinal,
        "text": text,
        "x0": _quantize(word.get("x0")),
        "top": _quantize(word.get("top")),
        "x1": _quantize(word.get("x1")),
        "bottom": _quantize(word.get("bottom")),
        "size": _quantize(word.get("size")),
        "font": str(word.get("font") or word.get("fontname") or ""),
        "upright": bool(word.get("upright", True)),
        "direction": str(word.get("direction") or "ltr"),
    }


def _native_rows(pages: Sequence[Mapping[str, Any]]) -> list[tuple[int, list[tuple[int, Mapping[str, Any]]]]]:
    """Group native words into visual rows for transient PII/furniture checks."""
    rows: list[tuple[int, list[tuple[int, Mapping[str, Any]]]]] = []
    for page in pages:
        page_number = int(page["number"])
        ordered = sorted(
            [(index, word) for index, word in enumerate(page["words"]) if isinstance(word, Mapping)],
            key=lambda item: (float(item[1]["top"]), float(item[1]["x0"]), item[0]),
        )
        groups: list[list[tuple[int, Mapping[str, Any]]]] = []
        for item in ordered:
            if not groups or abs(float(item[1]["top"]) - float(groups[-1][0][1]["top"])) > 2.2:
                groups.append([item])
            else:
                groups[-1].append(item)
        rows.extend((page_number, row) for row in groups)
    return rows


def _raw_row_text(row: Sequence[tuple[int, Mapping[str, Any]]]) -> str:
    return " ".join(str(word.get("text", "")) for _ordinal, word in row).strip()


def _row_is_margin(page: Mapping[str, Any], row: Sequence[tuple[int, Mapping[str, Any]]]) -> bool:
    height = float(page["height"])
    top = min(float(word["top"]) for _ordinal, word in row)
    bottom = max(float(word["bottom"]) for _ordinal, word in row)
    return top <= height * 0.18 or bottom >= height * 0.82


def _row_is_watermark_margin(page: Mapping[str, Any], row: Sequence[tuple[int, Mapping[str, Any]]]) -> bool:
    """Use a narrower perimeter than ordinary printed furniture."""
    height = float(page["height"])
    top = min(float(word["top"]) for _ordinal, word in row)
    bottom = max(float(word["bottom"]) for _ordinal, word in row)
    return top <= height * 0.06 or bottom >= height * 0.85


def _watermark_word_reasons(pages: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], str]:
    """Quarantine only proven margin-located watermark spans.

    Values never escape this function. A 1–4 row window supports PDFs which
    split an account email across drawing rows while refusing body ``@`` text.
    """
    rows = _native_rows(pages)
    by_page = {int(page["number"]): page for page in pages}
    quarantined: dict[tuple[int, int], str] = {}
    row_texts = [(page, row, _raw_row_text(row)) for page, row in rows]
    for index, (page, _row, _text) in enumerate(row_texts):
        window: list[tuple[int, Sequence[tuple[int, Mapping[str, Any]]], str]] = []
        for candidate_index in range(index, min(index + 4, len(row_texts))):
            candidate_page, candidate_row, candidate_text = row_texts[candidate_index]
            if candidate_page != page or not _row_is_watermark_margin(by_page[page], candidate_row):
                break
            if window and float(candidate_row[0][1]["top"]) - max(
                float(word["bottom"]) for _ordinal, word in window[-1][1]
            ) > 28.0:
                break
            window.append((candidate_page, candidate_row, candidate_text))
            compact = _SPACE_RE.sub("", "".join(item[2] for item in window))
            email_match = _EMAIL_RE.search(compact)
            # ``@actor.level`` must remain ordinary rule syntax.  The regex
            # can otherwise start after ``@`` and match ``actor.level``.
            if email_match is not None and "@" in email_match.group(0):
                for _candidate_page, candidate_row, _candidate_text in window:
                    for ordinal, _word in candidate_row:
                        quarantined[(page, ordinal)] = "watermark-email-span-v1"
                break
    repeated: dict[str, set[int]] = {}
    for page, _row, text in row_texts:
        normalized = _SPACE_RE.sub(" ", text).casefold().strip()
        if normalized:
            repeated.setdefault(normalized, set()).add(page)
    for page, row, text in row_texts:
        normalized = _SPACE_RE.sub(" ", text).casefold().strip()
        # A repeated name/order line is a watermark even when it carries no
        # email address.  Ordinary repeated headers lack these personal cues.
        header_words = {"pathfinder", "player", "core", "rulebook", "gm", "monster"}
        looks_like_product_header = bool(set(normalized.split()) & header_words)
        if _row_is_watermark_margin(by_page[page], row) and len(repeated.get(normalized, set())) >= 2 and (
            _ORDER_OR_ID_RE.search(text) is not None
            or (_PERSON_NAME_RE.search(text) is not None and not looks_like_product_header)
        ):
            for ordinal, _word in row:
                quarantined[(page, ordinal)] = "watermark-identity-row-v1"
    return quarantined


def native_word_inventory(
    payload: Mapping[str, Any], product_code: str, *, strict: bool = False
) -> NativeWordInventory:
    """Recompute the canonical, watermark-independent native-word contract.

    This operates on raw exporter words *before* section grouping.  Proven
    watermark words receive private anchors and quarantined reasons so they
    cannot disappear from completeness accounting.
    """
    if strict:
        _validate_native_export_payload(payload, product_code=product_code)
    elif payload.get("schema_version") != PDF_EXPORT_SCHEMA_VERSION or not re.fullmatch(r"PZO\d+", product_code):
        raise ValueError("unsupported native PDF export schema or product")
    records: list[tuple[dict[str, object], int, int]] = []
    word_anchors: dict[tuple[int, int], str] = {}
    pages = payload.get("pages")
    if not isinstance(pages, Sequence) or isinstance(pages, (str, bytes)):
        raise ValueError("native PDF export pages must be a list")
    typed_pages = [page for page in pages if isinstance(page, Mapping)]
    if len(typed_pages) != len(pages):
        raise ValueError("native PDF export page must be an object")
    page_by_number = {int(page.get("number", 0)): page for page in typed_pages}
    watermark_reasons = _watermark_word_reasons(typed_pages)
    for page in typed_pages:
        if not isinstance(page, Mapping):
            raise ValueError("native PDF export page must be an object")
        try:
            page_number = int(page.get("number"))
        except (TypeError, ValueError) as exc:
            raise ValueError("native PDF export page is missing its number") from exc
        words = page.get("words")
        if not isinstance(words, Sequence) or isinstance(words, (str, bytes)):
            raise ValueError("native PDF export page words must be a list")
        for raw_ordinal, word in enumerate(words):
            record = _native_word_record(page_number, 0, word)
            if record is None:
                continue
            # The inventory still contains this source word, but its opaque
            # anchor must not depend on personalized watermark characters.
            # Keep a shape-neutral placeholder only after transient detection.
            if (page_number, raw_ordinal) in watermark_reasons:
                record["text"] = "<watermark-quarantine-v1>"
            records.append((record, page_number, raw_ordinal))
    if not records:
        raise ValueError("native PDF export contains no non-PII words")

    # Raw extraction ordering is not semantic.  Canonicalize records by
    # quantized geometry/text first, then number only true identical siblings.
    records.sort(key=lambda item: tuple(item[0][key] for key in ("page", "top", "x0", "bottom", "x1", "text", "font", "size", "direction")))
    record_anchors: list[tuple[dict[str, object], str]] = []
    occurrences: dict[str, int] = {}
    for record, page_number, raw_ordinal in records:
        anchor_record = {key: value for key, value in record.items() if key != "ordinal"}
        canonical = json.dumps(anchor_record, sort_keys=True, separators=(",", ":"))
        occurrence = occurrences.get(canonical, 0)
        occurrences[canonical] = occurrence + 1
        anchor = hashlib.sha256((f"native-word-anchor-v1\n{occurrence}\n" + canonical).encode()).hexdigest()
        record_anchors.append((record, anchor))
        word_anchors[(page_number, raw_ordinal)] = anchor

    # Only narrowly evidenced furniture may be ignored.  It must recur on at
    # least two pages at the outer margin, while page numbers are constrained
    # to their conventional bottom margin.  Everything else must later be
    # assigned to exactly one parser section.
    row_anchors: list[tuple[int, list[str], str, float, float]] = []
    for page_number, row in _native_rows(typed_pages):
        anchors = [word_anchors[(page_number, ordinal)] for ordinal, _word in row if (page_number, ordinal) in word_anchors]
        if anchors:
            row_anchors.append((page_number, anchors, _raw_row_text(row), float(row[0][1]["top"]), max(float(word["bottom"]) for _ordinal, word in row)))
    repeated_rows: dict[str, set[int]] = {}
    for page_number, _anchors, text, top, bottom in row_anchors:
        page = page_by_number[page_number]
        height = float(page["height"])
        if top < height * 0.12 or bottom > height * 0.88:
            normalized = _SPACE_RE.sub(" ", re.sub(r"\b\d+\b", "", text)).casefold().strip()
            if normalized:
                repeated_rows.setdefault(normalized, set()).add(page_number)
    ignored_reasons: dict[str, str] = {
        word_anchors[key]: reason for key, reason in watermark_reasons.items() if key in word_anchors
    }
    for page_number, anchors, text, top, bottom in row_anchors:
        page = page_by_number[page_number]
        height = float(page["height"])
        reason: str | None = None
        if height and bottom > height * 0.88 and _PAGE_NUMBER_RE.match(text):
            reason = "printed-page-number-v1"
        elif height and (top < height * 0.12 or bottom > height * 0.88):
            key = _SPACE_RE.sub(" ", re.sub(r"\b\d+\b", "", text)).casefold().strip()
            if key and len(repeated_rows.get(key, set())) >= 2:
                reason = "repeated-margin-furniture-v1"
        if reason:
            for anchor in anchors:
                ignored_reasons.setdefault(anchor, reason)
    ignored = tuple(
        IgnoredAnchor(anchor_hash=anchor, reason=ignored_reasons[anchor])
        for anchor in sorted(ignored_reasons)
    )
    ignored_hashes = set(ignored_reasons)
    # Revision identity captures source words, but deliberately excludes
    # extractor ordering, geometry/fonts, repeated furniture, and PII.
    semantic_pages: list[dict[str, object]] = []
    for page_number, row in _native_rows(typed_pages):
        words: list[str] = []
        for ordinal, word in sorted(row, key=lambda item: (float(item[1]["x0"]), item[0])):
            anchor = word_anchors.get((page_number, ordinal))
            text = _normalize_word_text(word.get("text"))
            if anchor is not None and anchor not in ignored_hashes and text is not None:
                words.append(text.casefold())
        if words:
            if not semantic_pages or semantic_pages[-1]["page"] != page_number:
                semantic_pages.append({"page": page_number, "lines": []})
            semantic_pages[-1]["lines"].append(words)
    canonical = json.dumps(
        semantic_pages,
        sort_keys=True, separators=(",", ":"),
    )
    fingerprint = hashlib.sha256((f"native-semantic-revision-v1\n{product_code}\n" + canonical).encode()).hexdigest()
    return NativeWordInventory(
        content_fingerprint=fingerprint,
        anchors=tuple(anchor for _record, anchor in record_anchors),
        ignored_anchors=ignored,
        word_anchors=MappingProxyType(word_anchors),
        ignored_anchor_reasons=MappingProxyType(ignored_reasons),
    )


def annotate_native_words(payload: Mapping[str, Any], inventory: NativeWordInventory) -> dict[str, Any]:
    """Copy an export and attach private anchors without rewriting the artifact."""
    copied = _plain(payload)
    if not isinstance(copied, dict):
        raise ValueError("native PDF export root must be an object")
    for page in copied.get("pages", []):
        if not isinstance(page, dict):
            continue
        page_number = int(page.get("number", 0))
        for ordinal, word in enumerate(page.get("words", [])):
            if not isinstance(word, dict):
                continue
            anchor = inventory.word_anchors.get((page_number, ordinal))
            if anchor is None:
                word["_native_excluded"] = True
            else:
                word["_native_anchor"] = anchor
                reason = inventory.ignored_anchor_reasons.get(anchor)
                if reason is not None:
                    word["_native_excluded"] = True
                    word["_native_ignore_reason"] = reason
    return copied


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
