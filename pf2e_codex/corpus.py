"""Discovery, revision selection, and parsing for user-owned PF2E rulebooks.

The module intentionally consumes :mod:`pf2e_codex.pdf_export` artifacts.  It
does not run OCR or invoke another PDF text extractor.  Source files and the
selection state belong below the caller's ignored corpus root.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import median
from types import MappingProxyType
from typing import Any

from .pdf_export import (
    _TRUSTED_PDF_ORIGIN,
    PDF_EXPORT_SCHEMA_VERSION,
    NativeWordInventory,
    VerifiedNativeExport,
    annotate_native_words,
    export_pdf,
    native_word_inventory,
    trusted_payload_digest,
    verified_native_export_from_pdf,
)

SELECTION_STATE_FILENAME = ".pf2e-codex-corpus-selection.json"
CORPUS_SCHEMA_VERSION = 1
PAIZO_NATIVE_PARSER_V1 = "paizo-native-v1"
PAIZO_NATIVE_PARSER_V2 = "paizo-native-v2"
# V3 is deliberately opt-in for staged review runs.  It retains V2's reading
# order and only rejects a narrowly evidenced class of false headings inside
# recurring two-cell regions.
PAIZO_NATIVE_PARSER_V3 = "paizo-native-v3"
PAIZO_NATIVE_LAYOUT_V1 = "paizo-native-v3+pp-doclayout-v3-v1"
PAIZO_NATIVE_PARSER_V4 = "paizo-native-v4"
# V5 is the deterministic repair profile. It preserves the V4 native-word and
# layout evidence contract, but promotes ambiguous rule-bearing layout into
# exact native-text sections carrying fail-closed review flags. Non-rule
# quarantine remains quarantine.
PAIZO_NATIVE_PARSER_V5 = "paizo-native-v5"
PAIZO_NATIVE_PARSER_VERSION = PAIZO_NATIVE_PARSER_V1


@dataclass(frozen=True)
class ProductSpec:
    """Known Paizo product metadata used for provenance and filtering."""

    code: str
    title: str
    component: str
    rules_era: str
    license: str
    remaster: bool
    artifact_subdir: str


PRODUCT_CATALOG: Mapping[str, ProductSpec] = {
    "PZO2101": ProductSpec(
        code="PZO2101",
        title="Pathfinder Core Rulebook",
        component="core-rulebook",
        rules_era="legacy",
        license="OGL",
        remaster=False,
        artifact_subdir="pre-remaster/core-rulebook-4th-printing",
    ),
    "PZO12001": ProductSpec(
        code="PZO12001",
        title="Pathfinder Player Core",
        component="player-core",
        rules_era="remaster",
        license="ORC",
        remaster=True,
        artifact_subdir="remaster/player-core",
    ),
    "PZO12002": ProductSpec(
        code="PZO12002",
        title="Pathfinder GM Core",
        component="gm-core",
        rules_era="remaster",
        license="ORC",
        remaster=True,
        artifact_subdir="remaster/gm-core",
    ),
    "PZO12003": ProductSpec(
        code="PZO12003",
        title="Pathfinder Monster Core",
        component="monster-core",
        rules_era="remaster",
        license="ORC",
        remaster=True,
        artifact_subdir="remaster/monster-core",
    ),
    "PZO12004": ProductSpec(
        code="PZO12004",
        title="Pathfinder Player Core 2",
        component="player-core-2",
        rules_era="remaster",
        license="ORC",
        remaster=True,
        artifact_subdir="remaster/player-core-2",
    ),
}


@dataclass(frozen=True)
class CorpusSource:
    """A discovered PDF, including a PDF member inside a ZIP archive."""

    product: ProductSpec
    path: Path
    member: str | None
    part: str | None
    combined: bool
    printing: int | None
    source_name: str
    source_sha256: str
    mtime_ns: int

    @property
    def display_name(self) -> str:
        # ``member`` is deliberately kept only for archive access.  ZIPs may
        # contain personalized directory names, so all user-visible and
        # persisted source names use the sanitized member basename instead.
        # The outer archive name is intentionally omitted: it can itself be a
        # personalized filename (for example, ``buyer@example.invalid.zip``).
        return _safe_basename(self.source_name)

    @property
    def candidate_id(self) -> str:
        # The content hash is provenance, not product identity.  It makes the
        # persisted selection robust to a copied source while the normalized
        # name keeps two same-byte parts distinguishable.
        return ":".join(
            (
                self.product.code,
                _normalize_name(self.source_name),
                self.part or "combined",
                self.source_sha256,
            )
        )

    @property
    def selection_key(self) -> str:
        """Path-independent key used to retain a persisted local choice."""
        return f"{self.part or 'combined'}:{self.source_sha256}"


@dataclass(frozen=True)
class SelectedRevision:
    """Active source set for one catalog product."""

    product: ProductSpec
    sources: tuple[CorpusSource, ...]

    @property
    def combined(self) -> bool:
        return bool(self.sources) and self.sources[0].combined

    @property
    def printing(self) -> int | None:
        values = {source.printing for source in self.sources}
        values.discard(None)
        return max(values) if values else None


@dataclass(frozen=True)
class PreparedExport:
    """Native JSON artifact associated with one selected source."""

    source: CorpusSource
    output_path: Path
    source_sha256: str
    exported: bool
    stale: bool


@dataclass(frozen=True)
class RulebookSection:
    """A provenance-rich searchable section produced from native export JSON."""

    id: str
    name: str
    text: str
    product_code: str
    book: str
    component: str
    rules_era: str
    license: str
    remaster: bool
    source_filename: str
    source_sha256: str
    pages: tuple[int, ...]
    page_start: int
    page_end: int
    printed_page: str | None
    section_hash: str
    provenance: Mapping[str, Any]
    ordinal: int
    parser_version: str = PAIZO_NATIVE_PARSER_VERSION

    def as_chunk(self) -> dict[str, Any]:
        """Return the dict shape consumed by chunk/index integrations."""
        return {
            "id": self.id,
            "name": self.name,
            "text": self.text,
            "type": "rulebook_section",
            "pack": f"corpus-{self.component}",
            "slug": _slug(self.name),
            "level": None,
            "traits": [],
            "raw_rules_count": 0,
            "refs": [],
            "origin": "corpus",
            "source_id": f"paizo:{self.product_code}:{self.component}",
            "source": {
                "source_id": f"paizo:{self.product_code}:{self.component}",
                "source": "paizo-pdf",
                "product": self.product_code,
                "revision": self.provenance.get("content_fingerprint"),
                "parser": self.parser_version,
                "license": self.license,
                "era": self.rules_era,
                "provenance": dict(self.provenance),
            },
            "book": self.book,
            "component": self.component,
            "product_code": self.product_code,
            "rules_era": self.rules_era,
            "license": self.license,
            "remaster": self.remaster,
            "source_filename": self.source_filename,
            "source_sha256": self.source_sha256,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "source_page_start": self.page_start,
            "source_page_end": self.page_end,
            "printed_page": self.printed_page,
            "pages": list(self.pages),
            "section_hash": self.section_hash,
            "source_hash": self.section_hash,
            "source_pages": list(self.pages),
            "publication": {"license": self.license, "remaster": self.remaster},
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True, repr=False)
class TrustedBlock:
    """One ordered native-text block inside a trusted section."""

    kind: str
    physical_page: int
    ordinal: int
    text: str = field(repr=False)
    text_hash: str
    coverage_anchors: tuple[str, ...] = field(repr=False)
    table_cells: tuple[tuple[str, ...], ...] = field(default=(), repr=False)


@dataclass(frozen=True, repr=False)
class TrustedQuarantine:
    """Private native text excluded from semantic review, never from coverage."""

    quarantine_id: str
    reason: str
    physical_page: int
    text: str = field(repr=False)
    text_hash: str
    coverage_anchors: tuple[str, ...] = field(repr=False)


@dataclass(frozen=True, repr=False)
class TrustedSection:
    """Immutable private section record; its repr never exposes source text."""

    id: str
    heading: str
    text: str = field(repr=False)
    text_hash: str
    physical_pages: tuple[int, ...]
    printed_page: str | None
    stable_section_identity: str
    layout_flags: tuple[str, ...]
    coverage_anchors: tuple[str, ...] = field(repr=False)
    blocks: tuple[TrustedBlock, ...] = field(default=(), repr=False)
    # Public-safe provenance handle: product/component/physical page/ordinal
    # only. It contains neither title text nor local asset information.
    source_section_id: str = ""

    def __repr__(self) -> str:
        return f"TrustedSection(page={self.physical_pages[:1] or (0,)}, anchors={len(self.coverage_anchors)})"


@dataclass(frozen=True, repr=False)
class TrustedParseBundle:
    """Deep-sealed private handoff from a PDF-verified parse."""

    product_code: str
    parser_version: str
    exporter_profile_version: int
    semantic_fingerprint: str = field(repr=False)
    artifact_attestation: Mapping[str, object] = field(repr=False)
    inventory: NativeWordInventory = field(repr=False)
    sections: tuple[TrustedSection, ...] = field(repr=False)
    parser_output_digest: str = field(repr=False)
    sealed_digest: str = field(repr=False)
    artifact_attestation_digest: str = field(repr=False, default="")
    layout_binding_digest: str | None = field(repr=False, default=None)
    quarantine: tuple[TrustedQuarantine, ...] = field(default=(), repr=False)

    def __repr__(self) -> str:
        return (
            f"TrustedParseBundle(product={self.product_code}, parser={self.parser_version}, "
            f"sections={len(self.sections)}, anchors={len(self.inventory.anchors)}, "
            f"quarantine={len(self.quarantine)}, ignored={len(self.inventory.ignored_anchors)})"
        )

    def verify_seal(self) -> None:
        """Raise if this immutable bundle no longer matches its canonical seal."""
        if self.parser_version in {PAIZO_NATIVE_PARSER_V4, PAIZO_NATIVE_PARSER_V5}:
            for section in self.sections:
                block_anchors = tuple(
                    anchor for block in section.blocks for anchor in block.coverage_anchors
                )
                if (
                    not section.blocks
                    or tuple(block.ordinal for block in section.blocks)
                    != tuple(range(len(section.blocks)))
                    or block_anchors != section.coverage_anchors
                    or _clean_text(" ".join(block.text for block in section.blocks))
                    != section.text
                ):
                    raise ValueError(
                        "trusted structural section does not match its ordered native blocks"
                    )
            assigned = Counter(
                anchor
                for anchors in (
                    *(section.coverage_anchors for section in self.sections),
                    *(item.coverage_anchors for item in self.quarantine),
                )
                for anchor in anchors
            )
            expected = set(self.inventory.anchors) - set(self.inventory.ignored_anchor_reasons)
            if set(assigned) != expected or any(count != 1 for count in assigned.values()):
                raise ValueError(
                    "trusted parser bundle does not account for every native anchor exactly once"
                )
        parser_digest = _trusted_parser_output_digest(self.sections, self.quarantine)
        if parser_digest != self.parser_output_digest:
            raise ValueError("trusted parser bundle output digest does not match its sections")
        expected = _trusted_bundle_seal(
            product_code=self.product_code, parser_version=self.parser_version,
            exporter_profile_version=self.exporter_profile_version,
            semantic_fingerprint=self.semantic_fingerprint,
            artifact_attestation=self.artifact_attestation,
            artifact_attestation_digest=self.artifact_attestation_digest,
            inventory=self.inventory, sections=self.sections,
            quarantine=self.quarantine,
            parser_output_digest=self.parser_output_digest,
            layout_binding_digest=self.layout_binding_digest,
        )
        if expected != self.sealed_digest:
            raise ValueError("trusted parser bundle seal does not match its immutable fields")


_SOURCE_RE = re.compile(
    r"(?i)(?<![A-Z0-9])(PZO(?:2101|12001|12002|12003|12004))(?P<ebook>E)?(?P<suffix>[^.]*)$"
)
_PRINTING_RE = re.compile(r"(?i)(\d+)\s*(?:st|nd|rd|th)?\s*printing\b")
_PART_RE = re.compile(
    r"(?ix)(?:^|[ _.-])(?:part|pt|volume|vol|book|section)[ _.-]*(\d+)(?:$|[ _.-])"
)
_NUMERIC_PART_RE = re.compile(r"(?:^|[ _.-])(\d+)$")
_PAGE_RANGE_RE = re.compile(r"(?i)(?:^|\s)(\d{3})-(\d{3}|cover)(?:\s|$)")
_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_PAGE_NUMBER_RE = re.compile(r"^[-–—]?\s*\d{1,4}\s*[-–—]?$")
_WHITESPACE_RE = re.compile(r"\s+")
_TRUSTED_SOURCE_SECTION_ID_RE = re.compile(
    r"^pzo\d+:[a-z0-9-]+:p[1-9]\d*:h[0-9a-f]{16}:i\d+$"
)


def _normalize_name(value: str) -> str:
    value = value.casefold().replace("!", " ")
    return _WHITESPACE_RE.sub(" ", re.sub(r"[^a-z0-9]+", " ", value)).strip()


def _contains_email(value: str) -> bool:
    compact = re.sub(r"\s+", "", value)
    return bool(_EMAIL_RE.search(value) or _EMAIL_RE.search(compact))


def _safe_basename(value: str) -> str:
    """Return a path-free source name suitable for logs and provenance."""
    return value.replace("\\", "/").rsplit("/", 1)[-1] or "source.pdf"


def _source_bytes(source: CorpusSource) -> bytes:
    if source.member is None:
        return source.path.read_bytes()
    with zipfile.ZipFile(source.path) as archive:
        with archive.open(source.member) as stream:
            return stream.read()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _printing_marker(suffix: str) -> int | None:
    match = _PRINTING_RE.search(suffix)
    return int(match.group(1)) if match else None


def _zip_member_mtime_ns(info: zipfile.ZipInfo) -> int:
    """Convert ZIP's UTC-independent DOS timestamp tuple to nanoseconds.

    ZIP stores a local-looking ``date_time`` tuple with one-second precision.
    Treating it as UTC through ``timegm`` keeps selection deterministic across
    hosts with different local timezones while retaining every component.
    """
    return calendar.timegm((*info.date_time, 0)) * 1_000_000_000


def _part_number(suffix: str) -> str | None:
    page_range = _PAGE_RANGE_RE.search(suffix)
    if page_range:
        return f"{page_range.group(1).lower()}-{page_range.group(2).lower()}"
    match = _PART_RE.search(suffix)
    if match:
        return match.group(1)
    # PZO12001E-1.pdf and PZO12001E_2.pdf are common split naming forms.  A
    # printing marker wins over this fallback, so ``-4th Printing`` is whole.
    if _printing_marker(suffix) is None:
        match = _NUMERIC_PART_RE.search(suffix.strip())
        if match:
            return match.group(1)
    return None


def _classify_name(name: str) -> tuple[str, str | None, bool, int | None] | None:
    match = _SOURCE_RE.search(Path(name).stem)
    if not match:
        return None
    code = match.group(1).upper()
    suffix = match.group("suffix")
    printing = _printing_marker(suffix)
    part = _part_number(suffix)
    # Paizo combined ebooks use the E suffix. Split archives use the bare
    # product code plus a physical page range, sometimes followed by title or
    # printing text. Synthetic ``E-Part-N`` forms remain accepted for tests
    # and user-renamed copies.
    combined = part is None and bool(match.group("ebook"))
    if part is None and not combined:
        return None
    return code, part, combined, printing


def _make_source(
    path: Path,
    *,
    member: str | None,
    name: str,
    code: str,
) -> CorpusSource:
    product = PRODUCT_CATALOG[code]
    if member is None:
        digest = _sha256_file(path)
        mtime_ns = path.stat().st_mtime_ns
    else:
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo(member)
            digest = hashlib.sha256()
            with archive.open(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            mtime_ns = _zip_member_mtime_ns(info)
    classified = _classify_name(name)
    if classified is None:  # pragma: no cover - guarded by callers
        raise ValueError(f"unrecognized corpus source: {name}")
    _code, part, combined, printing = classified
    return CorpusSource(
        product=product,
        path=path,
        member=member,
        part=part,
        combined=combined,
        printing=printing,
        source_name=_safe_basename(name),
        source_sha256=digest.hexdigest() if hasattr(digest, "hexdigest") else str(digest),
        mtime_ns=mtime_ns,
    )


def discover_sources(
    corpus_root: Path | str,
    *,
    include: str | Iterable[str] | None = None,
    exclude: str | Iterable[str] | None = None,
) -> list[CorpusSource]:
    """Recursively discover catalog PDFs and matching PDF members in ZIPs."""
    root = Path(corpus_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    found: list[CorpusSource] = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item).casefold()):
        if path.is_file() and path.suffix.casefold() == ".pdf":
            classified = _classify_name(path.name)
            if classified:
                found.append(_make_source(path, member=None, name=path.name, code=classified[0]))
        elif path.is_file() and path.suffix.casefold() == ".zip":
            try:
                with zipfile.ZipFile(path) as archive:
                    members = sorted(
                        (info.filename for info in archive.infolist() if not info.is_dir()),
                        key=str.casefold,
                    )
            except (OSError, zipfile.BadZipFile):
                continue
            for member in members:
                if not member.casefold().endswith(".pdf"):
                    continue
                classified = _classify_name(Path(member).name)
                if classified:
                    found.append(
                        _make_source(path, member=member, name=Path(member).name, code=classified[0])
                    )
    return _filter_sources(found, include=include, exclude=exclude)


def group_sources(sources: Iterable[CorpusSource]) -> dict[tuple[str, str], list[CorpusSource]]:
    """Group discovered sources by stable product code and component."""
    grouped: dict[tuple[str, str], list[CorpusSource]] = defaultdict(list)
    for source in sources:
        grouped[(source.product.code, source.product.component)].append(source)
    return dict(grouped)


def _matches(source: CorpusSource, value: str) -> bool:
    token = value.casefold()
    haystack = " ".join(
        (
            source.product.code,
            source.product.component,
            source.product.title,
            source.display_name,
            str(source.path),
        )
    ).casefold()
    return token in haystack


def _filter_sources(
    sources: Iterable[CorpusSource],
    *,
    include: str | Iterable[str] | None,
    exclude: str | Iterable[str] | None,
) -> list[CorpusSource]:
    def terms(value: str | Iterable[str] | None) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        return tuple(value)

    include_terms = terms(include)
    exclude_terms = terms(exclude)
    return [
        source
        for source in sources
        if (not include_terms or any(_matches(source, term) for term in include_terms))
        and not any(_matches(source, term) for term in exclude_terms)
    ]


def _preference_terms(prefer: str | Iterable[str] | Mapping[str, str] | None, code: str) -> tuple[str, ...]:
    if prefer is None:
        return ()
    if isinstance(prefer, Mapping):
        value = prefer.get(code) or prefer.get(code.casefold())
        return (value,) if value else ()
    if isinstance(prefer, str):
        return (prefer,)
    return tuple(prefer)


def _candidate_rank(source: CorpusSource, prefer_terms: Sequence[str]) -> tuple[Any, ...]:
    preferred = 0 if any(_matches(source, term) for term in prefer_terms) else 1
    return (
        preferred,
        0 if source.printing is not None else 1,
        -(source.printing or 0),
        -source.mtime_ns,
        _normalize_name(source.source_name),
        source.source_sha256,
    )


def _candidate_content_fingerprint(
    source: CorpusSource,
    cache_root: Path | None,
) -> str:
    """Return a watermark-independent fingerprint when native JSON is available.

    Successful native exports collapse watermark-only copies. When a corpus
    root is supplied, dependency and parser failures remain explicit: silently
    falling back to the personalized raw hash would misclassify equivalent
    copies as distinct errata revisions.
    """
    if cache_root is None:
        return source.source_sha256
    cache_path = (
        cache_root.expanduser().resolve()
        / ".corpus-selection-cache"
        / f"{source.source_sha256}.json"
    )
    if not _artifact_current(cache_path, source.source_sha256):
        materialized = _materialize_source(source, cache_root)
        export_pdf(materialized, cache_path, overwrite=cache_path.exists())
    chunks = parse_rulebook_export(cache_path, product=source.product, source=source)
    fingerprints = {
        chunk.get("provenance", {}).get("content_fingerprint")
        for chunk in chunks
        if chunk.get("provenance", {}).get("content_fingerprint")
    }
    if len(fingerprints) != 1:
        raise ValueError(
            f"unable to compute one normalized content fingerprint for {source.display_name}"
        )
    return next(iter(fingerprints))


def _choose_candidate(
    candidates: Sequence[CorpusSource],
    prefer_terms: Sequence[str],
    prior_keys: set[str],
    prior_fingerprints: set[str],
    fingerprint_root: Path | None,
) -> CorpusSource:
    """Choose with explicit override and printing metadata ahead of persistence."""
    preferred = [
        source for source in candidates
        if any(_matches(source, term) for term in prefer_terms)
    ]
    pool = preferred or list(candidates)
    if not preferred:
        printings = [source.printing for source in pool if source.printing is not None]
        if printings:
            newest = max(printings)
            pool = [source for source in pool if source.printing == newest]
        fingerprints = {
            source: _candidate_content_fingerprint(source, fingerprint_root)
            for source in pool
        }
        groups: dict[str, list[CorpusSource]] = defaultdict(list)
        for source, fingerprint in fingerprints.items():
            groups[fingerprint].append(source)
        winning_fingerprint = min(
            groups,
            key=lambda fingerprint: _candidate_rank(
                min(groups[fingerprint], key=lambda source: _candidate_rank(source, prefer_terms)),
                prefer_terms,
            ),
        )
        pool = groups[winning_fingerprint]
        if prior_fingerprints and winning_fingerprint in prior_fingerprints:
            persisted = [
                source for source in pool if source.selection_key in prior_keys
            ]
            if persisted:
                pool = persisted
    return min(pool, key=lambda source: _candidate_rank(source, prefer_terms))


def select_revisions(
    sources: Iterable[CorpusSource],
    *,
    include: str | Iterable[str] | None = None,
    exclude: str | Iterable[str] | None = None,
    prefer: str | Iterable[str] | Mapping[str, str] | None = None,
    state_root: Path | str | None = None,
) -> list[SelectedRevision]:
    """Group sources by catalog product and choose active revisions.

    A combined PDF always wins over split files.  For split books, each part is
    selected independently after applying printing, preference, mtime, and
    normalized-name tie-breaks.  The resulting decision is persisted below
    ``state_root`` when supplied.
    """
    filtered = _filter_sources(sources, include=include, exclude=exclude)
    grouped: dict[str, list[CorpusSource]] = defaultdict(list)
    for source in filtered:
        grouped[source.product.code].append(source)
    selected: list[SelectedRevision] = []
    previous: dict[str, Any] = {}
    fingerprint_root: Path | None = None
    if state_root is not None:
        fingerprint_root = Path(state_root).expanduser().resolve()
        previous_path = fingerprint_root / SELECTION_STATE_FILENAME
        try:
            previous = json.loads(previous_path.read_text(encoding="utf-8")).get("selections", {})
        except (OSError, ValueError, TypeError):
            previous = {}
    state: dict[str, Any] = {"schema_version": CORPUS_SCHEMA_VERSION, "selections": {}}
    for code in sorted(grouped):
        product_sources = grouped[code]
        prefer_terms = _preference_terms(prefer, code)
        prior_fingerprints = set(previous.get(code, {}).get("content_fingerprints", []))
        combined = [source for source in product_sources if source.combined]
        if combined:
            prior_keys = set(previous.get(code, {}).get("selection_keys", []))
            chosen = _choose_candidate(
                combined,
                prefer_terms,
                prior_keys,
                prior_fingerprints,
                fingerprint_root,
            )
            active = (chosen,)
        else:
            by_part: dict[str, list[CorpusSource]] = defaultdict(list)
            for source in product_sources:
                by_part[source.part or "unknown"].append(source)
            prior_keys = set(previous.get(code, {}).get("selection_keys", []))
            active = tuple(
                _choose_candidate(
                    part_sources,
                    prefer_terms,
                    prior_keys,
                    prior_fingerprints,
                    fingerprint_root,
                )
                for _part, part_sources in sorted(by_part.items(), key=lambda item: item[0])
            )
        selected.append(SelectedRevision(product=PRODUCT_CATALOG[code], sources=active))
        state["selections"][code] = {
            "component": PRODUCT_CATALOG[code].component,
            "candidate_ids": [source.candidate_id for source in active],
            "selection_keys": [source.selection_key for source in active],
            "content_fingerprints": [
                _candidate_content_fingerprint(source, fingerprint_root) for source in active
            ],
        }
    if state_root is not None:
        root = Path(state_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        state_path = root / SELECTION_STATE_FILENAME
        temporary = state_path.with_suffix(state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(state_path)
    return selected


def _materialize_source(source: CorpusSource, root: Path) -> Path:
    if source.member is None:
        return source.path
    target = (
        root
        / ".materialized"
        / source.product.code
        / source.source_sha256[:16]
        / source.source_name
    )
    if not target.exists() or _sha256_file(target) != source.source_sha256:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_source_bytes(source))
    return target


def _artifact_current(path: Path, source_hash: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return (
        payload.get("schema_version") == PDF_EXPORT_SCHEMA_VERSION
        and payload.get("source", {}).get("sha256") == source_hash
    )


def prepare_exports(
    corpus_root: Path | str,
    revisions: Iterable[SelectedRevision],
    *,
    output_root: Path | str | None = None,
) -> list[PreparedExport]:
    """Create missing/stale native JSON artifacts using ``export_pdf``."""
    root = Path(corpus_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve() if output_root else root / "parsed"
    prepared: list[PreparedExport] = []
    for revision in revisions:
        for source in revision.sources:
            part = source.part or "combined"
            product_dir = destination / revision.product.artifact_subdir
            output = product_dir / ("native-pages.json" if source.combined else f"{part}.json")
            existed = output.is_file()
            current = existed and _artifact_current(output, source.source_sha256)
            if not current:
                materialized = _materialize_source(source, root)
                export_pdf(materialized, output, overwrite=output.exists())
            prepared.append(
                PreparedExport(
                    source=source,
                    output_path=output,
                    source_sha256=source.source_sha256,
                    exported=not current,
                    stale=existed and not current,
                )
            )
    return prepared


def _clean_text(value: str) -> str:
    value = _EMAIL_RE.sub("", re.sub(r"\s+", "", value) if _contains_email(value) else value)
    value = _WHITESPACE_RE.sub(" ", value).strip()
    tokens = value.split()
    if len(tokens) >= 2 and len(tokens) % 2 == 0:
        duplicate_pairs = sum(
            tokens[index].casefold() == tokens[index + 1].casefold()
            for index in range(0, len(tokens), 2)
        )
        if duplicate_pairs / (len(tokens) / 2) >= 0.75:
            value = " ".join(tokens[::2])
    return value


@dataclass(frozen=True)
class _Line:
    text: str
    top: float
    bottom: float
    x0: float
    x1: float
    size: float
    fonts: tuple[str, ...]
    layout_kind: str = "body"
    native_word_anchors: tuple[str, ...] = ()


@dataclass(frozen=True)
class _PageXBandModel:
    """Page-local evidence for reading starts and recurring interior cells."""

    dominant_starts: tuple[float, ...]
    interior_cell_starts: tuple[float, ...]
    tolerance: float

    def is_interior_cell_start(self, line: _Line) -> bool:
        return any(
            abs(line.x0 - start) <= self.tolerance
            for start in self.interior_cell_starts
        )


def _row_break_gap(words: Sequence[Mapping[str, Any]]) -> tuple[list[float], float]:
    """Return local gaps and a type-scaled threshold for one visual row."""
    if len(words) < 2:
        return [], 0.0
    gaps = [
        float(right.get("x0", 0)) - float(left.get("x1", left.get("x0", 0)))
        for left, right in zip(words, words[1:], strict=False)
    ]
    ordinary = [gap for gap in gaps if gap >= 0]
    typical_gap = median(ordinary) if ordinary else 0.0
    sizes = [float(word.get("size", 0)) for word in words]
    type_size = median([size for size in sizes if size > 0]) if any(sizes) else 9.0
    return gaps, max(type_size * 1.25, min(typical_gap * 3.0, type_size * 2.0))


def _page_split_starts(
    rows: Sequence[Sequence[Mapping[str, Any]]], width: float
) -> tuple[float, ...]:
    """Infer recurring region starts before splitting a page's word rows.

    An individual broad gap can be justified body text.  A column/table start
    recurs across independent baselines; use that page-level evidence before
    treating local spacing as a structural split.
    """
    candidates: list[tuple[float, float]] = []
    for row in rows:
        gaps, break_gap = _row_break_gap(row)
        structural = [gap for gap in gaps if gap > break_gap]
        if not structural:
            continue
        largest = max(structural)
        for word, gap in zip(row[1:], gaps, strict=False):
            x0 = float(word.get("x0", 0))
            # A real gutter is the dominant discontinuity in its visual row.
            # Retain near-ties for multi-cell tables, but ignore incidental
            # wide word spacing elsewhere in a paragraph.
            if gap >= largest * 0.7 and x0 >= width * 0.15:
                candidates.append((x0, float(word.get("top", 0))))
    if not candidates:
        return ()
    tolerance = max(width * 0.05, 18.0)
    clusters: list[list[tuple[float, float]]] = []
    for candidate in sorted(candidates):
        if not clusters or candidate[0] - clusters[-1][-1][0] > tolerance:
            clusters.append([candidate])
        else:
            clusters[-1].append(candidate)
    return tuple(
        median([x0 for x0, _top in cluster])
        for cluster in clusters
        if len({round(top, 1) for _x0, top in cluster}) >= 2
    )


def _split_word_row(
    words: Sequence[Mapping[str, Any]], split_starts: Sequence[float], width: float
) -> list[list[Mapping[str, Any]]]:
    """Split a row only at native region starts proven elsewhere on the page."""
    if len(words) < 2 or not split_starts:
        return [list(words)]
    gaps, break_gap = _row_break_gap(words)
    tolerance = max(width * 0.05, 18.0)
    groups: list[list[Mapping[str, Any]]] = [[words[0]]]
    for word, gap in zip(words[1:], gaps, strict=False):
        x0 = float(word.get("x0", 0))
        if gap > break_gap and any(abs(x0 - start) <= tolerance for start in split_starts):
            groups.append([word])
        else:
            groups[-1].append(word)
    return groups


def _lines_for_page_v1(page: Mapping[str, Any]) -> list[_Line]:
    """Frozen v1 reconstruction path retained for the active corpus."""
    words = [
        word for word in page.get("words", [])
        if isinstance(word, Mapping) and not word.get("_native_excluded")
    ]
    words.sort(key=lambda word: (float(word.get("top", 0)), float(word.get("x0", 0))))
    groups: list[list[Mapping[str, Any]]] = []
    for word in words:
        top = float(word.get("top", 0))
        if not groups or abs(top - float(groups[-1][0].get("top", 0))) > 2.2:
            groups.append([word])
        else:
            groups[-1].append(word)
    lines: list[_Line] = []
    for group in groups:
        group.sort(key=lambda word: float(word.get("x0", 0)))
        subgroups: list[list[Mapping[str, Any]]] = []
        for word in group:
            if not subgroups:
                subgroups.append([word])
                continue
            previous = subgroups[-1][-1]
            gap = float(word.get("x0", 0)) - float(
                previous.get("x1", previous.get("x0", 0))
            )
            if gap > 48:
                subgroups.append([word])
            else:
                subgroups[-1].append(word)
        lines.extend(_lines_from_subgroups(subgroups))
    return lines


def _lines_from_subgroups(
    subgroups: Sequence[Sequence[Mapping[str, Any]]],
) -> list[_Line]:
    lines: list[_Line] = []
    for subgroup in subgroups:
        text = _clean_text(" ".join(str(word.get("text", "")) for word in subgroup))
        if not text:
            continue
        lines.append(
            _Line(
                text=text,
                top=min(float(word.get("top", 0)) for word in subgroup),
                bottom=max(float(word.get("bottom", word.get("top", 0))) for word in subgroup),
                x0=min(float(word.get("x0", 0)) for word in subgroup),
                x1=max(float(word.get("x1", word.get("x0", 0))) for word in subgroup),
                size=max(float(word.get("size", 0)) for word in subgroup),
                fonts=tuple(str(word.get("font") or "") for word in subgroup),
                native_word_anchors=tuple(
                    str(word["_native_anchor"])
                    for word in subgroup
                    if isinstance(word.get("_native_anchor"), str)
                ),
            )
        )
    return lines


def _lines_for_page_v2(page: Mapping[str, Any]) -> list[_Line]:
    words = [
        word for word in page.get("words", [])
        if isinstance(word, Mapping) and not word.get("_native_excluded")
    ]
    words.sort(key=lambda word: (float(word.get("top", 0)), float(word.get("x0", 0))))
    groups: list[list[Mapping[str, Any]]] = []
    for word in words:
        top = float(word.get("top", 0))
        if not groups or abs(top - float(groups[-1][0].get("top", 0))) > 2.2:
            groups.append([word])
        else:
            groups[-1].append(word)
    width = float(page.get("width", 0) or 0)
    split_starts = _page_split_starts(groups, width) if width else ()
    lines: list[_Line] = []
    for group in groups:
        group.sort(key=lambda word: float(word.get("x0", 0)))
        # At the same y-coordinate, two newspaper-style columns appear as one
        # flat word row.  Use local geometry instead of a fixed page gutter:
        # native exports contain both 20--30pt and 50--60pt gutters.
        subgroups = _split_word_row(group, split_starts, width)
        lines.extend(_lines_from_subgroups(subgroups))
    return lines


def _rows_for_lines(lines: Sequence[_Line]) -> list[list[_Line]]:
    """Group already reconstructed lines that share one visual baseline."""
    rows: list[list[_Line]] = []
    for line in sorted(lines, key=lambda value: (value.top, value.x0)):
        if not rows or abs(line.top - rows[-1][0].top) > 2.2:
            rows.append([line])
        else:
            rows[-1].append(line)
    return rows


def _page_x_band_model(lines: Sequence[_Line], width: float) -> _PageXBandModel:
    """Separate repeated reading starts from recurring two-cell interiors.

    This is intentionally independent of the V2 reading-order bands.  Those
    bands use a generous indentation tolerance, which is right for reading
    order but too broad to tell a cell label from its containing column.  V3
    needs the narrower model only to reject a false heading when *both* its
    start and same-baseline cell relationship recur on this page.
    """
    sizes = [line.size for line in lines if line.size > 0]
    tolerance = max(6.0, min(width * 0.025, (median(sizes) if sizes else 9.0) * 1.25))
    if not lines or not width:
        return _PageXBandModel((), (), tolerance)

    bands: list[list[_Line]] = []
    for line in sorted(lines, key=lambda value: value.x0):
        if not bands or line.x0 - bands[-1][-1].x0 > tolerance:
            bands.append([line])
        else:
            bands[-1].append(line)
    repeated = [
        band
        for band in bands
        if len({round(line.top, 1) for line in band}) >= 2
    ]
    if not repeated:
        return _PageXBandModel((), (), tolerance)
    # A page can have one or two ordinary reading columns.  The outermost
    # repeated start on each page half is the reading start; a compact table's
    # denser interior labels must not displace it by sheer row count.
    dominant: list[float] = []
    for lower, upper in ((float("-inf"), width / 2), (width / 2, float("inf"))):
        candidates = [
            band for band in repeated
            if lower <= median([line.x0 for line in band]) < upper
        ]
        if candidates:
            chosen = min(
                candidates,
                key=lambda band: median([line.x0 for line in band]),
            )
            dominant.append(median([line.x0 for line in chosen]))
    dominant = sorted(set(dominant))
    if not dominant:
        return _PageXBandModel((), (), tolerance)

    def is_dominant(line: _Line) -> bool:
        return any(abs(line.x0 - start) <= tolerance for start in dominant)

    interior: list[float] = []
    for band in repeated:
        start = median([line.x0 for line in band])
        if any(abs(start - primary) <= tolerance for primary in dominant):
            continue
        # An indented list can recur, but it normally occupies its own
        # baseline.  A cell start is only actionable when it repeatedly shares
        # a baseline with a dominant reading start.
        shared_baselines = sum(
            1
            for row in _rows_for_lines(lines)
            if any(item in band for item in row)
            and any(
                is_dominant(item)
                and (item.x0 < width / 2) == (start < width / 2)
                for item in row
            )
        )
        if shared_baselines >= 2:
            interior.append(start)
    return _PageXBandModel(tuple(dominant), tuple(sorted(interior)), tolerance)


def _stable_x_bands(lines: Sequence[_Line], width: float) -> list[list[_Line]]:
    """Return persistent left-edge bands, ignoring one-off indents.

    Column starts are much more stable than the right edge of ragged text.  A
    band needs repeated baselines before it is treated as a page region; this
    keeps table cells and isolated callouts from manufacturing a column.
    """
    if not lines:
        return []
    font_sizes = [line.size for line in lines if line.size > 0]
    # Treat ordinary indentation as one reading band; real Paizo columns are
    # substantially farther apart than a body/list indent.
    tolerance = max(width * 0.10, (median(font_sizes) if font_sizes else 9.0) * 2.5)
    bands: list[list[_Line]] = []
    for line in sorted(lines, key=lambda value: value.x0):
        if not bands or line.x0 - bands[-1][-1].x0 > tolerance:
            bands.append([line])
        else:
            bands[-1].append(line)
    return [
        band
        for band in bands
        if len(band) >= 2 and len({round(line.top, 1) for line in band}) >= 2
    ]


def _dominant_x_bands(
    lines: Sequence[_Line], width: float, *, minimum_count: int
) -> list[list[_Line]]:
    """Discard sparse callout/sidebar starts before choosing reading bands."""
    bands = _stable_x_bands(lines, width)
    if not bands:
        return []
    threshold = max(minimum_count, max(len(band) for band in bands) * 0.25)
    return [band for band in bands if len(band) >= threshold]


def _table_block(row: Sequence[_Line], *, layout_kind: str) -> _Line:
    """Flatten one bounded table row in its observed left-to-right order."""
    ordered = sorted(row, key=lambda line: line.x0)
    return _Line(
        text=_clean_text(" ".join(line.text for line in ordered)),
        top=min(line.top for line in ordered),
        bottom=max(line.bottom for line in ordered),
        x0=min(line.x0 for line in ordered),
        x1=max(line.x1 for line in ordered),
        size=max(line.size for line in ordered),
        fonts=tuple(font for line in ordered for font in line.fonts),
        layout_kind=layout_kind,
        native_word_anchors=tuple(anchor for line in ordered for anchor in line.native_word_anchors),
    )


def _persistent_region_bands(
    lines: Sequence[_Line], width: float, height: float
) -> list[list[_Line]]:
    """Return region bands that persist enough of a page to be reading columns."""
    minimum_span = max(height * 0.18, width * 0.12)
    minimum_count = 3 if len(lines) < 30 else 8
    return [
        band
        for band in _dominant_x_bands(lines, width, minimum_count=minimum_count)
        if len({round(line.top, 1) for line in band}) >= 3
        and max(line.bottom for line in band) - min(line.top for line in band) >= minimum_span
    ]


def _row_grid_signature(row: Sequence[_Line], tolerance: float) -> tuple[int, ...]:
    """Normalize cell starts so aligned table rows compare without source text."""
    return tuple(round(line.x0 / tolerance) for line in sorted(row, key=lambda line: line.x0))


def _is_table_grid(
    rows: Sequence[Sequence[_Line]], width: float, height: float
) -> bool:
    """Recognize a bounded, aligned multi-row table using geometry only."""
    if len(rows) < 2 or any(len(row) < 3 for row in rows):
        return False
    font_sizes = [line.size for row in rows for line in row if line.size > 0]
    tolerance = max(width * 0.035, (median(font_sizes) if font_sizes else 9.0) * 2.0)
    signatures = [_row_grid_signature(row, tolerance) for row in rows]
    cell_counts = {len(signature) for signature in signatures}
    if len(cell_counts) != 1:
        return False
    if any(signature != signatures[0] for signature in signatures[1:]):
        return False
    cell_tokens = [len(line.text.split()) for row in rows for line in row]
    if median(cell_tokens) > 12:
        return False
    if font_sizes and max(font_sizes) - min(font_sizes) > 3.0:
        return False
    ordered_rows = sorted(rows, key=lambda row: row[0].top)
    row_gaps = [
        following[0].top - max(line.bottom for line in preceding)
        for preceding, following in zip(ordered_rows, ordered_rows[1:], strict=False)
    ]
    row_size = median(font_sizes) if font_sizes else 9.0
    if any(gap < 0 or gap > max(row_size * 3.5, height * 0.08) for gap in row_gaps):
        return False
    vertical_span = max(line.bottom for row in rows for line in row) - min(
        line.top for row in rows for line in row
    )
    # A page-spanning matrix may be reading columns, not a table.  A bounded
    # aligned grid is safe to retain as one table block for later review.
    return vertical_span <= height * 0.65


def _table_row_groups(rows: Sequence[Sequence[_Line]], height: float) -> list[list[list[_Line]]]:
    """Group nearby multi-cell rows into bounded candidate table blocks."""
    groups: list[list[list[_Line]]] = []
    for row in rows:
        if len(row) < 3:
            continue
        if not groups:
            groups.append([row])
            continue
        previous = groups[-1][-1]
        gap = row[0].top - max(line.bottom for line in previous)
        if 0 <= gap <= height * 0.08:
            groups[-1].append(row)
        else:
            groups.append([row])
    return groups


def _collapse_table_rows(
    page: Mapping[str, Any], lines: Sequence[_Line]
) -> tuple[list[_Line], list[_Line], bool]:
    """Keep bounded 3+ cell rows as blocks, rejecting only persistent columns."""
    width = float(page.get("width", 0) or 0)
    height = float(page.get("height", 0) or 0)
    rows = _rows_for_lines(lines)
    for row in rows:
        ordered = sorted(row, key=lambda line: line.x0)
        if any(
            following.x0 < preceding.x1 - max(preceding.size, following.size) * 0.25
            for preceding, following in zip(ordered, ordered[1:], strict=False)
        ):
            raise ValueError("unsupported Paizo page layout: overlapping regions")
    table_groups = _table_row_groups(rows, height)
    table_row_ids: set[int] = set()
    table_kinds: dict[int, str] = {}
    unbounded_complex = False
    for group in table_groups:
        kind = "table-grid" if _is_table_grid(group, width, height) else "table-ambiguous"
        vertical_span = max(line.bottom for row in group for line in row) - min(
            line.top for row in group for line in row
        )
        if kind == "table-ambiguous" and vertical_span > height * 0.65:
            unbounded_complex = True
        # An unaligned but bounded multi-cell block is retained for section
        # review.  It is not evidence that the entire page has three reading
        # columns, and aborting the book would lose unrelated rules.
        for row in group:
            table_row_ids.add(id(row))
            table_kinds[id(row)] = kind
    non_table_lines = [
        line
        for row in rows
        if id(row) not in table_row_ids
        for line in row
    ]
    complex_regions = unbounded_complex or bool(
        width and len(_persistent_region_bands(non_table_lines, width, height)) >= 3
    )
    ordinary: list[_Line] = []
    tables: list[_Line] = []
    for row in rows:
        if len(row) >= 3:
            tables.append(_table_block(row, layout_kind=table_kinds[id(row)]))
        else:
            ordinary.extend(row)
    layout_lines = ordinary + tables
    if complex_regions:
        # Preserve a representable but non-two-column page in visual row order
        # and expose that uncertainty to the later section-review gate.  It is
        # safer than aborting every unrelated section in the source book.
        layout_lines = [
            replace(line, layout_kind="complex-layout")
            if line.layout_kind == "body"
            else line
            for line in layout_lines
        ]
    return layout_lines, ordinary, complex_regions


def _spanning_blocks(
    lines: Sequence[_Line], bands: Sequence[Sequence[_Line]], width: float
) -> set[_Line]:
    """Identify blocks crossing both proven columns, including ragged tails."""
    starts = [median([line.x0 for line in band]) for band in bands]
    tolerance = max(width * 0.05, 12.0)

    def crosses_columns(line: _Line) -> bool:
        return line.x0 <= starts[0] + tolerance and line.x1 >= starts[1] - tolerance

    ordered = sorted(lines, key=lambda line: (line.top, line.x0))
    initial_spanning = {line for line in ordered if crosses_columns(line)}
    spanning = set(initial_spanning)
    for index, line in enumerate(ordered[:-1]):
        if line not in initial_spanning:
            continue
        candidate = ordered[index + 1]
        vertical_gap = candidate.top - line.bottom
        if (
            0 <= vertical_gap <= max(line.size, candidate.size) * 2.5
            and abs(candidate.x0 - line.x0) <= tolerance
            and candidate.x1 < starts[1] - tolerance
        ):
            spanning.add(candidate)
    return spanning


def _reading_order_v1(page: Mapping[str, Any], lines: list[_Line]) -> list[_Line]:
    """Frozen v1 page order retained for normal/local-full parsing."""
    if not lines:
        return []
    width = float(page.get("width", 0) or 0)
    if not width:
        return sorted(lines, key=lambda line: (line.top, line.x0))
    split = width / 2
    full = [line for line in lines if line.x1 - line.x0 >= width * 0.62]
    body = [line for line in lines if line not in full]
    left = [line for line in body if (line.x0 + line.x1) / 2 < split]
    right = [line for line in body if (line.x0 + line.x1) / 2 >= split]
    if not left or not right or len(body) < 6:
        return sorted(lines, key=lambda line: (line.top, line.x0))
    if full:
        first_body = min(line.top for line in body)
        before = sorted((line for line in full if line.top <= first_body), key=lambda line: line.top)
        after = sorted((line for line in full if line.top > first_body), key=lambda line: line.top)
        return before + sorted(left, key=lambda line: line.top) + sorted(right, key=lambda line: line.top) + after
    return sorted(left, key=lambda line: line.top) + sorted(right, key=lambda line: line.top)


def _reading_order_v2(page: Mapping[str, Any], lines: list[_Line]) -> list[_Line]:
    if not lines:
        return []
    width = float(page.get("width", 0) or 0)
    if not width:
        return sorted(lines, key=lambda line: (line.top, line.x0))
    layout_lines, ordinary_lines, force_visual_order = _collapse_table_rows(page, lines)
    if force_visual_order:
        return sorted(layout_lines, key=lambda line: (line.top, line.x0))
    bands = _dominant_x_bands(ordinary_lines, width, minimum_count=2)
    if len(bands) != 2 or len(ordinary_lines) < 4:
        return sorted(layout_lines, key=lambda line: (line.top, line.x0))
    # The midpoint between the repeated left-edge bands is the gutter.  It is
    # inferred from native geometry and works for both narrow and wide gutters.
    band_starts = [median([line.x0 for line in band]) for band in bands]
    gutter = (band_starts[0] + band_starts[1]) / 2
    full = _spanning_blocks(layout_lines, bands, width)

    def order_column_band(items: Sequence[_Line]) -> list[_Line]:
        left = [line for line in items if (line.x0 + line.x1) / 2 < gutter]
        right = [line for line in items if line not in left]
        return sorted(left, key=lambda line: (line.top, line.x0)) + sorted(
            right, key=lambda line: (line.top, line.x0)
        )

    # A spanning block is a vertical boundary: read the column band above it,
    # emit the block at its observed position, then begin a new band below it.
    # The old parser appended every full-width block after both columns.
    ordered: list[_Line] = []
    pending: list[_Line] = []
    for line in sorted(layout_lines, key=lambda value: (value.top, value.x0)):
        if line in full:
            ordered.extend(order_column_band(pending))
            pending = []
            ordered.append(line)
        else:
            pending.append(line)
    ordered.extend(order_column_band(pending))
    return ordered


def _repeated_furniture(
    pages: Sequence[Mapping[str, Any]],
    line_builder: Callable[[Mapping[str, Any]], list[_Line]],
) -> set[str]:
    candidates: list[str] = []
    for page in pages:
        height = float(page.get("height", 0) or 0)
        for line in line_builder(page):
            if height and (line.top < height * 0.12 or line.bottom > height * 0.88):
                normalized = _normalize_name(re.sub(r"\b\d+\b", "", line.text))
                if normalized and not _PAGE_NUMBER_RE.match(line.text):
                    candidates.append(normalized)
    return {value for value, count in Counter(candidates).items() if count >= 2}


def _is_heading_v1(line: _Line, body_size: float) -> bool:
    text = line.text
    if len(text) > 160 or _EMAIL_RE.search(text):
        return False
    if sum(char.isalnum() for char in text) < 3:
        return False
    if line.fonts and all(
        any(token in font.casefold() for token in ("pathfinder-icons", "taroca"))
        for font in line.fonts
    ):
        return False
    bold_tokens = ("bold", "semibold", "heavy", "black", "display", "condbold")
    bold_words = sum(
        any(token in font.casefold() for token in bold_tokens)
        for font in line.fonts
    )
    bold_fraction = bold_words / len(line.fonts) if line.fonts else 0.0
    all_caps = text.upper() == text and any(char.isalpha() for char in text)
    # Inline bold terms and table labels are common in Paizo body copy. A line
    # becomes a section boundary only with strong whole-line evidence: clearly
    # larger type, predominantly bold type above body size, or a short caps
    # heading. This deliberately avoids fragmenting every emphasized phrase.
    return (
        line.size >= body_size + 2.5
        or (bold_fraction >= 0.8 and line.size >= body_size + 0.75)
        or (all_caps and len(text) < 90 and line.size >= body_size)
    )


def _is_heading_v2(line: _Line, body_size: float) -> bool:
    text = line.text
    if len(text) > 160 or _EMAIL_RE.search(text):
        return False
    if sum(char.isalnum() for char in text) < 3:
        return False
    if line.fonts and all(
        any(token in font.casefold() for token in ("pathfinder-icons", "taroca"))
        for font in line.fonts
    ):
        return False
    bold_tokens = ("bold", "semibold", "heavy", "black", "display", "condbold")
    bold_words = sum(
        any(token in font.casefold() for token in bold_tokens)
        for font in line.fonts
    )
    bold_fraction = bold_words / len(line.fonts) if line.fonts else 0.0
    all_caps = text.upper() == text and any(char.isalpha() for char in text)
    return (
        line.size >= body_size + 2.5
        or (bold_fraction >= 0.8 and line.size >= body_size + 0.75)
        # Condensed all-caps labels are common inside body-sized tables.  Caps
        # alone is not a boundary unless it is also visibly display-sized.
        or (
            all_caps
            and len(text) < 90
            and bold_fraction >= 0.8
            and line.size >= body_size + 1.5
        )
    )


def _is_condensed_bold_body_candidate(line: _Line, body_size: float) -> bool:
    """Identify the V3 heading-shaped labels evidenced inside Paizo cells."""
    if line.size > body_size + 3.5 or not line.fonts:
        return False
    condensed = sum(
        "condbold" in font.casefold() or "condensed" in font.casefold()
        for font in line.fonts
    )
    return condensed / len(line.fonts) >= 0.8


_V4_BODY_LABELS = {
    "abstract", "content", "text", "reference", "reference_content",
}
_V4_SIDEBAR_LABELS = {"aside_text", "footnote"}
_V4_HEADING_LABELS = {"doc_title", "paragraph_title"}
_V4_QUARANTINE_REASONS = {
    "repeated-furniture", "page-number", "contents-index", "credits-legal",
    "unresolved-table", "unbound-layout", "heading-artifact",
    "unresolved-continuation", "unresolved-layout", "layout-order-conflict",
    "oversize-block",
}
_V4_SECTION_TARGET_CHARS = 7_800


@dataclass(frozen=True)
class _V4ParseResult:
    chunks: tuple[dict[str, Any], ...]
    blocks_by_section_id: Mapping[str, tuple[TrustedBlock, ...]]
    quarantine: tuple[TrustedQuarantine, ...]


def _v4_page_quarantine_reason(page_number: int, lines: Sequence[_Line]) -> str | None:
    """Classify bounded whole-page non-rule matter without exposing its text."""
    normalized = [_normalize_name(line.text) for line in lines]
    exact = set(normalized)
    if exact & {"contents", "table of contents", "index", "subject index"}:
        return "contents-index"
    leader_rows = sum(
        bool(re.search(r"(?:\.{2,}|\s)\d{1,4}\s*$", line.text))
        for line in lines
    )
    if len(lines) >= 8 and leader_rows >= max(5, len(lines) // 3):
        return "contents-index"
    legal_markers = {
        "credits", "open game license version 1 0a", "orc notice",
        "orc license", "product identity", "designation of product identity",
    }
    if exact & legal_markers or any(
        marker in value for marker in legal_markers for value in normalized
    ):
        return "credits-legal"
    # Front-matter credit blocks often have no single canonical title, but a
    # dense cluster of role labels is deterministic and book-independent.
    credit_roles = ("authors", "designers", "developers", "editors", "art director")
    if page_number <= 12 and sum(
        any(role in value for role in credit_roles) for value in normalized
    ) >= 3:
        return "credits-legal"
    return None


def _v4_is_heading(line: _Line, body_size: float, region_label: str) -> bool:
    text = line.text.strip()
    if not text or _PAGE_NUMBER_RE.fullmatch(text):
        return False
    words = text.split()
    sentence_like = (
        len(text) > 80
        or len(words) >= 16
        or (len(words) >= 8 and text[-1:] in {".", "?", "!", ";", ":"})
    )
    if sentence_like:
        return False
    if region_label == "heading" or region_label in _V4_HEADING_LABELS:
        return len(text) <= 160 and sum(char.isalnum() for char in text) >= 3
    if line.layout_kind.startswith("table"):
        return False
    return _is_heading_v2(line, body_size)


def _v4_table_rows(
    page: Mapping[str, Any], raw_lines: Sequence[_Line]
) -> list[list[_Line]] | None:
    """Return a stable native table grid or ``None`` when geometry is ambiguous."""
    width = float(page.get("width", 0) or 0)
    height = float(page.get("height", 0) or 0)
    rows = _rows_for_lines(raw_lines)
    data_rows = [sorted(row, key=lambda item: item.x0) for row in rows if len(row) >= 2]
    if len(data_rows) < 2 or not width:
        return None
    counts = Counter(len(row) for row in data_rows)
    cell_count, support = counts.most_common(1)[0]
    if support < 2 or cell_count > 8:
        return None
    aligned = [row for row in data_rows if len(row) == cell_count]
    sizes = [line.size for row in aligned for line in row if line.size > 0]
    tolerance = max(width * 0.035, (median(sizes) if sizes else 9.0) * 2.0)
    signature = _row_grid_signature(aligned[0], tolerance)
    if any(_row_grid_signature(row, tolerance) != signature for row in aligned[1:]):
        return None
    vertical_span = max(line.bottom for row in aligned for line in row) - min(
        line.top for row in aligned for line in row
    )
    if height and vertical_span > height * 0.8:
        return None
    # One-cell rows are retained as spanning table headings or notes; any
    # other inconsistent row shape makes the entire region unresolved.
    if any(len(row) not in {1, cell_count} for row in rows):
        return None
    return [sorted(row, key=lambda item: item.x0) for row in rows]


def _v4_native_fallback_order(line: _Line, regions: Sequence[Any]) -> float:
    """Place a model-uncovered native line between nearby bound regions.

    PP-DocLayout can omit dense stat-block text while still detecting the
    headings immediately above and below it. Native geometry is authoritative,
    so use those same-column regions as deterministic order brackets instead
    of quarantining the uncovered words or inventing OCR text.
    """
    textual = [
        region
        for region in regions
        if region.label
        in _V4_BODY_LABELS | _V4_SIDEBAR_LABELS | _V4_HEADING_LABELS | {"table"}
    ]
    if not textual:
        return float("inf")

    def horizontal_distance(region: Any) -> float:
        x0, _y0, x1, _y1 = (float(value) for value in region.box)
        return max(x0 - line.x1, line.x0 - x1, 0.0)

    nearest_distance = min(horizontal_distance(region) for region in textual)
    nearby = [
        region
        for region in textual
        if horizontal_distance(region) <= nearest_distance + max(line.size * 2.0, 12.0)
    ]
    overlapping = [
        region
        for region in nearby
        if float(region.box[1]) <= (line.top + line.bottom) / 2 <= float(region.box[3])
    ]
    if overlapping:
        chosen = min(
            overlapping,
            key=lambda region: (
                horizontal_distance(region),
                max(0.0, float(region.box[2]) - float(region.box[0]))
                * max(0.0, float(region.box[3]) - float(region.box[1])),
                int(region.order),
            ),
        )
        return float(chosen.order)

    above = [region for region in nearby if float(region.box[3]) <= line.top + 2.0]
    below = [region for region in nearby if float(region.box[1]) >= line.bottom - 2.0]
    previous = max(above, key=lambda region: (float(region.box[3]), int(region.order))) if above else None
    following = min(below, key=lambda region: (float(region.box[1]), int(region.order))) if below else None
    if previous is not None and following is not None:
        before = float(previous.order)
        after = float(following.order)
        if before < after:
            return (before + after) / 2.0
        return before + 0.5
    if previous is not None:
        return float(previous.order) + 0.5
    if following is not None:
        return float(following.order) - 0.5
    return float("inf")


def _v4_quarantine(
    product: ProductSpec,
    reason: str,
    page: int,
    ordinal: int,
    text: str,
    anchors: Sequence[str],
) -> TrustedQuarantine:
    if reason not in _V4_QUARANTINE_REASONS or not anchors:
        raise ValueError("invalid native-text quarantine record")
    cleaned = _clean_text(text)
    digest = hashlib.sha256("\n".join(anchors).encode()).hexdigest()[:20]
    return TrustedQuarantine(
        quarantine_id=(
            f"{product.code.casefold()}:{product.component}:p{page}:q{ordinal}:{digest}"
        ),
        reason=reason,
        physical_page=page,
        text=cleaned,
        text_hash=hashlib.sha256(cleaned.encode()).hexdigest(),
        coverage_anchors=tuple(anchors),
    )


def _parse_rulebook_payload_v4(
    payload: Mapping[str, Any],
    *,
    product: ProductSpec,
    binding: Any,
    fallback_filename: str,
    parser_version: str = PAIZO_NATIVE_PARSER_V4,
) -> _V4ParseResult:
    """Reconstruct review sections from native words ordered by bound layout regions."""
    repair_ambiguous = parser_version == PAIZO_NATIVE_PARSER_V5
    if parser_version not in {PAIZO_NATIVE_PARSER_V4, PAIZO_NATIVE_PARSER_V5}:
        raise ValueError("structural parser requires V4 or V5")
    if binding.product_code != product.code:
        raise ValueError("native layout binding does not match the parser product")
    inventory = native_word_inventory(payload, product.code, strict=True)
    annotated = annotate_native_words(payload, inventory)
    pages = {
        int(page["number"]): page
        for page in annotated.get("pages", [])
        if isinstance(page, Mapping)
    }
    if tuple(sorted(pages)) != tuple(binding.selected_pages):
        raise ValueError("v4 layout evidence must cover every native PDF page")

    words_by_anchor: dict[str, Mapping[str, Any]] = {}
    page_by_anchor: dict[str, int] = {}
    for page_number, page in pages.items():
        for word in page.get("words", []):
            if not isinstance(word, Mapping):
                continue
            anchor = word.get("_native_anchor")
            if isinstance(anchor, str) and anchor not in inventory.ignored_anchor_reasons:
                words_by_anchor[anchor] = word
                page_by_anchor[anchor] = page_number

    regions_by_page: dict[int, list[Any]] = defaultdict(list)
    for region in binding.regions:
        regions_by_page[int(region.page)].append(region)
    for regions in regions_by_page.values():
        regions.sort(key=lambda item: (item.order, item.box, item.label))

    furniture = _repeated_furniture(tuple(pages.values()), _lines_for_page_v2)
    ordered: list[tuple[int, str, _Line, tuple[tuple[str, ...], ...]]] = []
    quarantined: list[TrustedQuarantine] = []
    quarantine_ordinals: Counter[tuple[int, str]] = Counter()
    assigned: set[str] = set()
    fallback_anchors: set[str] = set()
    repair_flags_by_anchor: dict[str, str] = {}

    def quarantine(reason: str, page: int, text: str, anchors: Sequence[str]) -> None:
        unique = tuple(anchor for anchor in anchors if anchor not in assigned)
        if not unique:
            return
        key = (page, reason)
        quarantined.append(
            _v4_quarantine(product, reason, page, quarantine_ordinals[key], text, unique)
        )
        quarantine_ordinals[key] += 1
        assigned.update(unique)

    for page_number in sorted(pages):
        page = pages[page_number]
        all_lines = _lines_for_page_v2(page)
        page_reason = _v4_page_quarantine_reason(page_number, all_lines)
        page_anchors = tuple(
            str(word["_native_anchor"])
            for word in page.get("words", [])
            if isinstance(word, Mapping)
            and isinstance(word.get("_native_anchor"), str)
            and word["_native_anchor"] not in inventory.ignored_anchor_reasons
        )
        if page_reason is not None:
            quarantine(page_reason, page_number, " ".join(line.text for line in all_lines), page_anchors)
            continue
        height = float(page.get("height", 0) or 0)
        page_regions = regions_by_page.get(page_number, [])
        page_ordered: list[
            tuple[float, float, float, str, _Line, tuple[tuple[str, ...], ...]]
        ] = []
        order_conflicts: set[int] = set()
        for index, (earlier, later) in enumerate(
            zip(page_regions, page_regions[1:], strict=False)
        ):
            earlier_width = max(1.0, float(earlier.box[2]) - float(earlier.box[0]))
            later_width = max(1.0, float(later.box[2]) - float(later.box[0]))
            overlap = max(
                0.0,
                min(float(earlier.box[2]), float(later.box[2]))
                - max(float(earlier.box[0]), float(later.box[0])),
            )
            if (
                overlap / min(earlier_width, later_width) >= 0.5
                and float(later.box[1]) + 2.0 < float(earlier.box[1])
            ):
                order_conflicts.update((index, index + 1))
        if order_conflicts:
            conflict_anchors = tuple(
                anchor
                for region_index in sorted(order_conflicts)
                for anchor in page_regions[region_index].native_word_anchors
                if anchor in words_by_anchor and anchor not in assigned
            )
            if repair_ambiguous:
                conflict_page = dict(page)
                conflict_page["words"] = [words_by_anchor[anchor] for anchor in conflict_anchors]
                conflict_order = min(
                    float(page_regions[index].order) for index in order_conflicts
                )
                for line in _reading_order_v2(
                    conflict_page, _lines_for_page_v2(conflict_page)
                ):
                    line_anchors = tuple(
                        anchor for anchor in line.native_word_anchors if anchor not in assigned
                    )
                    if not line_anchors:
                        continue
                    page_ordered.append(
                        (
                            conflict_order, line.top, line.x0, "native-repair",
                            replace(line, layout_kind="native-repair"), (),
                        )
                    )
                    assigned.update(line_anchors)
                    repair_flags_by_anchor.update(
                        (anchor, "layout-order-conflict") for anchor in line_anchors
                    )
            else:
                quarantine(
                    "layout-order-conflict",
                    page_number,
                    " ".join(
                        str(words_by_anchor[anchor].get("text", ""))
                        for anchor in conflict_anchors
                    ),
                    conflict_anchors,
                )
        for region_index, region in enumerate(page_regions):
            anchors = tuple(
                anchor
                for anchor in region.native_word_anchors
                if anchor in words_by_anchor and anchor not in assigned
            )
            if not anchors:
                continue
            if region_index in order_conflicts:
                continue
            region_page = dict(page)
            region_page["words"] = [words_by_anchor[anchor] for anchor in anchors]
            raw_lines = _lines_for_page_v2(region_page)
            if not raw_lines:
                quarantine("unresolved-layout", page_number, "", anchors)
                continue
            if region.label == "table":
                rows = _v4_table_rows(region_page, raw_lines)
                if rows is None:
                    if repair_ambiguous:
                        for line in _reading_order_v2(region_page, raw_lines):
                            line_anchors = tuple(
                                anchor for anchor in line.native_word_anchors
                                if anchor not in assigned
                            )
                            if not line_anchors:
                                continue
                            page_ordered.append(
                                (
                                    float(region.order), line.top, line.x0,
                                    "table-repair", replace(line, layout_kind="table-repair"), (),
                                )
                            )
                            assigned.update(line_anchors)
                            repair_flags_by_anchor.update(
                                (anchor, "table-ambiguous") for anchor in line_anchors
                            )
                    else:
                        quarantine(
                            "unresolved-table", page_number,
                            " ".join(line.text for line in raw_lines), anchors,
                        )
                    continue
                for row in rows:
                    line = _table_block(row, layout_kind="table-grid")
                    row_cells = (tuple(item.text for item in row),)
                    page_ordered.append(
                        (
                            float(region.order), line.top, line.x0,
                            "table", line, row_cells,
                        )
                    )
                    assigned.update(line.native_word_anchors)
                continue
            if region.label not in _V4_BODY_LABELS | _V4_SIDEBAR_LABELS | _V4_HEADING_LABELS:
                if repair_ambiguous:
                    for line in _reading_order_v2(region_page, raw_lines):
                        line_anchors = tuple(
                            anchor for anchor in line.native_word_anchors if anchor not in assigned
                        )
                        if not line_anchors:
                            continue
                        page_ordered.append(
                            (
                                float(region.order), line.top, line.x0, "native-repair",
                                replace(line, layout_kind="native-repair"), (),
                            )
                        )
                        assigned.update(line_anchors)
                        repair_flags_by_anchor.update(
                            (anchor, "unsupported-layout") for anchor in line_anchors
                        )
                else:
                    quarantine(
                        "unresolved-layout", page_number,
                        " ".join(line.text for line in raw_lines), anchors,
                    )
                continue
            lines = _reading_order_v2(region_page, raw_lines)
            for line in lines:
                line_anchors = tuple(anchor for anchor in line.native_word_anchors if anchor not in assigned)
                if not line_anchors:
                    continue
                compact_heading = re.sub(r"[\s\W_]", "", line.text, flags=re.UNICODE)
                if (
                    repair_ambiguous
                    and region.label in _V4_HEADING_LABELS
                    and compact_heading
                    and compact_heading.isdecimal()
                ):
                    quarantine("page-number", page_number, line.text, line_anchors)
                    continue
                normalized = _normalize_name(re.sub(r"\b\d+\b", "", line.text))
                margin = bool(height and (line.top < height * 0.12 or line.bottom > height * 0.88))
                if margin and _PAGE_NUMBER_RE.fullmatch(line.text.strip()):
                    quarantine("page-number", page_number, line.text, line_anchors)
                    continue
                if margin and normalized in furniture:
                    quarantine("repeated-furniture", page_number, line.text, line_anchors)
                    continue
                kind = "sidebar" if region.label in _V4_SIDEBAR_LABELS else "body"
                if region.label in _V4_HEADING_LABELS:
                    kind = "heading"
                page_ordered.append(
                    (
                        float(region.order), line.top, line.x0,
                        kind, replace(line, layout_kind=kind), (),
                    )
                )
                assigned.update(line_anchors)

        # Model-unbound native words remain authoritative. Reconstruct their
        # lines from the PDF geometry and bracket them between nearby detected
        # regions; never invent OCR text or silently discard a stat block.
        remaining = [anchor for anchor in page_anchors if anchor not in assigned]
        if remaining:
            fallback_page = dict(page)
            fallback_page["words"] = [words_by_anchor[anchor] for anchor in remaining]
            for line in _reading_order_v2(fallback_page, _lines_for_page_v2(fallback_page)):
                line_anchors = tuple(
                    anchor for anchor in line.native_word_anchors if anchor not in assigned
                )
                if not line_anchors:
                    continue
                normalized = _normalize_name(re.sub(r"\b\d+\b", "", line.text))
                margin = bool(
                    height
                    and (line.top < height * 0.12 or line.bottom > height * 0.88)
                )
                if margin and _PAGE_NUMBER_RE.fullmatch(line.text.strip()):
                    quarantine("page-number", page_number, line.text, line_anchors)
                    continue
                if margin and normalized in furniture:
                    quarantine("repeated-furniture", page_number, line.text, line_anchors)
                    continue
                page_ordered.append(
                    (
                        _v4_native_fallback_order(line, page_regions),
                        line.top,
                        line.x0,
                        "native-fallback",
                        replace(line, layout_kind="native-fallback"),
                        (),
                    )
                )
                assigned.update(line_anchors)
                fallback_anchors.update(line_anchors)
        ordered.extend(
            (page_number, kind, line, cells)
            for _order, _top, _x0, kind, line, cells in sorted(
                page_ordered, key=lambda item: item[:3]
            )
        )

    sections: list[dict[str, Any]] = []
    blocks_by_section_id: dict[str, tuple[TrustedBlock, ...]] = {}
    section_ordinals: Counter[tuple[int, str]] = Counter()
    title: str | None = None
    body_lines: list[str] = []
    blocks: list[TrustedBlock] = []
    anchors: list[str] = []
    pages_in_section: list[int] = []
    layout_flags: set[str] = set()

    def quarantine_open(reason: str) -> None:
        nonlocal title, body_lines, blocks, anchors, pages_in_section, layout_flags
        if anchors:
            key = (pages_in_section[0], reason)
            quarantined.append(
                _v4_quarantine(
                    product, reason, pages_in_section[0], quarantine_ordinals[key],
                    " ".join(block.text for block in blocks), anchors,
                )
            )
            quarantine_ordinals[key] += 1
        title = None
        body_lines = []
        blocks = []
        anchors = []
        pages_in_section = []
        layout_flags = set()

    def flush() -> None:
        nonlocal title, body_lines, blocks, anchors, pages_in_section, layout_flags
        if title is None or not body_lines:
            reason = "heading-artifact" if title else "unresolved-continuation"
            if not repair_ambiguous or not blocks:
                quarantine_open(reason)
                return
            title = title or f"Continuation on page {pages_in_section[0]}"
            layout_flags.add(reason)
        if any(len(block.text) > _V4_SECTION_TARGET_CHARS for block in blocks):
            if not repair_ambiguous:
                quarantine_open("oversize-block")
                return
            layout_flags.add("oversize-block")
        heading_chain = tuple(
            block.text for block in blocks if block.kind == "heading"
        ) or (title,)
        identity_heading = "\n".join(heading_chain) or title
        groups: list[list[TrustedBlock]] = []
        current: list[TrustedBlock] = []
        current_chars = 0
        for block in blocks:
            next_chars = current_chars + len(block.text) + int(bool(current))
            if (
                current
                and next_chars > _V4_SECTION_TARGET_CHARS
                and any(item.kind != "heading" for item in current)
            ):
                groups.append(current)
                current = []
                current_chars = 0
            current.append(block)
            current_chars += len(block.text) + int(len(current) > 1)
        if current:
            groups.append(current)
        split = len(groups) > 1
        for group in groups:
            group_blocks = tuple(
                replace(block, ordinal=ordinal)
                for ordinal, block in enumerate(group)
            )
            group_anchors = tuple(
                anchor for block in group_blocks for anchor in block.coverage_anchors
            )
            page_values = tuple(
                dict.fromkeys(block.physical_page for block in group_blocks)
            )
            page_start = page_values[0]
            key = (page_start, _slug(title))
            ordinal = section_ordinals[key]
            section_ordinals[key] += 1
            section_id = (
                f"corpus:{product.code}:{product.component}:p{page_start}:"
                f"{key[1]}:{ordinal}"
            )
            text = _clean_text(" ".join(block.text for block in group_blocks))
            group_flags = {
                *("structured-table" for block in group_blocks if block.kind == "table"),
                *("structured-sidebar" for block in group_blocks if block.kind == "sidebar"),
            }
            if split:
                group_flags.add("oversize-split")
            if fallback_anchors.intersection(group_anchors):
                group_flags.add("native-layout-fallback")
            group_flags.update(
                repair_flags_by_anchor[anchor]
                for anchor in group_anchors
                if anchor in repair_flags_by_anchor
            )
            group_flags.update(layout_flags)
            provenance = {
                "export_schema_version": payload.get("schema_version"),
                "title": product.title,
                "physical_pages": list(page_values),
                "printed_pages": [],
                "content_fingerprint": inventory.content_fingerprint,
                "native_word_anchors": list(group_anchors),
                "heading_chain": list(heading_chain),
                "stable_section_identity": _stable_section_identity(
                    product, page_start, None, identity_heading, ordinal
                ),
            }
            if group_flags:
                provenance["layout_flags"] = sorted(group_flags)
            section_hash = hashlib.sha256(text.encode()).hexdigest()
            sections.append(
                RulebookSection(
                    id=section_id, name=title, text=text, product_code=product.code,
                    book=product.title, component=product.component,
                    rules_era=product.rules_era, license=product.license,
                    remaster=product.remaster,
                    source_filename=_safe_basename(fallback_filename),
                    source_sha256="", pages=page_values, page_start=page_start,
                    page_end=page_values[-1], printed_page=None,
                    section_hash=section_hash, provenance=provenance,
                    ordinal=ordinal, parser_version=parser_version,
                ).as_chunk()
            )
            blocks_by_section_id[section_id] = group_blocks
        title = None
        body_lines = []
        blocks = []
        anchors = []
        pages_in_section = []
        layout_flags = set()

    page_sizes: dict[int, float] = {}
    for page_number, page in pages.items():
        sizes = [
            float(word.get("size", 0)) for word in page.get("words", [])
            if isinstance(word, Mapping) and float(word.get("size", 0)) > 0
        ]
        page_sizes[page_number] = median(sizes) if sizes else 9.0

    for page_number, kind, line, table_cells in ordered:
        heading = _v4_is_heading(line, page_sizes[page_number], kind)
        compact_heading = re.sub(r"[\s\W_]", "", line.text, flags=re.UNICODE)
        if (
            repair_ambiguous
            and heading
            and compact_heading
            and compact_heading.isdecimal()
        ):
            key = (page_number, "page-number")
            quarantined.append(
                _v4_quarantine(
                    product, "page-number", page_number,
                    quarantine_ordinals[key], line.text, line.native_word_anchors,
                )
            )
            quarantine_ordinals[key] += 1
            continue
        if heading:
            if (
                title is not None
                and not body_lines
                and pages_in_section[-1] == page_number
            ):
                title = line.text
                anchors.extend(line.native_word_anchors)
                pages_in_section.append(page_number)
                blocks.append(
                    TrustedBlock(
                        kind="heading", physical_page=page_number,
                        ordinal=len(blocks), text=line.text,
                        text_hash=hashlib.sha256(line.text.encode()).hexdigest(),
                        coverage_anchors=line.native_word_anchors,
                    )
                )
                continue
            flush()
            title = line.text
            anchors = list(line.native_word_anchors)
            pages_in_section = [page_number]
            blocks = [
                TrustedBlock(
                    kind="heading", physical_page=page_number, ordinal=0,
                    text=line.text, text_hash=hashlib.sha256(line.text.encode()).hexdigest(),
                    coverage_anchors=line.native_word_anchors,
                )
            ]
            continue
        block_kind = (
            "table" if kind == "table-repair"
            else "body" if kind in {"native-fallback", "native-repair"}
            else kind
        )
        if title is None:
            # Keep collecting until a real heading is observed; the group is
            # quarantined deterministically at the next boundary/end.
            pages_in_section.append(page_number)
            anchors.extend(line.native_word_anchors)
            body_lines.append(line.text)
            blocks.append(
                TrustedBlock(
                    kind=block_kind, physical_page=page_number, ordinal=len(blocks),
                    text=line.text, text_hash=hashlib.sha256(line.text.encode()).hexdigest(),
                    coverage_anchors=line.native_word_anchors, table_cells=table_cells,
                )
            )
            continue
        body_lines.append(line.text)
        pages_in_section.append(page_number)
        anchors.extend(line.native_word_anchors)
        blocks.append(
            TrustedBlock(
                kind=block_kind, physical_page=page_number, ordinal=len(blocks),
                text=line.text, text_hash=hashlib.sha256(line.text.encode()).hexdigest(),
                coverage_anchors=line.native_word_anchors, table_cells=table_cells,
            )
        )
        if kind in {"table", "table-repair"}:
            layout_flags.add("structured-table")
        elif kind == "sidebar":
            layout_flags.add("structured-sidebar")
    flush()

    expected = set(inventory.anchors) - set(inventory.ignored_anchor_reasons)
    ownership = Counter(
        anchor
        for anchor_group in (
            *(chunk["provenance"]["native_word_anchors"] for chunk in sections),
            *(item.coverage_anchors for item in quarantined),
        )
        for anchor in anchor_group
    )
    if set(ownership) != expected or any(value != 1 for value in ownership.values()):
        raise ValueError("structural parser did not assign every native anchor exactly once")
    return _V4ParseResult(tuple(sections), MappingProxyType(blocks_by_section_id), tuple(quarantined))


def _slug(value: str) -> str:
    value = _normalize_name(value).replace(" ", "-")
    return value[:80] or "section"


def _stable_section_identity(
    product: ProductSpec, page_start: object, printed_page: object, heading: object, ordinal: object
) -> str:
    """Versioned parser identity, deliberately separate from text/provenance hashes."""
    material = "\n".join(
        (
            "paizo-section-identity-v2", product.code, product.component,
            str(page_start), str(printed_page or ""), _normalize_name(str(heading)), str(ordinal),
        )
    )
    return hashlib.sha256(material.encode()).hexdigest()


def _disambiguate_v5_stable_identities(sections: list[dict[str, Any]]) -> None:
    """Resolve rare flattened-heading collisions without renumbering other sections.

    The frozen V2 identity normalizes a complete heading chain as one string.
    Consequently, two different chains such as ``["A B", "C"]`` and
    ``["A", "B C"]`` can have the same flattened identity. Keep every
    non-colliding identity unchanged so exact V4 work remains reusable, while
    deriving collision-only identities from the length-delimited chain and the
    first authoritative native anchor.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for section in sections:
        provenance = section.get("provenance")
        if isinstance(provenance, dict):
            identity = provenance.get("stable_section_identity")
            if isinstance(identity, str):
                grouped[identity].append(section)
    for identity, collisions in grouped.items():
        if len(collisions) < 2:
            continue
        replacement_ids: set[str] = set()
        for section in collisions:
            provenance = section["provenance"]
            heading_chain = provenance.get("heading_chain", [section.get("name", "")])
            anchors = provenance.get("native_word_anchors", [])
            if (
                not isinstance(heading_chain, list)
                or any(not isinstance(value, str) for value in heading_chain)
                or not isinstance(anchors, list)
                or not anchors
                or not isinstance(anchors[0], str)
            ):
                raise ValueError("V5 identity collision lacks structural provenance")
            material = json.dumps({
                "version": "paizo-section-identity-v2-collision-1",
                "base": identity,
                "heading_chain": [_normalize_name(value) for value in heading_chain],
                "first_native_anchor": anchors[0],
            }, sort_keys=True, separators=(",", ":"))
            replacement = hashlib.sha256(material.encode("utf-8")).hexdigest()
            if replacement in replacement_ids:
                raise ValueError("V5 stable identity collision remains ambiguous")
            replacement_ids.add(replacement)
            provenance["stable_section_identity"] = replacement


def _trusted_section_canonical(section: TrustedSection) -> dict[str, object]:
    return {
        "id": section.id,
        "source_section_id": section.source_section_id,
        "text_hash": section.text_hash,
        "heading": section.heading,
        "pages": section.physical_pages,
        "printed_page": section.printed_page,
        "stable_section_identity": section.stable_section_identity,
        "layout_flags": section.layout_flags,
        "anchors": section.coverage_anchors,
        "blocks": [
            {
                "kind": block.kind,
                "page": block.physical_page,
                "ordinal": block.ordinal,
                "text_hash": block.text_hash,
                "anchors": block.coverage_anchors,
                "table_shape": tuple(len(row) for row in block.table_cells),
            }
            for block in section.blocks
        ],
    }


def _trusted_quarantine_canonical(item: TrustedQuarantine) -> dict[str, object]:
    return {
        "id": item.quarantine_id,
        "reason": item.reason,
        "page": item.physical_page,
        "text_hash": item.text_hash,
        "anchors": item.coverage_anchors,
    }


def _trusted_parser_output_digest(
    sections: Sequence[TrustedSection],
    quarantine: Sequence[TrustedQuarantine] = (),
) -> str:
    canonical: object
    if quarantine:
        canonical = {
            "sections": [_trusted_section_canonical(section) for section in sections],
            "quarantine": [_trusted_quarantine_canonical(item) for item in quarantine],
        }
    else:
        canonical = [_trusted_section_canonical(section) for section in sections]
    return hashlib.sha256(
        (
            ("trusted-parser-output-v3\n" if quarantine else "trusted-parser-output-v2\n")
            + json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        ).encode()
    ).hexdigest()


def _trusted_bundle_seal(
    *,
    product_code: str,
    parser_version: str,
    exporter_profile_version: int,
    semantic_fingerprint: str,
    artifact_attestation: Mapping[str, object],
    artifact_attestation_digest: str,
    inventory: NativeWordInventory,
    sections: Sequence[TrustedSection],
    parser_output_digest: str,
    quarantine: Sequence[TrustedQuarantine] = (),
    layout_binding_digest: str | None = None,
) -> str:
    material = {
        "version": "trusted-parse-bundle-v3",
        "product": product_code,
        "artifact_attestation": artifact_attestation_digest,
        "artifact_evidence": dict(artifact_attestation),
        "exporter_profile": exporter_profile_version,
        "parser": parser_version,
        "layout_binding": layout_binding_digest,
        "semantic_fingerprint": semantic_fingerprint,
        "ignored_policy": sorted((item.anchor_hash, item.reason) for item in inventory.ignored_anchors),
        "sections": [_trusted_section_canonical(section) for section in sections],
        "quarantine": [_trusted_quarantine_canonical(item) for item in quarantine],
        "parser_output_digest": parser_output_digest,
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parse_rulebook_export(
    export_path: Path | str,
    *,
    product: ProductSpec | None = None,
    source: CorpusSource | None = None,
    parser_version: str = PAIZO_NATIVE_PARSER_VERSION,
) -> list[dict[str, Any]]:
    """Parse one exporter JSON artifact into ``rulebook_section`` chunks."""
    path = Path(export_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return parse_rulebook_payload(
        payload, product=product, source=source, parser_version=parser_version,
        fallback_filename=path.name,
    )


def parse_rulebook_payload(
    payload: Mapping[str, Any],
    *,
    product: ProductSpec | None = None,
    source: CorpusSource | None = None,
    parser_version: str = PAIZO_NATIVE_PARSER_VERSION,
    fallback_filename: str = "native-export.json",
) -> list[dict[str, Any]]:
    """Parse an already-loaded native export; trusted callers never reread it."""
    if parser_version == PAIZO_NATIVE_PARSER_V1:
        line_builder = _lines_for_page_v1
        reading_order = _reading_order_v1
        is_heading = _is_heading_v1
    elif parser_version == PAIZO_NATIVE_PARSER_V2:
        line_builder = _lines_for_page_v2
        reading_order = _reading_order_v2
        is_heading = _is_heading_v2
    elif parser_version == PAIZO_NATIVE_PARSER_V3:
        # V3 keeps V2's reconstruction exactly; its page-local x-band model is
        # consumed only at heading classification below.
        line_builder = _lines_for_page_v2
        reading_order = _reading_order_v2
        is_heading = _is_heading_v2
    else:
        raise ValueError(f"unsupported Paizo parser version: {parser_version}")
    if payload.get("schema_version") != PDF_EXPORT_SCHEMA_VERSION:
        raise ValueError("unsupported native PDF export schema")
    source_meta = payload.get("source") or {}
    if product is None:
        classified = _classify_name(str(source_meta.get("filename", fallback_filename)))
        if classified is None:
            raise ValueError("native export filename does not identify a catalog product")
        product = PRODUCT_CATALOG[classified[0]]
    source_name = source.display_name if source else _safe_basename(str(source_meta.get("filename", fallback_filename)))
    source_hash = source.source_sha256 if source else str(source_meta.get("sha256", ""))
    # Build the source-wide inventory before parser segmentation.  V1/V2 keep
    # their normal local-full text behavior, while carrying opaque anchors in
    # private provenance so the trusted staging path can prove coverage.
    inventory = native_word_inventory(payload, product.code)
    annotated_payload = annotate_native_words(payload, inventory)
    pages = [page for page in annotated_payload.get("pages", []) if isinstance(page, Mapping)]
    raw_pages_by_number = {
        int(page.get("number", 0)): page
        for page in payload.get("pages", [])
        if isinstance(page, Mapping)
    }
    page_records: list[tuple[int, str | None, list[_Line], _PageXBandModel | None]] = []
    all_sizes = [
        float(word.get("size", 0))
        for page in pages
        for word in page.get("words", [])
        if isinstance(word, Mapping) and float(word.get("size", 0)) > 0
    ]
    body_size = median(all_sizes) if all_sizes else 9.0
    for page in pages:
        page_number = int(page.get("number", 0))
        if source and source.part:
            part_start = source.part.split("-", 1)[0]
            if part_start.isdigit():
                page_number = int(part_start) + max(page_number - 1, 0)
        lines = []
        height = float(page.get("height", 0) or 0)
        raw_lines = line_builder(page)
        width = float(page.get("width", 0) or 0)
        original_page = raw_pages_by_number.get(int(page.get("number", 0)), page)
        printed_lines = line_builder(original_page)
        printed_candidates = [
            line.text.strip(" -–—")
            for line in printed_lines
            if _PAGE_NUMBER_RE.match(line.text)
            and (not height or line.bottom > height * 0.88)
            and (
                not width
                or line.x0 < width * 0.25
                or line.x1 > width * 0.75
            )
        ]
        printed_page = printed_candidates[-1] if printed_candidates else None
        # Quarantined watermark/furniture anchors have already been removed by
        # annotation. A body ``@`` or repeated table header stays visible and
        # must either attach to a rule or become explicit unclassified text.
        lines.extend(reading_order(page, raw_lines))
        x_band_model = (
            _page_x_band_model(raw_lines, width)
            if parser_version == PAIZO_NATIVE_PARSER_V3
            else None
        )
        page_records.append((page_number, printed_page, lines, x_band_model))

    content_fingerprint = inventory.content_fingerprint

    sections: list[dict[str, Any]] = []
    current_title = product.title
    current_lines: list[str] = []
    current_pages: list[int] = []
    current_printed_pages: list[str] = []
    current_layout_flags: set[str] = set()
    current_anchors: list[str] = []
    anchor_ordinals: Counter[tuple[int, str]] = Counter()

    def flush() -> None:
        nonlocal current_lines, current_pages, current_printed_pages, current_title
        nonlocal current_layout_flags, current_anchors
        body = _clean_text(" ".join(current_lines))
        if not body:
            current_lines = []
            current_pages = []
            current_printed_pages = []
            current_layout_flags = set()
            current_anchors = []
            return
        text = _clean_text(f"{current_title} {body}")
        page_values = tuple(dict.fromkeys(current_pages))
        section_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        page_start = min(page_values) if page_values else 0
        page_end = max(page_values) if page_values else 0
        printed_values = tuple(dict.fromkeys(current_printed_pages))
        if not printed_values:
            printed_page = None
        elif len(printed_values) == 1:
            printed_page = printed_values[0]
        else:
            printed_page = f"{printed_values[0]}-{printed_values[-1]}"
        anchor = (page_start, _slug(current_title))
        ordinal = anchor_ordinals[anchor]
        anchor_ordinals[anchor] += 1
        section_id = (
            f"corpus:{product.code}:{product.component}:p{page_start}:"
            f"{anchor[1]}:{ordinal}"
        )
        provenance = {
            "export_schema_version": payload.get("schema_version"),
            "extractor": payload.get("extractor", {}),
            "title": product.title,
            "source_path": source_name,
            "source_sha256": source_hash,
            "printing": source.printing if source else None,
            "component_part": source.part if source else None,
            "physical_pages": list(page_values),
            "printed_pages": list(printed_values),
            "content_fingerprint": content_fingerprint,
            # Private only: the review staging bridge consumes these and the
            # public projection intentionally drops them.
            "native_word_anchors": list(current_anchors),
        }
        if current_layout_flags:
            provenance["layout_flags"] = sorted(current_layout_flags)
        sections.append(
            RulebookSection(
                id=section_id,
                name=current_title,
                text=text,
                product_code=product.code,
                book=product.title,
                component=product.component,
                rules_era=product.rules_era,
                license=product.license,
                remaster=product.remaster,
                source_filename=source_name,
                source_sha256=source_hash,
                pages=page_values,
                page_start=page_start,
                page_end=page_end,
                printed_page=printed_page,
                section_hash=section_hash,
                provenance=provenance,
                ordinal=ordinal,
                parser_version=parser_version,
            ).as_chunk()
        )
        current_lines = []
        current_pages = []
        current_printed_pages = []
        current_layout_flags = set()
        current_anchors = []

    for page_number, printed_page, lines, x_band_model in page_records:
        for line in lines:
            heading = is_heading(line, body_size)
            if (
                parser_version == PAIZO_NATIVE_PARSER_V3
                and heading
                and x_band_model is not None
                and x_band_model.is_interior_cell_start(line)
                and _is_condensed_bold_body_candidate(line, body_size)
            ):
                # Keep this label in its surrounding section and require an
                # explicit layout-aware review before anything derived from it
                # can enter the public projection.
                heading = False
                if line.layout_kind == "body":
                    line = replace(line, layout_kind="table-cell")
            if heading:
                flush()
                current_title = line.text
                current_pages = [page_number]
                current_printed_pages = [printed_page] if printed_page else []
                current_anchors = list(line.native_word_anchors)
                continue
            current_lines.append(line.text)
            if parser_version in {PAIZO_NATIVE_PARSER_V2, PAIZO_NATIVE_PARSER_V3} and line.layout_kind != "body":
                current_layout_flags.add(line.layout_kind)
            current_pages.append(page_number)
            if printed_page:
                current_printed_pages.append(printed_page)
            current_anchors.extend(line.native_word_anchors)
    flush()
    assigned_counts = Counter(
        anchor
        for section in sections
        for anchor in section["provenance"].get("native_word_anchors", [])
    )
    expected = set(inventory.anchors) - set(inventory.ignored_anchor_reasons)
    duplicate = {anchor for anchor, count in assigned_counts.items() if count != 1}
    unknown = set(assigned_counts) - expected
    if duplicate or unknown:
        raise ValueError("parser assigned duplicate, invalid, or ignored native-word anchors")
    unassigned = expected - set(assigned_counts)
    if unassigned:
        # Never collapse dropped source text into an "ignored" category.  It
        # becomes a visible review section that can be excluded deliberately.
        words_by_anchor: dict[str, str] = {}
        pages_by_anchor: dict[str, int] = {}
        for page in pages:
            page_number = int(page.get("number", 0))
            for word in page.get("words", []):
                if not isinstance(word, Mapping):
                    continue
                anchor = word.get("_native_anchor")
                if isinstance(anchor, str) and anchor in unassigned:
                    words_by_anchor[anchor] = str(word.get("text", ""))
                    pages_by_anchor[anchor] = page_number
        by_page: dict[int, list[str]] = defaultdict(list)
        for anchor in sorted(unassigned, key=lambda value: (pages_by_anchor.get(value, 0), value)):
            if anchor not in words_by_anchor:
                raise ValueError("parser left an unclassified native word without recoverable text")
            by_page[pages_by_anchor[anchor]].append(anchor)
        for page_number, anchors in sorted(by_page.items()):
            text = _clean_text(" ".join(words_by_anchor[anchor] for anchor in anchors))
            if not text:
                raise ValueError("parser left an empty unclassified native-word section")
            section_id = f"corpus:{product.code}:{product.component}:p{page_number}:unclassified:0"
            section_hash = hashlib.sha256(text.encode()).hexdigest()
            provenance = {
                "export_schema_version": payload.get("schema_version"),
                "title": product.title,
                "physical_pages": [page_number],
                "printed_pages": [],
                "content_fingerprint": content_fingerprint,
                "native_word_anchors": anchors,
                "layout_flags": ["unclassified-native-coverage"],
            }
            sections.append(
                RulebookSection(
                    id=section_id, name="Unclassified native text", text=text,
                    product_code=product.code, book=product.title, component=product.component,
                    rules_era=product.rules_era, license=product.license, remaster=product.remaster,
                    source_filename=source_name, source_sha256=source_hash, pages=(page_number,),
                    page_start=page_number, page_end=page_number, printed_page=None,
                    section_hash=section_hash, provenance=provenance, ordinal=0,
                    parser_version=parser_version,
                ).as_chunk()
            )
    for section in sections:
        provenance = section.get("provenance")
        if isinstance(provenance, dict):
            provenance["stable_section_identity"] = _stable_section_identity(
                product, section.get("page_start"), section.get("printed_page"),
                section.get("name"), section.get("id", "").rsplit(":", 1)[-1],
            )
    return sections


def parse_verified_native_export(
    artifact: VerifiedNativeExport,
    *,
    parser_version: str = PAIZO_NATIVE_PARSER_VERSION,
    layout_binding: Any | None = None,
) -> TrustedParseBundle:
    """Parse one verified in-memory export into a private staging contract."""
    if (
        not artifact.pdf_verified
        or artifact._verification_token is not _TRUSTED_PDF_ORIGIN
        or not artifact._payload_digest
        or artifact._payload_digest != trusted_payload_digest(artifact.payload)
    ):
        raise ValueError("trusted parsing requires a PDF-verified native export")
    product = PRODUCT_CATALOG.get(artifact.product_code)
    if product is None:
        raise ValueError("verified export references an unsupported PZO product")
    inventory = native_word_inventory(artifact.payload, product.code, strict=True)
    v4_result: _V4ParseResult | None = None
    if parser_version in {PAIZO_NATIVE_PARSER_V4, PAIZO_NATIVE_PARSER_V5}:
        if layout_binding is None:
            raise ValueError(f"{parser_version} requires complete bound layout evidence")
        v4_result = _parse_rulebook_payload_v4(
            artifact.payload, product=product, binding=layout_binding,
            fallback_filename=artifact.source_basename, parser_version=parser_version,
        )
        sections = list(v4_result.chunks)
    else:
        if layout_binding is not None:
            raise ValueError("bound layout input is only accepted by structural parsers")
        sections = parse_rulebook_payload(
            artifact.payload, product=product, parser_version=parser_version,
            fallback_filename=artifact.source_basename,
        )
    if parser_version == PAIZO_NATIVE_PARSER_V5:
        _disambiguate_v5_stable_identities(sections)
    trusted_sections: list[TrustedSection] = []
    for section in sections:
        provenance = section.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("parser section lacks provenance")
        anchors = provenance.get("native_word_anchors")
        if not isinstance(anchors, list) or not all(isinstance(anchor, str) for anchor in anchors):
            raise ValueError("parser section lacks native anchor coverage")
        flags = provenance.get("layout_flags", [])
        if not isinstance(flags, list) or not all(isinstance(flag, str) for flag in flags):
            raise ValueError("parser section layout flags are invalid")
        stable = provenance.get("stable_section_identity")
        if not isinstance(stable, str):
            raise ValueError("parser section lacks stable identity")
        pages = section.get("pages")
        if not isinstance(pages, list) or not all(isinstance(page, int) for page in pages):
            raise ValueError("parser section pages are invalid")
        ordinal_text = str(section["id"]).rsplit(":", 1)[-1]
        if not ordinal_text.isdigit():
            raise ValueError("parser section lacks a structural ordinal")
        heading_chain = provenance.get("heading_chain", [section["name"]])
        if (
            not isinstance(heading_chain, list)
            or not heading_chain
            or any(not isinstance(value, str) or not value for value in heading_chain)
        ):
            raise ValueError("parser section heading chain is invalid")
        heading_anchor = hashlib.sha256(
            "\n".join(_normalize_name(value) for value in heading_chain).encode("utf-8")
        ).hexdigest()[:16]
        source_section_id = (
            f"{product.code.casefold()}:{product.component}:p{pages[0]}:"
            f"h{heading_anchor}:i{ordinal_text}"
        )
        if not _TRUSTED_SOURCE_SECTION_ID_RE.fullmatch(source_section_id):
            raise ValueError("parser generated an unsafe source section identifier")
        trusted_sections.append(
            TrustedSection(
                id=str(section["id"]), source_section_id=source_section_id,
                heading=str(section["name"]), text=str(section["text"]),
                text_hash=str(section["section_hash"]), physical_pages=tuple(pages),
                printed_page=section.get("printed_page") if isinstance(section.get("printed_page"), str) else None,
                stable_section_identity=stable, layout_flags=tuple(sorted(flags)),
                coverage_anchors=tuple(anchors),
                blocks=(
                    v4_result.blocks_by_section_id.get(str(section["id"]), ())
                    if v4_result is not None else ()
                ),
            )
        )
    quarantine = v4_result.quarantine if v4_result is not None else ()
    parser_output_digest = _trusted_parser_output_digest(trusted_sections, quarantine)
    attestation = MappingProxyType({
        "product_verified": True,
        "page_count": artifact.page_count,
        "title_marker_verified": bool(artifact.product_evidence.get("title_marker_verified")),
        "matched_product_count": len(artifact.product_evidence.get("matched_product_codes", ())),
        "conflict_product_count": len(artifact.product_evidence.get("conflict_product_codes", ())),
    })
    sealed_digest = _trusted_bundle_seal(
        product_code=product.code, parser_version=parser_version,
        exporter_profile_version=artifact.extractor_profile_version,
        semantic_fingerprint=inventory.content_fingerprint, artifact_attestation=attestation,
        artifact_attestation_digest=artifact.attestation_digest, inventory=inventory,
        sections=trusted_sections, parser_output_digest=parser_output_digest,
        quarantine=quarantine,
        layout_binding_digest=(
            str(layout_binding.binding_digest) if layout_binding is not None else None
        ),
    )
    return TrustedParseBundle(
        product_code=product.code,
        parser_version=parser_version,
        exporter_profile_version=artifact.extractor_profile_version,
        semantic_fingerprint=inventory.content_fingerprint,
        artifact_attestation=attestation,
        artifact_attestation_digest=artifact.attestation_digest,
        inventory=inventory,
        sections=tuple(trusted_sections),
        parser_output_digest=parser_output_digest,
        sealed_digest=sealed_digest,
        layout_binding_digest=(
            str(layout_binding.binding_digest) if layout_binding is not None else None
        ),
        quarantine=quarantine,
    )


def load_and_parse_verified_pdf(
    source_pdf: Path | str,
    *,
    product_code: str,
    parser_version: str = PAIZO_NATIVE_PARSER_VERSION,
    layout_artifact: Path | str | Mapping[str, object] | None = None,
) -> TrustedParseBundle:
    """The one-read trusted bridge from a selected PDF to sealed parser output."""
    product = PRODUCT_CATALOG.get(product_code)
    if product is None:
        raise ValueError("selected PDF references an unsupported PZO product")
    title_markers = (product.title, product.title.removeprefix("Pathfinder "))
    catalog_title_markers = {
        code: (spec.title, spec.title.removeprefix("Pathfinder "))
        for code, spec in PRODUCT_CATALOG.items()
    }
    artifact = verified_native_export_from_pdf(
        source_pdf, product_code=product_code, expected_title_markers=title_markers,
        catalog_title_markers=catalog_title_markers,
    )
    if layout_artifact is None:
        if parser_version in {PAIZO_NATIVE_PARSER_V4, PAIZO_NATIVE_PARSER_V5}:
            raise ValueError(f"{parser_version} requires complete layout evidence")
        return parse_verified_native_export(artifact, parser_version=parser_version)
    if parser_version not in {
        PAIZO_NATIVE_PARSER_V3, PAIZO_NATIVE_PARSER_V4, PAIZO_NATIVE_PARSER_V5,
    }:
        raise ValueError("layout evidence requires the current licensed-review parser")
    from .pdf_layout import bind_layout_to_native_export

    binding = bind_layout_to_native_export(artifact, layout_artifact)
    native_pages = tuple(
        sorted(
            int(page["number"])
            for page in artifact.payload["pages"]
            if isinstance(page, Mapping)
        )
    )
    if binding.selected_pages != native_pages:
        raise ValueError(
            "trusted layout evidence must cover every exported native PDF page"
        )
    if parser_version in {PAIZO_NATIVE_PARSER_V4, PAIZO_NATIVE_PARSER_V5}:
        return parse_verified_native_export(
            artifact, parser_version=parser_version, layout_binding=binding
        )
    bundle = parse_verified_native_export(artifact, parser_version=parser_version)
    return apply_layout_evidence(bundle, binding)


def apply_layout_evidence(bundle: TrustedParseBundle, binding: Any) -> TrustedParseBundle:
    """Add bounded layout-review flags without changing parser text or anchors."""
    bundle.verify_seal()
    if bundle.parser_version != PAIZO_NATIVE_PARSER_V3:
        raise ValueError("layout evidence can only adapt a paizo-native-v3 bundle")
    if binding.product_code != bundle.product_code or not re.fullmatch(
        r"[0-9a-f]{64}", str(binding.binding_digest)
    ):
        raise ValueError("native layout binding does not match the trusted parser bundle")

    anchor_owner: dict[str, int] = {}
    for section_index, section in enumerate(bundle.sections):
        for anchor in section.coverage_anchors:
            if anchor in anchor_owner:
                raise ValueError("trusted parser sections contain duplicate native anchors")
            anchor_owner[anchor] = section_index
    region_by_anchor: dict[str, Any] = {}
    split_sections: dict[int, set[str]] = defaultdict(set)
    textual_labels = {
        "abstract", "algorithm", "aside_text", "content", "doc_title", "figure_title",
        "footnote", "formula", "paragraph_title", "reference", "reference_content",
        "table", "text", "vision_footnote",
    }
    complex_labels = {
        "algorithm", "aside_text", "chart", "formula", "image", "table",
        "footnote", "vision_footnote",
    }
    for region in binding.regions:
        owners: set[int] = set()
        for anchor in region.native_word_anchors:
            if anchor in region_by_anchor:
                raise ValueError("native layout binding assigns one anchor to multiple regions")
            region_by_anchor[anchor] = region
            owner = anchor_owner.get(anchor)
            if owner is not None:
                owners.add(owner)
        ordered = sorted(owners)
        if (
            region.label in textual_labels
            and 2 <= len(ordered) <= 3
            and ordered == list(range(ordered[0], ordered[-1] + 1))
        ):
            token = f"layout-region-split:p{region.page}-o{region.order}"
            for owner in ordered:
                split_sections[owner].add(token)

    unbound = set(binding.unbound_native_anchors)
    sections: list[TrustedSection] = []
    for section_index, section in enumerate(bundle.sections):
        flags = set(section.layout_flags)
        section_regions = [
            region_by_anchor[anchor]
            for anchor in section.coverage_anchors
            if anchor in region_by_anchor
        ]
        if any(anchor in unbound for anchor in section.coverage_anchors):
            flags.add("layout-model-unbound")
        if any(region.label in complex_labels for region in section_regions):
            flags.add("layout-model-complex")
        if any(region.label == "table" for region in section_regions):
            flags.add("layout-model-table")
        collapsed_orders: list[tuple[int, int]] = []
        for region in section_regions:
            key = (region.page, region.order)
            if not collapsed_orders or collapsed_orders[-1] != key:
                collapsed_orders.append(key)
        if any(right < left for left, right in zip(collapsed_orders, collapsed_orders[1:], strict=False)):
            flags.add("layout-order-conflict")
        if section_index in split_sections:
            flags.add("layout-region-split")
            flags.update(split_sections[section_index])
        sections.append(replace(section, layout_flags=tuple(sorted(flags))))

    parser_output_digest = _trusted_parser_output_digest(sections)
    sealed_digest = _trusted_bundle_seal(
        product_code=bundle.product_code,
        parser_version=PAIZO_NATIVE_LAYOUT_V1,
        exporter_profile_version=bundle.exporter_profile_version,
        semantic_fingerprint=bundle.semantic_fingerprint,
        artifact_attestation=bundle.artifact_attestation,
        artifact_attestation_digest=bundle.artifact_attestation_digest,
        inventory=bundle.inventory,
        sections=sections,
        parser_output_digest=parser_output_digest,
        layout_binding_digest=binding.binding_digest,
    )
    adapted = TrustedParseBundle(
        product_code=bundle.product_code,
        parser_version=PAIZO_NATIVE_LAYOUT_V1,
        exporter_profile_version=bundle.exporter_profile_version,
        semantic_fingerprint=bundle.semantic_fingerprint,
        artifact_attestation=bundle.artifact_attestation,
        artifact_attestation_digest=bundle.artifact_attestation_digest,
        inventory=bundle.inventory,
        sections=tuple(sections),
        parser_output_digest=parser_output_digest,
        sealed_digest=sealed_digest,
        layout_binding_digest=binding.binding_digest,
    )
    adapted.verify_seal()
    return adapted


def repair_trusted_bundle(
    bundle: TrustedParseBundle,
    merge_groups: Sequence[Sequence[str]],
) -> TrustedParseBundle:
    """Return a newly sealed bundle after complete adjacent-section unions.

    ``merge_groups`` contains source-section IDs from the freshly re-read,
    sealed bundle. Groups must contain two or three consecutive sections and
    may not overlap. No anchor may be added, removed, or duplicated.
    """
    bundle.verify_seal()
    sections = list(bundle.sections)
    positions = {section.source_section_id: index for index, section in enumerate(sections)}
    claimed: set[int] = set()
    normalized: dict[int, tuple[int, ...]] = {}
    for raw_group in merge_groups:
        group = tuple(raw_group)
        if len(group) not in {2, 3} or len(set(group)) != len(group):
            raise ValueError("trusted stitch groups must contain two or three distinct sections")
        try:
            indexes = tuple(positions[value] for value in group)
        except KeyError as exc:
            raise ValueError("trusted stitch references an unknown source section") from exc
        if indexes != tuple(range(indexes[0], indexes[0] + len(indexes))):
            raise ValueError("trusted stitch groups must preserve consecutive parser order")
        if claimed.intersection(indexes):
            raise ValueError("trusted stitch groups must not overlap")
        claimed.update(indexes)
        normalized[indexes[0]] = indexes

    repaired: list[TrustedSection] = []
    index = 0
    while index < len(sections):
        indexes = normalized.get(index)
        if indexes is None:
            repaired.append(sections[index])
            index += 1
            continue
        group = [sections[position] for position in indexes]
        pages = tuple(sorted({page for section in group for page in section.physical_pages}))
        anchors = tuple(anchor for section in group for anchor in section.coverage_anchors)
        merged_blocks = tuple(
            replace(block, ordinal=ordinal)
            for ordinal, block in enumerate(
                block for section in group for block in section.blocks
            )
        )
        if len(set(anchors)) != len(anchors):
            raise ValueError("trusted stitch would duplicate native anchors")
        text = "\n\n".join(section.text.rstrip() for section in group).strip()
        stable = hashlib.sha256(
            ("paizo-stitched-section-v1\n" + "\n".join(
                section.stable_section_identity for section in group
            )).encode("utf-8")
        ).hexdigest()
        repaired.append(
            TrustedSection(
                id=group[0].id,
                source_section_id=group[0].source_section_id,
                heading=group[0].heading,
                text=text,
                text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                physical_pages=pages,
                printed_page=group[0].printed_page,
                stable_section_identity=stable,
                layout_flags=tuple(sorted({
                    "stitched-adjacent-v1",
                    *(flag for section in group for flag in section.layout_flags),
                })),
                coverage_anchors=anchors,
                blocks=merged_blocks,
            )
        )
        index += len(indexes)

    original_anchors = [anchor for section in sections for anchor in section.coverage_anchors]
    repaired_anchors = [anchor for section in repaired for anchor in section.coverage_anchors]
    if sorted(original_anchors) != sorted(repaired_anchors):
        raise ValueError("trusted stitch changed native anchor coverage")
    parser_output_digest = _trusted_parser_output_digest(repaired, bundle.quarantine)
    sealed_digest = _trusted_bundle_seal(
        product_code=bundle.product_code,
        parser_version=bundle.parser_version,
        exporter_profile_version=bundle.exporter_profile_version,
        semantic_fingerprint=bundle.semantic_fingerprint,
        artifact_attestation=bundle.artifact_attestation,
        artifact_attestation_digest=bundle.artifact_attestation_digest,
        inventory=bundle.inventory,
        sections=repaired,
        parser_output_digest=parser_output_digest,
        quarantine=bundle.quarantine,
        layout_binding_digest=bundle.layout_binding_digest,
    )
    result = TrustedParseBundle(
        product_code=bundle.product_code,
        parser_version=bundle.parser_version,
        exporter_profile_version=bundle.exporter_profile_version,
        semantic_fingerprint=bundle.semantic_fingerprint,
        artifact_attestation=bundle.artifact_attestation,
        artifact_attestation_digest=bundle.artifact_attestation_digest,
        inventory=bundle.inventory,
        sections=tuple(repaired),
        parser_output_digest=parser_output_digest,
        sealed_digest=sealed_digest,
        layout_binding_digest=bundle.layout_binding_digest,
        quarantine=bundle.quarantine,
    )
    result.verify_seal()
    return result


def load_and_parse_verified_native_export(*args: Any, **kwargs: Any) -> TrustedParseBundle:
    """Removed trusted JSON bridge; callers must use ``load_and_parse_verified_pdf``."""
    raise ValueError("cached native exports are untrusted; provide the source PDF to load_and_parse_verified_pdf")


def parse_exports(
    prepared: Iterable[PreparedExport],
    *,
    parser_version: str = PAIZO_NATIVE_PARSER_VERSION,
) -> list[dict[str, Any]]:
    """Parse selected prepared artifacts in source/page reading order."""
    chunks: list[dict[str, Any]] = []
    for item in prepared:
        chunks.extend(
            parse_rulebook_export(
                item.output_path,
                source=item.source,
                parser_version=parser_version,
            )
        )
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        by_source[chunk["source_id"]].append(chunk)
    for source_chunks in by_source.values():
        normalized = "\n".join(
            f"{chunk['id']}:{chunk['section_hash']}"
            for chunk in sorted(source_chunks, key=lambda value: value["id"])
        )
        fingerprint = hashlib.sha256(normalized.encode()).hexdigest()
        for chunk in source_chunks:
            chunk["source"]["revision"] = fingerprint
            chunk["source"]["provenance"]["content_fingerprint"] = fingerprint
    return chunks


# Short aliases make CLI/pipeline integration discoverable without hiding the
# explicit parser name used by tests and callers.
discover = discover_sources
select = select_revisions
parse_export = parse_rulebook_export


__all__ = [
    "CORPUS_SCHEMA_VERSION",
    "PAIZO_NATIVE_PARSER_V1",
    "PAIZO_NATIVE_PARSER_V2",
    "PAIZO_NATIVE_PARSER_V3",
    "PAIZO_NATIVE_PARSER_V4",
    "PAIZO_NATIVE_PARSER_V5",
    "PAIZO_NATIVE_LAYOUT_V1",
    "PAIZO_NATIVE_PARSER_VERSION",
    "PRODUCT_CATALOG",
    "SELECTION_STATE_FILENAME",
    "CorpusSource",
    "PreparedExport",
    "ProductSpec",
    "RulebookSection",
    "TrustedSection",
    "TrustedBlock",
    "TrustedQuarantine",
    "TrustedParseBundle",
    "SelectedRevision",
    "discover",
    "discover_sources",
    "group_sources",
    "parse_export",
    "parse_exports",
    "parse_rulebook_payload",
    "parse_rulebook_export",
    "parse_verified_native_export",
    "apply_layout_evidence",
    "load_and_parse_verified_native_export",
    "load_and_parse_verified_pdf",
    "prepare_exports",
    "select",
    "select_revisions",
]
