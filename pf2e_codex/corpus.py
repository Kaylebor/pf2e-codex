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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from .pdf_export import PDF_EXPORT_SCHEMA_VERSION, export_pdf

SELECTION_STATE_FILENAME = ".pf2e-codex-corpus-selection.json"
CORPUS_SCHEMA_VERSION = 1


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
                "parser": "paizo-native-v1",
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


def _lines_for_page(page: Mapping[str, Any]) -> list[_Line]:
    words = [word for word in page.get("words", []) if isinstance(word, Mapping)]
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
        # At the same y-coordinate, two newspaper-style columns appear as one
        # flat word row.  A large horizontal gap is geometric evidence that it
        # is actually two lines and must not be interleaved.
        subgroups: list[list[Mapping[str, Any]]] = []
        for word in group:
            if not subgroups:
                subgroups.append([word])
                continue
            previous = subgroups[-1][-1]
            gap = float(word.get("x0", 0)) - float(previous.get("x1", previous.get("x0", 0)))
            if gap > 48:
                subgroups.append([word])
            else:
                subgroups[-1].append(word)
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
                )
            )
    return lines


def _reading_order(page: Mapping[str, Any], lines: list[_Line]) -> list[_Line]:
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
    has_columns = bool(left and right)
    if not has_columns or len(body) < 6:
        return sorted(lines, key=lambda line: (line.top, line.x0))
    # Full-width headings are emitted at their vertical position; column body
    # is read down the left column and then down the right column.
    if full:
        first_body = min(line.top for line in body)
        before = sorted((line for line in full if line.top <= first_body), key=lambda line: line.top)
        after = sorted((line for line in full if line.top > first_body), key=lambda line: line.top)
        ordered = before
        ordered.extend(sorted(left, key=lambda line: line.top))
        ordered.extend(sorted(right, key=lambda line: line.top))
        ordered.extend(after)
    else:
        ordered = sorted(left, key=lambda line: line.top) + sorted(right, key=lambda line: line.top)
    return ordered


def _repeated_furniture(pages: Sequence[Mapping[str, Any]]) -> set[str]:
    candidates: list[str] = []
    for page in pages:
        height = float(page.get("height", 0) or 0)
        for line in _lines_for_page(page):
            if height and (line.top < height * 0.12 or line.bottom > height * 0.88):
                normalized = _normalize_name(re.sub(r"\b\d+\b", "", line.text))
                if normalized and not _PAGE_NUMBER_RE.match(line.text):
                    candidates.append(normalized)
    return {value for value, count in Counter(candidates).items() if count >= 2}


def _is_heading(line: _Line, body_size: float) -> bool:
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


def _slug(value: str) -> str:
    value = _normalize_name(value).replace(" ", "-")
    return value[:80] or "section"


def parse_rulebook_export(
    export_path: Path | str,
    *,
    product: ProductSpec | None = None,
    source: CorpusSource | None = None,
) -> list[dict[str, Any]]:
    """Parse one exporter JSON artifact into ``rulebook_section`` chunks."""
    path = Path(export_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PDF_EXPORT_SCHEMA_VERSION:
        raise ValueError("unsupported native PDF export schema")
    source_meta = payload.get("source") or {}
    if product is None:
        classified = _classify_name(str(source_meta.get("filename", path.name)))
        if classified is None:
            raise ValueError("native export filename does not identify a catalog product")
        product = PRODUCT_CATALOG[classified[0]]
    source_name = source.display_name if source else str(source_meta.get("filename", path.name))
    source_hash = source.source_sha256 if source else str(source_meta.get("sha256", ""))
    pages = [page for page in payload.get("pages", []) if isinstance(page, Mapping)]
    repeated = _repeated_furniture(pages)
    page_records: list[tuple[int, str | None, list[_Line]]] = []
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
        raw_lines = _lines_for_page(page)
        width = float(page.get("width", 0) or 0)
        printed_candidates = [
            line.text.strip(" -–—")
            for line in raw_lines
            if _PAGE_NUMBER_RE.match(line.text)
            and (not height or line.bottom > height * 0.88)
            and (
                not width
                or line.x0 < width * 0.25
                or line.x1 > width * 0.75
            )
        ]
        printed_page = printed_candidates[-1] if printed_candidates else None
        for line in _reading_order(page, raw_lines):
            normalized = _normalize_name(line.text)
            if normalized in repeated or _PAGE_NUMBER_RE.match(line.text):
                continue
            if _contains_email(line.text):
                continue
            # A lone email-like watermark can be rotated or split in native
            # extraction; remove any line whose compacted text still contains
            # the characteristic marker without retaining/logging its value.
            if height and (line.top < height * 0.04 or line.bottom > height * 0.96) and "@" in line.text:
                continue
            lines.append(line)
        page_records.append((page_number, printed_page, lines))

    # This intentionally excludes raw PDF bytes, paths, page furniture, and
    # watermark-like lines. It is therefore stable across personalized or
    # regenerated copies while still changing when searchable rules change.
    normalized_source_text = "\n".join(
        f"{page_number}:{line.text}"
        for page_number, _printed_page, lines in page_records
        for line in lines
    )
    content_fingerprint = hashlib.sha256(
        f"{product.code}\n{normalized_source_text}".encode()
    ).hexdigest()

    sections: list[dict[str, Any]] = []
    current_title = product.title
    current_lines: list[str] = []
    current_pages: list[int] = []
    current_printed_pages: list[str] = []
    anchor_ordinals: Counter[tuple[int, str]] = Counter()

    def flush() -> None:
        nonlocal current_lines, current_pages, current_printed_pages, current_title
        body = _clean_text(" ".join(current_lines))
        if not body:
            current_lines = []
            current_pages = []
            current_printed_pages = []
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
        }
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
            ).as_chunk()
        )
        current_lines = []
        current_pages = []
        current_printed_pages = []

    for page_number, printed_page, lines in page_records:
        for line in lines:
            if _is_heading(line, body_size):
                flush()
                current_title = line.text
                current_pages = [page_number]
                current_printed_pages = [printed_page] if printed_page else []
                continue
            current_lines.append(line.text)
            current_pages.append(page_number)
            if printed_page:
                current_printed_pages.append(printed_page)
    flush()
    return sections


def parse_exports(
    prepared: Iterable[PreparedExport],
) -> list[dict[str, Any]]:
    """Parse selected prepared artifacts in source/page reading order."""
    chunks: list[dict[str, Any]] = []
    for item in prepared:
        chunks.extend(parse_rulebook_export(item.output_path, source=item.source))
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
    "PRODUCT_CATALOG",
    "SELECTION_STATE_FILENAME",
    "CorpusSource",
    "PreparedExport",
    "ProductSpec",
    "RulebookSection",
    "SelectedRevision",
    "discover",
    "discover_sources",
    "group_sources",
    "parse_export",
    "parse_exports",
    "parse_rulebook_export",
    "prepare_exports",
    "select",
    "select_revisions",
]
