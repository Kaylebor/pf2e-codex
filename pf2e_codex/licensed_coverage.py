"""Deterministic private-corpus normalization and clean-Foundry coverage evidence.

The functions in this module never decide whether private text is publishable.
They canonicalize exact duplicates and prepare bounded public Foundry candidates
for the Spark screening judgment owned by :mod:`pf2e_codex.review_runner`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import threading
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .distribution import audit_database_slot
from .fetcher import extract_all_packs
from .foundry_scope import REDISTRIBUTABLE_FOUNDRY_PUBLICATIONS, is_redistributable_foundry_entry

NORMALIZER_VERSION = "licensed-coverage-v2"
MAX_FOUNDRY_CANDIDATES = 3
_SNAPSHOT_CACHE: dict[tuple[str, int, int], FoundrySnapshot] = {}
_SNAPSHOT_CACHE_LOCK = threading.Lock()

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_NUMBER_RE = re.compile(
    r"(?<!\w)(?:\d+d\d+(?:\s*[+-]\s*\d+)?|\d+(?:\.\d+)?(?:st|nd|rd|th)?)(?!\w)",
    re.IGNORECASE,
)
_SPACE_RE = re.compile(r"\s+")
_BOILERPLATE_RE = re.compile(
    r"^(?:description|source|page|chapter|section|contents?)\s*:\s*",
    re.IGNORECASE,
)
_PUNCT_TRANSLATION = str.maketrans({
    "’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-",
    "−": "-", "•": " ", "◆": " action ", "◇": " free-action ",
    "▶": " action ", "◀": " action ", "↺": " reaction ",
})


def _digest(namespace: str, *values: object) -> str:
    payload = "\n".join((namespace, *(str(value) for value in values)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_heading(value: object) -> str:
    """Return a conservative identity normalization for a rule heading."""
    text = unicodedata.normalize("NFKC", str(value or "")).translate(_PUNCT_TRANSLATION)
    text = " ".join(_WORD_RE.findall(text.casefold()))
    return _SPACE_RE.sub(" ", text).strip()


def normalize_rule_text(value: object) -> str:
    """Return a deterministic comparison form without retaining it in SQLite."""
    text = unicodedata.normalize("NFKC", str(value or "")).translate(_PUNCT_TRANSLATION)
    lines = []
    for raw in text.splitlines() or [text]:
        cleaned = _BOILERPLATE_RE.sub("", raw.strip())
        if cleaned:
            lines.append(cleaned)
    return " ".join(_WORD_RE.findall(" ".join(lines).casefold()))


def normalized_hash(value: object, *, heading: bool = False) -> str:
    normalized = normalize_heading(value) if heading else normalize_rule_text(value)
    return _digest(NORMALIZER_VERSION, normalized)


def foundry_identity_hashes(name: object, text: object) -> tuple[str, ...]:
    """Return exact normalized identities for deterministic Foundry wrappers."""
    raw = str(text or "")
    variants = {raw}
    marker = "\n\nDescription:\n"
    if marker in raw:
        description = raw.split(marker, 1)[1].split("\n\nSource:", 1)[0].strip()
        if description:
            variants.add(description)
            variants.add(f"{name}\n{description}")
    return tuple(sorted(normalized_hash(value) for value in variants))


def _tokens(value: object) -> tuple[str, ...]:
    return tuple(normalize_rule_text(value).split())


def _trigrams(tokens: Sequence[str]) -> frozenset[tuple[str, str, str]]:
    return frozenset(zip(tokens, tokens[1:], tokens[2:], strict=False)) if len(tokens) >= 3 else frozenset()


def _ratio(left: Iterable[object], right: Iterable[object]) -> float:
    lhs, rhs = set(left), set(right)
    return len(lhs & rhs) / max(1, len(lhs))


def comparison_metrics(private_text: object, foundry_texts: Sequence[object]) -> dict[str, object]:
    """Return content-free lexical and numeric coverage measurements."""
    private_tokens = _tokens(private_text)
    public_tokens = tuple(token for value in foundry_texts for token in _tokens(value))
    private_numbers = tuple(_NUMBER_RE.findall(str(private_text or "")))
    public_numbers = tuple(
        number for value in foundry_texts for number in _NUMBER_RE.findall(str(value or ""))
    )
    return {
        "private_token_count": len(private_tokens),
        "foundry_token_count": len(public_tokens),
        "token_coverage": round(_ratio(private_tokens, public_tokens), 6),
        "trigram_coverage": round(_ratio(_trigrams(private_tokens), _trigrams(public_tokens)), 6),
        "numeric_coverage": round(_ratio((n.casefold() for n in private_numbers), (n.casefold() for n in public_numbers)), 6),
        "private_numeric_count": len(private_numbers),
    }


def duplicate_identity(*, heading: object, text: object, license_name: object, era: object) -> str:
    """Identity for exact same-heading mechanics within one license and era."""
    return _digest(
        "licensed-duplicate-group-v1",
        NORMALIZER_VERSION,
        str(license_name),
        str(era),
        normalized_hash(heading, heading=True),
        normalized_hash(text),
    )


@dataclass(frozen=True)
class FoundryRow:
    chunk_id: str
    name: str
    text: str
    source_hash: str
    normalized_hash: str
    heading_hash: str
    publication_title: str
    license: str
    era: str
    type: str

    def manifest_record(self) -> dict[str, object]:
        return {
            "id": self.chunk_id,
            "source_hash": self.source_hash,
            "normalized_hash": self.normalized_hash,
            "identity_hashes": list(foundry_identity_hashes(self.name, self.text)),
            "heading_hash": self.heading_hash,
            "publication_title": self.publication_title,
            "license": self.license,
            "era": self.era,
            "type": self.type,
        }


@dataclass(frozen=True)
class FoundrySnapshot:
    release: str
    digest: str
    rows: tuple[FoundryRow, ...]


def load_clean_foundry(path: Path | str) -> FoundrySnapshot:
    """Validate and load stable public evidence from a clean model database."""
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    cache_key = (str(resolved), stat.st_mtime_ns, stat.st_size)
    with _SNAPSHOT_CACHE_LOCK:
        cached = _SNAPSHOT_CACHE.get(cache_key)
        if cached is not None:
            return cached
    audit_database_slot(resolved, "clean")
    conn = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        scope = conn.execute("SELECT value FROM _meta WHERE key='distribution_scope'").fetchone()
        if scope is None or str(scope[0]) != "redistributable":
            raise ValueError("Foundry coverage requires a redistributable database")
        leaked = conn.execute(
            "SELECT 1 FROM chunks WHERE origin<>'foundry' LIMIT 1"
        ).fetchone()
        if leaked is not None:
            raise ValueError("Foundry coverage database must contain only Foundry rows")
        release_row = conn.execute("SELECT value FROM _meta WHERE key='pf2e_release'").fetchone()
        release = str(release_row[0]) if release_row else "unknown"
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(chunks)")}
        publication = "publication_title" if "publication_title" in columns else "'' AS publication_title"
        rows = []
        for row in conn.execute(
            f"""SELECT id, name, text, source_hash, {publication}, license, remaster, type
                  FROM chunks WHERE origin='foundry' ORDER BY id"""
        ):
            era = "remaster" if row["remaster"] == 1 else "legacy" if row["remaster"] == 0 else "unknown"
            text = str(row["text"] or "")
            name = str(row["name"] or "")
            publication_title = str(row["publication_title"] or "")
            if not publication_title:
                source_suffix = text.rsplit("Source:", 1)[-1]
                publication_title = next(
                    (
                        title for title in sorted(
                            REDISTRIBUTABLE_FOUNDRY_PUBLICATIONS, key=len, reverse=True
                        )
                        if title in source_suffix
                    ),
                    "",
                )
            if not publication_title:
                raise ValueError(
                    f"clean Foundry row lacks deterministic publication ownership: {row['id']}"
                )
            rows.append(FoundryRow(
                chunk_id=str(row["id"]), name=name, text=text,
                source_hash=str(row["source_hash"] or hashlib.sha256(text.encode()).hexdigest()),
                normalized_hash=normalized_hash(text),
                heading_hash=normalized_hash(name, heading=True),
                publication_title=publication_title,
                license=str(row["license"] or "NONE"), era=era,
                type=str(row["type"] or ""),
            ))
    finally:
        conn.close()
    manifest = [row.manifest_record() for row in rows]
    digest = _digest(
        "clean-foundry-snapshot-v1", release,
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
    )
    snapshot = FoundrySnapshot(release, digest, tuple(rows))
    with _SNAPSHOT_CACHE_LOCK:
        _SNAPSHOT_CACHE.clear()
        _SNAPSHOT_CACHE[cache_key] = snapshot
    return snapshot


class FoundryMatcher:
    """Deterministic exact-plus-bounded-lexical candidate selector."""

    def __init__(self, snapshot: FoundrySnapshot):
        self.snapshot = snapshot
        self._by_heading: dict[str, list[FoundryRow]] = defaultdict(list)
        self._by_term: dict[str, set[int]] = defaultdict(set)
        for index, row in enumerate(snapshot.rows):
            self._by_heading[row.heading_hash].append(row)
            for term in set(normalize_heading(row.name).split()):
                if len(term) >= 3:
                    self._by_term[term].add(index)

    def candidates(self, section: Mapping[str, object]) -> list[dict[str, object]]:
        era, license_name = str(section["era"]), str(section["license"])
        heading_hash = normalized_hash(section["heading"], heading=True)
        exact = list(self._by_heading.get(heading_hash, ()))
        candidate_indexes: set[int] = set()
        heading_terms = set(normalize_heading(section["heading"]).split())
        for term in heading_terms:
            candidate_indexes.update(self._by_term.get(term, ()))
        pool = {row.chunk_id: row for row in exact}
        for index in candidate_indexes:
            row = self.snapshot.rows[index]
            pool.setdefault(row.chunk_id, row)
        ranked: list[tuple[tuple[object, ...], FoundryRow, dict[str, object]]] = []
        expected_publication = str(section.get("publication_title") or "")
        for row in pool.values():
            if row.era != era or row.license != license_name:
                continue
            public_heading_terms = set(normalize_heading(row.name).split())
            heading_overlap = _ratio(heading_terms, public_heading_terms)
            exact_heading = row.heading_hash == heading_hash
            if not exact_heading and heading_overlap < 0.6:
                continue
            metrics = comparison_metrics(section["source_text"], [row.text])
            exact_identity = (
                normalized_hash(section["source_text"])
                in foundry_identity_hashes(row.name, row.text)
            )
            metrics["exact_identity"] = exact_identity
            if not exact_heading and float(metrics["token_coverage"]) < 0.6:
                continue
            same_publication = bool(expected_publication and row.publication_title == expected_publication)
            score = (
                int(exact_identity), int(exact_heading), int(same_publication),
                float(metrics["numeric_coverage"]), float(metrics["trigram_coverage"]),
                float(metrics["token_coverage"]), -abs(
                    int(metrics["private_token_count"]) - int(metrics["foundry_token_count"])
                ),
            )
            ranked.append((score, row, {**metrics, "heading_overlap": round(heading_overlap, 6)}))
        ranked.sort(key=lambda item: (tuple(-value for value in item[0]), item[1].chunk_id))
        return [
            {
                "foundry_id": row.chunk_id,
                "name": row.name,
                "text": row.text,
                "type": row.type,
                "publication_title": row.publication_title,
                "license": row.license,
                "era": row.era,
                "source_hash": row.source_hash,
                "normalized_hash": row.normalized_hash,
                "metrics": metrics,
            }
            for _score, row, metrics in ranked[:MAX_FOUNDRY_CANDIDATES]
        ]


def build_foundry_evidence_database(
    archive: Path | str,
    output: Path | str,
    *,
    release: str,
) -> dict[str, object]:
    """Build a vector-free, strictly filtered Foundry evidence projection."""
    from .chunker import ChunkBuilder, UUIDResolver

    source = Path(archive).expanduser().resolve()
    if not source.is_file() or not release:
        raise ValueError("Foundry evidence requires a cached archive and release")
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(f".{target.name}.staging-{os.getpid()}")
    staged.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="pf2e-foundry-evidence-") as raw:
        entries = extract_all_packs(source, Path(raw) / "extract")
        builder = ChunkBuilder(UUIDResolver(entries))
        chunks: list[dict[str, object]] = []
        for pack_name in sorted(entries):
            for entry in entries[pack_name]:
                if not is_redistributable_foundry_entry(entry):
                    continue
                for chunk in builder.build_all(entry, pack_name):
                    title = str(chunk.get("publication_title") or "")
                    license_name = str(chunk.get("license") or "")
                    if title not in REDISTRIBUTABLE_FOUNDRY_PUBLICATIONS or license_name not in {"OGL", "ORC"}:
                        raise ValueError("filtered Foundry chunk lost owning-publication provenance")
                    chunks.append(chunk)
    conn = sqlite3.connect(staged)
    try:
        conn.executescript(
            """
            CREATE TABLE _meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE chunks(
                id TEXT PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL,
                pack TEXT NOT NULL, text TEXT NOT NULL, source_hash TEXT NOT NULL,
                license TEXT NOT NULL, remaster INTEGER, publication_title TEXT NOT NULL,
                origin TEXT NOT NULL CHECK(origin='foundry'));
            """
        )
        conn.executemany("INSERT INTO _meta VALUES (?, ?)", [
            ("distribution_scope", "redistributable"),
            ("foundry_scope", "core-publications-v1"),
            ("pf2e_release", release),
            ("embedding_model", "none-evidence-only"),
            ("total_chunks", str(len(chunks))),
        ])
        conn.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'foundry')",
            (
                (
                    str(chunk["id"]), str(chunk["name"]), str(chunk["type"]),
                    str(chunk["pack"]), str(chunk["text"]), str(chunk["source_hash"]),
                    str(chunk["license"]), chunk.get("remaster"),
                    str(chunk["publication_title"]),
                )
                for chunk in sorted(chunks, key=lambda item: str(item["id"]))
            ),
        )
        conn.commit()
        conn.execute("VACUUM")
    except BaseException:
        conn.close()
        staged.unlink(missing_ok=True)
        raise
    else:
        conn.close()
    os.replace(staged, target)
    audit_database_slot(target, "clean")
    snapshot = load_clean_foundry(target)
    return {
        "output": target.name,
        "release": release,
        "rows": len(snapshot.rows),
        "snapshot_digest": snapshot.digest,
    }


__all__ = [
    "FoundryMatcher", "FoundryRow", "FoundrySnapshot", "MAX_FOUNDRY_CANDIDATES",
    "NORMALIZER_VERSION", "comparison_metrics", "duplicate_identity",
    "build_foundry_evidence_database", "load_clean_foundry", "normalize_heading",
    "normalize_rule_text", "normalized_hash",
]
