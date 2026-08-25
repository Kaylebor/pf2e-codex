"""Load and validate the reviewed, redistributable licensed-core projection."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from .licensed_policy import (
    LICENSED_CORE_POLICY_VERSION,
    licensed_policy_digest,
)

LICENSED_CORE_SCHEMA_VERSION = 1
LICENSED_CORE_RESOURCE = "data/licensed_core.sqlite3"
LICENSED_CORE_ORIGIN = "licensed-core"
_LICENSES = {"OGL", "ORC"}
_ERAS = {"legacy", "remaster", "unknown"}
_PRODUCT_CODE = re.compile(r"^PZO[0-9]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_SECTION_ID = re.compile(
    r"^(?P<product>pzo[0-9]+):[a-z0-9]+(?:-[a-z0-9]+)*:"
    r"p(?P<page>[1-9][0-9]*):h[0-9a-f]{16}:i[0-9]+$"
)
_EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_WINDOWS_PATH = re.compile(r"(?i)(?:^|\s)[A-Z]:[\\/]")
_PRIVATE_TEXT_MARKERS = (
    ".local-corpus",
    "source_sha256",
    "source_path",
    "file://",
    "/home/",
    "/users/",
    "\\users\\",
)


@dataclass(frozen=True)
class LicensedCoreBundle:
    """Validated content and release metadata read from the static corpus."""

    chunks: tuple[dict[str, Any], ...]
    source_revisions: tuple[dict[str, Any], ...]
    notices: tuple[dict[str, str], ...]
    schema_version: int = LICENSED_CORE_SCHEMA_VERSION


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _validate_public_text(value: str, *, field: str) -> None:
    """Reject obvious watermark and private-local provenance at the load boundary."""
    lowered = value.casefold()
    if (
        "@" in value
        or _EMAIL.search(value)
        or _WINDOWS_PATH.search(value)
        or any(marker in lowered for marker in _PRIVATE_TEXT_MARKERS)
        or any(ord(char) < 32 and char not in "\n\t" for char in value)
    ):
        raise ValueError(f"licensed-core {field} contains private or unsafe text")


def _validate_public_scalar(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"licensed-core {field} must be a non-empty string")
    _validate_public_text(value, field=field)
    return value


@contextmanager
def _bundled_path() -> Iterator[Path | None]:
    resource = resources.files("pf2e_codex").joinpath(LICENSED_CORE_RESOURCE)
    if not resource.is_file():
        yield None
        return
    with resources.as_file(resource) as path:
        yield path


def _read_bundle(path: Path, excluded: frozenset[str]) -> LicensedCoreBundle:
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        required = {"metadata", "source_revisions", "notices", "licensed_sections"}
        missing = required - tables
        if missing:
            raise ValueError(
                "licensed-core database is missing tables: " + ", ".join(sorted(missing))
            )

        metadata = {
            str(row[0]): str(row[1])
            for row in conn.execute("SELECT key, value FROM metadata")
        }
        if metadata.get("content_scope") != "licensed-core-reviewed":
            raise ValueError("licensed-core database has an invalid content scope")
        if (
            metadata.get("policy_version") != LICENSED_CORE_POLICY_VERSION
            or metadata.get("policy_digest") != licensed_policy_digest()
        ):
            raise ValueError("licensed-core database has an invalid review policy")
        try:
            schema_version = int(metadata.get("public_schema_version", ""))
        except ValueError as exc:
            raise ValueError("licensed-core database has no valid schema version") from exc
        if schema_version != LICENSED_CORE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported licensed-core schema {schema_version}; "
                f"expected {LICENSED_CORE_SCHEMA_VERSION}"
            )

        notices_by_key: dict[str, dict[str, str]] = {}
        for row in conn.execute(
            "SELECT notice_key, license, text FROM notices ORDER BY notice_key"
        ):
            notice = {
                "notice_key": _validate_public_scalar(
                    row["notice_key"], field="notice key"
                ),
                "license": str(row["license"]),
                "text": _validate_public_scalar(row["text"], field="notice"),
            }
            if notice["license"] not in _LICENSES:
                raise ValueError(f"licensed-core notice {notice['notice_key']!r} is invalid")
            notices_by_key[notice["notice_key"]] = notice

        revision_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(source_revisions)")
        }
        section_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(licensed_sections)")
        }
        revision_printing = (
            "printing_revision" if "printing_revision" in revision_columns
            else "NULL AS printing_revision"
        )
        section_printing = (
            "printing_revision" if "printing_revision" in section_columns
            else "NULL AS printing_revision"
        )
        revisions_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for row in conn.execute(
            f"""SELECT product_code, content_fingerprint, license, era,
                       parser_version, source_schema_version, {revision_printing}
                FROM source_revisions ORDER BY product_code, content_fingerprint"""
        ):
            product = str(row["product_code"])
            if product in excluded:
                continue
            revision: dict[str, Any] = {
                "product_code": product,
                "content_fingerprint": str(row["content_fingerprint"]),
                "license": str(row["license"]),
                "era": str(row["era"]),
                "parser_version": str(row["parser_version"]),
                "source_schema_version": (
                    str(row["source_schema_version"])
                    if row["source_schema_version"] is not None
                    else None
                ),
            }
            if "printing_revision" in revision_columns:
                revision["printing_revision"] = str(row["printing_revision"])
            _validate_public_scalar(revision["parser_version"], field="parser version")
            if "printing_revision" in revision:
                _validate_public_scalar(revision["printing_revision"], field="printing revision")
            if revision["source_schema_version"] is not None:
                _validate_public_scalar(
                    revision["source_schema_version"], field="source schema version"
                )
            if (
                _PRODUCT_CODE.fullmatch(product) is None
                or _SHA256.fullmatch(revision["content_fingerprint"]) is None
                or revision["license"] not in _LICENSES
                or revision["era"] not in _ERAS
                or not revision["parser_version"]
            ):
                raise ValueError(f"licensed-core revision for {product!r} is invalid")
            revisions_by_key[(product, revision["content_fingerprint"])] = revision

        chunks: list[dict[str, Any]] = []
        policy_by_revision: dict[tuple[str, str], set[str]] = {}
        query = f"""SELECT public_id, product_code, content_fingerprint,
                           source_section_id, source_section_hash, page_start, page_end,
                           printed_page, heading, text, content_hash, license, era,
                           extraction_method, policy_version, parser_version,
                           {section_printing}, notice_key
                    FROM licensed_sections ORDER BY public_id"""
        for row in conn.execute(query):
            product = str(row["product_code"])
            if product in excluded:
                continue
            fingerprint = str(row["content_fingerprint"])
            revision_key = (product, fingerprint)
            revision = revisions_by_key.get(revision_key)
            if revision is None:
                raise ValueError(f"licensed-core section has no source revision: {product}")
            public_id = _validate_public_scalar(row["public_id"], field="public ID")
            text = str(row["text"])
            content_hash = str(row["content_hash"])
            license_name = str(row["license"])
            era = str(row["era"])
            parser_version = str(row["parser_version"])
            printing_revision = (
                str(row["printing_revision"])
                if "printing_revision" in section_columns else None
            )
            policy_version = _validate_public_scalar(
                row["policy_version"], field="policy version"
            )
            notice_key = _validate_public_scalar(row["notice_key"], field="notice key")
            source_section_id = str(row["source_section_id"])
            source_section_hash = str(row["source_section_hash"])
            page_start = row["page_start"]
            page_end = row["page_end"]
            notice = notices_by_key.get(notice_key)
            if not public_id.startswith("licensed:"):
                raise ValueError(f"licensed-core public ID has an invalid namespace: {public_id}")
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != content_hash:
                raise ValueError(f"licensed-core content hash mismatch: {public_id}")
            _validate_public_text(text, field="section text")
            if (
                not text.strip()
                or license_name != revision["license"]
                or era != revision["era"]
                or parser_version != revision["parser_version"]
                or (
                    printing_revision is not None
                    and printing_revision != revision.get("printing_revision")
                )
                or policy_version != LICENSED_CORE_POLICY_VERSION
                or notice is None
                or notice["license"] != license_name
            ):
                raise ValueError(f"licensed-core provenance mismatch: {public_id}")
            source_match = _SOURCE_SECTION_ID.fullmatch(source_section_id)
            if (
                source_match is None
                or source_match.group("product") != product.lower()
                or _SHA256.fullmatch(source_section_hash) is None
            ):
                raise ValueError(
                    f"licensed-core source provenance is invalid: {public_id}"
                )
            if (
                isinstance(page_start, bool)
                or not isinstance(page_start, int)
                or page_start < 1
                or isinstance(page_end, bool)
                or not isinstance(page_end, int)
                or page_end < page_start
                or int(source_match.group("page")) != page_start
            ):
                raise ValueError(
                    f"licensed-core page provenance is invalid: {public_id}"
                )
            policy_by_revision.setdefault(revision_key, set()).add(policy_version)
            source_id = f"licensed:{product}:{fingerprint[:16]}"
            heading = str(row["heading"])
            _validate_public_scalar(heading, field="heading")
            extraction_method = (
                _validate_public_scalar(
                    row["extraction_method"], field="extraction method"
                )
                if row["extraction_method"] is not None
                else None
            )
            printed_page = (
                _validate_public_scalar(row["printed_page"], field="printed page")
                if row["printed_page"] is not None
                else None
            )
            chunks.append(
                {
                    "id": public_id,
                    "name": heading,
                    "type": "rulebook_section",
                    "pack": f"licensed-core-{product.lower()}",
                    "slug": _slug(heading),
                    "level": None,
                    "traits": [],
                    "text": text,
                    "raw_rules_count": 0,
                    "source_hash": content_hash,
                    "license": license_name,
                    "remaster": True if era == "remaster" else (False if era == "legacy" else None),
                    "refs": [],
                    "origin": LICENSED_CORE_ORIGIN,
                    "source_id": source_id,
                    "source": {
                        "source_id": source_id,
                        "source": LICENSED_CORE_ORIGIN,
                        "product": product,
                        "revision": fingerprint,
                        "parser": parser_version,
                        "license": license_name,
                        "era": era,
                        "provenance": {
                            "content_fingerprint": fingerprint,
                            "public_schema_version": schema_version,
                            **(
                                {"printing_revision": printing_revision}
                                if printing_revision is not None else {}
                            ),
                        },
                    },
                    "source_page_start": page_start,
                    "source_page_end": page_end,
                    "printed_page": printed_page,
                    "section_hash": content_hash,
                    "licensed_provenance": {
                        "product_code": product,
                        "content_fingerprint": fingerprint,
                        "source_section_id": source_section_id,
                        "source_section_hash": source_section_hash,
                        "content_hash": content_hash,
                        "license": license_name,
                        "era": era,
                        "extraction_method": extraction_method,
                        "policy_version": policy_version,
                        "parser_version": parser_version,
                        **(
                            {"printing_revision": printing_revision}
                            if printing_revision is not None else {}
                        ),
                        "source_schema_version": revision["source_schema_version"],
                        "notice_key": notice_key,
                    },
                    "licensed_notice": dict(notice),
                }
            )
    finally:
        conn.close()

    source_revisions: list[dict[str, Any]] = []
    for key, revision in sorted(revisions_by_key.items()):
        if key not in policy_by_revision:
            continue
        source_revisions.append(
            {**revision, "policy_versions": sorted(policy_by_revision[key])}
        )
    notices = tuple(notices_by_key[key] for key in sorted(notices_by_key))
    return LicensedCoreBundle(tuple(chunks), tuple(source_revisions), notices, schema_version)


def load_licensed_core(
    path: Path | str | None = None,
    *,
    exclude_products: set[str] | frozenset[str] = frozenset(),
) -> LicensedCoreBundle:
    """Load a supplied or package-bundled static corpus.

    A missing bundled resource is treated as an empty development projection;
    strict release auditing separately requires the licensed-core marker once
    the resource is populated. An explicitly supplied path must exist.
    """
    excluded = frozenset(exclude_products)
    if path is not None:
        explicit = Path(path).expanduser().resolve()
        if not explicit.is_file():
            raise FileNotFoundError(f"licensed-core database not found: {explicit}")
        return _read_bundle(explicit, excluded)
    with _bundled_path() as bundled:
        if bundled is None:
            return LicensedCoreBundle((), (), ())
        return _read_bundle(bundled, excluded)


def licensed_core_digest(bundle: LicensedCoreBundle) -> str:
    """Return a deterministic digest of a validated projection's public contract."""
    sections = [
        {
            "id": chunk["id"],
            "heading": chunk["name"],
            "page_start": chunk["source_page_start"],
            "page_end": chunk["source_page_end"],
            "printed_page": chunk["printed_page"],
            "content_hash": chunk["licensed_provenance"]["content_hash"],
            "provenance": chunk["licensed_provenance"],
        }
        for chunk in bundle.chunks
    ]
    return licensed_core_contract_digest(
        schema_version=bundle.schema_version,
        source_revisions=list(bundle.source_revisions),
        notices=[
            {
                "notice_key": notice["notice_key"],
                "license": notice["license"],
                "content_hash": hashlib.sha256(notice["text"].encode("utf-8")).hexdigest(),
            }
            for notice in bundle.notices
        ],
        sections=sections,
    )


def licensed_core_contract_digest(
    *,
    schema_version: int,
    source_revisions: list[dict[str, Any]],
    notices: list[dict[str, Any]],
    sections: list[dict[str, Any]],
) -> str:
    """Hash the complete model-independent contract for a public projection."""
    payload = {
        "schema_version": schema_version,
        "source_revisions": source_revisions,
        "notices": notices,
        "sections": sections,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "LICENSED_CORE_ORIGIN",
    "LICENSED_CORE_RESOURCE",
    "LICENSED_CORE_SCHEMA_VERSION",
    "LicensedCoreBundle",
    "licensed_core_contract_digest",
    "licensed_core_digest",
    "load_licensed_core",
]
