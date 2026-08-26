"""Fail-closed distribution audit for pre-built database artifacts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DistributionAudit:
    """Distribution-relevant facts read from one SQLite database."""

    scope: str | None
    corpus_chunks: int
    licensed_core_chunks: int
    non_foundry_chunks: int
    unknown_origin_chunks: int
    unapproved_foundry_chunks: int
    foundry_scope_verified: bool
    foundry_scope_marker: str | None
    licensed_core_issues: tuple[str, ...]
    licensed_core_digest: str | None
    explicit_marker: bool
    pf2e_release: str | None
    embedding_model: str | None


def _model_licensed_core_digest(conn: sqlite3.Connection) -> str:
    """Recompute the embedded projection contract instead of trusting row hashes."""
    from .licensed_core import (
        LICENSED_CORE_SCHEMA_VERSION,
        licensed_core_contract_digest,
    )

    revisions: list[dict[str, object]] = []
    for row in conn.execute(
        """SELECT product_code, content_fingerprint, license, era, parser_version,
                   source_schema_version, policy_versions, printing_revision
            FROM licensed_revisions ORDER BY product_code, content_fingerprint"""
    ):
        policies = json.loads(row[6])
        if not isinstance(policies, list):
            raise ValueError("licensed revision policies are not a list")
        revisions.append(
            {
                "product_code": str(row[0]),
                "content_fingerprint": str(row[1]),
                "license": str(row[2]),
                "era": str(row[3]),
                "parser_version": str(row[4]),
                "source_schema_version": str(row[5]) if row[5] is not None else None,
                "printing_revision": str(row[7]),
                "policy_versions": sorted(str(value) for value in policies),
            }
        )
    notices = [
        {"notice_key": str(row[0]), "license": str(row[1]), "content_hash": str(row[2])}
        for row in conn.execute(
            "SELECT notice_key, license, content_hash FROM license_notices ORDER BY notice_key"
        )
    ]
    sections: list[dict[str, object]] = []
    for row in conn.execute(
        """SELECT ls.public_id, ls.heading, ls.page_start, ls.page_end,
                  ls.printed_page, ls.product_code, ls.content_fingerprint,
                  ls.source_section_id, ls.source_section_hash, ls.content_hash,
                  ls.license, ls.era, ls.extraction_method, ls.policy_version,
                  ls.parser_version, lr.source_schema_version, ls.notice_key,
                  ls.printing_revision
           FROM licensed_sections AS ls JOIN licensed_revisions AS lr
             ON lr.product_code=ls.product_code
            AND lr.content_fingerprint=ls.content_fingerprint
           ORDER BY ls.public_id"""
    ):
        provenance = {
            "product_code": str(row[5]),
            "content_fingerprint": str(row[6]),
            "source_section_id": str(row[7]),
            "source_section_hash": str(row[8]),
            "content_hash": str(row[9]),
            "license": str(row[10]),
            "era": str(row[11]),
            "extraction_method": str(row[12]) if row[12] is not None else None,
            "policy_version": str(row[13]),
            "parser_version": str(row[14]),
            "source_schema_version": str(row[15]) if row[15] is not None else None,
            "notice_key": str(row[16]),
        }
        provenance["printing_revision"] = str(row[17])
        provenance["sources"] = [
            {
                "product_code": str(source[0]),
                "content_fingerprint": str(source[1]),
                "source_section_id": str(source[2]),
                "source_section_hash": str(source[3]),
                "page_start": source[4],
                "page_end": source[5],
                "printed_page": source[6],
                "parser_version": str(source[7]),
                "printing_revision": str(source[8]),
                "source_schema_version": (
                    str(source[9]) if source[9] is not None else None
                ),
                "notice_key": str(source[10]),
            }
            for source in conn.execute(
                """SELECT lss.product_code, lss.content_fingerprint,
                          lss.source_section_id, lss.source_section_hash,
                          lss.page_start, lss.page_end, lss.printed_page,
                          lss.parser_version, lss.printing_revision,
                          lr2.source_schema_version, lss.notice_key
                     FROM licensed_section_sources AS lss
                     JOIN licensed_revisions AS lr2
                       ON lr2.product_code=lss.product_code
                      AND lr2.content_fingerprint=lss.content_fingerprint
                    WHERE lss.public_id=? ORDER BY lss.source_ordinal""",
                (str(row[0]),),
            )
        ]
        sections.append(
            {
                "id": str(row[0]),
                "heading": str(row[1]),
                "page_start": row[2],
                "page_end": row[3],
                "printed_page": row[4],
                "content_hash": str(row[9]),
                "provenance": provenance,
            }
        )
    required_foundry_rows = [
        {
            "foundry_id": str(row[0]),
            "source_hash": str(row[1]),
            "normalized_hash": str(row[2]),
            "publication_title": str(row[3]),
            "license": str(row[4]),
            "era": str(row[5]),
        }
        for row in conn.execute(
            """SELECT foundry_id, source_hash, normalized_hash, publication_title,
                      license, era
                 FROM required_foundry_rows ORDER BY foundry_id"""
        )
    ]
    scope_row = conn.execute(
        "SELECT value FROM _meta WHERE key='licensed_core_covered_products'"
    ).fetchone()
    if scope_row is None:
        raise ValueError("licensed-core product scope is missing")
    covered_products = json.loads(str(scope_row[0]))
    if not isinstance(covered_products, list) or any(
        not isinstance(value, str) for value in covered_products
    ):
        raise ValueError("licensed-core product scope is invalid")
    return licensed_core_contract_digest(
        schema_version=LICENSED_CORE_SCHEMA_VERSION,
        source_revisions=revisions,
        notices=notices,
        sections=sections,
        required_foundry_rows=required_foundry_rows,
        covered_products=covered_products,
    )


def inspect_database_scope(db_path: Path | str) -> DistributionAudit:
    """Read distribution ownership facts without deciding publication policy."""
    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if "chunks" not in tables:
            raise RuntimeError("distribution audit failed: chunks table is missing")

        scope: str | None = None
        pf2e_release: str | None = None
        embedding_model: str | None = None
        licensed_core_digest: str | None = None
        foundry_scope_marker: str | None = None
        if "_meta" in tables:
            row = conn.execute(
                "SELECT value FROM _meta WHERE key = 'distribution_scope'"
            ).fetchone()
            scope = str(row[0]) if row else None
            row = conn.execute(
                "SELECT value FROM _meta WHERE key = 'pf2e_release'"
            ).fetchone()
            pf2e_release = str(row[0]) if row else None
            row = conn.execute(
                "SELECT value FROM _meta WHERE key = 'embedding_model'"
            ).fetchone()
            embedding_model = str(row[0]) if row else None
            row = conn.execute(
                "SELECT value FROM _meta WHERE key = 'licensed_core_digest'"
            ).fetchone()
            licensed_core_digest = str(row[0]) if row else None
            row = conn.execute(
                "SELECT value FROM _meta WHERE key = 'foundry_scope'"
            ).fetchone()
            foundry_scope_marker = str(row[0]) if row else None

        columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
        corpus_chunks = 0
        licensed_core_chunks = 0
        non_foundry_chunks = 0
        unknown_origin_chunks = 0
        unapproved_foundry_chunks = 0
        foundry_scope_verified = False
        licensed_core_issues: list[str] = []
        if "origin" in columns:
            corpus_chunks = int(
                conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE origin = 'corpus'"
                ).fetchone()[0]
            )
            non_foundry_chunks = int(
                conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE origin IS NULL "
                    "OR origin NOT IN ('foundry', 'licensed-core')"
                ).fetchone()[0]
            )
            licensed_core_chunks = int(
                conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE origin = 'licensed-core'"
                ).fetchone()[0]
            )
            unknown_origin_chunks = int(
                conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE origin IS NULL "
                    "OR origin NOT IN ('foundry', 'licensed-core', 'corpus')"
                ).fetchone()[0]
            )
            if {"license", "publication_title"}.issubset(columns):
                from .foundry_scope import (
                    REDISTRIBUTABLE_FOUNDRY_PUBLICATIONS,
                    REDISTRIBUTABLE_LICENSES,
                )

                titles = sorted(REDISTRIBUTABLE_FOUNDRY_PUBLICATIONS)
                licenses = sorted(REDISTRIBUTABLE_LICENSES)
                title_slots = ",".join("?" for _ in titles)
                license_slots = ",".join("?" for _ in licenses)
                unapproved_foundry_chunks = int(
                    conn.execute(
                        f"""SELECT COUNT(*) FROM chunks WHERE origin='foundry' AND (
                            publication_title IS NULL
                            OR publication_title NOT IN ({title_slots})
                            OR license IS NULL OR license NOT IN ({license_slots})
                        )""",
                        (*titles, *licenses),
                    ).fetchone()[0]
                )
                foundry_scope_verified = True
        if licensed_core_chunks:
            required = {
                "licensed_sections", "licensed_section_sources", "licensed_revisions",
                "license_notices", "required_foundry_rows", "sources",
            }
            missing = required - tables
            if missing:
                licensed_core_issues.append(
                    "missing tables: " + ", ".join(sorted(missing))
                )
            else:
                section_count = int(
                    conn.execute("SELECT COUNT(*) FROM licensed_sections").fetchone()[0]
                )
                if section_count != licensed_core_chunks:
                    licensed_core_issues.append("chunk/provenance count mismatch")
                missing_source_provenance = int(
                    conn.execute(
                        """SELECT COUNT(*) FROM licensed_sections AS ls
                           WHERE NOT EXISTS (
                               SELECT 1 FROM licensed_section_sources AS lss
                               WHERE lss.public_id=ls.public_id
                           )"""
                    ).fetchone()[0]
                )
                if missing_source_provenance:
                    licensed_core_issues.append("missing multi-source provenance")
                bad_links = int(
                    conn.execute(
                        """SELECT COUNT(*) FROM chunks AS c
                           LEFT JOIN licensed_sections AS ls ON ls.public_id=c.id
                           LEFT JOIN licensed_revisions AS lr
                             ON lr.product_code=ls.product_code
                            AND lr.content_fingerprint=ls.content_fingerprint
                           LEFT JOIN license_notices AS n ON n.notice_key=ls.notice_key
                           LEFT JOIN sources AS s ON s.source_id=c.source_id
                           WHERE c.origin='licensed-core' AND (
                               ls.public_id IS NULL OR lr.product_code IS NULL
                               OR n.notice_key IS NULL OR ls.license NOT IN ('OGL', 'ORC')
                               OR ls.license <> lr.license OR ls.license <> n.license
                               OR s.source_id IS NULL OR s.source <> 'licensed-core'
                               OR ls.policy_version = '' OR ls.parser_version = ''
                               OR ls.content_hash = ''
                           )"""
                    ).fetchone()[0]
                )
                if bad_links:
                    licensed_core_issues.append("invalid manifest or notice links")
                for row in conn.execute(
                    """SELECT c.id, c.text, ls.content_hash
                       FROM chunks AS c JOIN licensed_sections AS ls ON ls.public_id=c.id
                       WHERE c.origin='licensed-core'"""
                ):
                    if hashlib.sha256(str(row[1]).encode("utf-8")).hexdigest() != row[2]:
                        licensed_core_issues.append(f"content hash mismatch: {row[0]}")
                        break
                for row in conn.execute(
                    "SELECT notice_key, text, content_hash FROM license_notices"
                ):
                    if hashlib.sha256(str(row[1]).encode("utf-8")).hexdigest() != row[2]:
                        licensed_core_issues.append(f"notice hash mismatch: {row[0]}")
                        break
                for row in conn.execute(
                    """SELECT source_id, source, product, revision, provenance
                       FROM sources WHERE source='licensed-core'"""
                ):
                    try:
                        provenance = json.loads(row[4]) if row[4] else {}
                    except (TypeError, json.JSONDecodeError):
                        licensed_core_issues.append(f"invalid source provenance: {row[0]}")
                        break
                    if (
                        row[1] != "licensed-core"
                        or not str(row[2]).startswith("PZO")
                        or len(str(row[3])) != 64
                        or set(provenance) - {
                            "content_fingerprint", "public_schema_version",
                            "printing_revision",
                        }
                        or provenance.get("content_fingerprint") != row[3]
                    ):
                        licensed_core_issues.append(f"unsafe source provenance: {row[0]}")
                        break
                try:
                    computed_digest = _model_licensed_core_digest(conn)
                except (json.JSONDecodeError, sqlite3.DatabaseError, ValueError) as exc:
                    licensed_core_issues.append(f"invalid projection contract: {exc}")
                else:
                    if licensed_core_digest != computed_digest:
                        licensed_core_issues.append("projection digest mismatch")
    finally:
        conn.close()

    return DistributionAudit(
        scope=scope,
        corpus_chunks=corpus_chunks,
        licensed_core_chunks=licensed_core_chunks,
        non_foundry_chunks=non_foundry_chunks,
        unknown_origin_chunks=unknown_origin_chunks,
        unapproved_foundry_chunks=unapproved_foundry_chunks,
        foundry_scope_verified=foundry_scope_verified,
        foundry_scope_marker=foundry_scope_marker,
        licensed_core_issues=tuple(licensed_core_issues),
        licensed_core_digest=licensed_core_digest,
        explicit_marker=scope == "redistributable",
        pf2e_release=pf2e_release,
        embedding_model=embedding_model,
    )


def audit_database_slot(db_path: Path | str, expected_scope: str) -> DistributionAudit:
    """Fail when database ownership does not match its physical slot."""
    audit = inspect_database_scope(db_path)
    if expected_scope == "clean":
        if audit.scope == "local-full" or audit.non_foundry_chunks:
            raise RuntimeError(
                "clean database slot contains private or unowned rows; move it "
                "to the local slot or rebuild the clean database"
            )
        if audit.licensed_core_issues:
            raise RuntimeError(
                "clean database slot has invalid licensed-core metadata: "
                + "; ".join(audit.licensed_core_issues)
            )
        if audit.foundry_scope_marker is not None and (
            audit.foundry_scope_marker != "core-publications-v1"
            or audit.scope != "redistributable"
            or not audit.foundry_scope_verified
            or audit.unapproved_foundry_chunks
        ):
            raise RuntimeError(
                "clean database slot contains Foundry rows outside the approved core publications"
            )
        if audit.scope not in (None, "redistributable"):
            raise RuntimeError(f"database scope marker is invalid: {audit.scope!r}")
    elif expected_scope == "local":
        if audit.unknown_origin_chunks:
            raise RuntimeError(
                "local database slot contains rows with unknown ownership"
            )
        if audit.licensed_core_issues:
            raise RuntimeError(
                "local database slot has invalid licensed-core metadata: "
                + "; ".join(audit.licensed_core_issues)
            )
        # Legacy combined DBs predate the marker but are recognizable by their
        # owned corpus rows. New empty private snapshots retain local-full.
        if audit.scope != "local-full" and not audit.corpus_chunks:
            raise RuntimeError(
                "local database slot does not contain a private local-full database"
            )
        if audit.scope not in (None, "local-full"):
            raise RuntimeError(f"database scope marker is invalid: {audit.scope!r}")
    else:
        raise ValueError(f"unknown database slot scope: {expected_scope!r}")
    return audit


def audit_redistributable_database(
    db_path: Path | str,
    *,
    require_explicit_marker: bool = False,
    expected_release: str | None = None,
    expected_model: str | None = None,
    trusted_projection: Path | str | None = None,
) -> DistributionAudit:
    """Reject DBs containing private corpus rows or a private scope marker.

    Legacy Foundry-only databases predate the marker and remain pull-compatible
    when they have no corpus ownership column. Release publication uses the
    strict marker requirement so newly uploaded artifacts prove which seed
    policy produced them. Required source-license and trademark notices remain
    a release-process responsibility rather than fabricated database metadata.
    """
    audit = inspect_database_scope(db_path)
    scope = audit.scope
    non_foundry_chunks = audit.non_foundry_chunks

    if scope == "local-full" or non_foundry_chunks:
        raise RuntimeError(
            "database is private, unowned, or local-full and must not be published "
            "or auto-pulled"
        )
    if scope not in (None, "redistributable"):
        raise RuntimeError(f"distribution audit failed: unknown scope {scope!r}")
    if audit.licensed_core_issues:
        raise RuntimeError(
            "distribution audit failed: invalid licensed-core metadata: "
            + "; ".join(audit.licensed_core_issues)
        )
    if require_explicit_marker and scope != "redistributable":
        raise RuntimeError(
            "distribution audit failed: explicit redistributable seed marker is missing"
        )
    if require_explicit_marker and (
        audit.foundry_scope_marker != "core-publications-v1"
        or not audit.foundry_scope_verified
        or audit.unapproved_foundry_chunks
    ):
        raise RuntimeError(
            "distribution audit failed: Foundry rows are not limited to the approved "
            "core publications and licenses"
        )
    if require_explicit_marker and not audit.licensed_core_chunks:
        raise RuntimeError(
            "distribution audit failed: reviewed licensed-core content is missing"
        )
    if require_explicit_marker:
        from .licensed_core import licensed_core_digest, load_licensed_core

        trusted = load_licensed_core(trusted_projection)
        if not trusted.chunks:
            raise RuntimeError(
                "distribution audit failed: trusted licensed-core projection is missing"
            )
        trusted_digest = licensed_core_digest(trusted)
        if audit.licensed_core_digest != trusted_digest:
            raise RuntimeError(
                "distribution audit failed: model database does not match the trusted "
                "licensed-core projection"
            )
    if expected_release is not None and audit.pf2e_release != expected_release:
        raise RuntimeError(
            "distribution audit failed: database release "
            f"{audit.pf2e_release!r} does not match {expected_release!r}"
        )
    if expected_model is not None and audit.embedding_model != expected_model:
        raise RuntimeError(
            "distribution audit failed: embedding model "
            f"{audit.embedding_model!r} does not match {expected_model!r}"
        )
    return audit


def audit_downloadable_database(
    db_path: Path | str,
    *,
    expected_release: str | None = None,
    expected_model: str | None = None,
    trusted_projection: Path | str | None = None,
) -> DistributionAudit:
    """Audit a downloaded clean-slot candidate before it can be activated.

    New release artifacts declare ``distribution_scope=redistributable`` and
    must therefore pass the complete release audit, including an exact match
    against the bundled reviewed licensed-core projection.  Markerless
    Foundry-only artifacts are accepted solely for backwards compatibility
    with releases made before distribution scope metadata existed.
    """
    audit = inspect_database_scope(db_path)
    if audit.scope is None:
        if audit.licensed_core_chunks:
            raise RuntimeError(
                "distribution audit failed: markerless legacy downloads must be Foundry-only"
            )
        return audit_redistributable_database(
            db_path,
            expected_release=expected_release,
            expected_model=expected_model,
            trusted_projection=trusted_projection,
        )
    return audit_redistributable_database(
        db_path,
        require_explicit_marker=True,
        expected_release=expected_release,
        expected_model=expected_model,
        trusted_projection=trusted_projection,
    )
