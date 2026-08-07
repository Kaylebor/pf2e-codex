"""Fail-closed distribution audit for pre-built database artifacts."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DistributionAudit:
    """Distribution-relevant facts read from one SQLite database."""

    scope: str | None
    corpus_chunks: int
    non_foundry_chunks: int
    unknown_origin_chunks: int
    explicit_marker: bool
    pf2e_release: str | None
    embedding_model: str | None


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

        columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
        corpus_chunks = 0
        non_foundry_chunks = 0
        unknown_origin_chunks = 0
        if "origin" in columns:
            corpus_chunks = int(
                conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE origin = 'corpus'"
                ).fetchone()[0]
            )
            non_foundry_chunks = int(
                conn.execute(
                    "SELECT COUNT(*) FROM chunks "
                    "WHERE origin IS NULL OR origin <> 'foundry'"
                ).fetchone()[0]
            )
            unknown_origin_chunks = int(
                conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE origin IS NULL "
                    "OR origin NOT IN ('foundry', 'corpus')"
                ).fetchone()[0]
            )
    finally:
        conn.close()

    return DistributionAudit(
        scope=scope,
        corpus_chunks=corpus_chunks,
        non_foundry_chunks=non_foundry_chunks,
        unknown_origin_chunks=unknown_origin_chunks,
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
                "clean database slot contains non-Foundry or unowned rows; move it "
                "to the local slot or rebuild the clean database"
            )
        if audit.scope not in (None, "redistributable"):
            raise RuntimeError(f"database scope marker is invalid: {audit.scope!r}")
    elif expected_scope == "local":
        if audit.unknown_origin_chunks:
            raise RuntimeError(
                "local database slot contains rows with unknown ownership"
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
    if require_explicit_marker and scope != "redistributable":
        raise RuntimeError(
            "distribution audit failed: explicit redistributable seed marker is missing"
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
