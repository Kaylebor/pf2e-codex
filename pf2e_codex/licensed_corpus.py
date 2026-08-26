"""Private review workspace and fail-closed builder for licensed core text.

This module deliberately has no dependency on the indexing pipeline.  It turns a
``local-full`` corpus database into a *private* review workspace, then produces
a separate, compact database from independently approved candidate text.  The
working database is allowed to contain purchased-book text; the built database
is not.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from .corpus import (
    PAIZO_NATIVE_PARSER_V4,
    PAIZO_NATIVE_PARSER_V5,
    PRODUCT_CATALOG,
    TrustedParseBundle,
    load_and_parse_verified_pdf,
    repair_trusted_bundle,
)
from .licensed_coverage import (
    NORMALIZER_VERSION,
    FoundryMatcher,
    duplicate_identity,
    load_clean_foundry,
    normalized_hash,
)
from .licensed_policy import LICENSED_CORE_POLICY_VERSION, licensed_policy_digest

REVIEW_SCHEMA_VERSION = 20
PUBLIC_SCHEMA_VERSION = 3
REVIEW_SCOPE_VERSION = "semantic-products-v1"
POLICY_DECISIONS = {
    "PUBLIC_AS_IS",
    "MIXED_NEEDS_EXTRACTION",
    "EXCLUDE",
    "UNCERTAIN",
}
REVIEW_VERDICTS = {"APPROVE", "REJECT", "REVISE"}
CLAIM_MODES = {"ordinary", "rework"}
SCREENING_DECISIONS = {"ADD", "REJECT", "DEFER"}
SCREENING_CLAIM_MODES = {"ordinary", "escalation"}
SCREENING_DEFER_REASONS = {
    "layout",
    "scope",
    "complex-rule",
    "insufficient-context",
}
SCREENING_REJECT_REASONS = {"no-mechanics", "duplicate", "setting-prose"}
SCREENING_REOPEN_REASONS = {"parser-quality", "scope-correction", "maintainer-review"}
REVIEW_SCOPE_HOLD_REASONS = {"legacy-study", "maintainer-hold"}
PUBLIC_DECISIONS = {"PUBLIC_AS_IS", "MIXED_NEEDS_EXTRACTION"}
_REVIEW_VERDICT_ALIASES = {
    "APPROVE_PUBLIC": "APPROVE",
    "CONFIRM_EXCLUSION": "REJECT",
}
_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_WINDOWS_PATH_RE = re.compile(r"(?i)(?:^|\s)[A-Z]:[\\/]")
_PUBLIC_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._-]{0,255}$")
_STRUCTURAL_SOURCE_SECTION_ID_RE = re.compile(
    r"^pzo[0-9]+:[a-z0-9]+(?:-[a-z0-9]+)*:p([1-9][0-9]*):h[0-9a-f]{16}:i[0-9]+$"
)
_PRIVATE_TEXT_MARKERS = (
    ".local-corpus",
    "source_sha256",
    "source_path",
    "file://",
    "/home/",
    "/users/",
    "\\users\\",
)
_TRUSTED_IGNORED_ANCHOR_REASONS = {
    "printed-page-number-v1",
    "repeated-margin-furniture-v1",
    "watermark-email-span-v1",
    "watermark-identity-row-v1",
}

# A bundle is deliberately not a serializable public API.  The direct-PDF
# bridge registers its exact in-memory object here and consumes it once during
# staging.  This prevents a cached JSON export, or a caller-constructed
# dataclass lookalike, from certifying a complete parser run.
_DIRECT_BUNDLE_CAPABILITIES: dict[int, tuple[TrustedParseBundle, object]] = {}
_TRUSTED_MANIFEST_VERSION = "trusted-native-pdf-v1"
_TRUSTED_RUN_ORIGIN = "trusted-direct-pdf-v1"
_LEGACY_RUN_ORIGIN = "legacy-untrusted"


_RUNNER_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS review_product_scope (
    product_code TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL CHECK(enabled IN (0, 1)),
    reason TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS runner_sessions (
    queue_name TEXT NOT NULL,
    slot INTEGER NOT NULL,
    role TEXT NOT NULL,
    model TEXT NOT NULL,
    cli_version TEXT NOT NULL,
    thread_id TEXT,
    prompt_digest TEXT NOT NULL,
    schema_digest TEXT NOT NULL,
    policy_digest TEXT NOT NULL,
    completed_batches INTEGER NOT NULL DEFAULT 0,
    submitted_evidence_bytes INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (queue_name, slot)
);
CREATE TABLE IF NOT EXISTS runner_attempts (
    attempt_id TEXT PRIMARY KEY,
    queue_name TEXT NOT NULL,
    batch_key TEXT NOT NULL,
    slot INTEGER NOT NULL,
    model TEXT NOT NULL,
    cli_version TEXT NOT NULL,
    thread_id TEXT,
    attempt INTEGER NOT NULL,
    input_digest TEXT NOT NULL,
    result_digest TEXT,
    status TEXT NOT NULL CHECK(status IN
        ('running', 'transport-failure', 'schema-failure', 'accepted')),
    exit_code INTEGER,
    usage_json TEXT,
    started_at INTEGER NOT NULL,
    completed_at INTEGER,
    error_kind TEXT,
    UNIQUE(queue_name, batch_key, attempt)
);
CREATE INDEX IF NOT EXISTS runner_attempts_by_queue
    ON runner_attempts(queue_name, status, started_at);
CREATE TABLE IF NOT EXISTS runner_maintenance (
    item_id TEXT PRIMARY KEY,
    queue_name TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    resolved_at INTEGER,
    resolution TEXT,
    UNIQUE(queue_name, subject_id, reason)
);
CREATE INDEX IF NOT EXISTS runner_maintenance_open
    ON runner_maintenance(resolved_at, queue_name);
CREATE TABLE IF NOT EXISTS runner_classifications (
    section_key TEXT PRIMARY KEY REFERENCES source_sections(section_key),
    parser_run_id TEXT NOT NULL REFERENCES parser_runs(parser_run_id),
    decision TEXT NOT NULL CHECK(decision IN
        ('PUBLIC_AS_IS', 'MIXED_NEEDS_EXTRACTION', 'EXCLUDE', 'UNCERTAIN')),
    reason_tags TEXT NOT NULL,
    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
    worker TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    decided_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS runner_classifications_by_decision
    ON runner_classifications(parser_run_id, decision);
CREATE TABLE IF NOT EXISTS runner_screen_escalations (
    section_key TEXT PRIMARY KEY REFERENCES source_sections(section_key),
    luna_worker TEXT NOT NULL,
    attempted_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS runner_screen_rejections (
    section_key TEXT PRIMARY KEY REFERENCES source_sections(section_key),
    reason TEXT NOT NULL CHECK(reason IN ('no-mechanics', 'duplicate', 'setting-prose')),
    worker TEXT NOT NULL,
    decided_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS aon_cache (
    query_digest TEXT PRIMARY KEY,
    normalized_query TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('match', 'no-match', 'inconclusive')),
    results_json TEXT NOT NULL,
    checked_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS stitch_candidates (
    candidate_id TEXT PRIMARY KEY,
    parser_run_id TEXT NOT NULL REFERENCES parser_runs(parser_run_id),
    product_code TEXT NOT NULL,
    section_keys TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    UNIQUE(parser_run_id, section_keys)
);
CREATE TABLE IF NOT EXISTS stitch_votes (
    candidate_id TEXT NOT NULL REFERENCES stitch_candidates(candidate_id),
    role TEXT NOT NULL CHECK(role IN ('selector', 'confirmer')),
    worker TEXT NOT NULL,
    decision TEXT NOT NULL CHECK(decision IN ('merge', 'no-merge')),
    reason TEXT NOT NULL,
    decided_at INTEGER NOT NULL,
    PRIMARY KEY(candidate_id, role)
);
CREATE TABLE IF NOT EXISTS stitch_claims (
    candidate_id TEXT NOT NULL REFERENCES stitch_candidates(candidate_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('selector', 'confirmer')),
    claimant TEXT NOT NULL,
    claimed_at INTEGER NOT NULL,
    lease_expires_at INTEGER NOT NULL,
    PRIMARY KEY(candidate_id, role)
);
CREATE INDEX IF NOT EXISTS stitch_claims_by_claimant
    ON stitch_claims(claimant, lease_expires_at);
CREATE TABLE IF NOT EXISTS duplicate_groups (
    group_id TEXT PRIMARY KEY,
    normalizer_version TEXT NOT NULL,
    license TEXT NOT NULL CHECK(license IN ('OGL','ORC')),
    era TEXT NOT NULL CHECK(era IN ('legacy','remaster','unknown')),
    heading_hash TEXT NOT NULL,
    body_hash TEXT NOT NULL,
    canonical_section_key TEXT NOT NULL REFERENCES source_sections(section_key),
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS duplicate_group_members (
    group_id TEXT NOT NULL REFERENCES duplicate_groups(group_id) ON DELETE CASCADE,
    section_key TEXT NOT NULL REFERENCES source_sections(section_key),
    source_ordinal INTEGER NOT NULL CHECK(source_ordinal >= 0),
    PRIMARY KEY(group_id, section_key),
    UNIQUE(section_key)
);
CREATE INDEX IF NOT EXISTS duplicate_members_by_canonical
    ON duplicate_group_members(group_id, source_ordinal);
CREATE TABLE IF NOT EXISTS foundry_snapshots (
    snapshot_digest TEXT PRIMARY KEY,
    pf2e_release TEXT NOT NULL,
    row_count INTEGER NOT NULL CHECK(row_count >= 0),
    normalizer_version TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS foundry_snapshot_rows (
    snapshot_digest TEXT NOT NULL REFERENCES foundry_snapshots(snapshot_digest) ON DELETE CASCADE,
    foundry_id TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    normalized_hash TEXT NOT NULL,
    heading_hash TEXT NOT NULL,
    publication_title TEXT NOT NULL,
    license TEXT NOT NULL,
    era TEXT NOT NULL,
    entry_type TEXT NOT NULL,
    PRIMARY KEY(snapshot_digest, foundry_id)
);
CREATE TABLE IF NOT EXISTS foundry_coverage_candidates (
    section_key TEXT NOT NULL REFERENCES source_sections(section_key),
    snapshot_digest TEXT NOT NULL REFERENCES foundry_snapshots(snapshot_digest),
    candidate_rank INTEGER NOT NULL CHECK(candidate_rank BETWEEN 0 AND 2),
    foundry_id TEXT NOT NULL,
    proof_digest TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    PRIMARY KEY(section_key, snapshot_digest, foundry_id),
    UNIQUE(section_key, snapshot_digest, candidate_rank),
    FOREIGN KEY(snapshot_digest, foundry_id)
        REFERENCES foundry_snapshot_rows(snapshot_digest, foundry_id)
);
CREATE TABLE IF NOT EXISTS foundry_coverage_confirmations (
    section_key TEXT NOT NULL REFERENCES source_sections(section_key),
    snapshot_digest TEXT NOT NULL REFERENCES foundry_snapshots(snapshot_digest),
    foundry_ids_json TEXT NOT NULL,
    proof_digest TEXT NOT NULL,
    worker TEXT NOT NULL,
    decided_at INTEGER NOT NULL,
    PRIMARY KEY(section_key, snapshot_digest)
);
CREATE INDEX IF NOT EXISTS foundry_confirmations_by_snapshot
    ON foundry_coverage_confirmations(snapshot_digest, section_key);
"""


def _validate_public_text(value: str, *, field: str) -> None:
    """Reject obvious private provenance and watermark material fail-closed."""
    lowered = value.casefold()
    if (
        "@" in value
        or _EMAIL_RE.search(value)
        or _WINDOWS_PATH_RE.search(value)
        or any(marker in lowered for marker in _PRIVATE_TEXT_MARKERS)
        or any(ord(char) < 32 and char not in "\n\t" for char in value)
    ):
        raise ValueError(f"public {field} contains private or unsafe provenance text")


def _validate_public_scalar(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"public {field} must be a non-empty string")
    _validate_public_text(value, field=field)
    return value


def _connect(path: Path | str, *, readonly: bool = False) -> sqlite3.Connection:
    path = Path(path).expanduser().resolve()
    if readonly:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA foreign_keys = ON")
    if not readonly:
        conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _parser_output_digest(records: Iterable[Mapping[str, object]]) -> str:
    """Hash parser output records in a versioned, path-free canonical form."""
    canonical = [
        {
            key: record.get(key)
            for key in (
                "source_section_id", "source_section_hash", "text_hash", "heading",
                "stable_identity", "provenance_hash", "page_start", "page_end", "printed_page", "layout_flags",
            )
        }
        for record in records
    ]
    canonical.sort(key=lambda record: (str(record["stable_identity"]), str(record["source_section_id"])))
    return _digest("parser-output-v1", _canonical_json(canonical))


def _native_word_coverage_digest(
    records: Iterable[Mapping[str, object]],
    quarantine: Iterable[Mapping[str, object]] = (),
) -> str:
    """Bind a complete manifest to independently recorded native-word coverage.

    The parser seam does not itself have native-word pages available.  A run
    therefore cannot activate unless the upstream exporter supplies per-section
    coverage records.  This is deliberately a digest, not a caller-provided
    boolean.
    """
    canonical = [
        {
            "stable_identity": record.get("stable_identity"),
            "native_word_count": record.get("native_word_count"),
            "native_word_digest": record.get("native_word_digest"),
            "native_word_anchors": sorted(
                str(anchor)
                for anchor in (record.get("native_word_anchors") or [])
            ),
        }
        for record in records
    ]
    canonical.sort(key=lambda record: str(record["stable_identity"]))
    quarantined = [
        {
            "quarantine_id": record.get("quarantine_id"),
            "reason": record.get("reason"),
            "physical_page": record.get("physical_page"),
            "native_word_count": record.get("native_word_count"),
            "native_word_digest": record.get("native_word_digest"),
            "native_word_anchors": sorted(
                str(anchor) for anchor in (record.get("native_word_anchors") or [])
            ),
        }
        for record in quarantine
    ]
    quarantined.sort(key=lambda record: str(record["quarantine_id"]))
    if not quarantined:
        return _digest("native-word-coverage-v1", _canonical_json(canonical))
    return _digest(
        "native-word-coverage-v2",
        _canonical_json({"sections": canonical, "quarantine": quarantined}),
    )


def _anchor_digest(namespace: str, anchors: Iterable[str]) -> str:
    return _digest(namespace, *sorted(anchors))


def _ignored_anchor_digest(anchors: Iterable[Mapping[str, object]]) -> str:
    canonical = sorted(
        (
            {"anchor_hash": item.get("anchor_hash"), "reason": item.get("reason")}
            for item in anchors
        ),
        key=lambda item: str(item["anchor_hash"]),
    )
    return _digest("ignored-native-word-anchors-v1", _canonical_json(canonical))


def _section_commitment(record: Mapping[str, object]) -> str:
    """Private, opaque commitment for later activation-time revalidation."""
    canonical = {
        key: record.get(key)
        for key in (
            "source_section_id", "source_section_hash", "text_hash", "heading",
            "stable_identity", "provenance_hash", "page_start", "page_end", "printed_page",
            "layout_flags", "native_word_anchors",
        )
    }
    canonical["layout_flags"] = sorted(str(flag) for flag in (canonical["layout_flags"] or []))
    canonical["native_word_anchors"] = sorted(str(anchor) for anchor in (canonical["native_word_anchors"] or []))
    return _digest(
        "trusted-section-commitment-v1",
        _canonical_json(canonical),
    )


def _candidate_commitment(
    *,
    source_section_hash: object,
    decision: object,
    candidate_text: object,
    public_heading: object,
    extraction_method: object,
    reason_tags: object,
) -> str:
    """Canonical commitment for every reviewable public-policy decision."""
    try:
        tags = json.loads(reason_tags) if isinstance(reason_tags, str) else reason_tags
    except json.JSONDecodeError as exc:
        raise ValueError("candidate reason_tags are corrupted") from exc
    if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag for tag in tags):
        raise ValueError("candidate reason_tags are invalid")
    return _digest(
        "candidate-commitment-v2",
        _canonical_json({
            "source_section_hash": source_section_hash,
            "decision": decision,
            "candidate_text": candidate_text,
            "public_heading": public_heading,
            "extraction_method": extraction_method,
            "reason_tags": sorted(set(tags)),
        }),
    )


def _validate_candidate_commitment(candidate: Mapping[str, object]) -> str:
    expected = _candidate_commitment(
        source_section_hash=candidate["source_section_hash"], decision=candidate["decision"],
        candidate_text=candidate["candidate_text"], public_heading=candidate["public_heading"],
        extraction_method=candidate["extraction_method"], reason_tags=candidate["reason_tags"],
    )
    if candidate["candidate_hash"] != expected:
        raise ValueError("candidate commitment no longer matches its reviewed fields")
    return expected


def _validate_candidate_layout(candidate: Mapping[str, object], section: Mapping[str, object]) -> None:
    try:
        flags = set(json.loads(str(section["layout_flags"] or "[]")))
        tags = set(json.loads(str(candidate["reason_tags"])))
    except json.JSONDecodeError as exc:
        raise ValueError("candidate or source layout metadata is corrupted") from exc
    complex_layout = {
        "unclassified-native-coverage", "complex-layout", "table-ambiguous", "table-cell",
        "layout-model-complex", "layout-model-table", "layout-model-unbound",
        "layout-order-conflict", "layout-region-split", "unsupported-layout",
        "unresolved-continuation", "heading-artifact", "oversize-block",
    }
    if candidate["decision"] == "PUBLIC_AS_IS" and flags.intersection(complex_layout):
        raise ValueError("PUBLIC_AS_IS candidate has unreviewed complex layout")
    if candidate["decision"] == "MIXED_NEEDS_EXTRACTION" and "layout-reviewed" not in tags:
        raise ValueError("MIXED_NEEDS_EXTRACTION candidate lacks layout-reviewed evidence")


def _validate_page_provenance(
    source_section_id: object, page_start: object, page_end: object
) -> None:
    """Require structural source IDs to name the same physical start page."""
    if (
        isinstance(page_start, bool)
        or isinstance(page_end, bool)
        or not isinstance(page_start, int)
        or not isinstance(page_end, int)
        or page_start < 1
        or page_end < page_start
    ):
        raise ValueError("source section pages must be non-bool positive integers in order")
    match = (
        _STRUCTURAL_SOURCE_SECTION_ID_RE.fullmatch(source_section_id)
        if isinstance(source_section_id, str)
        else None
    )
    if match is None or int(match.group(1)) != page_start:
        raise ValueError("source section ID must structurally match its page_start")


def _review_commitment(
    *,
    candidate_id: object,
    reviewer: object,
    verdict: object,
    policy_version: object,
    reason_tags: object,
    reviewed_at: object,
    review_lineage: object,
) -> str:
    """Canonical, versioned commitment to the fields that make a review trusted."""
    try:
        tags = json.loads(reason_tags) if isinstance(reason_tags, str) else reason_tags
    except json.JSONDecodeError as exc:
        raise ValueError("review reason_tags are corrupted") from exc
    if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag for tag in tags):
        raise ValueError("review reason_tags are invalid")
    if isinstance(reviewed_at, bool) or not isinstance(reviewed_at, int):
        raise ValueError("reviewed_at is invalid")
    lineage = _review_lineage(review_lineage)
    return _digest(
        "review-commitment-v2",
        _canonical_json(
            {
                "candidate_id": candidate_id,
                "reviewer": reviewer,
                "verdict": verdict,
                "policy_version": policy_version,
                "reason_tags": sorted(set(tags)),
                "reviewed_at": reviewed_at,
                "review_lineage": lineage,
            }
        ),
    )


def _review_lineage(value: object) -> list[str]:
    try:
        lineage = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise ValueError("review lineage is corrupted") from exc
    if (
        not isinstance(lineage, list)
        or not lineage
        or any(not isinstance(item, str) or not item for item in lineage)
        or len(set(lineage)) != len(lineage)
    ):
        raise ValueError("review lineage is invalid")
    return lineage


def _lineage_is_invalidated(conn: sqlite3.Connection, lineage: Iterable[str]) -> bool:
    values = list(lineage)
    placeholders = ", ".join("?" for _ in values)
    return conn.execute(
        f"SELECT 1 FROM review_invalidations WHERE review_id IN ({placeholders}) LIMIT 1", values
    ).fetchone() is not None


def _validate_review_commitment(
    conn: sqlite3.Connection, review: Mapping[str, object], candidate: Mapping[str, object]
) -> None:
    """Reject altered, stale-policy, self-reviewed, or invalidated review rows."""
    candidate_id = review["candidate_id"]
    reviewer = review["reviewer"]
    if not isinstance(candidate_id, str) or not candidate_id or not isinstance(reviewer, str) or not reviewer:
        raise ValueError("review identity is invalid")
    if review["review_id"] != "review:" + _digest(candidate_id, reviewer):
        raise ValueError("review ID does not match its candidate and reviewer")
    if reviewer == candidate["worker"]:
        raise ValueError("reviewer is not independent from candidate worker")
    if review["verdict"] not in REVIEW_VERDICTS:
        raise ValueError("review verdict is invalid")
    if review["policy_version"] != LICENSED_CORE_POLICY_VERSION:
        raise ValueError("review policy is not the current supported policy")
    expected = _review_commitment(
        candidate_id=candidate_id,
        reviewer=reviewer,
        verdict=review["verdict"],
        policy_version=review["policy_version"],
        reason_tags=review["reason_tags"],
        reviewed_at=review["reviewed_at"],
        review_lineage=review["review_lineage"],
    )
    if review["review_commitment"] != expected:
        raise ValueError("review commitment no longer matches its trusted fields")
    lineage = _review_lineage(review["review_lineage"])
    if lineage[-1] != review["review_id"]:
        raise ValueError("review lineage does not end at its review ID")
    if _lineage_is_invalidated(conn, lineage):
        raise ValueError("invalidated review cannot be trusted")


def _active_valid_reviews(
    conn: sqlite3.Connection, candidate: Mapping[str, object]
) -> list[sqlite3.Row]:
    """Load the candidate's still-active reviews and validate every trust field."""
    reviews = conn.execute(
        f"""SELECT * FROM reviews AS r WHERE r.candidate_id=?
        AND ({_active_review_sql('r')}) ORDER BY r.reviewer""",
        (candidate["candidate_id"],),
    ).fetchall()
    trusted: list[sqlite3.Row] = []
    for review in reviews:
        # Pre-v6/v7 rows remain in the audit log but have no immutable
        # commitment/lineage. They cannot govern publication or block rework.
        if review["review_commitment"] is None or review["review_lineage"] is None:
            continue
        if _lineage_is_invalidated(conn, _review_lineage(review["review_lineage"])):
            continue
        _validate_review_commitment(conn, review, candidate)
        trusted.append(review)
    return trusted


def _has_blocking_active_review(
    conn: sqlite3.Connection, candidate: Mapping[str, object]
) -> bool:
    """Return whether a candidate has a current trusted review that still governs it.

    Historical pre-commitment rows and clone rows whose lineage was invalidated
    remain auditable but do not prevent a fresh independent review.
    """
    reviews = conn.execute(
        f"""SELECT * FROM reviews AS r WHERE r.candidate_id=?
        AND ({_active_review_sql('r')}) ORDER BY r.reviewer""",
        (candidate["candidate_id"],),
    ).fetchall()
    for review in reviews:
        if review["review_commitment"] is None or review["review_lineage"] is None:
            continue
        lineage = _review_lineage(review["review_lineage"])
        if _lineage_is_invalidated(conn, lineage):
            continue
        _validate_review_commitment(conn, review, candidate)
        return True
    return False


def _active_review_sql(review_alias: str) -> str:
    """Predicate for a review which has not been invalidated by audit action."""
    return f"""NOT EXISTS (
        SELECT 1 FROM review_invalidations AS invalidation
        WHERE invalidation.review_id={review_alias}.review_id
    )"""


def _source_product_code(source_id: str, product: object) -> str:
    if isinstance(product, str) and product.startswith("PZO"):
        return product
    parts = source_id.split(":")
    if len(parts) >= 2 and parts[0] == "paizo" and parts[1].startswith("PZO"):
        return parts[1]
    raise ValueError(f"corpus source is missing a PZO product code: {source_id!r}")


def _fingerprint(source: sqlite3.Row) -> str:
    raw = source["provenance"]
    try:
        provenance = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise ValueError("corpus source provenance is invalid JSON") from exc
    value = provenance.get("content_fingerprint")
    if not isinstance(value, str) or not value:
        raise ValueError("corpus source is missing a normalized content fingerprint")
    revision = source["revision"]
    if isinstance(revision, str) and revision and revision != value:
        raise ValueError("corpus source revision does not match its normalized content fingerprint")
    return value


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE source_revisions (
            product_code TEXT NOT NULL,
            content_fingerprint TEXT NOT NULL,
            license TEXT NOT NULL CHECK (license IN ('OGL', 'ORC')),
            era TEXT NOT NULL,
            parser_version TEXT NOT NULL,
            source_schema_version TEXT,
            printing_revision TEXT NOT NULL DEFAULT 'unknown',
            PRIMARY KEY (product_code, content_fingerprint)
        );
        CREATE TABLE review_shards (
            shard_id INTEGER PRIMARY KEY,
            product_code TEXT NOT NULL,
            content_fingerprint TEXT NOT NULL,
            shard_ordinal INTEGER NOT NULL,
            section_count INTEGER NOT NULL,
            claimant TEXT,
            claimed_at INTEGER,
            lease_expires_at INTEGER,
            claim_mode TEXT CHECK (claim_mode IN ('ordinary', 'rework') OR claim_mode IS NULL),
            UNIQUE (product_code, content_fingerprint, shard_ordinal),
            FOREIGN KEY (product_code, content_fingerprint)
                REFERENCES source_revisions(product_code, content_fingerprint)
        );
        CREATE TABLE source_sections (
            section_key TEXT PRIMARY KEY,
            product_code TEXT NOT NULL,
            content_fingerprint TEXT NOT NULL,
            source_section_id TEXT NOT NULL,
            source_section_hash TEXT NOT NULL,
            page_start INTEGER,
            page_end INTEGER,
            printed_page TEXT,
            heading TEXT NOT NULL,
            source_text TEXT NOT NULL,
            shard_id INTEGER NOT NULL,
            UNIQUE (product_code, content_fingerprint, source_section_id),
            FOREIGN KEY (product_code, content_fingerprint)
                REFERENCES source_revisions(product_code, content_fingerprint),
            FOREIGN KEY (shard_id) REFERENCES review_shards(shard_id)
        );
        CREATE INDEX source_sections_by_shard ON source_sections(shard_id);
        CREATE TABLE candidates (
            candidate_id TEXT PRIMARY KEY,
            section_key TEXT NOT NULL,
            source_section_hash TEXT NOT NULL,
            candidate_ordinal INTEGER NOT NULL,
            decision TEXT NOT NULL CHECK (decision IN
                ('PUBLIC_AS_IS', 'MIXED_NEEDS_EXTRACTION', 'EXCLUDE', 'UNCERTAIN')),
            candidate_text TEXT,
            public_heading TEXT,
            candidate_hash TEXT NOT NULL,
            extraction_method TEXT,
            reason_tags TEXT NOT NULL,
            confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
            worker TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            submitted_at INTEGER NOT NULL,
            UNIQUE (section_key, candidate_ordinal),
            FOREIGN KEY (section_key) REFERENCES source_sections(section_key)
        );
        CREATE INDEX candidates_by_section ON candidates(section_key);
        CREATE TABLE review_claims (
            candidate_id TEXT PRIMARY KEY REFERENCES candidates(candidate_id),
            claimant TEXT NOT NULL,
            claimed_at INTEGER NOT NULL,
            lease_expires_at INTEGER NOT NULL
        );
        CREATE TABLE reviews (
            review_id TEXT PRIMARY KEY,
            candidate_id TEXT NOT NULL,
            reviewer TEXT NOT NULL,
            verdict TEXT NOT NULL CHECK (verdict IN ('APPROVE', 'REJECT', 'REVISE')),
            policy_version TEXT NOT NULL,
            reason_tags TEXT NOT NULL,
            notes TEXT,
            reviewed_at INTEGER NOT NULL,
            review_commitment TEXT NOT NULL,
            review_lineage TEXT NOT NULL,
            UNIQUE (candidate_id, reviewer),
            FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
        );
        CREATE INDEX reviews_by_candidate ON reviews(candidate_id);
        CREATE TABLE review_invalidations (
            review_id TEXT PRIMARY KEY REFERENCES reviews(review_id),
            invalidation_id TEXT NOT NULL UNIQUE,
            batch_id TEXT NOT NULL,
            selection_digest TEXT NOT NULL,
            invalidated_by TEXT NOT NULL,
            reason TEXT NOT NULL,
            invalidated_at INTEGER NOT NULL
        );
        CREATE INDEX review_invalidations_by_batch ON review_invalidations(batch_id);
        CREATE TABLE evidence (
            evidence_id TEXT PRIMARY KEY,
            candidate_id TEXT,
            review_id TEXT,
            evidence_kind TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('match', 'no_match', 'uncertain')),
            url TEXT,
            checked_at INTEGER NOT NULL,
            note TEXT,
            CHECK ((candidate_id IS NOT NULL) != (review_id IS NOT NULL)),
            FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id),
            FOREIGN KEY (review_id) REFERENCES reviews(review_id)
        );
        CREATE INDEX evidence_by_candidate ON evidence(candidate_id);
        CREATE INDEX evidence_by_review ON evidence(review_id);
        """
    )


def _migrate_review_workspace(conn: sqlite3.Connection) -> None:
    """Safely migrate older private workspaces while retaining audit history.

    Version 2 added persisted candidate-claim modes.  Existing live leases are
    classified exactly once: a shard is rework only when every current
    decision was reviewed and one was explicitly revised; every other active
    lease remains ordinary so a worker can safely finish the assignment it
    originally claimed.  Version 3 adds append-only review invalidations.
    Version 4 appends parser-run and source-asset provenance without re-keying
    prior candidate, review, invalidation, or evidence rows.  Versions 6 and 7
    add review commitments and immutable clone lineage; historical review rows
    remain auditable but are not publishable until re-reviewed under the
    current contract.  Version 8 repairs only the derived native-word coverage
    digest of sealed, complete, review-disabled staged runs created before
    per-section anchor order was canonicalized. Version 9 adds a private,
    binary first-pass screen whose decisions never enter the public builder.
    Version 10 adds nonterminal deferral and a separate escalation queue while
    preserving every version-9 terminal decision in place. Version 11 adds
    content-free runner sessions/attempts, maintainer items, AON provenance,
    and deterministic stitch decisions. Private text remains in source and
    candidate tables only. Version 12 binds every session and attempt to the
    exact Codex CLI version used by the supervisor. Version 13 records the
    Luna-to-Terra semantic escalation boundary for complex screening records.
    Version 14 retains the bounded Spark rejection reason used by each
    deterministic EXCLUDE candidate. Version 15 adds database-backed stitch
    leases so concurrent selector/confirmer workers cannot claim one proposal.
    Version 16 replaces mutable screening rows with an append-only event log;
    legacy rows are copied once and a read-only latest-state view is derived.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "review_shards" not in tables:
            conn.commit()
            return
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(review_shards)")}
        schema_row = conn.execute(
            "SELECT value FROM metadata WHERE key='review_schema_version'"
        ).fetchone()
        if schema_row is not None:
            try:
                stored_schema = int(str(schema_row["value"]))
            except ValueError as exc:
                raise ValueError("review workspace has an invalid schema version") from exc
            if stored_schema > REVIEW_SCHEMA_VERSION:
                raise ValueError(
                    "review workspace uses unsupported future schema "
                    f"{stored_schema}; this build supports {REVIEW_SCHEMA_VERSION}"
                )
        if "review_invalidations" not in tables:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS review_invalidations (
                    review_id TEXT PRIMARY KEY REFERENCES reviews(review_id),
                    invalidation_id TEXT NOT NULL UNIQUE,
                    batch_id TEXT NOT NULL,
                    selection_digest TEXT NOT NULL,
                    invalidated_by TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    invalidated_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS review_invalidations_by_batch ON review_invalidations(batch_id);
                """
            )
        # Keep assets (the parser-independent source) distinct from parser
        # outputs.  A later parser may therefore be staged safely beside the
        # review target instead of overwriting it.
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS source_assets (
                asset_id TEXT PRIMARY KEY,
                product_code TEXT NOT NULL,
                source_fingerprint TEXT NOT NULL,
                source_content_fingerprint TEXT,
                provenance_hash TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                inventory_profile TEXT,
                native_inventory_version TEXT,
                native_word_anchor_digest TEXT,
                native_word_anchor_count INTEGER,
                ignored_anchor_policy TEXT,
                ignored_anchor_digest TEXT,
                inventory_manifest_digest TEXT,
                inventory_bound_at INTEGER,
                UNIQUE(product_code, source_fingerprint)
            );
            CREATE TABLE IF NOT EXISTS parser_runs (
                parser_run_id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL REFERENCES source_assets(asset_id),
                product_code TEXT NOT NULL,
                source_fingerprint TEXT NOT NULL,
                parser_version TEXT NOT NULL,
                parser_output_digest TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('staged', 'active', 'retired', 'rejected')),
                review_enabled INTEGER NOT NULL DEFAULT 0 CHECK (review_enabled IN (0, 1)),
                complete INTEGER NOT NULL DEFAULT 1 CHECK (complete IN (0, 1)),
                created_at INTEGER NOT NULL,
                activated_at INTEGER,
                manifest_version TEXT,
                manifest_digest TEXT,
                declared_section_count INTEGER,
                native_word_coverage_digest TEXT,
                source_inventory_digest TEXT,
                origin TEXT NOT NULL DEFAULT 'legacy-untrusted',
                bundle_seal TEXT,
                bundle_parser_output_digest TEXT,
                bundle_commitment TEXT,
                UNIQUE(asset_id, parser_output_digest)
            );
            CREATE INDEX IF NOT EXISTS parser_runs_by_product
                ON parser_runs(product_code, state, review_enabled);
            CREATE TABLE IF NOT EXISTS parser_run_sections (
                parser_run_id TEXT NOT NULL REFERENCES parser_runs(parser_run_id),
                section_key TEXT NOT NULL REFERENCES source_sections(section_key),
                stable_identity TEXT NOT NULL,
                provenance_hash TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                section_commitment TEXT,
                membership_state TEXT NOT NULL DEFAULT 'present'
                    CHECK (membership_state IN ('present', 'retired')),
                reused_from_section_key TEXT REFERENCES source_sections(section_key),
                PRIMARY KEY(parser_run_id, section_key)
            );
            CREATE INDEX IF NOT EXISTS parser_run_sections_by_identity
                ON parser_run_sections(parser_run_id, stable_identity, provenance_hash, text_hash);
            CREATE TABLE IF NOT EXISTS parser_section_anchors (
                parser_run_id TEXT NOT NULL REFERENCES parser_runs(parser_run_id),
                section_key TEXT NOT NULL REFERENCES source_sections(section_key),
                anchor_hash TEXT NOT NULL,
                PRIMARY KEY(parser_run_id, anchor_hash),
                UNIQUE(parser_run_id, section_key, anchor_hash)
            );
            CREATE TABLE IF NOT EXISTS parser_ignored_anchors (
                parser_run_id TEXT NOT NULL REFERENCES parser_runs(parser_run_id),
                anchor_hash TEXT NOT NULL,
                reason TEXT NOT NULL,
                PRIMARY KEY(parser_run_id, anchor_hash)
            );
            CREATE TABLE IF NOT EXISTS parser_section_blocks (
                parser_run_id TEXT NOT NULL REFERENCES parser_runs(parser_run_id),
                section_key TEXT NOT NULL REFERENCES source_sections(section_key),
                block_ordinal INTEGER NOT NULL CHECK(block_ordinal >= 0),
                kind TEXT NOT NULL CHECK(kind IN ('heading', 'body', 'sidebar', 'table')),
                physical_page INTEGER NOT NULL CHECK(physical_page >= 1),
                source_text TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                table_json TEXT,
                anchor_digest TEXT NOT NULL,
                PRIMARY KEY(parser_run_id, section_key, block_ordinal)
            );
            CREATE TABLE IF NOT EXISTS parser_section_block_anchors (
                parser_run_id TEXT NOT NULL,
                section_key TEXT NOT NULL,
                block_ordinal INTEGER NOT NULL,
                anchor_ordinal INTEGER NOT NULL CHECK(anchor_ordinal >= 0),
                anchor_hash TEXT NOT NULL,
                PRIMARY KEY(parser_run_id, section_key, block_ordinal, anchor_ordinal),
                UNIQUE(parser_run_id, section_key, anchor_hash),
                FOREIGN KEY(parser_run_id, section_key, block_ordinal)
                    REFERENCES parser_section_blocks(parser_run_id, section_key, block_ordinal)
            );
            CREATE TABLE IF NOT EXISTS parser_quarantine (
                parser_run_id TEXT NOT NULL REFERENCES parser_runs(parser_run_id),
                quarantine_id TEXT NOT NULL,
                product_code TEXT NOT NULL,
                reason TEXT NOT NULL CHECK(reason IN (
                    'repeated-furniture', 'page-number', 'contents-index',
                    'credits-legal', 'unresolved-table', 'unbound-layout',
                    'heading-artifact', 'unresolved-continuation', 'unresolved-layout',
                    'layout-order-conflict', 'oversize-block'
                )),
                physical_page INTEGER NOT NULL CHECK(physical_page >= 1),
                source_text TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                anchor_count INTEGER NOT NULL CHECK(anchor_count >= 1),
                anchor_digest TEXT NOT NULL,
                PRIMARY KEY(parser_run_id, quarantine_id)
            );
            CREATE INDEX IF NOT EXISTS parser_quarantine_by_product
                ON parser_quarantine(parser_run_id, product_code, reason);
            CREATE TABLE IF NOT EXISTS parser_quarantine_anchors (
                parser_run_id TEXT NOT NULL,
                quarantine_id TEXT NOT NULL,
                anchor_hash TEXT NOT NULL,
                PRIMARY KEY(parser_run_id, anchor_hash),
                FOREIGN KEY(parser_run_id, quarantine_id)
                    REFERENCES parser_quarantine(parser_run_id, quarantine_id)
            );
            CREATE TABLE IF NOT EXISTS draft_screening_claims (
                parser_run_id TEXT NOT NULL REFERENCES parser_runs(parser_run_id),
                shard_id INTEGER NOT NULL REFERENCES review_shards(shard_id),
                claimant TEXT NOT NULL,
                claimed_at INTEGER NOT NULL,
                lease_expires_at INTEGER NOT NULL,
                claim_mode TEXT NOT NULL DEFAULT 'ordinary'
                    CHECK(claim_mode IN ('ordinary', 'escalation')),
                PRIMARY KEY(parser_run_id, shard_id)
            );
            CREATE INDEX IF NOT EXISTS draft_screening_claims_by_claimant
                ON draft_screening_claims(claimant, lease_expires_at);
            CREATE TABLE IF NOT EXISTS draft_screening_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                parser_run_id TEXT NOT NULL REFERENCES parser_runs(parser_run_id),
                section_key TEXT NOT NULL REFERENCES source_sections(section_key),
                event_type TEXT NOT NULL CHECK(event_type IN ('DECISION', 'REOPEN')),
                requested_decision TEXT CHECK(requested_decision IN ('ADD', 'REJECT', 'DEFER')
                    OR requested_decision IS NULL),
                decision TEXT CHECK(decision IN ('ADD', 'REJECT', 'DEFER') OR decision IS NULL),
                duplicate_of_section_key TEXT REFERENCES source_sections(section_key),
                reject_reason TEXT CHECK(reject_reason IN
                    ('no-mechanics', 'duplicate', 'setting-prose')
                    OR reject_reason IS NULL),
                defer_reason TEXT CHECK(defer_reason IN
                    ('layout', 'scope', 'complex-rule', 'insufficient-context')
                    OR defer_reason IS NULL),
                deferred_by TEXT,
                deferred_at INTEGER,
                worker TEXT NOT NULL,
                decided_at INTEGER NOT NULL,
                reopen_reason TEXT,
                supersedes_event_id INTEGER REFERENCES draft_screening_events(event_id),
                CHECK(
                    (event_type='DECISION' AND requested_decision IS NOT NULL
                     AND decision IS NOT NULL
                     AND ((decision='DEFER' AND defer_reason IS NOT NULL
                           AND deferred_by IS NOT NULL AND deferred_at IS NOT NULL)
                          OR decision<>'DEFER')
                        AND (decision='REJECT' OR reject_reason IS NULL)
                        AND reopen_reason IS NULL)
                    OR (event_type='REOPEN' AND requested_decision IS NULL
                        AND decision IS NULL AND reopen_reason IS NOT NULL
                        AND duplicate_of_section_key IS NULL AND defer_reason IS NULL
                        AND deferred_by IS NULL AND deferred_at IS NULL)
                ),
                UNIQUE(parser_run_id, section_key, event_type, decided_at, worker)
            );
            CREATE INDEX IF NOT EXISTS draft_screening_events_by_section
                ON draft_screening_events(parser_run_id, section_key, event_id);
            CREATE INDEX IF NOT EXISTS draft_screening_events_by_result
                ON draft_screening_events(parser_run_id, decision, event_id);
            """
        )
        # sqlite3.executescript() commits any pending transaction before it
        # runs.  Re-enter one write transaction so schema upgrades and all
        # following backfills either complete together or roll back together.
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
        conn.executescript(_RUNNER_SCHEMA_SQL)
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
        quarantine_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='parser_quarantine'"
        ).fetchone()
        if quarantine_sql is not None and "oversize-block" not in str(quarantine_sql[0]):
            conn.execute(
                """CREATE TEMP TABLE parser_quarantine_anchors_v18 AS
                   SELECT parser_run_id, quarantine_id, anchor_hash
                   FROM parser_quarantine_anchors"""
            )
            conn.execute(
                """CREATE TABLE parser_quarantine_v18 (
                    parser_run_id TEXT NOT NULL REFERENCES parser_runs(parser_run_id),
                    quarantine_id TEXT NOT NULL,
                    product_code TEXT NOT NULL,
                    reason TEXT NOT NULL CHECK(reason IN (
                        'repeated-furniture', 'page-number', 'contents-index',
                        'credits-legal', 'unresolved-table', 'unbound-layout',
                        'heading-artifact', 'unresolved-continuation', 'unresolved-layout',
                        'layout-order-conflict', 'oversize-block'
                    )),
                    physical_page INTEGER NOT NULL CHECK(physical_page >= 1),
                    source_text TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    anchor_count INTEGER NOT NULL CHECK(anchor_count >= 1),
                    anchor_digest TEXT NOT NULL,
                    PRIMARY KEY(parser_run_id, quarantine_id)
                )"""
            )
            conn.execute(
                """INSERT INTO parser_quarantine_v18
                   SELECT parser_run_id, quarantine_id, product_code, reason,
                          physical_page, source_text, text_hash, anchor_count, anchor_digest
                   FROM parser_quarantine"""
            )
            conn.execute("DROP TABLE parser_quarantine_anchors")
            conn.execute("DROP TABLE parser_quarantine")
            conn.execute("ALTER TABLE parser_quarantine_v18 RENAME TO parser_quarantine")
            conn.execute(
                """CREATE INDEX parser_quarantine_by_product
                   ON parser_quarantine(parser_run_id, product_code, reason)"""
            )
            conn.execute(
                """CREATE TABLE parser_quarantine_anchors (
                    parser_run_id TEXT NOT NULL,
                    quarantine_id TEXT NOT NULL,
                    anchor_hash TEXT NOT NULL,
                    PRIMARY KEY(parser_run_id, anchor_hash),
                    FOREIGN KEY(parser_run_id, quarantine_id)
                        REFERENCES parser_quarantine(parser_run_id, quarantine_id)
                )"""
            )
            conn.execute(
                """INSERT INTO parser_quarantine_anchors
                   SELECT parser_run_id, quarantine_id, anchor_hash
                   FROM parser_quarantine_anchors_v18"""
            )
            conn.execute("DROP TABLE parser_quarantine_anchors_v18")
        session_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(runner_sessions)")}
        if "cli_version" not in session_columns:
            conn.execute("ALTER TABLE runner_sessions ADD COLUMN cli_version TEXT NOT NULL DEFAULT ''")
        attempt_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(runner_attempts)")}
        if "cli_version" not in attempt_columns:
            conn.execute("ALTER TABLE runner_attempts ADD COLUMN cli_version TEXT NOT NULL DEFAULT ''")
        screening_claim_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(draft_screening_claims)")
        }
        if "claim_mode" not in screening_claim_columns:
            conn.execute(
                """ALTER TABLE draft_screening_claims ADD COLUMN claim_mode TEXT
                   NOT NULL DEFAULT 'ordinary'
                   CHECK(claim_mode IN ('ordinary', 'escalation'))"""
            )
        # Screening decisions are an append-only event log.  Older workspaces
        # had one mutable row per section; copy those rows into the event log
        # once, then remove the mutable table.  The current-state view below
        # is read-only and always derives from the newest event.
        screening_object = conn.execute(
            "SELECT type FROM sqlite_master WHERE name='draft_screening_decisions'"
        ).fetchone()
        if screening_object is not None and str(screening_object[0]) == "table":
            legacy_columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(draft_screening_decisions)")
            }
            conn.execute("ALTER TABLE draft_screening_decisions RENAME TO draft_screening_decisions_legacy")
            if not conn.in_transaction:
                conn.execute("BEGIN IMMEDIATE")
            legacy_rows = conn.execute("SELECT * FROM draft_screening_decisions_legacy").fetchall()

            def _legacy_value(
                row: sqlite3.Row, name: str, default: object = None
            ) -> object:
                return row[name] if name in legacy_columns else default

            for legacy in legacy_rows:
                requested = _legacy_value(legacy, "requested_decision")
                decision = _legacy_value(legacy, "decision")
                if requested not in SCREENING_DECISIONS or decision not in SCREENING_DECISIONS:
                    raise ValueError("review workspace has an invalid legacy screening decision")
                legacy_worker = _legacy_value(legacy, "worker", "legacy-migration") or "legacy-migration"
                legacy_decided_at = _legacy_value(legacy, "decided_at", int(time.time())) or int(time.time())
                legacy_defer_reason = _legacy_value(legacy, "defer_reason")
                legacy_deferred_by = _legacy_value(legacy, "deferred_by")
                legacy_deferred_at = _legacy_value(legacy, "deferred_at")
                if decision == "DEFER":
                    legacy_defer_reason = legacy_defer_reason or "scope"
                    legacy_deferred_by = legacy_deferred_by or legacy_worker
                    legacy_deferred_at = legacy_deferred_at or legacy_decided_at
                conn.execute(
                    """INSERT INTO draft_screening_events
                       (parser_run_id, section_key, event_type, requested_decision, decision,
                        duplicate_of_section_key, reject_reason, defer_reason, deferred_by, deferred_at,
                        worker, decided_at)
                       VALUES (?, ?, 'DECISION', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        legacy["parser_run_id"], legacy["section_key"], requested, decision,
                        _legacy_value(legacy, "duplicate_of_section_key"),
                        _legacy_value(legacy, "reject_reason"), legacy_defer_reason,
                        legacy_deferred_by, legacy_deferred_at, legacy_worker, legacy_decided_at,
                    ),
                )
            conn.execute("DROP TABLE draft_screening_decisions_legacy")
        # A manually-created current-state view from a pre-event fixture is
        # not part of the runtime schema; replace it with the canonical view.
        if conn.execute(
            "SELECT type FROM sqlite_master WHERE name='draft_screening_decisions'"
        ).fetchone() is not None:
            conn.execute("DROP VIEW IF EXISTS draft_screening_decisions")
        conn.execute(
            """CREATE VIEW IF NOT EXISTS draft_screening_current AS
               SELECT e.parser_run_id, e.section_key, e.requested_decision,
                      e.decision, e.duplicate_of_section_key, e.reject_reason, e.defer_reason,
                      e.deferred_by, e.deferred_at, e.worker, e.decided_at
                 FROM draft_screening_events AS e
                WHERE e.event_type='DECISION'
                  AND e.event_id=(
                      SELECT MAX(newer.event_id)
                        FROM draft_screening_events AS newer
                       WHERE newer.parser_run_id=e.parser_run_id
                         AND newer.section_key=e.section_key
                  )"""
        )
        section_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(source_sections)")}
        revision_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(source_revisions)")}
        if "printing_revision" not in revision_columns:
            conn.execute(
                "ALTER TABLE source_revisions ADD COLUMN printing_revision TEXT NOT NULL DEFAULT 'unknown'"
            )
        for column in ("parser_run_id", "stable_identity", "provenance_hash", "text_hash", "layout_flags"):
            if column not in section_columns:
                conn.execute(f"ALTER TABLE source_sections ADD COLUMN {column} TEXT")
        shard_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(review_shards)")}
        if "parser_run_id" not in shard_columns:
            conn.execute("ALTER TABLE review_shards ADD COLUMN parser_run_id TEXT")
        asset_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(source_assets)")}
        if "source_content_fingerprint" not in asset_columns:
            conn.execute("ALTER TABLE source_assets ADD COLUMN source_content_fingerprint TEXT")
            conn.execute(
                "UPDATE source_assets SET source_content_fingerprint=source_fingerprint "
                "WHERE source_content_fingerprint IS NULL"
            )
        for column, definition in (
            ("inventory_profile", "TEXT"),
            ("native_inventory_version", "TEXT"),
            ("native_word_anchor_digest", "TEXT"),
            ("native_word_anchor_count", "INTEGER"),
            ("ignored_anchor_policy", "TEXT"),
            ("ignored_anchor_digest", "TEXT"),
            ("inventory_manifest_digest", "TEXT"),
            ("inventory_bound_at", "INTEGER"),
        ):
            if column not in asset_columns:
                conn.execute(f"ALTER TABLE source_assets ADD COLUMN {column} {definition}")
        run_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(parser_runs)")}
        for column, definition in (
            ("manifest_version", "TEXT"),
            ("manifest_digest", "TEXT"),
            ("declared_section_count", "INTEGER"),
            ("native_word_coverage_digest", "TEXT"),
            ("source_inventory_digest", "TEXT"),
            ("origin", "TEXT NOT NULL DEFAULT 'legacy-untrusted'"),
            ("bundle_seal", "TEXT"),
            ("bundle_parser_output_digest", "TEXT"),
            ("bundle_commitment", "TEXT"),
        ):
            if column not in run_columns:
                conn.execute(f"ALTER TABLE parser_runs ADD COLUMN {column} {definition}")
        conn.execute(
            "UPDATE parser_runs SET origin=? WHERE origin IS NULL OR origin=''",
            (_LEGACY_RUN_ORIGIN,),
        )
        membership_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(parser_run_sections)")}
        if "section_commitment" not in membership_columns:
            conn.execute("ALTER TABLE parser_run_sections ADD COLUMN section_commitment TEXT")
        candidate_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(candidates)")}
        if "public_heading" not in candidate_columns:
            conn.execute("ALTER TABLE candidates ADD COLUMN public_heading TEXT")
        review_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(reviews)")}
        if "review_commitment" not in review_columns:
            # Existing reviews retain their history but cannot become public
            # approvals until independently re-reviewed under this commitment.
            conn.execute("ALTER TABLE reviews ADD COLUMN review_commitment TEXT")
        if "review_lineage" not in review_columns:
            # A lineage cannot be reconstructed without changing historical
            # evidence, so historical rows remain auditable but nonpublishable.
            conn.execute("ALTER TABLE reviews ADD COLUMN review_lineage TEXT")
        if "claim_mode" not in columns:
            try:
                conn.execute(
                    """ALTER TABLE review_shards ADD COLUMN claim_mode TEXT
                    CHECK (claim_mode IN ('ordinary', 'rework') OR claim_mode IS NULL)"""
                )
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc):
                    raise
            now = int(time.time())
            conn.execute(
                f"""UPDATE review_shards AS sh
                SET claim_mode=CASE
                    WHEN EXISTS (
                        SELECT 1
                        FROM source_sections AS revised_section
                        JOIN candidates AS revised_candidate
                          ON revised_candidate.section_key=revised_section.section_key
                        JOIN reviews AS revised_review
                          ON revised_review.candidate_id=revised_candidate.candidate_id
                        WHERE revised_section.shard_id=sh.shard_id
                          AND revised_candidate.candidate_ordinal=(
                              SELECT MAX(candidate_ordinal) FROM candidates AS latest_candidate
                              WHERE latest_candidate.section_key=revised_section.section_key
                          )
                          AND revised_review.verdict='REVISE'
                          AND ({_active_review_sql('revised_review')})
                    )
                    AND NOT EXISTS (
                        SELECT 1
                        FROM source_sections AS unresolved_section
                        WHERE unresolved_section.shard_id=sh.shard_id
                          AND NOT EXISTS (
                              SELECT 1
                              FROM candidates AS current_candidate
                              JOIN reviews AS current_review
                                ON current_review.candidate_id=current_candidate.candidate_id
                              WHERE current_candidate.section_key=unresolved_section.section_key
                                AND current_candidate.candidate_ordinal=(
                                    SELECT MAX(candidate_ordinal) FROM candidates AS latest_candidate
                                    WHERE latest_candidate.section_key=current_candidate.section_key
                                )
                              AND ({_active_review_sql('current_review')})
                          )
                    )
                    THEN 'rework'
                    ELSE 'ordinary'
                END
                WHERE claimant IS NOT NULL AND lease_expires_at >= ?""",
                (now,),
            )
        # Backfill exactly one active v1 run per product.  Old workspaces did
        # not persist a revision-selection decision, so the deterministic
        # lexicographically greatest fingerprint is merely a bootstrap choice;
        # all other historical revisions remain retained as retired runs.
        now = int(time.time())
        revisions = conn.execute(
            "SELECT product_code, content_fingerprint, parser_version FROM source_revisions "
            "ORDER BY product_code, content_fingerprint"
        ).fetchall()
        active_fingerprint = {
            str(row["product_code"]): str(row["content_fingerprint"])
            for row in revisions
        }
        for revision in revisions:
            product = str(revision["product_code"])
            fingerprint = str(revision["content_fingerprint"])
            # A staged V4 parser output still has a compatibility revision row
            # for old foreign keys, but its sections are already attached to
            # its real parser run.  Never reinterpret that row as a fresh V1
            # import on a later read/migration pass.
            needs_backfill = conn.execute(
                """SELECT 1 FROM source_sections WHERE product_code=? AND content_fingerprint=?
                AND parser_run_id IS NULL LIMIT 1""", (product, fingerprint)
            ).fetchone()
            if needs_backfill is None:
                continue
            asset_id = "asset:" + _digest(product, fingerprint)
            run_id = "parser-run:v1:" + _digest(product, fingerprint)
            conn.execute(
                """INSERT OR IGNORE INTO source_assets
                (asset_id, product_code, source_fingerprint, source_content_fingerprint, provenance_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (asset_id, product, fingerprint, fingerprint,
                 _digest("v1-source-asset", product, fingerprint), now),
            )
            active = fingerprint == active_fingerprint[product]
            conn.execute(
                """INSERT OR IGNORE INTO parser_runs
                (parser_run_id, asset_id, product_code, source_fingerprint, parser_version,
                 parser_output_digest, state, review_enabled, complete, created_at, activated_at, origin)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)""",
                (run_id, asset_id, product, fingerprint, str(revision["parser_version"]),
                 _digest("v1-parser-output", product, fingerprint),
                 "active" if active else "retired", 1 if active else 0, now, now if active else None,
                 _LEGACY_RUN_ORIGIN),
            )
            sections = conn.execute(
                """SELECT section_key, source_section_id, source_text, page_start, page_end,
                          printed_page, heading FROM source_sections
                   WHERE product_code=? AND content_fingerprint=?""",
                (product, fingerprint),
            ).fetchall()
            for section in sections:
                stable_identity = _digest(
                    "v1-anchor", product, str(section["source_section_id"]),
                    str(section["page_start"] or ""), str(section["heading"] or ""),
                )
                provenance_hash = _digest(
                    "v1-provenance", str(section["page_start"] or ""),
                    str(section["page_end"] or ""), str(section["printed_page"] or ""),
                    str(section["heading"] or ""),
                )
                text_hash = hashlib.sha256(str(section["source_text"]).encode("utf-8")).hexdigest()
                conn.execute(
                    """UPDATE source_sections SET parser_run_id=?, stable_identity=?,
                       provenance_hash=?, text_hash=? WHERE section_key=?""",
                    (run_id, stable_identity, provenance_hash, text_hash, section["section_key"]),
                )
                conn.execute(
                    """INSERT OR IGNORE INTO parser_run_sections
                    (parser_run_id, section_key, stable_identity, provenance_hash, text_hash)
                    VALUES (?, ?, ?, ?, ?)""",
                    (run_id, section["section_key"], stable_identity, provenance_hash, text_hash),
                )
            conn.execute(
                """UPDATE review_shards SET parser_run_id=?
                WHERE product_code=? AND content_fingerprint=? AND parser_run_id IS NULL""",
                (run_id, product, fingerprint),
            )
        duplicate_target = conn.execute(
            """SELECT product_code FROM parser_runs WHERE state='active' AND review_enabled=1
            GROUP BY product_code HAVING COUNT(*) > 1 LIMIT 1"""
        ).fetchone()
        if duplicate_target is not None:
            raise ValueError("ambiguous active parser-run review target")
        conn.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS parser_runs_one_active_target
            ON parser_runs(product_code) WHERE state='active' AND review_enabled=1"""
        )
        # Semantic scheduling is deliberately independent from parser-run
        # activation.  Existing workspaces retain their prior all-product
        # behavior until a maintainer explicitly narrows the scope.
        now = int(time.time())
        conn.execute(
            """INSERT OR IGNORE INTO review_product_scope
               (product_code, enabled, reason, updated_at)
               SELECT DISTINCT product_code, 1, 'enabled', ?
                 FROM parser_runs
                WHERE state='active' AND review_enabled=1""",
            (now,),
        )
        if schema_row is None or schema_row["value"] != str(REVIEW_SCHEMA_VERSION):
            _repair_staged_native_coverage_digests(conn)
        if schema_row is None or schema_row["value"] != str(REVIEW_SCHEMA_VERSION):
            conn.execute(
                """INSERT INTO metadata(key, value) VALUES ('review_schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
                (str(REVIEW_SCHEMA_VERSION),),
            )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def _ensure_workspace_migrated(workspace: Path | str) -> None:
    """Apply the one-way current private-workspace upgrade before access."""
    conn = _connect(workspace)
    try:
        _migrate_review_workspace(conn)
    finally:
        conn.close()


def _semantic_scope_sql(product_expression: str) -> str:
    """Return the SQL predicate for a product in the configured semantic scope."""
    return f"""EXISTS (
        SELECT 1 FROM review_product_scope AS semantic_scope
        WHERE semantic_scope.product_code={product_expression}
          AND semantic_scope.enabled=1
    )"""


def _review_scope_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT scope.product_code, scope.enabled, scope.reason, scope.updated_at
             FROM review_product_scope AS scope
             JOIN parser_runs AS run ON run.product_code=scope.product_code
              AND run.state='active' AND run.review_enabled=1
            GROUP BY scope.product_code
            ORDER BY scope.product_code"""
    ).fetchall()


def review_product_scope(workspace: Path | str) -> dict[str, object]:
    """Return the persistent semantic product scope without private source data."""
    _ensure_workspace_migrated(workspace)
    conn = _connect(workspace, readonly=True)
    try:
        rows = _review_scope_rows(conn)
        products = [
            {
                "product_code": str(row["product_code"]),
                "state": "enabled" if int(row["enabled"]) else "held",
                "reason": str(row["reason"]),
            }
            for row in rows
        ]
        digest = _digest(REVIEW_SCOPE_VERSION, _canonical_json(products))
        return {
            "version": REVIEW_SCOPE_VERSION,
            "digest": digest,
            "enabled_products": [
                str(row["product_code"]) for row in rows if int(row["enabled"])
            ],
            "held_products": [
                str(row["product_code"]) for row in rows if not int(row["enabled"])
            ],
            "products": products,
        }
    finally:
        conn.close()


def set_review_product_scope(
    workspace: Path | str,
    enabled_products: Sequence[str],
    *,
    held_reason: str = "maintainer-hold",
) -> dict[str, object]:
    """Atomically set semantic scheduling scope without deleting review work."""
    selected = tuple(sorted(set(enabled_products)))
    if not selected or len(selected) != len(tuple(enabled_products)):
        raise ValueError("review scope requires a non-empty unique product list")
    if any(product not in PRODUCT_CATALOG for product in selected):
        raise ValueError("review scope contains an unknown product")
    if held_reason not in REVIEW_SCOPE_HOLD_REASONS:
        raise ValueError("review scope requires a bounded hold reason")
    _ensure_workspace_migrated(workspace)
    conn = _connect(workspace)
    try:
        conn.execute("BEGIN IMMEDIATE")
        active = {
            str(row[0])
            for row in conn.execute(
                """SELECT product_code FROM parser_runs
                    WHERE state='active' AND review_enabled=1 ORDER BY product_code"""
            )
        }
        unknown = set(selected) - active
        if unknown:
            raise ValueError(
                "review scope products have no active trusted parser run: "
                + ", ".join(sorted(unknown))
            )
        now = int(time.time())
        live_claims = int(conn.execute(
            """SELECT
                (SELECT COUNT(*) FROM review_shards
                  WHERE claimant IS NOT NULL AND lease_expires_at>=?)
              + (SELECT COUNT(*) FROM review_claims WHERE lease_expires_at>=?)
              + (SELECT COUNT(*) FROM draft_screening_claims WHERE lease_expires_at>=?)
              + (SELECT COUNT(*) FROM stitch_claims WHERE lease_expires_at>=?)""",
            (now, now, now, now),
        ).fetchone()[0])
        if live_claims:
            raise ValueError("review scope cannot change while claims are live")
        for product in sorted(active):
            enabled = int(product in selected)
            conn.execute(
                """INSERT INTO review_product_scope
                   (product_code, enabled, reason, updated_at) VALUES (?, ?, ?, ?)
                   ON CONFLICT(product_code) DO UPDATE SET
                     enabled=excluded.enabled, reason=excluded.reason,
                     updated_at=excluded.updated_at""",
                (product, enabled, "enabled" if enabled else held_reason, now),
            )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()
    return review_product_scope(workspace)


def prepare_deterministic_review(
    workspace: Path | str,
    foundry_database: Path | str,
) -> dict[str, object]:
    """Refresh exact duplicate groups and bounded clean-Foundry candidates.

    The operation stores no normalized private text and never treats a
    Foundry match as a decision. Exact same-era/same-license PDF shadows are
    terminal deterministic duplicates; their canonical section alone enters
    semantic screening.
    """
    _ensure_workspace_migrated(workspace)
    snapshot = load_clean_foundry(foundry_database)
    matcher = FoundryMatcher(snapshot)
    conn = _connect(workspace)
    try:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            """SELECT s.*, r.license, r.era
                 FROM source_sections AS s
                 JOIN parser_runs AS p ON p.parser_run_id=s.parser_run_id
                 JOIN source_revisions AS r USING(product_code, content_fingerprint)
                WHERE p.state='active' AND p.review_enabled=1
                ORDER BY s.product_code, s.page_start, s.stable_identity, s.section_key"""
        ).fetchall()
        catalog_order = {code: ordinal for ordinal, code in enumerate(PRODUCT_CATALOG)}
        grouped: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            identity = duplicate_identity(
                heading=row["heading"], text=row["source_text"],
                license_name=row["license"], era=row["era"],
            )
            grouped.setdefault(identity, []).append(row)

        conn.execute("DELETE FROM duplicate_group_members")
        conn.execute("DELETE FROM duplicate_groups")
        now = int(time.time())
        canonical_keys: set[str] = set()
        shadow_count = 0
        for group_id, members in sorted(grouped.items()):
            members.sort(key=lambda row: (
                catalog_order.get(str(row["product_code"]), 999),
                int(row["page_start"] or 0), str(row["stable_identity"]),
                str(row["section_key"]),
            ))
            canonical = members[0]
            canonical_key = str(canonical["section_key"])
            canonical_keys.add(canonical_key)
            conn.execute(
                """INSERT INTO duplicate_groups
                   (group_id, normalizer_version, license, era, heading_hash,
                    body_hash, canonical_section_key, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    group_id, NORMALIZER_VERSION, canonical["license"], canonical["era"],
                    normalized_hash(canonical["heading"], heading=True),
                    normalized_hash(canonical["source_text"]), canonical_key, now,
                ),
            )
            for ordinal, member in enumerate(members):
                section_key = str(member["section_key"])
                conn.execute(
                    "INSERT INTO duplicate_group_members VALUES (?, ?, ?)",
                    (group_id, section_key, ordinal),
                )
                if ordinal == 0:
                    continue
                shadow_count += 1
                current = conn.execute(
                    """SELECT decision, duplicate_of_section_key FROM draft_screening_current
                       WHERE parser_run_id=? AND section_key=?""",
                    (member["parser_run_id"], section_key),
                ).fetchone()
                if current is not None and (
                    str(current["decision"]) != "REJECT"
                    or str(current["duplicate_of_section_key"] or "") != canonical_key
                ):
                    latest = conn.execute(
                        """SELECT event_id FROM draft_screening_events
                           WHERE parser_run_id=? AND section_key=? ORDER BY event_id DESC LIMIT 1""",
                        (member["parser_run_id"], section_key),
                    ).fetchone()
                    conn.execute(
                        """INSERT INTO draft_screening_events
                           (parser_run_id, section_key, event_type, worker, decided_at,
                            reopen_reason, supersedes_event_id)
                           VALUES (?, ?, 'REOPEN', 'deterministic-dedup', ?,
                                   'scope-correction', ?)""",
                        (member["parser_run_id"], section_key, now, latest["event_id"]),
                    )
                    current = None
                if current is None:
                    conn.execute(
                        """INSERT INTO draft_screening_events
                           (parser_run_id, section_key, event_type, requested_decision,
                            decision, duplicate_of_section_key, reject_reason, worker, decided_at)
                           VALUES (?, ?, 'DECISION', 'REJECT', 'REJECT', ?, 'duplicate',
                                   'deterministic-dedup', ?)""",
                        (member["parser_run_id"], section_key, canonical_key, now),
                    )
                    conn.execute(
                        """INSERT INTO runner_screen_rejections(section_key, reason, worker, decided_at)
                           VALUES (?, 'duplicate', 'deterministic-dedup', ?)
                           ON CONFLICT(section_key) DO UPDATE SET reason=excluded.reason,
                               worker=excluded.worker, decided_at=excluded.decided_at""",
                        (section_key, now),
                    )

        # A Foundry snapshot digest commits to the release and every stable
        # source/normalized row hash. Any terminal decision backed by a
        # different digest is therefore reopened before the new candidates
        # are installed. Historical confirmations remain audit evidence.
        shadow_keys = {
            str(row[0])
            for row in conn.execute(
                "SELECT section_key FROM duplicate_group_members WHERE source_ordinal>0"
            )
        }
        stale_sections = conn.execute(
            """SELECT DISTINCT confirmation.section_key, section.parser_run_id
                 FROM foundry_coverage_confirmations AS confirmation
                 JOIN source_sections AS section
                   ON section.section_key=confirmation.section_key
                 JOIN parser_runs AS run ON run.parser_run_id=section.parser_run_id
                 JOIN draft_screening_current AS current
                   ON current.parser_run_id=section.parser_run_id
                  AND current.section_key=section.section_key
                WHERE confirmation.snapshot_digest<>?
                  AND run.state='active' AND run.review_enabled=1
                ORDER BY confirmation.section_key""",
            (snapshot.digest,),
        ).fetchall()
        stale_reopened = 0
        for stale in stale_sections:
            section_key = str(stale["section_key"])
            if section_key in shadow_keys:
                continue
            latest = conn.execute(
                """SELECT event_id FROM draft_screening_events
                    WHERE parser_run_id=? AND section_key=?
                    ORDER BY event_id DESC LIMIT 1""",
                (stale["parser_run_id"], section_key),
            ).fetchone()
            if latest is None:
                continue
            conn.execute(
                """INSERT INTO draft_screening_events
                   (parser_run_id, section_key, event_type, worker, decided_at,
                    reopen_reason, supersedes_event_id)
                   VALUES (?, ?, 'REOPEN', 'deterministic-foundry-refresh', ?,
                           'scope-correction', ?)""",
                (stale["parser_run_id"], section_key, now, latest["event_id"]),
            )
            conn.execute(
                "DELETE FROM runner_screen_rejections WHERE section_key=?",
                (section_key,),
            )
            stale_reopened += 1

        conn.execute(
            """INSERT OR IGNORE INTO foundry_snapshots
               (snapshot_digest, pf2e_release, row_count, normalizer_version, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (snapshot.digest, snapshot.release, len(snapshot.rows), NORMALIZER_VERSION, now),
        )
        conn.executemany(
            """INSERT OR IGNORE INTO foundry_snapshot_rows
               (snapshot_digest, foundry_id, source_hash, normalized_hash, heading_hash,
                publication_title, license, era, entry_type)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                (
                    snapshot.digest, row.chunk_id, row.source_hash, row.normalized_hash,
                    row.heading_hash, row.publication_title, row.license, row.era, row.type,
                )
                for row in snapshot.rows
            ),
        )
        conn.execute(
            "DELETE FROM foundry_coverage_candidates WHERE snapshot_digest=?",
            (snapshot.digest,),
        )
        candidate_count = 0
        for row in rows:
            section_key = str(row["section_key"])
            if section_key not in canonical_keys:
                continue
            spec = PRODUCT_CATALOG[str(row["product_code"])]
            section = {
                **dict(row),
                "publication_title": spec.title,
            }
            for rank, candidate in enumerate(matcher.candidates(section)):
                proof_digest = _digest(
                    "foundry-coverage-candidate-v1", NORMALIZER_VERSION,
                    section_key, snapshot.digest, str(candidate["foundry_id"]),
                    normalized_hash(row["source_text"]),
                    str(candidate["normalized_hash"]),
                    _canonical_json(candidate["metrics"]),
                )
                conn.execute(
                    """INSERT INTO foundry_coverage_candidates
                       (section_key, snapshot_digest, candidate_rank, foundry_id,
                        proof_digest, metrics_json) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        section_key, snapshot.digest, rank, candidate["foundry_id"],
                        proof_digest, _canonical_json(candidate["metrics"]),
                    ),
                )
                candidate_count += 1
        conn.execute(
            """INSERT INTO metadata(key, value) VALUES ('active_foundry_snapshot', ?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
            (snapshot.digest,),
        )
        product_counts = []
        for product in sorted({str(row["product_code"]) for row in rows}):
            product_counts.append({
                "product_code": product,
                "canonical_sections": int(conn.execute(
                    """SELECT COUNT(*) FROM duplicate_groups AS groups_
                       JOIN source_sections AS section
                         ON section.section_key=groups_.canonical_section_key
                      WHERE section.product_code=?""",
                    (product,),
                ).fetchone()[0]),
                "shadow_duplicates": int(conn.execute(
                    """SELECT COUNT(*) FROM duplicate_group_members AS member
                       JOIN source_sections AS section ON section.section_key=member.section_key
                      WHERE member.source_ordinal>0 AND section.product_code=?""",
                    (product,),
                ).fetchone()[0]),
                "coverage_candidates": int(conn.execute(
                    """SELECT COUNT(*) FROM foundry_coverage_candidates AS candidate
                       JOIN source_sections AS section ON section.section_key=candidate.section_key
                      WHERE candidate.snapshot_digest=? AND section.product_code=?""",
                    (snapshot.digest, product),
                ).fetchone()[0]),
            })
        scope_by_product = {
            str(row["product_code"]): bool(row["enabled"])
            for row in _review_scope_rows(conn)
        }
        prepared_manifest = {
            "version": "deterministic-review-preparation-v1",
            "normalizer_version": NORMALIZER_VERSION,
            "snapshot_digest": snapshot.digest,
            "duplicate_groups": [
                tuple(row) for row in conn.execute(
                    """SELECT group_id, canonical_section_key FROM duplicate_groups
                       ORDER BY group_id"""
                )
            ],
            "duplicate_members": [
                tuple(row) for row in conn.execute(
                    """SELECT group_id, section_key, source_ordinal
                       FROM duplicate_group_members
                       ORDER BY group_id, source_ordinal, section_key"""
                )
            ],
            "coverage_candidates": [
                tuple(row) for row in conn.execute(
                    """SELECT section_key, candidate_rank, foundry_id, proof_digest
                       FROM foundry_coverage_candidates WHERE snapshot_digest=?
                       ORDER BY section_key, candidate_rank, foundry_id""",
                    (snapshot.digest,),
                )
            ],
        }
        preparation_digest = _digest(_canonical_json(prepared_manifest))
        conn.commit()
        return {
            "preparation_version": "deterministic-review-preparation-v1",
            "preparation_digest": preparation_digest,
            "normalizer_version": NORMALIZER_VERSION,
            "foundry_release": snapshot.release,
            "snapshot_digest": snapshot.digest,
            "foundry_rows": len(snapshot.rows),
            "duplicate_groups": len(grouped),
            "shadow_duplicates": shadow_count,
            "canonical_sections": len(canonical_keys),
            "coverage_candidates": candidate_count,
            "stale_coverage_reopened": stale_reopened,
            "products": [
                {
                    **counts,
                    "scope_state": (
                        "enabled" if scope_by_product.get(str(counts["product_code"]), False)
                        else "held"
                    ),
                }
                for counts in product_counts
            ],
        }
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def foundry_coverage_evidence(
    workspace: Path | str,
    foundry_database: Path | str,
    section_keys: Sequence[str],
) -> dict[str, list[dict[str, object]]]:
    """Return bounded public Foundry text for prepared claimed sections."""
    snapshot = load_clean_foundry(foundry_database)
    return _foundry_coverage_evidence_for_snapshot(workspace, snapshot, section_keys)


def _foundry_coverage_evidence_for_snapshot(
    workspace: Path | str,
    snapshot: object,
    section_keys: Sequence[str],
) -> dict[str, list[dict[str, object]]]:
    """Internal shared serializer for live claims and read-only previews."""
    rows_by_id = {row.chunk_id: row for row in snapshot.rows}
    conn = _connect(workspace, readonly=True)
    try:
        active = conn.execute(
            "SELECT value FROM metadata WHERE key='active_foundry_snapshot'"
        ).fetchone()
        if active is None or str(active[0]) != snapshot.digest:
            raise ValueError("Foundry coverage snapshot is stale or unprepared")
        result: dict[str, list[dict[str, object]]] = {}
        for section_key in section_keys:
            prepared = conn.execute(
                """SELECT * FROM foundry_coverage_candidates
                   WHERE section_key=? AND snapshot_digest=? ORDER BY candidate_rank""",
                (section_key, snapshot.digest),
            ).fetchall()
            values = []
            for item in prepared:
                row = rows_by_id[str(item["foundry_id"])]
                values.append({
                    "id": row.chunk_id,
                    "name": row.name,
                    "type": row.type,
                    "publication_title": row.publication_title,
                    "license": row.license,
                    "era": row.era,
                    "text": row.text,
                    "metrics": json.loads(str(item["metrics_json"])),
                })
            result[section_key] = values
        return result
    finally:
        conn.close()


def _review_target_sql(section_alias: str = "s") -> str:
    """Restrict semantic work to active parser runs in the configured scope."""
    return f"""EXISTS (
        SELECT 1 FROM parser_runs AS parser_run
        WHERE parser_run.parser_run_id={section_alias}.parser_run_id
          AND parser_run.state='active' AND parser_run.review_enabled=1
          AND {_semantic_scope_sql('parser_run.product_code')}
    )"""


def _require_unambiguous_targets(conn: sqlite3.Connection) -> None:
    """Fail closed if a migration/manual edit leaves a product ambiguous."""
    rows = conn.execute(
        """SELECT product_code, COUNT(*) AS count FROM parser_runs
        WHERE state='active' AND review_enabled=1 GROUP BY product_code HAVING COUNT(*) != 1"""
    ).fetchall()
    if rows:
        raise ValueError("ambiguous active parser-run review target")
    missing = conn.execute(
        """SELECT DISTINCT product_code FROM source_sections AS s
        WHERE NOT EXISTS (SELECT 1 FROM parser_runs AS p
                          WHERE p.parser_run_id=s.parser_run_id
                            AND p.state='active' AND p.review_enabled=1)
          AND NOT EXISTS (SELECT 1 FROM parser_runs AS p2
                          WHERE p2.product_code=s.product_code
                            AND p2.state='active' AND p2.review_enabled=1)
        LIMIT 1"""
    ).fetchone()
    if missing is not None:
        raise ValueError("parser-run workspace has no active review target")


def _insert_evidence(
    conn: sqlite3.Connection,
    evidence: object,
    *,
    candidate_id: str | None = None,
    review_id: str | None = None,
) -> None:
    """Persist optional human corroboration without making it a policy decision."""
    if evidence is None:
        return
    if not isinstance(evidence, list):
        raise ValueError("evidence must be a list of objects")
    for ordinal, item in enumerate(evidence, start=1):
        if not isinstance(item, Mapping):
            raise ValueError("every evidence record must be an object")
        kind = item.get("evidence_kind")
        status = item.get("status")
        if not isinstance(kind, str) or not kind:
            raise ValueError("evidence_kind is required")
        if status not in {"match", "no_match", "uncertain"}:
            raise ValueError("evidence status must be match, no_match, or uncertain")
        url = item.get("url")
        note = item.get("note")
        if url is not None and not isinstance(url, str):
            raise ValueError("evidence URL must be a string")
        if note is not None and not isinstance(note, str):
            raise ValueError("evidence note must be a string")
        checked_at = item.get("checked_at", int(time.time()))
        if not isinstance(checked_at, int):
            raise ValueError("evidence checked_at must be an integer timestamp")
        owner = candidate_id or review_id
        assert owner is not None
        evidence_id = "evidence:" + _digest(owner, str(ordinal), kind, str(status), url or "", note or "")
        conn.execute(
            "INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (evidence_id, candidate_id, review_id, kind, status, url, checked_at, note),
        )


def initialize_workspace(
    workspace: Path | str,
    source_database: Path | str,
    *,
    shard_size: int = 100,
) -> dict[str, int]:
    """Create a private review database from corpus-owned local-full sections."""
    if shard_size < 1:
        raise ValueError("shard_size must be at least one")
    workspace_path = Path(workspace).expanduser().resolve()
    if workspace_path.exists():
        raise FileExistsError(f"review workspace already exists: {workspace_path}")

    source = _connect(source_database, readonly=True)
    try:
        tables = {row[0] for row in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"_meta", "chunks", "sources"}.issubset(tables):
            raise ValueError("source database is not a corpus-capable PF2E index")
        scope_row = source.execute(
            "SELECT value FROM _meta WHERE key = 'distribution_scope'"
        ).fetchone()
        if not scope_row or scope_row[0] != "local-full":
            raise ValueError("source database must be a local-full private corpus database")
        source_rows = source.execute(
            """
            SELECT c.id, c.section_hash, c.text, c.name, c.source_page_start,
                   c.source_page_end, c.printed_page, c.license, c.source_id,
                   s.product, s.revision, s.parser, s.license AS source_license,
                   s.era, s.provenance
            FROM chunks AS c
            JOIN sources AS s ON s.source_id = c.source_id
            WHERE c.origin = 'corpus'
            ORDER BY c.source_id, c.id
            """
        ).fetchall()
    finally:
        source.close()
    if not source_rows:
        raise ValueError("source database has no corpus-owned sections")

    parsed: list[dict[str, object]] = []
    revisions: dict[tuple[str, str], dict[str, str | None]] = {}
    for row in source_rows:
        product = _source_product_code(str(row["source_id"]), row["product"])
        fingerprint = _fingerprint(row)
        license_name = str(row["source_license"] or row["license"] or "")
        if license_name not in {"OGL", "ORC"}:
            raise ValueError(f"corpus source {product} has unsupported license {license_name!r}")
        parser_version = str(row["parser"] or "")
        if not parser_version:
            raise ValueError(f"corpus source {product} is missing parser provenance")
        raw_provenance = row["provenance"]
        try:
            provenance = json.loads(raw_provenance) if raw_provenance else {}
        except json.JSONDecodeError as exc:
            raise ValueError("corpus source provenance is invalid JSON") from exc
        schema_version = provenance.get("export_schema_version")
        revision_key = (product, fingerprint)
        revision = {
            "license": license_name,
            "era": str(row["era"] or "unknown"),
            "parser_version": parser_version,
            "source_schema_version": str(schema_version) if schema_version is not None else None,
            "printing_revision": f"legacy-import-{fingerprint[:16]}",
        }
        old_revision = revisions.setdefault(revision_key, revision)
        if old_revision != revision:
            raise ValueError(f"inconsistent provenance for {product} revision {fingerprint}")
        section_hash = str(row["section_hash"] or "")
        if not section_hash:
            raise ValueError(f"corpus section {row['id']!r} is missing its section hash")
        parsed.append(
            {
                "product": product,
                "fingerprint": fingerprint,
                "id": str(row["id"]),
                "hash": section_hash,
                "text": str(row["text"]),
                "heading": str(row["name"] or ""),
                "page_start": row["source_page_start"],
                "page_end": row["source_page_end"],
                "printed_page": row["printed_page"],
            }
        )

    conn = _connect(workspace_path)
    try:
        _init_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        conn.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("review_schema_version", str(REVIEW_SCHEMA_VERSION)),
                ("workspace_scope", "private-review"),
            ],
        )
        for (product, fingerprint), revision in sorted(revisions.items()):
            conn.execute(
                """INSERT INTO source_revisions
                   (product_code, content_fingerprint, license, era, parser_version,
                    source_schema_version, printing_revision)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    product,
                    fingerprint,
                    revision["license"],
                    revision["era"],
                    revision["parser_version"],
                    revision["source_schema_version"],
                    revision["printing_revision"],
                ),
            )
        shard_count = 0
        for revision_key in sorted(revisions):
            product, fingerprint = revision_key
            sections = sorted(
                (item for item in parsed if (item["product"], item["fingerprint"]) == revision_key),
                key=lambda item: str(item["id"]),
            )
            for shard_ordinal, offset in enumerate(range(0, len(sections), shard_size)):
                shard = sections[offset : offset + shard_size]
                cursor = conn.execute(
                    """INSERT INTO review_shards
                    (product_code, content_fingerprint, shard_ordinal, section_count)
                    VALUES (?, ?, ?, ?)""",
                    (product, fingerprint, shard_ordinal, len(shard)),
                )
                shard_id = int(cursor.lastrowid)
                shard_count += 1
                for item in shard:
                    section_key = _digest(
                        product, fingerprint, str(item["id"]), str(item["hash"])
                    )
                    conn.execute(
                        """INSERT INTO source_sections VALUES
                        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            section_key,
                            product,
                            fingerprint,
                            item["id"],
                            item["hash"],
                            item["page_start"],
                            item["page_end"],
                            item["printed_page"],
                            item["heading"],
                            item["text"],
                            shard_id,
                        ),
                    )
        conn.commit()
    except BaseException:
        conn.rollback()
        conn.close()
        workspace_path.unlink(missing_ok=True)
        raise
    else:
        conn.close()
    # Populate the v4 parser-run seam after the legacy-shaped import.  Doing
    # this as a normal migration keeps historical fixture/workspace creation
    # byte-for-byte compatible with the V3 import path.
    _ensure_workspace_migrated(workspace_path)
    return {"revisions": len(revisions), "sections": len(parsed), "shards": shard_count}


def initialize_trusted_workspace(workspace: Path | str) -> dict[str, int]:
    """Create an empty private workspace for direct-PDF trusted staging.

    The deterministic runner uses this entry point so a fresh review never
    depends on a legacy ``local-full`` embedding database.  The workspace is
    intentionally empty until sealed direct-PDF parser runs are staged and
    activated.
    """
    workspace_path = Path(workspace).expanduser().resolve()
    if workspace_path.exists():
        raise FileExistsError(f"review workspace already exists: {workspace_path}")
    conn = _connect(workspace_path)
    try:
        _init_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        conn.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("review_schema_version", str(REVIEW_SCHEMA_VERSION)),
                ("workspace_scope", "private-review"),
                ("workspace_origin", "trusted-direct-pdf"),
            ],
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        conn.close()
        workspace_path.unlink(missing_ok=True)
        raise
    else:
        conn.close()
    _ensure_workspace_migrated(workspace_path)
    return {"revisions": 0, "sections": 0, "shards": 0}


_PENDING_SECTION_SQL = f"""
    NOT EXISTS (SELECT 1 FROM candidates AS c WHERE c.section_key = s.section_key)
    OR EXISTS (
        SELECT 1
        FROM candidates AS latest
        JOIN reviews AS latest_review ON latest_review.candidate_id = latest.candidate_id
        WHERE latest.section_key = s.section_key
          AND latest.candidate_ordinal = (
              SELECT MAX(c2.candidate_ordinal)
              FROM candidates AS c2 WHERE c2.section_key = s.section_key
          )
          AND latest_review.verdict = 'REVISE'
          AND ({_active_review_sql('latest_review')})
    )
"""


def _claimable_shard_sql(shard_table: str) -> str:
    """Return the fail-closed predicate for a new or rework shard claim.

    A lease can expire while a worker has only submitted part of a shard.  That
    shard is no longer pristine, though: treating its remaining undecided
    sections as a fresh assignment lets another worker resubmit decisions for
    the already-decided sections.  Keep those interrupted mixed shards out of
    ordinary scheduling.  They require explicit reconciliation instead.

    A fully reviewed ``REVISE`` is the one deliberate exception.  In that
    case every section has a reviewed current decision and the shard can be
    claimed for the replacement candidate(s) only.
    """
    return f"""
        (
            NOT EXISTS (
                SELECT 1
                FROM source_sections AS existing_section
                JOIN candidates AS existing_candidate
                  ON existing_candidate.section_key = existing_section.section_key
                WHERE existing_section.shard_id = {shard_table}.shard_id
            )
            OR (
                EXISTS (
                    SELECT 1
                    FROM source_sections AS revised_section
                    JOIN candidates AS revised_candidate
                      ON revised_candidate.section_key = revised_section.section_key
                    JOIN reviews AS revised_review
                      ON revised_review.candidate_id = revised_candidate.candidate_id
                    WHERE revised_section.shard_id = {shard_table}.shard_id
                      AND revised_candidate.candidate_ordinal = (
                          SELECT MAX(candidate_ordinal)
                          FROM candidates AS latest_candidate
                          WHERE latest_candidate.section_key = revised_section.section_key
                      )
                      AND revised_review.verdict = 'REVISE'
                      AND ({_active_review_sql('revised_review')})
                )
                AND NOT EXISTS (
                    SELECT 1
                    FROM source_sections AS unresolved_section
                    WHERE unresolved_section.shard_id = {shard_table}.shard_id
                      AND NOT EXISTS (
                          SELECT 1
                          FROM candidates AS current_candidate
                          JOIN reviews AS current_review
                            ON current_review.candidate_id = current_candidate.candidate_id
                          WHERE current_candidate.section_key = unresolved_section.section_key
                            AND current_candidate.candidate_ordinal = (
                                SELECT MAX(candidate_ordinal)
                                FROM candidates AS latest_candidate
                                WHERE latest_candidate.section_key = unresolved_section.section_key
                            )
                          AND ({_active_review_sql('current_review')})
                      )
                )
            )
        )
    """


def _release_shard_if_complete(conn: sqlite3.Connection, shard_id: int) -> None:
    pending = conn.execute(
        f"""SELECT 1 FROM source_sections AS s
        WHERE s.shard_id=? AND ({_PENDING_SECTION_SQL}) LIMIT 1""",
        (shard_id,),
    ).fetchone()
    if pending is None:
        conn.execute(
            """UPDATE review_shards
            SET claimant=NULL, claimed_at=NULL, lease_expires_at=NULL, claim_mode=NULL
            WHERE shard_id=?""",
            (shard_id,),
        )


def workspace_status(workspace: Path | str) -> dict[str, object]:
    """Return aggregate, non-content workspace state suitable for coordination."""
    _ensure_workspace_migrated(workspace)
    conn = _connect(workspace, readonly=True)
    try:
        _require_unambiguous_targets(conn)
        now = int(time.time())
        products = []
        for (product_code,) in conn.execute(
            """SELECT DISTINCT s.product_code FROM source_sections AS s
            WHERE """ + _review_target_sql("s") + " ORDER BY s.product_code"
        ):
            sections = int(conn.execute(
                "SELECT COUNT(*) FROM source_sections AS s WHERE product_code=? AND "
                + _review_target_sql("s"), (product_code,)
            ).fetchone()[0])
            available_shards = int(conn.execute(
                f"""SELECT COUNT(*) FROM review_shards AS sh
                WHERE sh.product_code=?
                  AND EXISTS (SELECT 1 FROM parser_runs AS p WHERE p.parser_run_id=sh.parser_run_id
                              AND p.state='active' AND p.review_enabled=1)
                  AND (sh.claimant IS NULL OR sh.lease_expires_at < ?)
                  AND ({_claimable_shard_sql('sh')})""",
                (product_code, now),
            ).fetchone()[0])
            candidates = int(conn.execute(
                """SELECT COUNT(*) FROM candidates AS c
                JOIN source_sections AS s ON s.section_key=c.section_key
                WHERE s.product_code=? AND """ + _review_target_sql("s"), (product_code,)
            ).fetchone()[0])
            reviews = int(conn.execute(
                """SELECT COUNT(*) FROM reviews AS r
                JOIN candidates AS c ON c.candidate_id=r.candidate_id
                JOIN source_sections AS s ON s.section_key=c.section_key
                WHERE s.product_code=? AND """ + _review_target_sql("s")
                + f" AND ({_active_review_sql('r')})", (product_code,)
            ).fetchone()[0])
            invalidated_reviews = int(conn.execute(
                """SELECT COUNT(*) FROM review_invalidations AS invalidation
                JOIN reviews AS r ON r.review_id=invalidation.review_id
                JOIN candidates AS c ON c.candidate_id=r.candidate_id
                JOIN source_sections AS s ON s.section_key=c.section_key
                WHERE s.product_code=? AND """ + _review_target_sql("s"), (product_code,)
            ).fetchone()[0])
            available_reviews = int(conn.execute(
                """SELECT COUNT(*) FROM candidates AS c
                JOIN source_sections AS s ON s.section_key=c.section_key
                LEFT JOIN reviews AS r ON r.candidate_id=c.candidate_id
                  AND (""" + _active_review_sql("r") + """)
                LEFT JOIN review_claims AS rc ON rc.candidate_id=c.candidate_id
                WHERE s.product_code=? AND """ + _review_target_sql("s") + """ AND r.review_id IS NULL
                  AND c.candidate_ordinal=(
                      SELECT MAX(c2.candidate_ordinal) FROM candidates AS c2
                      WHERE c2.section_key=c.section_key
                  )
                  AND (rc.candidate_id IS NULL OR rc.lease_expires_at < ?)""",
                (product_code, now),
            ).fetchone()[0])
            products.append({
                "product_code": product_code,
                "sections": sections,
                "available_shards": available_shards,
                "candidates": candidates,
                "reviews": reviews,
                "invalidated_reviews": invalidated_reviews,
                "available_reviews": available_reviews,
            })
        return {"products": products, "workspace_scope": "private-review"}
    finally:
        conn.close()


def claim_shard(
    workspace: Path | str,
    claimant: str,
    *,
    lease_seconds: int = 3600,
    preferred_shard_id: int | None = None,
) -> dict[str, object] | None:
    """Atomically assign one deterministic or explicitly targeted shard."""
    if not claimant or lease_seconds < 1 or (preferred_shard_id is not None and preferred_shard_id < 1):
        raise ValueError("claimant, positive lease/shard values are required")
    now = int(time.time())
    conn = _connect(workspace)
    try:
        _migrate_review_workspace(conn)
        conn.execute("BEGIN IMMEDIATE")
        _require_unambiguous_targets(conn)
        row = conn.execute(
            f"""SELECT * FROM review_shards
            WHERE claimant = ? AND lease_expires_at >= ?
              AND EXISTS (SELECT 1 FROM parser_runs AS p WHERE p.parser_run_id=review_shards.parser_run_id
                          AND p.state='active' AND p.review_enabled=1
                          AND {_semantic_scope_sql('p.product_code')})
              AND EXISTS (
                  SELECT 1 FROM source_sections AS s
                  WHERE s.shard_id=review_shards.shard_id AND ({_PENDING_SECTION_SQL})
              )
            ORDER BY shard_id LIMIT 1""",
            (claimant, now),
        ).fetchone()
        if row is not None and (
            preferred_shard_id is not None and row["shard_id"] != preferred_shard_id
        ):
            raise ValueError(
                f"claimant already holds pending shard {row['shard_id']}; "
                "finish it before targeting another shard"
            )
        if row is None:
            target_clause = "AND shard_id = ?" if preferred_shard_id is not None else ""
            parameters: tuple[object, ...] = (
                (now, preferred_shard_id)
                if preferred_shard_id is not None
                else (now,)
            )
            row = conn.execute(
                f"""SELECT review_shards.*,
                    CASE WHEN NOT EXISTS (
                        SELECT 1 FROM source_sections AS pristine_section
                        JOIN candidates AS pristine_candidate
                          ON pristine_candidate.section_key=pristine_section.section_key
                        WHERE pristine_section.shard_id=review_shards.shard_id
                    ) THEN 'ordinary' ELSE 'rework' END AS next_claim_mode
                FROM review_shards
                WHERE (claimant IS NULL OR lease_expires_at < ?)
                  AND EXISTS (SELECT 1 FROM parser_runs AS p WHERE p.parser_run_id=review_shards.parser_run_id
                              AND p.state='active' AND p.review_enabled=1
                              AND {_semantic_scope_sql('p.product_code')})
                  {target_clause}
                  AND ({_claimable_shard_sql('review_shards')})
                ORDER BY product_code, content_fingerprint, shard_ordinal, shard_id LIMIT 1""",
                parameters,
            ).fetchone()
            if row is None:
                conn.commit()
                if preferred_shard_id is not None:
                    raise ValueError(
                        f"requested shard {preferred_shard_id} is unavailable or complete"
                    )
                return None
            expiry = now + lease_seconds
            claim_mode = str(row["next_claim_mode"])
            conn.execute(
                """UPDATE review_shards
                SET claimant=?, claimed_at=?, lease_expires_at=?, claim_mode=?
                WHERE shard_id=?""",
                (claimant, now, expiry, claim_mode, row["shard_id"]),
            )
        else:
            expiry = int(row["lease_expires_at"])
            claim_mode = str(row["claim_mode"] or "")
            if claim_mode not in CLAIM_MODES:
                raise RuntimeError("active shard claim is missing a persisted claim mode")
        conn.commit()
        return {
            "shard_id": int(row["shard_id"]),
            "product_code": str(row["product_code"]),
            "section_count": int(row["section_count"]),
            "claimant": claimant,
            "lease_expires_at": expiry,
            "claim_mode": claim_mode,
        }
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def reclaim_interrupted_shard(
    workspace: Path | str,
    shard_id: int,
    claimant: str,
    *,
    lease_seconds: int = 3600,
) -> dict[str, object]:
    """Explicitly resume only the missing records of a partial ordinary shard.

    This is the deterministic runner's crash-recovery seam. Existing candidate
    rows remain immutable and ``submit_candidate`` continues to reject any
    attempt to overwrite them.
    """
    if shard_id < 1 or not claimant or lease_seconds < 1:
        raise ValueError("positive shard, claimant, and lease are required")
    now = int(time.time())
    conn = _connect(workspace)
    try:
        _migrate_review_workspace(conn)
        conn.execute("BEGIN IMMEDIATE")
        shard = conn.execute(
            f"""SELECT sh.* FROM review_shards AS sh
               JOIN parser_runs AS p ON p.parser_run_id=sh.parser_run_id
               WHERE sh.shard_id=? AND p.state='active' AND p.review_enabled=1
                 AND {_semantic_scope_sql('p.product_code')}""",
            (shard_id,),
        ).fetchone()
        if shard is None:
            raise ValueError("interrupted shard is not an active review target")
        if shard["claimant"] is not None and int(shard["lease_expires_at"] or 0) >= now:
            raise PermissionError("interrupted shard still has a live claim")
        counts = conn.execute(
            """SELECT COUNT(*) AS total,
                      SUM(CASE WHEN EXISTS(SELECT 1 FROM candidates AS c
                                           WHERE c.section_key=s.section_key) THEN 1 ELSE 0 END) AS decided
               FROM source_sections AS s WHERE s.shard_id=?""",
            (shard_id,),
        ).fetchone()
        total = int(counts["total"] or 0)
        decided = int(counts["decided"] or 0)
        if not (0 < decided < total):
            raise ValueError("shard is not a partial ordinary submission")
        expiry = now + lease_seconds
        conn.execute(
            """UPDATE review_shards SET claimant=?, claimed_at=?, lease_expires_at=?, claim_mode='ordinary'
               WHERE shard_id=?""",
            (claimant, now, expiry, shard_id),
        )
        conn.commit()
        return {
            "shard_id": shard_id,
            "product_code": str(shard["product_code"]),
            "section_count": total - decided,
            "claimant": claimant,
            "lease_expires_at": expiry,
            "claim_mode": "ordinary",
        }
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def read_claimed_shard(workspace: Path | str, shard_id: int, claimant: str) -> list[dict[str, object]]:
    """Explicit private read of a currently claimed shard, including source text."""
    conn = _connect(workspace, readonly=True)
    try:
        _require_unambiguous_targets(conn)
        now = int(time.time())
        claim = conn.execute(
            f"""SELECT 1 FROM review_shards
            WHERE shard_id=? AND claimant=? AND lease_expires_at >= ?
              AND EXISTS (SELECT 1 FROM parser_runs AS p WHERE p.parser_run_id=review_shards.parser_run_id
                          AND p.state='active' AND p.review_enabled=1
                          AND {_semantic_scope_sql('p.product_code')})""",
            (shard_id, claimant, now),
        ).fetchone()
        if claim is None:
            raise PermissionError("shard is not currently claimed by this worker")
        return [
            dict(row)
            for row in conn.execute(
                """SELECT section_key, product_code, content_fingerprint, source_section_id,
                           source_section_hash, page_start, page_end, printed_page, heading, source_text,
                           stable_identity, provenance_hash, text_hash
                    FROM source_sections WHERE shard_id=? ORDER BY source_section_id""",
                (shard_id,),
            )
        ]
    finally:
        conn.close()


def _screening_run_sql(alias: str) -> str:
    return (
        f"{alias}.state='active' AND {alias}.review_enabled=1 "
        f"AND {alias}.complete=1 AND {alias}.origin='{_TRUSTED_RUN_ORIGIN}' "
        f"AND {_semantic_scope_sql(f'{alias}.product_code')}"
    )


def _screening_latest_event(
    conn: sqlite3.Connection, run_id: str, section_key: str
) -> sqlite3.Row | None:
    """Return the newest screening event for one parser-run section."""
    return conn.execute(
        """SELECT * FROM draft_screening_events
           WHERE parser_run_id=? AND section_key=?
           ORDER BY event_id DESC LIMIT 1""",
        (run_id, section_key),
    ).fetchone()


def draft_screening_status(workspace: Path | str) -> dict[str, object]:
    """Return aggregate state for the private quad-state draft screen."""
    _ensure_workspace_migrated(workspace)
    conn = _connect(workspace, readonly=True)
    try:
        now = int(time.time())
        products: list[dict[str, object]] = []
        for run in conn.execute(
            "SELECT parser_run_id, product_code FROM parser_runs AS p WHERE "
            + _screening_run_sql("p")
            + " ORDER BY product_code"
        ):
            run_id = str(run["parser_run_id"])
            counts = conn.execute(
                """SELECT COUNT(*) AS sections,
                          SUM(CASE WHEN d.decision='ADD' THEN 1 ELSE 0 END) AS accepted,
                          SUM(CASE WHEN d.decision='REJECT' THEN 1 ELSE 0 END) AS rejected,
                          SUM(CASE WHEN d.decision='DEFER' THEN 1 ELSE 0 END) AS deferred,
                          SUM(CASE WHEN d.duplicate_of_section_key IS NOT NULL THEN 1 ELSE 0 END)
                              AS duplicate_rejected
                   FROM parser_run_sections AS membership
                   LEFT JOIN draft_screening_current AS d
                     ON d.parser_run_id=membership.parser_run_id
                    AND d.section_key=membership.section_key
                   WHERE membership.parser_run_id=?
                     AND membership.membership_state='present'""",
                (run_id,),
            ).fetchone()
            sections = int(counts["sections"] or 0)
            accepted = int(counts["accepted"] or 0)
            rejected = int(counts["rejected"] or 0)
            deferred = int(counts["deferred"] or 0)
            unprocessed = sections - accepted - rejected - deferred
            available_batches = int(
                conn.execute(
                    """SELECT COUNT(*) FROM review_shards AS sh
                       WHERE sh.parser_run_id=?
                         AND NOT EXISTS (
                             SELECT 1 FROM draft_screening_claims AS claim
                             WHERE claim.parser_run_id=sh.parser_run_id
                               AND claim.shard_id=sh.shard_id
                               AND claim.lease_expires_at >= ?
                         )
                         AND EXISTS (
                             SELECT 1 FROM source_sections AS s
                             WHERE s.shard_id=sh.shard_id
                               AND NOT EXISTS (
                                   SELECT 1 FROM draft_screening_current AS d
                                   WHERE d.parser_run_id=sh.parser_run_id
                                     AND d.section_key=s.section_key
                               )
                         )""",
                    (run_id, now),
                ).fetchone()[0]
            )
            deferred_batches = int(
                conn.execute(
                    """SELECT COUNT(*) FROM review_shards AS sh
                       WHERE sh.parser_run_id=?
                         AND NOT EXISTS (
                             SELECT 1 FROM draft_screening_claims AS claim
                             WHERE claim.parser_run_id=sh.parser_run_id
                               AND claim.shard_id=sh.shard_id
                               AND claim.lease_expires_at >= ?
                         )
                         AND EXISTS (
                             SELECT 1 FROM source_sections AS s
                             JOIN draft_screening_current AS d
                               ON d.parser_run_id=sh.parser_run_id
                              AND d.section_key=s.section_key
                             WHERE s.shard_id=sh.shard_id AND d.decision='DEFER'
                         )""",
                    (run_id, now),
                ).fetchone()[0]
            )
            live_claims = int(
                conn.execute(
                    """SELECT COUNT(*) FROM draft_screening_claims
                       WHERE parser_run_id=? AND lease_expires_at >= ?""",
                    (run_id, now),
                ).fetchone()[0]
            )
            products.append(
                {
                    "product_code": str(run["product_code"]),
                    "sections": sections,
                    "unprocessed": unprocessed,
                    "accepted": accepted,
                    "rejected": rejected,
                    "deferred": deferred,
                    "duplicate_rejected": int(counts["duplicate_rejected"] or 0),
                    "unprocessed_batches": available_batches,
                    "deferred_batches": deferred_batches,
                    "live_claims": live_claims,
                }
            )
        return {"products": products, "workspace_scope": "private-draft-screen"}
    finally:
        conn.close()


def claim_draft_screening_batch(
    workspace: Path | str,
    claimant: str,
    *,
    product_code: str | None = None,
    lease_seconds: int = 3600,
    preferred_shard_id: int | None = None,
    queue: str = "unprocessed",
) -> dict[str, object] | None:
    """Atomically claim one unprocessed or deferred active-run batch."""
    if (
        not claimant
        or lease_seconds < 1
        or (product_code is not None and product_code not in PRODUCT_CATALOG)
        or (preferred_shard_id is not None and preferred_shard_id < 1)
        or queue not in {"unprocessed", "deferred"}
    ):
        raise ValueError("valid claimant, product, lease, shard, and queue values are required")
    claim_mode = "ordinary" if queue == "unprocessed" else "escalation"
    now = int(time.time())
    conn = _connect(workspace)
    try:
        _migrate_review_workspace(conn)
        conn.execute("BEGIN IMMEDIATE")
        active = conn.execute(
            """SELECT claim.*, sh.product_code, sh.section_count
               FROM draft_screening_claims AS claim
               JOIN review_shards AS sh ON sh.shard_id=claim.shard_id
               JOIN parser_runs AS p ON p.parser_run_id=claim.parser_run_id
               WHERE claim.claimant=? AND claim.lease_expires_at >= ? AND """
            + _screening_run_sql("p")
            + " ORDER BY claim.shard_id LIMIT 1",
            (claimant, now),
        ).fetchone()
        if active is not None:
            if product_code is not None and str(active["product_code"]) != product_code:
                raise ValueError(
                    f"claimant already holds a screening batch for {active['product_code']}"
                )
            if str(active["claim_mode"]) != claim_mode:
                raise ValueError(
                    f"claimant already holds a {active['claim_mode']} screening batch"
                )
            if preferred_shard_id is not None and int(active["shard_id"]) != preferred_shard_id:
                raise ValueError(
                    f"claimant already holds screening batch {active['shard_id']}"
                )
            active_eligibility = (
                "NOT EXISTS (SELECT 1 FROM draft_screening_current AS d "
                "WHERE d.parser_run_id=? AND d.section_key=s.section_key)"
                if queue == "unprocessed"
                else "EXISTS (SELECT 1 FROM draft_screening_current AS d "
                "WHERE d.parser_run_id=? AND d.section_key=s.section_key "
                "AND d.decision='DEFER')"
            )
            eligible_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM source_sections AS s WHERE s.shard_id=? AND "
                    + active_eligibility,
                    (active["shard_id"], active["parser_run_id"]),
                ).fetchone()[0]
            )
            conn.commit()
            return {
                "shard_id": int(active["shard_id"]),
                "product_code": str(active["product_code"]),
                "section_count": int(active["section_count"]),
                "claimant": claimant,
                "queue": queue,
                "eligible_count": eligible_count,
                "lease_expires_at": int(active["lease_expires_at"]),
            }

        eligibility = (
            "NOT EXISTS (SELECT 1 FROM draft_screening_current AS d "
            "WHERE d.parser_run_id=sh.parser_run_id AND d.section_key=s.section_key)"
            if queue == "unprocessed"
            else "EXISTS (SELECT 1 FROM draft_screening_current AS d "
            "WHERE d.parser_run_id=sh.parser_run_id AND d.section_key=s.section_key "
            "AND d.decision='DEFER')"
        )
        clauses = [
            _screening_run_sql("p"),
            "NOT EXISTS (SELECT 1 FROM draft_screening_claims AS claim "
            "WHERE claim.parser_run_id=sh.parser_run_id AND claim.shard_id=sh.shard_id "
            "AND claim.lease_expires_at >= ?)",
            "EXISTS (SELECT 1 FROM source_sections AS s WHERE s.shard_id=sh.shard_id AND "
            + eligibility
            + ")",
        ]
        parameters: list[object] = [now]
        if product_code is not None:
            clauses.append("p.product_code=?")
            parameters.append(product_code)
        if preferred_shard_id is not None:
            clauses.append("sh.shard_id=?")
            parameters.append(preferred_shard_id)
        row = conn.execute(
            """SELECT sh.*, p.parser_run_id FROM review_shards AS sh
               JOIN parser_runs AS p ON p.parser_run_id=sh.parser_run_id
               WHERE """
            + " AND ".join(clauses)
            + " ORDER BY p.product_code, sh.shard_ordinal, sh.shard_id LIMIT 1",
            tuple(parameters),
        ).fetchone()
        if row is None:
            conn.commit()
            if preferred_shard_id is not None:
                raise ValueError(
                    f"requested screening batch {preferred_shard_id} is unavailable or complete"
                )
            return None
        expiry = now + lease_seconds
        conn.execute(
            """INSERT INTO draft_screening_claims
               (parser_run_id, shard_id, claimant, claimed_at, lease_expires_at, claim_mode)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(parser_run_id, shard_id) DO UPDATE SET
                   claimant=excluded.claimant,
                   claimed_at=excluded.claimed_at,
                   lease_expires_at=excluded.lease_expires_at,
                   claim_mode=excluded.claim_mode""",
            (row["parser_run_id"], row["shard_id"], claimant, now, expiry, claim_mode),
        )
        count_eligibility = (
            "NOT EXISTS (SELECT 1 FROM draft_screening_current AS d "
            "WHERE d.parser_run_id=? AND d.section_key=s.section_key)"
            if queue == "unprocessed"
            else "EXISTS (SELECT 1 FROM draft_screening_current AS d "
            "WHERE d.parser_run_id=? AND d.section_key=s.section_key "
            "AND d.decision='DEFER')"
        )
        eligible_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM source_sections AS s WHERE s.shard_id=? AND "
                + count_eligibility,
                (row["shard_id"], row["parser_run_id"]),
            ).fetchone()[0]
        )
        conn.commit()
        return {
            "shard_id": int(row["shard_id"]),
            "product_code": str(row["product_code"]),
            "section_count": int(row["section_count"]),
            "claimant": claimant,
            "queue": queue,
            "eligible_count": eligible_count,
            "lease_expires_at": expiry,
        }
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def read_draft_screening_record(
    workspace: Path | str, shard_id: int, claimant: str, index: int
) -> dict[str, object]:
    """Read exactly one private section from a currently claimed screen batch."""
    if shard_id < 1 or not claimant or index < 0:
        raise ValueError("positive shard, claimant, and non-negative index are required")
    conn = _connect(workspace, readonly=True)
    try:
        now = int(time.time())
        claim = conn.execute(
            """SELECT claim.parser_run_id, claim.claim_mode FROM draft_screening_claims AS claim
               JOIN parser_runs AS p ON p.parser_run_id=claim.parser_run_id
               WHERE claim.shard_id=? AND claim.claimant=?
                 AND claim.lease_expires_at >= ? AND """
            + _screening_run_sql("p"),
            (shard_id, claimant, now),
        ).fetchone()
        if claim is None:
            raise PermissionError("screening batch is not currently claimed by this worker")
        rows = conn.execute(
            """SELECT section_key, heading, source_text, page_start, page_end,
                      printed_page, layout_flags
               FROM source_sections WHERE shard_id=? ORDER BY source_section_id""",
            (shard_id,),
        ).fetchall()
        if index >= len(rows):
            raise ValueError(f"screening record index {index} is outside the batch")
        row = rows[index]
        existing = conn.execute(
            """SELECT requested_decision, decision, duplicate_of_section_key,
                      defer_reason, deferred_by
               FROM draft_screening_current
               WHERE parser_run_id=? AND section_key=?""",
            (claim["parser_run_id"], row["section_key"]),
        ).fetchone()
        claim_mode = str(claim["claim_mode"])
        if existing is not None and (
            claim_mode != "escalation" or str(existing["decision"]) != "DEFER"
        ):
            return {
                "index": index,
                "section_count": len(rows),
                "state": "decided",
                "requested_decision": str(existing["requested_decision"]).lower(),
                "decision": str(existing["decision"]).lower(),
                "duplicate_rejected": existing["duplicate_of_section_key"] is not None,
            }
        if claim_mode == "escalation" and existing is None:
            return {
                "index": index,
                "section_count": len(rows),
                "state": "unprocessed",
            }
        return {
            "index": index,
            "section_count": len(rows),
            "state": "deferred" if claim_mode == "escalation" else "pending",
            "heading": row["heading"],
            "source_text": row["source_text"],
            "page_start": row["page_start"],
            "page_end": row["page_end"],
            "printed_page": row["printed_page"],
            "layout_flags": json.loads(str(row["layout_flags"] or "[]")),
            **(
                {"defer_reason": str(existing["defer_reason"])}
                if existing is not None
                else {}
            ),
        }
    finally:
        conn.close()


def _screening_stored_decision(
    conn: sqlite3.Connection,
    run_id: str,
    section: sqlite3.Row,
    requested: str,
) -> tuple[str, str | None]:
    """Resolve only versioned same-heading duplicate-group shadows."""
    if requested != "ADD":
        return requested, None
    canonical = conn.execute(
        """SELECT groups.canonical_section_key
             FROM duplicate_group_members AS member
             JOIN duplicate_groups AS groups ON groups.group_id=member.group_id
            WHERE member.section_key=?""",
        (section["section_key"],),
    ).fetchone()
    if canonical is None:
        # Low-level tests and manual screening may precede deterministic
        # preparation; absence of a group must fail open, never deduplicate.
        return requested, None
    canonical_key = str(canonical["canonical_section_key"])
    if canonical_key != str(section["section_key"]):
        return "REJECT", canonical_key
    return requested, None


def _screening_batch_complete(
    conn: sqlite3.Connection, shard_id: int, run_id: str, claim_mode: str
) -> bool:
    if claim_mode == "ordinary":
        pending = conn.execute(
            """SELECT 1 FROM source_sections AS s
               WHERE s.shard_id=? AND NOT EXISTS (
                   SELECT 1 FROM draft_screening_current AS d
                   WHERE d.parser_run_id=? AND d.section_key=s.section_key
               ) LIMIT 1""",
            (shard_id, run_id),
        ).fetchone()
    else:
        pending = conn.execute(
            """SELECT 1 FROM source_sections AS s
               JOIN draft_screening_current AS d
                 ON d.parser_run_id=? AND d.section_key=s.section_key
               WHERE s.shard_id=? AND d.decision='DEFER' LIMIT 1""",
            (run_id, shard_id),
        ).fetchone()
    return pending is None


def _store_foundry_confirmation(
    conn: sqlite3.Connection,
    *,
    section_key: str,
    foundry_ids: Sequence[str],
    worker: str,
    decided_at: int,
) -> str | None:
    if not foundry_ids:
        return None
    active = conn.execute(
        "SELECT value FROM metadata WHERE key='active_foundry_snapshot'"
    ).fetchone()
    if active is None:
        raise ValueError("Foundry duplicate confirmation requires a prepared snapshot")
    snapshot = str(active[0])
    placeholders = ",".join("?" for _ in foundry_ids)
    candidates = conn.execute(
        f"""SELECT foundry_id, proof_digest FROM foundry_coverage_candidates
            WHERE section_key=? AND snapshot_digest=?
              AND foundry_id IN ({placeholders}) ORDER BY foundry_id""",
        (section_key, snapshot, *foundry_ids),
    ).fetchall()
    if len(candidates) != len(foundry_ids):
        raise ValueError("Foundry duplicate result selected an unsupplied or stale candidate")
    ordered_ids = sorted(foundry_ids)
    proof_digest = _digest(
        "foundry-coverage-confirmation-v1", section_key, snapshot,
        _canonical_json(ordered_ids),
        _canonical_json([str(row["proof_digest"]) for row in candidates]),
    )
    conn.execute(
        """INSERT INTO foundry_coverage_confirmations
           (section_key, snapshot_digest, foundry_ids_json, proof_digest, worker, decided_at)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(section_key, snapshot_digest) DO UPDATE SET
             foundry_ids_json=excluded.foundry_ids_json,
             proof_digest=excluded.proof_digest,
             worker=excluded.worker, decided_at=excluded.decided_at""",
        (section_key, snapshot, _canonical_json(ordered_ids), proof_digest, worker, decided_at),
    )
    return proof_digest


def submit_draft_screening_decision(
    workspace: Path | str,
    shard_id: int,
    claimant: str,
    index: int,
    decision: str,
    *,
    defer_reason: str | None = None,
    reject_reason: str | None = None,
    foundry_ids: Sequence[str] = (),
) -> dict[str, object]:
    """Persist one quad-state decision or resolve one deferred decision."""
    requested = decision.upper()
    if (
        shard_id < 1
        or not claimant
        or index < 0
        or requested not in SCREENING_DECISIONS
        or (requested == "DEFER" and defer_reason not in SCREENING_DEFER_REASONS)
        or (requested != "DEFER" and defer_reason is not None)
        or (requested == "REJECT" and reject_reason is not None and reject_reason not in SCREENING_REJECT_REASONS)
        or (requested != "REJECT" and reject_reason is not None)
        or len(foundry_ids) > 3
        or len(set(foundry_ids)) != len(foundry_ids)
        or any(not isinstance(value, str) or not value for value in foundry_ids)
        or (bool(foundry_ids) and (requested != "REJECT" or reject_reason != "duplicate"))
    ):
        raise ValueError(
            "screening requires shard, claimant, index, add/reject/defer, "
            "and a bounded reason for defer"
        )
    now = int(time.time())
    conn = _connect(workspace)
    try:
        _migrate_review_workspace(conn)
        conn.execute("BEGIN IMMEDIATE")
        run = conn.execute(
            """SELECT p.parser_run_id FROM review_shards AS sh
               JOIN parser_runs AS p ON p.parser_run_id=sh.parser_run_id
               WHERE sh.shard_id=? AND """
            + _screening_run_sql("p"),
            (shard_id,),
        ).fetchone()
        if run is None:
            raise ValueError("screening batch is not part of the active trusted parser run")
        run_id = str(run["parser_run_id"])
        rows = conn.execute(
            """SELECT section_key, text_hash, stable_identity, source_section_id
               FROM source_sections WHERE shard_id=? ORDER BY source_section_id""",
            (shard_id,),
        ).fetchall()
        if index >= len(rows):
            raise ValueError(f"screening record index {index} is outside the batch")
        section = rows[index]
        existing = conn.execute(
            """SELECT requested_decision, decision, duplicate_of_section_key,
                      defer_reason, deferred_by, deferred_at, worker, decided_at
               FROM draft_screening_current
               WHERE parser_run_id=? AND section_key=?""",
            (run_id, section["section_key"]),
        ).fetchone()
        latest_event = _screening_latest_event(conn, run_id, str(section["section_key"]))
        claim = conn.execute(
            """SELECT claim_mode FROM draft_screening_claims
               WHERE parser_run_id=? AND shard_id=? AND claimant=?
                 AND lease_expires_at >= ?""",
            (run_id, shard_id, claimant, now),
        ).fetchone()

        if existing is not None and str(existing["decision"]) == "DEFER" and requested in {
            "ADD", "REJECT",
        }:
            if claim is None or str(claim["claim_mode"]) != "escalation":
                raise PermissionError("deferred records require a live escalation claim")
            stored, duplicate_of = (
                ("REJECT", None)
                if foundry_ids else _screening_stored_decision(conn, run_id, section, requested)
            )
            _store_foundry_confirmation(
                conn, section_key=str(section["section_key"]), foundry_ids=foundry_ids,
                worker=claimant, decided_at=now,
            )
            if latest_event is None or str(latest_event["event_type"]) != "DECISION":
                raise ValueError("screening record has no deferred event to resolve")
            conn.execute(
                """INSERT INTO draft_screening_events
                   (parser_run_id, section_key, event_type, requested_decision, decision,
                    duplicate_of_section_key, reject_reason, defer_reason, deferred_by, deferred_at,
                    worker, decided_at, supersedes_event_id)
                   VALUES (?, ?, 'DECISION', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    section["section_key"],
                    requested,
                    stored,
                    duplicate_of,
                    reject_reason if requested == "REJECT" else None,
                    latest_event["defer_reason"],
                    latest_event["deferred_by"],
                    latest_event["deferred_at"],
                    claimant,
                    now,
                    latest_event["event_id"],
                ),
            )
            if requested == "REJECT" and reject_reason is not None:
                conn.execute(
                    """INSERT INTO runner_screen_rejections VALUES (?, ?, ?, ?)
                       ON CONFLICT(section_key) DO UPDATE SET
                         reason=excluded.reason, worker=excluded.worker,
                         decided_at=excluded.decided_at""",
                    (section["section_key"], reject_reason, claimant, now),
                )
            complete = _screening_batch_complete(
                conn, shard_id, run_id, "escalation"
            )
            if complete:
                conn.execute(
                    "DELETE FROM draft_screening_claims WHERE parser_run_id=? AND shard_id=?",
                    (run_id, shard_id),
                )
            conn.commit()
            return {
                "index": index,
                "state": "resolved",
                "decision": stored.lower(),
                "duplicate_rejected": duplicate_of is not None,
                "batch_complete": complete,
            }

        if existing is not None:
            if str(existing["requested_decision"]) != requested:
                raise ValueError("screening record already has a conflicting decision")
            if str(existing["worker"]) != claimant:
                raise PermissionError("screening record was decided by another worker")
            if requested == "DEFER" and str(existing["defer_reason"]) != defer_reason:
                raise ValueError("screening record already has a conflicting defer reason")
            retry_mode = (
                "escalation"
                if existing["deferred_by"] is not None and requested != "DEFER"
                else "ordinary"
            )
            complete = _screening_batch_complete(conn, shard_id, run_id, retry_mode)
            conn.commit()
            return {
                "index": index,
                "state": "unchanged",
                "decision": str(existing["decision"]).lower(),
                "duplicate_rejected": existing["duplicate_of_section_key"] is not None,
                "batch_complete": complete,
            }

        if claim is None or str(claim["claim_mode"]) != "ordinary":
            raise PermissionError("unprocessed records require a live ordinary claim")
        stored, duplicate_of = (
            ("REJECT", None)
            if foundry_ids else _screening_stored_decision(conn, run_id, section, requested)
        )
        _store_foundry_confirmation(
            conn, section_key=str(section["section_key"]), foundry_ids=foundry_ids,
            worker=claimant, decided_at=now,
        )
        conn.execute(
            """INSERT INTO draft_screening_events
               (parser_run_id, section_key, event_type, requested_decision, decision,
                duplicate_of_section_key, reject_reason, defer_reason, deferred_by, deferred_at,
                worker, decided_at)
               VALUES (?, ?, 'DECISION', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                section["section_key"],
                requested,
                stored,
                duplicate_of,
                reject_reason if requested == "REJECT" else None,
                defer_reason,
                claimant if requested == "DEFER" else None,
                now if requested == "DEFER" else None,
                claimant,
                now,
            ),
        )
        if requested == "REJECT" and reject_reason is not None:
            conn.execute(
                "INSERT INTO runner_screen_rejections VALUES (?, ?, ?, ?)",
                (section["section_key"], reject_reason, claimant, now),
            )
        complete = _screening_batch_complete(conn, shard_id, run_id, "ordinary")
        if complete:
            conn.execute(
                "DELETE FROM draft_screening_claims WHERE parser_run_id=? AND shard_id=?",
                (run_id, shard_id),
            )
        conn.commit()
        return {
            "index": index,
            "state": "inserted",
            "decision": stored.lower(),
            "duplicate_rejected": duplicate_of is not None,
            "batch_complete": complete,
        }
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def next_draft_screening_record(
    workspace: Path | str,
    shard_id: int,
    claimant: str,
    *,
    after_index: int = -1,
) -> dict[str, object] | None:
    """Return the next record eligible for the claim's ordinary/escalation queue."""
    if after_index < -1:
        raise ValueError("after_index must be at least -1")
    conn = _connect(workspace, readonly=True)
    try:
        section_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM source_sections WHERE shard_id=?", (shard_id,)
            ).fetchone()[0]
        )
    finally:
        conn.close()
    if after_index >= section_count:
        raise ValueError(f"screening record index {after_index} is outside the batch")
    indices = (*range(after_index + 1, section_count), *range(0, after_index + 1))
    for index in indices:
        record = read_draft_screening_record(workspace, shard_id, claimant, index)
        if record["state"] in {"pending", "deferred"}:
            return record
    return None


def step_draft_screening(
    workspace: Path | str,
    shard_id: int,
    claimant: str,
    index: int,
    decision: str,
    *,
    defer_reason: str | None = None,
) -> dict[str, object]:
    """Persist one decision and return the next eligible record in the batch."""
    result = submit_draft_screening_decision(
        workspace,
        shard_id,
        claimant,
        index,
        decision,
        defer_reason=defer_reason,
    )
    if result["batch_complete"]:
        return {"result": result, "next_record": None}
    record = next_draft_screening_record(
        workspace, shard_id, claimant, after_index=index
    )
    if record is None:
        raise RuntimeError("screening batch reports pending work but no pending record exists")
    return {"result": result, "next_record": record}


def reopen_draft_screening(
    workspace: Path | str,
    section_key: str,
    *,
    maintainer: str = "maintainer",
    reason: str,
) -> dict[str, object]:
    """Append a maintainer reopen event for an active screening section.

    Reopen is deliberately an explicit maintainer operation.  It never edits
    or deletes the previous decision and refuses while either the section's
    shard or its screening batch has a live lease.
    """
    if not section_key or not maintainer or reason not in SCREENING_REOPEN_REASONS:
        raise ValueError(
            "screening reopen requires a section key, maintainer, and bounded reason"
        )
    now = int(time.time())
    conn = _connect(workspace)
    try:
        _migrate_review_workspace(conn)
        conn.execute("BEGIN IMMEDIATE")
        requested_section_key = section_key
        grouped = conn.execute(
            """SELECT groups.canonical_section_key
                 FROM duplicate_group_members AS member
                 JOIN duplicate_groups AS groups ON groups.group_id=member.group_id
                WHERE member.section_key=?""",
            (section_key,),
        ).fetchone()
        if grouped is not None:
            section_key = str(grouped["canonical_section_key"])
        section = conn.execute(
            """SELECT s.section_key, s.parser_run_id, s.shard_id
                 FROM source_sections AS s
                 JOIN parser_runs AS p ON p.parser_run_id=s.parser_run_id
                WHERE s.section_key=? AND """ + _screening_run_sql("p"),
            (section_key,),
        ).fetchone()
        if section is None:
            raise ValueError("section is not part of the active trusted screening run")
        live = conn.execute(
            """SELECT 1 FROM review_shards
                WHERE shard_id=? AND claimant IS NOT NULL AND lease_expires_at>=?
               UNION ALL
              SELECT 1 FROM draft_screening_claims
                WHERE parser_run_id=? AND shard_id=? AND lease_expires_at>=?
               LIMIT 1""",
            (section["shard_id"], now, section["parser_run_id"], section["shard_id"], now),
        ).fetchone()
        if live is not None:
            raise PermissionError("cannot reopen screening while the section has a live claim")
        latest = _screening_latest_event(conn, str(section["parser_run_id"]), section_key)
        if latest is None or str(latest["event_type"]) != "DECISION":
            if requested_section_key != section_key:
                conn.commit()
                return {
                    "requested_section_key": requested_section_key,
                    "section_key": section_key,
                    "state": "already-open",
                    "previous_decision": None,
                    "event_id": None,
                }
            raise ValueError("section has no current terminal screening decision")
        current = conn.execute(
            """SELECT decision FROM draft_screening_current
               WHERE parser_run_id=? AND section_key=?""",
            (section["parser_run_id"], section_key),
        ).fetchone()
        if current is None:
            raise ValueError("section has no current terminal screening decision")
        # Retrying the exact maintainer action is idempotent; a different
        # reopen after one has already been recorded must be explicit in a
        # future decision event rather than silently extending the history.
        prior = conn.execute(
            """SELECT event_id, reopen_reason, worker FROM draft_screening_events
                WHERE parser_run_id=? AND section_key=? AND event_type='REOPEN'
                ORDER BY event_id DESC LIMIT 1""",
            (section["parser_run_id"], section_key),
        ).fetchone()
        if prior is not None:
            if str(prior["reopen_reason"]) == reason and str(prior["worker"]) == maintainer:
                conn.commit()
                return {
                    "requested_section_key": requested_section_key,
                    "section_key": section_key,
                    "state": "unchanged",
                    "previous_decision": str(current["decision"]).lower(),
                    "event_id": int(prior["event_id"]),
                }
            raise ValueError("screening section has already been reopened")
        cursor = conn.execute(
            """INSERT INTO draft_screening_events
               (parser_run_id, section_key, event_type, worker, decided_at,
                reopen_reason, supersedes_event_id)
               VALUES (?, ?, 'REOPEN', ?, ?, ?, ?)""",
            (
                section["parser_run_id"], section_key, maintainer, now, reason,
                latest["event_id"],
            ),
        )
        conn.commit()
        return {
            "requested_section_key": requested_section_key,
            "section_key": section_key,
            "state": "reopened",
            "previous_decision": str(current["decision"]).lower(),
            "event_id": int(cursor.lastrowid),
        }
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def release_draft_screening_batch(
    workspace: Path | str, shard_id: int, claimant: str
) -> dict[str, object]:
    """Release an incomplete screen lease without deleting any decisions."""
    if shard_id < 1 or not claimant:
        raise ValueError("positive shard and claimant are required")
    conn = _connect(workspace)
    try:
        _migrate_review_workspace(conn)
        conn.execute("BEGIN IMMEDIATE")
        deleted = conn.execute(
            "DELETE FROM draft_screening_claims WHERE shard_id=? AND claimant=?",
            (shard_id, claimant),
        ).rowcount
        conn.commit()
        return {"shard_id": shard_id, "released": bool(deleted)}
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def submit_candidate(workspace: Path | str, submission: Mapping[str, object]) -> dict[str, object]:
    """Submit one worker decision, rejecting stale text and foreign claims."""
    required = (
        "section_key", "source_section_id", "source_section_hash", "decision", "worker", "prompt_version"
    )
    missing = [key for key in required if not isinstance(submission.get(key), str) or not submission.get(key)]
    if missing:
        raise ValueError("candidate submission is missing: " + ", ".join(missing))
    decision = str(submission["decision"])
    if decision not in POLICY_DECISIONS:
        raise ValueError(f"invalid candidate decision: {decision!r}")
    text = submission.get("candidate_text")
    if text is not None and not isinstance(text, str):
        raise ValueError("candidate_text must be a string when supplied")
    if decision in PUBLIC_DECISIONS and not (isinstance(text, str) and text.strip()):
        raise ValueError("public candidate decisions require non-empty candidate_text")
    public_heading = submission.get("public_heading")
    if decision in PUBLIC_DECISIONS:
        public_heading = _validate_public_scalar(public_heading, field="candidate heading")
    tags = submission.get("reason_tags", [])
    if not isinstance(tags, Iterable) or isinstance(tags, (str, bytes)):
        raise ValueError("reason_tags must be a list of strings")
    normalized_tags = sorted({str(tag) for tag in tags if str(tag)})
    if decision in PUBLIC_DECISIONS:
        if not normalized_tags:
            raise ValueError("public candidate decisions require reason_tags")
        method = submission.get("extraction_method")
        method = _validate_public_scalar(method, field="extraction method")
        assert isinstance(text, str)
        _validate_public_text(text, field="candidate text")
    confidence = float(submission.get("confidence", 0))
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between zero and one")
    worker = str(submission["worker"])
    now = int(time.time())
    conn = _connect(workspace)
    try:
        _migrate_review_workspace(conn)
        conn.execute("BEGIN IMMEDIATE")
        _require_unambiguous_targets(conn)
        section = conn.execute(
            """SELECT s.*, sh.claimant, sh.lease_expires_at, sh.claim_mode
            FROM source_sections AS s JOIN review_shards AS sh ON sh.shard_id=s.shard_id
            WHERE s.section_key=? AND """ + _review_target_sql("s"),
            (submission["section_key"],),
        ).fetchone()
        if section is None:
            raise ValueError("unknown source section")
        if section["source_section_id"] != submission["source_section_id"]:
            raise ValueError("source section ID does not match section_key")
        if section["source_section_hash"] != submission["source_section_hash"]:
            raise ValueError("stale source section hash")
        if section["claimant"] != worker or int(section["lease_expires_at"] or 0) < now:
            raise PermissionError("source section is not claimed by submitting worker")
        claim_mode = str(section["claim_mode"] or "")
        if claim_mode not in CLAIM_MODES:
            raise PermissionError("source section claim is missing a persisted claim mode")
        has_current_candidate = conn.execute(
            "SELECT 1 FROM candidates WHERE section_key=? LIMIT 1",
            (section["section_key"],),
        ).fetchone() is not None
        if claim_mode == "ordinary":
            if has_current_candidate:
                raise PermissionError(
                    "ordinary assignments accept only the first decision for each section"
                )
        else:
            current_reviews = conn.execute(
                f"""SELECT r.verdict
                FROM candidates AS c
                JOIN reviews AS r ON r.candidate_id=c.candidate_id
                WHERE c.section_key=?
                  AND c.candidate_ordinal=(
                      SELECT MAX(c2.candidate_ordinal) FROM candidates AS c2
                      WHERE c2.section_key=c.section_key
                  )
                  AND ({_active_review_sql('r')})
                ORDER BY r.reviewer""",
                (section["section_key"],),
            ).fetchall()
            verdicts = {str(row["verdict"]) for row in current_reviews}
            if verdicts != {"REVISE"}:
                raise PermissionError(
                    "replacement submissions require the section's current candidate "
                    "to have a completed REVISE review"
                )
        if decision == "PUBLIC_AS_IS" and text != section["source_text"]:
            raise ValueError("PUBLIC_AS_IS candidate text must exactly match the source section")
        if decision == "MIXED_NEEDS_EXTRACTION" and str(text).strip() == str(section["source_text"]).strip():
            raise ValueError("mixed extraction must reconstruct rather than copy the source section")
        try:
            layout_flags = set(json.loads(str(section["layout_flags"] or "[]")))
        except json.JSONDecodeError as exc:
            raise ValueError("source section layout flags are invalid") from exc
        complex_layout = {
            "unclassified-native-coverage", "complex-layout", "table-ambiguous", "table-cell",
            "layout-model-complex", "layout-model-table", "layout-model-unbound",
            "layout-order-conflict", "layout-region-split", "unsupported-layout",
            "unresolved-continuation", "heading-artifact", "oversize-block",
        }
        if decision == "PUBLIC_AS_IS" and layout_flags.intersection(complex_layout):
            raise ValueError("PUBLIC_AS_IS is forbidden for unclassified or complex layout")
        if decision == "MIXED_NEEDS_EXTRACTION" and "layout-reviewed" not in normalized_tags:
            raise ValueError("MIXED_NEEDS_EXTRACTION requires an explicit layout-reviewed tag")
        ordinal = int(
            conn.execute(
                "SELECT COALESCE(MAX(candidate_ordinal), 0) + 1 FROM candidates WHERE section_key=?",
                (section["section_key"],),
            ).fetchone()[0]
        )
        candidate_text = str(text) if text is not None else None
        candidate_hash = _candidate_commitment(
            source_section_hash=section["source_section_hash"], decision=decision,
            candidate_text=candidate_text, public_heading=public_heading,
            extraction_method=submission.get("extraction_method"), reason_tags=normalized_tags,
        )
        candidate_id = "candidate:" + _digest(
            str(section["product_code"]), str(section["content_fingerprint"]),
            str(section["source_section_id"]), str(section["source_section_hash"]),
            str(ordinal), candidate_hash,
        )
        conn.execute(
            """INSERT INTO candidates
            (candidate_id, section_key, source_section_hash, candidate_ordinal, decision, candidate_text,
             public_heading, candidate_hash, extraction_method, reason_tags, confidence, worker,
             prompt_version, submitted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                candidate_id, section["section_key"], section["source_section_hash"], ordinal,
                decision, candidate_text, public_heading, candidate_hash,
                submission.get("extraction_method"), _canonical_json(normalized_tags), confidence,
                worker, str(submission["prompt_version"]), now,
            ),
        )
        _insert_evidence(conn, submission.get("evidence"), candidate_id=candidate_id)
        _release_shard_if_complete(conn, int(section["shard_id"]))
        conn.commit()
        return {"candidate_id": candidate_id, "candidate_ordinal": ordinal}
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def claim_review(
    workspace: Path | str, reviewer: str, *, lease_seconds: int = 3600
) -> dict[str, object] | None:
    """Atomically assign one pending candidate to an independent reviewer."""
    if not reviewer or lease_seconds < 1:
        raise ValueError("reviewer and a positive lease_seconds are required")
    now = int(time.time())
    conn = _connect(workspace)
    try:
        _migrate_review_workspace(conn)
        conn.execute("BEGIN IMMEDIATE")
        _require_unambiguous_targets(conn)
        row = conn.execute(
            """SELECT rc.candidate_id, rc.lease_expires_at
            FROM review_claims AS rc WHERE rc.claimant=? AND rc.lease_expires_at >= ?
            ORDER BY rc.candidate_id LIMIT 1""",
            (reviewer, now),
        ).fetchone()
        if row is None:
            candidate_rows = conn.execute(
                """SELECT c.*
                FROM candidates AS c
                JOIN source_sections AS s ON s.section_key=c.section_key
                LEFT JOIN review_claims AS rc ON rc.candidate_id = c.candidate_id
                WHERE c.worker <> ? AND """ + _review_target_sql("s") + """
                  AND NOT EXISTS (
                      SELECT 1 FROM reviews AS prior_review
                      WHERE prior_review.candidate_id=c.candidate_id
                        AND prior_review.reviewer=?
                  )
                  AND c.candidate_ordinal=(
                      SELECT MAX(c2.candidate_ordinal) FROM candidates AS c2
                      WHERE c2.section_key=c.section_key
                  )
                  AND (rc.candidate_id IS NULL OR rc.lease_expires_at < ?)
                ORDER BY c.submitted_at, c.candidate_id
                """,
                (reviewer, reviewer, now),
            ).fetchall()
            row = next(
                (
                    candidate for candidate in candidate_rows
                    if not _has_blocking_active_review(conn, candidate)
                ),
                None,
            )
            if row is None:
                conn.commit()
                return None
            expiry = now + lease_seconds
            conn.execute(
                """INSERT INTO review_claims(candidate_id, claimant, claimed_at, lease_expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    claimant=excluded.claimant, claimed_at=excluded.claimed_at,
                    lease_expires_at=excluded.lease_expires_at""",
                (row["candidate_id"], reviewer, now, expiry),
            )
        else:
            expiry = int(row["lease_expires_at"])
        candidate = conn.execute("SELECT * FROM candidates WHERE candidate_id=?", (row["candidate_id"],)).fetchone()
        if candidate is None:
            raise ValueError("review claim references an unknown candidate")
        _validate_candidate_commitment(candidate)
        conn.commit()
        return {
            "candidate_id": str(row["candidate_id"]),
            "reviewer": reviewer,
            "lease_expires_at": expiry,
        }
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def read_claimed_review(
    workspace: Path | str, candidate_id: str, reviewer: str
) -> dict[str, object]:
    """Explicit private read of an assigned candidate, its source, and evidence."""
    conn = _connect(workspace, readonly=True)
    try:
        _require_unambiguous_targets(conn)
        now = int(time.time())
        row = conn.execute(
            """SELECT c.*, s.product_code, s.content_fingerprint, s.source_section_id,
                       s.source_section_hash, s.page_start, s.page_end, s.printed_page,
                       s.heading, s.source_text, s.layout_flags
                FROM review_claims AS rc
                JOIN candidates AS c ON c.candidate_id=rc.candidate_id
                JOIN source_sections AS s ON s.section_key=c.section_key
                WHERE rc.candidate_id=? AND rc.claimant=? AND rc.lease_expires_at >= ?
                  AND """ + _review_target_sql("s"),
            (candidate_id, reviewer, now),
        ).fetchone()
        if row is None:
            raise PermissionError("candidate is not currently claimed by this reviewer")
        _validate_candidate_commitment(row)
        evidence = [
            dict(evidence_row)
            for evidence_row in conn.execute(
                """SELECT evidence_kind, status, url, checked_at, note
                FROM evidence WHERE candidate_id=? ORDER BY evidence_id""",
                (candidate_id,),
            )
        ]
        result = dict(row)
        result["evidence"] = evidence
        return result
    finally:
        conn.close()


def submit_review(workspace: Path | str, review: Mapping[str, object]) -> dict[str, str]:
    """Record an independent review of one candidate."""
    required = ("candidate_id", "reviewer", "verdict", "policy_version")
    missing = [key for key in required if not isinstance(review.get(key), str) or not review.get(key)]
    if missing:
        raise ValueError("review is missing: " + ", ".join(missing))
    verdict = _REVIEW_VERDICT_ALIASES.get(str(review["verdict"]), str(review["verdict"]))
    if verdict not in REVIEW_VERDICTS:
        raise ValueError(f"invalid review verdict: {verdict!r}")
    tags = review.get("reason_tags", [])
    if not isinstance(tags, Iterable) or isinstance(tags, (str, bytes)):
        raise ValueError("reason_tags must be a list of strings")
    normalized_tags = sorted({str(tag) for tag in tags if str(tag)})
    conn = _connect(workspace)
    try:
        _migrate_review_workspace(conn)
        conn.execute("BEGIN IMMEDIATE")
        _require_unambiguous_targets(conn)
        candidate = conn.execute("SELECT * FROM candidates WHERE candidate_id=?", (review["candidate_id"],)).fetchone()
        if candidate is None:
            raise ValueError("unknown candidate")
        _validate_candidate_commitment(candidate)
        reviewer = str(review["reviewer"])
        if reviewer == candidate["worker"]:
            raise PermissionError("candidate worker cannot independently review their own work")
        claim = conn.execute(
            """SELECT 1 FROM review_claims
            WHERE candidate_id=? AND claimant=? AND lease_expires_at >= ?""",
            (candidate["candidate_id"], reviewer, int(time.time())),
        ).fetchone()
        if claim is None:
            raise PermissionError("candidate is not currently claimed by this reviewer")
        if _has_blocking_active_review(conn, candidate):
            raise PermissionError("candidate already has an active independent review")
        if verdict == "APPROVE" and (
            candidate["decision"] not in PUBLIC_DECISIONS
            or not isinstance(candidate["candidate_text"], str)
            or not candidate["candidate_text"].strip()
            or not isinstance(candidate["public_heading"], str)
            or not candidate["public_heading"].strip()
        ):
            raise ValueError("only text-bearing public candidates may be approved")
        if verdict == "REJECT" and candidate["decision"] not in {"EXCLUDE", "UNCERTAIN"}:
            raise ValueError("only EXCLUDE or UNCERTAIN may be confirmed; use REVISE otherwise")
        policy_version = str(review["policy_version"])
        if policy_version != LICENSED_CORE_POLICY_VERSION:
            raise ValueError(f"unsupported review policy: {policy_version}")
        reviewed_at = int(time.time())
        review_id = "review:" + _digest(str(candidate["candidate_id"]), reviewer)
        review_lineage = [review_id]
        review_commitment = _review_commitment(
            candidate_id=candidate["candidate_id"], reviewer=reviewer, verdict=verdict,
            policy_version=policy_version, reason_tags=normalized_tags, reviewed_at=reviewed_at,
            review_lineage=review_lineage,
        )
        conn.execute(
            """INSERT INTO reviews
            (review_id, candidate_id, reviewer, verdict, policy_version, reason_tags, notes,
             reviewed_at, review_commitment, review_lineage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                review_id, candidate["candidate_id"], reviewer, verdict,
                policy_version, _canonical_json(normalized_tags),
                review.get("notes"), reviewed_at, review_commitment, _canonical_json(review_lineage),
            ),
        )
        _insert_evidence(conn, review.get("evidence"), review_id=review_id)
        conn.execute("DELETE FROM review_claims WHERE candidate_id=?", (candidate["candidate_id"],))
        conn.commit()
        return {"review_id": review_id}
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def invalidate_reviews(
    workspace: Path | str,
    reviewer: str,
    review_ids: Iterable[str],
    *,
    invalidated_by: str,
    reason: str,
) -> dict[str, object]:
    """Append an auditable invalidation for an exact reviewer-owned review set.

    Reviews are never deleted.  The selection must name every immutable review
    ID explicitly and every ID must belong to ``reviewer`` and still be active.
    This makes a bad batch reopen for independent review without accidentally
    invalidating that reviewer's other work.
    """
    if (
        not isinstance(reviewer, str)
        or not reviewer
        or not isinstance(invalidated_by, str)
        or not invalidated_by
        or not isinstance(reason, str)
        or not reason.strip()
    ):
        raise ValueError("reviewer, invalidated_by, and a non-empty reason are required")
    try:
        ids = list(review_ids)
    except TypeError as exc:
        raise ValueError("review_ids must be a non-empty list of non-empty strings") from exc
    if not ids or any(not isinstance(review_id, str) or not review_id for review_id in ids):
        raise ValueError("review_ids must be a non-empty list of non-empty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("review_ids must not contain duplicates")
    ordered_ids = sorted(ids)
    selection_digest = _digest("review-invalidation-v1", *ordered_ids)
    batch_id = "review-invalidation:" + selection_digest
    now = int(time.time())
    placeholders = ", ".join("?" for _ in ordered_ids)
    conn = _connect(workspace)
    try:
        _migrate_review_workspace(conn)
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            f"""SELECT r.review_id FROM reviews AS r
            LEFT JOIN review_invalidations AS invalidation ON invalidation.review_id=r.review_id
            WHERE r.reviewer=? AND r.review_id IN ({placeholders})
              AND invalidation.review_id IS NULL
            ORDER BY r.review_id""",
            (reviewer, *ordered_ids),
        ).fetchall()
        if len(rows) != len(ordered_ids):
            raise ValueError(
                "every requested review_id must belong to the named reviewer and remain active"
            )
        for row in rows:
            review_id = str(row["review_id"])
            conn.execute(
                """INSERT INTO review_invalidations VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    review_id,
                    "review-invalidation:" + _digest(
                        review_id, batch_id, invalidated_by, reason,
                    ),
                    batch_id,
                    selection_digest,
                    invalidated_by,
                    reason,
                    now,
                ),
            )
        conn.commit()
        return {
            "reviewer": reviewer,
            "invalidated": len(ordered_ids),
            "batch_id": batch_id,
            "selection_digest": selection_digest,
        }
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _latest_resolution(conn: sqlite3.Connection, section_key: str) -> tuple[sqlite3.Row, list[sqlite3.Row]] | None:
    """Return a completed reusable resolution, never a merely similar draft."""
    candidate = conn.execute(
        """SELECT * FROM candidates WHERE section_key=?
        ORDER BY candidate_ordinal DESC LIMIT 1""", (section_key,)
    ).fetchone()
    if candidate is None:
        return None
    _validate_candidate_commitment(candidate)
    reviews = _active_valid_reviews(conn, candidate)
    verdicts = {str(row["verdict"]) for row in reviews}
    if verdicts not in ({"APPROVE"}, {"REJECT"}):
        return None
    if verdicts == {"APPROVE"} and str(candidate["decision"]) not in PUBLIC_DECISIONS:
        return None
    if verdicts == {"REJECT"} and str(candidate["decision"]) not in {"EXCLUDE", "UNCERTAIN"}:
        return None
    return candidate, reviews


def _clone_resolution(
    conn: sqlite3.Connection, *, old_section_key: str, new_section: sqlite3.Row, now: int
) -> bool:
    """Copy only an exact, completed review result into a new parser output.

    The old audit rows remain immutable.  New IDs name the new section, so an
    invalidation/review on either run cannot accidentally affect the other.
    """
    trusted_old = conn.execute(
        """SELECT 1 FROM source_sections AS section JOIN parser_runs AS run
        ON run.parser_run_id=section.parser_run_id
        WHERE section.section_key=? AND run.origin=?""",
        (old_section_key, _TRUSTED_RUN_ORIGIN),
    ).fetchone()
    if trusted_old is None:
        return False
    resolution = _latest_resolution(conn, old_section_key)
    if resolution is None:
        return False
    old_candidate, old_reviews = resolution
    candidate_hash = _candidate_commitment(
        source_section_hash=new_section["source_section_hash"], decision=old_candidate["decision"],
        candidate_text=old_candidate["candidate_text"], public_heading=old_candidate["public_heading"],
        extraction_method=old_candidate["extraction_method"], reason_tags=old_candidate["reason_tags"],
    )
    candidate_id = "candidate:" + _digest(
        str(new_section["product_code"]), str(new_section["content_fingerprint"]),
        str(new_section["source_section_id"]), str(new_section["source_section_hash"]), "1", candidate_hash,
    )
    conn.execute(
        """INSERT INTO candidates
        (candidate_id, section_key, source_section_hash, candidate_ordinal, decision, candidate_text,
         public_heading, candidate_hash, extraction_method, reason_tags, confidence, worker,
         prompt_version, submitted_at)
        VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (candidate_id, new_section["section_key"], new_section["source_section_hash"],
         old_candidate["decision"], old_candidate["candidate_text"], old_candidate["public_heading"], candidate_hash,
         old_candidate["extraction_method"], old_candidate["reason_tags"], old_candidate["confidence"],
         old_candidate["worker"], old_candidate["prompt_version"], now),
    )
    for review in old_reviews:
        review_id = "review:" + _digest(candidate_id, str(review["reviewer"]))
        old_lineage = json.loads(str(review["review_lineage"]))
        review_lineage = [*old_lineage, review_id]
        review_commitment = _review_commitment(
            candidate_id=candidate_id, reviewer=review["reviewer"], verdict=review["verdict"],
            policy_version=review["policy_version"], reason_tags=review["reason_tags"], reviewed_at=now,
            review_lineage=review_lineage,
        )
        conn.execute(
            """INSERT INTO reviews
            (review_id, candidate_id, reviewer, verdict, policy_version, reason_tags, notes,
             reviewed_at, review_commitment, review_lineage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (review_id, candidate_id, review["reviewer"], review["verdict"], review["policy_version"],
             review["reason_tags"], review["notes"], now, review_commitment, _canonical_json(review_lineage)),
        )
    return True


def _clone_screening_resolutions(
    conn: sqlite3.Connection,
    *,
    parser_run_id: str,
    old_to_new: Mapping[str, str],
    now: int,
) -> int:
    """Carry exact terminal screening judgments into a replacement run.

    The original event history stays immutable on the retired run. Deferred
    and reopened sections are deliberately not copied because they are not a
    terminal semantic judgment. Duplicate rejections are copied only when the
    canonical section was also mapped exactly into the new run.
    """
    copied = 0
    for old_section_key, new_section_key in sorted(old_to_new.items()):
        event = conn.execute(
            """SELECT * FROM draft_screening_events
               WHERE section_key=? ORDER BY event_id DESC LIMIT 1""",
            (old_section_key,),
        ).fetchone()
        if (
            event is None
            or event["event_type"] != "DECISION"
            or event["decision"] not in {"ADD", "REJECT"}
        ):
            continue
        duplicate_of = event["duplicate_of_section_key"]
        if duplicate_of is not None:
            duplicate_of = old_to_new.get(str(duplicate_of))
            if duplicate_of is None:
                continue
        conn.execute(
            """INSERT INTO draft_screening_events
               (parser_run_id, section_key, event_type, requested_decision, decision,
                duplicate_of_section_key, reject_reason, defer_reason, deferred_by, deferred_at,
                worker, decided_at, supersedes_event_id)
               VALUES (?, ?, 'DECISION', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                parser_run_id,
                new_section_key,
                event["requested_decision"],
                event["decision"],
                duplicate_of,
                event["reject_reason"],
                event["defer_reason"],
                event["deferred_by"],
                event["deferred_at"],
                event["worker"],
                now,
                event["event_id"],
            ),
        )
        copied += 1
    return copied


def _bind_source_asset_inventory(
    workspace: Path | str,
    product_code: str,
    source_content_fingerprint: str,
    inventory: Mapping[str, object],
) -> dict[str, object]:
    """Irreversibly bind a *verified* exporter-native inventory to an asset.

    Raw anchor hashes are accepted only for this private transaction and are
    never stored.  Parser runs can subsequently prove their section coverage
    against the compact aggregate contract, but cannot author that contract.
    """
    if not product_code.startswith("PZO") or not _is_sha256(source_content_fingerprint):
        raise ValueError("a PZO product and canonical SHA-256 source fingerprint are required")
    profile = inventory.get("inventory_profile")
    version = inventory.get("version")
    policy = inventory.get("ignored_anchor_policy")
    anchors = inventory.get("native_word_anchors")
    ignored = inventory.get("ignored_anchors", [])
    if (
        not isinstance(profile, str) or not profile
        or not isinstance(version, str) or not version
        or not isinstance(policy, str) or not policy
        or not isinstance(anchors, list) or not anchors
        or not isinstance(ignored, list)
        or any(not _is_sha256(anchor) for anchor in anchors)
        or len(set(anchors)) != len(anchors)
    ):
        raise ValueError("inventory needs versioned profile, policy, and unique SHA-256 native-word anchors")
    ignored_hashes: list[str] = []
    for item in ignored:
        if (
            not isinstance(item, Mapping)
            or not _is_sha256(item.get("anchor_hash"))
            or not isinstance(item.get("reason"), str)
            or str(item["reason"]) not in _TRUSTED_IGNORED_ANCHOR_REASONS
        ):
            raise ValueError("every ignored native-word anchor needs a constrained furniture reason")
        ignored_hashes.append(str(item["anchor_hash"]))
    if len(set(ignored_hashes)) != len(ignored_hashes) or not set(ignored_hashes).issubset(set(anchors)):
        raise ValueError("ignored native-word anchors must be unique members of the inventory")
    inventory_digest = _anchor_digest("asset-native-word-inventory-v1", anchors)
    ignored_digest = _ignored_anchor_digest(ignored)
    manifest_digest = _digest(
        "source-asset-native-inventory-v1", str(profile), str(version), inventory_digest,
        str(len(anchors)), str(policy), ignored_digest,
    )
    conn = _connect(workspace)
    try:
        _migrate_review_workspace(conn)
        conn.execute("BEGIN IMMEDIATE")
        asset = conn.execute(
            """SELECT * FROM source_assets WHERE product_code=?
            AND source_content_fingerprint=?""", (product_code, source_content_fingerprint)
        ).fetchone()
        if asset is None:
            raise ValueError("source asset is not registered in this workspace")
        if asset["inventory_manifest_digest"] is not None:
            if (
                asset["inventory_manifest_digest"] == manifest_digest
                and asset["native_word_anchor_digest"] == inventory_digest
                and int(asset["native_word_anchor_count"] or 0) == len(anchors)
                and asset["ignored_anchor_digest"] == ignored_digest
            ):
                conn.commit()
                return {
                    "product_code": product_code,
                    "source_content_fingerprint": source_content_fingerprint,
                    "inventory_manifest_digest": manifest_digest,
                    "native_word_anchor_count": len(anchors),
                }
            raise ValueError("source asset native inventory is immutable once bound")
        conn.execute(
            """UPDATE source_assets SET inventory_profile=?, native_inventory_version=?,
            native_word_anchor_digest=?, native_word_anchor_count=?, ignored_anchor_policy=?,
            ignored_anchor_digest=?, inventory_manifest_digest=?, inventory_bound_at=? WHERE asset_id=?""",
            (profile, version, inventory_digest, len(anchors), policy, ignored_digest,
             manifest_digest, int(time.time()), asset["asset_id"]),
        )
        conn.commit()
        return {
            "product_code": product_code,
            "source_content_fingerprint": source_content_fingerprint,
            "inventory_manifest_digest": manifest_digest,
            "native_word_anchor_count": len(anchors),
        }
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _register_trusted_source_asset(
    workspace: Path | str,
    *,
    product_code: str,
    source_fingerprint: str,
    provenance_hash: str,
) -> None:
    """Register only the sanitized asset contract used by trusted staging."""
    conn = _connect(workspace)
    try:
        _migrate_review_workspace(conn)
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT provenance_hash FROM source_assets WHERE product_code=? AND source_fingerprint=?",
            (product_code, source_fingerprint),
        ).fetchone()
        if existing is None:
            conn.execute(
                """INSERT INTO source_assets
                (asset_id, product_code, source_fingerprint, source_content_fingerprint, provenance_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    "asset:" + _digest(product_code, source_fingerprint), product_code,
                    source_fingerprint, source_fingerprint, provenance_hash, int(time.time()),
                ),
            )
        elif str(existing["provenance_hash"]) != provenance_hash:
            raise ValueError("trusted native export does not match the registered source-asset contract")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _active_stable_identities(workspace: Path | str, product_code: str) -> set[str]:
    """Read only stable IDs needed to make replacement removals explicit."""
    conn = _connect(workspace)
    try:
        _migrate_review_workspace(conn)
        active = conn.execute(
            "SELECT parser_run_id FROM parser_runs WHERE product_code=? AND state='active' AND review_enabled=1",
            (product_code,),
        ).fetchone()
        if active is None:
            return set()
        return {
            str(row[0])
            for row in conn.execute(
                "SELECT stable_identity FROM parser_run_sections WHERE parser_run_id=? AND membership_state='present'",
                (active["parser_run_id"],),
            )
        }
    finally:
        conn.close()


def stage_trusted_native_pdf(
    workspace: Path | str,
    source_pdf: Path | str,
    *,
    product_code: str,
    parser_version: str,
    shard_size: int = 100,
    printing_revision: str | None = None,
    layout_artifact: Path | str | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Direct-PDF-only, one-read bridge for a complete trusted parser run.

    Cached exporter JSON is intentionally not accepted here: it cannot carry
    the private verification capability that binds native-word coverage to the
    purchased PDF read in this process.
    """
    bundle = load_and_parse_verified_pdf(
        source_pdf,
        product_code=product_code,
        parser_version=parser_version,
        layout_artifact=layout_artifact,
    )
    bundle.verify_seal()
    expected_parser = (
        parser_version
        if parser_version in {PAIZO_NATIVE_PARSER_V4, PAIZO_NATIVE_PARSER_V5}
        or layout_artifact is None
        else "paizo-native-v3+pp-doclayout-v3-v1"
    )
    if bundle.product_code != product_code or bundle.parser_version != expected_parser or not _is_sha256(bundle.sealed_digest):
        raise ValueError("direct PDF parser bundle is malformed")
    capability = object()
    _DIRECT_BUNDLE_CAPABILITIES[id(bundle)] = (bundle, capability)
    try:
        return _stage_trusted_bundle(
            workspace, bundle, capability=capability, shard_size=shard_size,
            printing_revision=printing_revision,
        )
    finally:
        # The bridge is deliberately one-shot even if validation or SQLite
        # insertion fails; callers must re-read the PDF for another attempt.
        _DIRECT_BUNDLE_CAPABILITIES.pop(id(bundle), None)


def stage_trusted_native_pdf_with_approved_stitches(
    workspace: Path | str,
    source_pdf: Path | str,
    *,
    product_code: str,
    parser_version: str,
    shard_size: int = 1,
    printing_revision: str | None = None,
    layout_artifact: Path | str | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Re-read a PDF and stage only independently confirmed adjacent unions."""
    conn = _connect(workspace, readonly=True)
    try:
        if printing_revision is None:
            current = conn.execute(
                """SELECT r.printing_revision FROM parser_runs AS p
                   JOIN source_sections AS s ON s.parser_run_id=p.parser_run_id
                   JOIN source_revisions AS r USING(product_code, content_fingerprint)
                   WHERE p.product_code=? AND p.state='active' AND p.review_enabled=1
                   LIMIT 1""",
                (product_code,),
            ).fetchone()
            if current is not None:
                printing_revision = str(current["printing_revision"])
        rows = conn.execute(
            """SELECT DISTINCT sc.section_keys
               FROM stitch_candidates AS sc
               JOIN stitch_votes AS selector
                 ON selector.candidate_id=sc.candidate_id AND selector.role='selector'
               LEFT JOIN stitch_votes AS confirmer
                 ON confirmer.candidate_id=sc.candidate_id AND confirmer.role='confirmer'
               WHERE sc.product_code=?
                 AND selector.decision='merge'
                 AND (
                   confirmer.decision='merge'
                   OR EXISTS (
                     SELECT 1 FROM runner_maintenance AS maintenance
                     WHERE maintenance.queue_name='stitch'
                       AND maintenance.subject_id=sc.candidate_id
                       AND maintenance.reason='independent-disagreement'
                       AND maintenance.resolved_at IS NOT NULL
                       AND maintenance.resolution='merge'
                   )
                 )
               ORDER BY sc.candidate_id""",
            (product_code,),
        ).fetchall()
        groups: list[list[str]] = []
        for row in rows:
            section_keys = json.loads(str(row["section_keys"]))
            placeholders = ",".join("?" for _ in section_keys)
            sources = conn.execute(
                f"SELECT section_key, source_section_id FROM source_sections WHERE section_key IN ({placeholders})",
                section_keys,
            ).fetchall()
            by_key = {str(item["section_key"]): str(item["source_section_id"]) for item in sources}
            if set(by_key) != set(section_keys):
                raise ValueError("approved stitch references a missing source section")
            groups.append([by_key[key] for key in section_keys])
    finally:
        conn.close()
    bundle = load_and_parse_verified_pdf(
        source_pdf,
        product_code=product_code,
        parser_version=parser_version,
        layout_artifact=layout_artifact,
    )
    bundle = repair_trusted_bundle(bundle, groups)
    capability = object()
    _DIRECT_BUNDLE_CAPABILITIES[id(bundle)] = (bundle, capability)
    try:
        return _stage_trusted_bundle(
            workspace, bundle, capability=capability, shard_size=shard_size,
            printing_revision=printing_revision,
        )
    finally:
        _DIRECT_BUNDLE_CAPABILITIES.pop(id(bundle), None)


def _stage_trusted_bundle(
    workspace: Path | str,
    bundle: TrustedParseBundle,
    *,
    capability: object,
    shard_size: int,
    printing_revision: str | None = None,
) -> dict[str, object]:
    """Adapt the sealed parser handoff without exposing its private inputs."""
    registered = _DIRECT_BUNDLE_CAPABILITIES.get(id(bundle))
    if registered is None or registered[0] is not bundle or registered[1] is not capability:
        raise ValueError("trusted parser bundle lacks the direct-PDF capability")
    product = PRODUCT_CATALOG.get(bundle.product_code)
    if product is None or not _is_sha256(bundle.semantic_fingerprint):
        raise ValueError("trusted parser bundle has unsupported product provenance")
    if printing_revision is None:
        printing_revision = f"content-{bundle.semantic_fingerprint[:16]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", printing_revision):
        raise ValueError("printing revision must be a bounded public-safe identifier")
    ignored = list(bundle.inventory.ignored_anchors)
    records: list[dict[str, object]] = []
    block_records: list[dict[str, object]] = []
    for section in bundle.sections:
        if (
            not section.id or not section.heading or not section.text
            or not _is_sha256(section.text_hash) or not _is_sha256(section.stable_section_identity)
            or not section.physical_pages or not section.coverage_anchors
        ):
            raise ValueError("trusted parser bundle contains an incomplete section")
        if any(
            isinstance(page, bool) or not isinstance(page, int) or page < 1
            for page in section.physical_pages
        ):
            raise ValueError("trusted parser bundle contains invalid physical pages")
        text_hash = hashlib.sha256(section.text.encode("utf-8")).hexdigest()
        if text_hash != section.text_hash:
            raise ValueError("trusted parser bundle text hash mismatch")
        _validate_page_provenance(
            section.source_section_id, min(section.physical_pages), max(section.physical_pages)
        )
        provenance_hash = _digest(
            "trusted-section-provenance-v2",
            section.source_section_id,
            str(min(section.physical_pages)),
            str(max(section.physical_pages)),
            str(section.printed_page or ""),
            section.heading,
        )
        records.append({
            "source_section_id": section.source_section_id,
            "source_section_hash": section.text_hash,
            "source_text": section.text,
            "heading": section.heading,
            "page_start": min(section.physical_pages),
            "page_end": max(section.physical_pages),
            "printed_page": section.printed_page,
            "stable_identity": section.stable_section_identity,
            "provenance_hash": provenance_hash,
            "text_hash": text_hash,
            "native_word_count": len(section.coverage_anchors),
            "native_word_digest": _anchor_digest("section-native-word-anchors-v1", section.coverage_anchors),
            "native_word_anchors": list(section.coverage_anchors),
            "layout_flags": list(section.layout_flags),
        })
        block_anchors: list[str] = []
        for expected_ordinal, block in enumerate(section.blocks):
            if (
                block.ordinal != expected_ordinal
                or block.kind not in {"heading", "body", "sidebar", "table"}
                or block.physical_page not in section.physical_pages
                or hashlib.sha256(block.text.encode("utf-8")).hexdigest() != block.text_hash
                or not block.coverage_anchors
            ):
                raise ValueError("trusted parser bundle contains an invalid structural block")
            block_anchors.extend(block.coverage_anchors)
            block_records.append({
                "source_section_id": section.source_section_id,
                "block_ordinal": block.ordinal,
                "kind": block.kind,
                "physical_page": block.physical_page,
                "source_text": block.text,
                "text_hash": block.text_hash,
                "table_json": (
                    _canonical_json([list(row) for row in block.table_cells])
                    if block.table_cells else None
                ),
                "native_word_anchors": list(block.coverage_anchors),
                "anchor_digest": _anchor_digest(
                    "section-block-native-word-anchors-v1", block.coverage_anchors
                ),
            })
        if section.blocks and block_anchors != list(section.coverage_anchors):
            raise ValueError("trusted structural blocks do not exactly cover their section")
    quarantine_records: list[dict[str, object]] = []
    for item in bundle.quarantine:
        if (
            not item.quarantine_id
            or not item.coverage_anchors
            or hashlib.sha256(item.text.encode("utf-8")).hexdigest() != item.text_hash
            or item.physical_page < 1
        ):
            raise ValueError("trusted parser bundle contains an invalid quarantine record")
        quarantine_records.append({
            "quarantine_id": item.quarantine_id,
            "reason": item.reason,
            "physical_page": item.physical_page,
            "source_text": item.text,
            "text_hash": item.text_hash,
            "native_word_count": len(item.coverage_anchors),
            "native_word_digest": _anchor_digest(
                "quarantine-native-word-anchors-v1", item.coverage_anchors
            ),
            "native_word_anchors": list(item.coverage_anchors),
        })
    output_digest = _parser_output_digest(records)
    anchors_digest = _anchor_digest("asset-native-word-inventory-v1", bundle.inventory.anchors)
    ignored_digest = _ignored_anchor_digest(ignored)
    asset_provenance = _digest(
        "trusted-direct-pdf-asset-v2",
        bundle.product_code,
        bundle.semantic_fingerprint,
        str(bundle.exporter_profile_version),
        anchors_digest,
        ignored_digest,
    )
    run = {
        "product_code": bundle.product_code,
        "source_fingerprint": bundle.semantic_fingerprint,
        "parser_version": bundle.parser_version,
        "parser_output_digest": output_digest,
        "trusted_parser_output_digest": bundle.parser_output_digest,
        "license": product.license,
        "era": product.rules_era,
        "source_schema_version": str(bundle.exporter_profile_version),
        "printing_revision": printing_revision,
        "asset_provenance_hash": asset_provenance,
        "bundle_seal": bundle.sealed_digest,
        "trusted_bundle": bundle,
        "trusted_blocks": block_records,
        "trusted_quarantine": quarantine_records,
        "complete_manifest": {
            "version": _TRUSTED_MANIFEST_VERSION,
            "declared_section_count": len(records),
            "output_digest": output_digest,
            "native_word_coverage_digest": _native_word_coverage_digest(
                records, quarantine_records
            ),
            # Calculated under the staging transaction against the selected
            # review target; callers cannot waive source removals.
            "removed_stable_identities": None,
            "ignored_anchor_policy": "constrained-private-native-v1",
            "ignored_anchors": ignored,
        },
    }
    return _stage_parser_run(workspace, run, records, shard_size=shard_size, capability=capability)


def _stage_parser_run(
    workspace: Path | str,
    run: Mapping[str, object],
    sections: Iterable[Mapping[str, object]],
    *,
    shard_size: int = 100,
    capability: object | None = None,
) -> dict[str, object]:
    """Stage one complete parser output without changing the review target.

    Stable identity and provenance are parser inputs, not guessed fuzzy
    matches. A section may inherit terminal work only when its native text and
    every explicit provenance field match. The one-time v3-to-v4 bridge checks
    those fields directly because the retired v3 provenance hash included its
    parser version; new hashes are parser-independent.
    """
    bundle = run.get("trusted_bundle")
    if not isinstance(bundle, TrustedParseBundle):
        raise ValueError("parser runs may only be staged from a sealed direct-PDF bundle")
    registered = _DIRECT_BUNDLE_CAPABILITIES.get(id(bundle))
    if registered is None or registered[0] is not bundle or registered[1] is not capability:
        raise ValueError("parser runs require a live direct-PDF capability")
    if shard_size < 1:
        raise ValueError("shard_size must be at least one")
    required = ("product_code", "source_fingerprint", "parser_version", "parser_output_digest", "license", "era")
    missing = [key for key in required if not isinstance(run.get(key), str) or not str(run[key])]
    if missing:
        raise ValueError("parser run is missing: " + ", ".join(missing))
    product = str(run["product_code"])
    fingerprint = str(run["source_fingerprint"])
    if (
        not product.startswith("PZO")
        or str(run["license"]) not in {"OGL", "ORC"}
        or not _is_sha256(fingerprint)
        or not _is_sha256(run["parser_output_digest"])
    ):
        raise ValueError("parser run requires a PZO product, OGL or ORC license, and SHA-256 fingerprints")
    records = list(sections)
    if not records:
        raise ValueError("parser run must contain at least one section")
    normalized: list[dict[str, object]] = []
    seen_identity: set[str] = set()
    for record in records:
        needed = ("source_section_id", "source_section_hash", "source_text", "heading", "stable_identity", "provenance_hash")
        absent = [key for key in needed if not isinstance(record.get(key), str) or not str(record[key])]
        if absent:
            raise ValueError("parser section is missing: " + ", ".join(absent))
        identity = str(record["stable_identity"])
        if identity in seen_identity:
            raise ValueError("ambiguous duplicate stable_identity in parser run")
        seen_identity.add(identity)
        text = str(record["source_text"])
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        supplied_hash = record.get("text_hash")
        if supplied_hash is not None and supplied_hash != text_hash:
            raise ValueError("parser section text_hash does not match source_text")
        _validate_page_provenance(
            record["source_section_id"], record.get("page_start"), record.get("page_end")
        )
        flags = record.get("layout_flags", [])
        if not isinstance(flags, list) or any(not isinstance(flag, str) or not flag for flag in flags):
            raise ValueError("parser section layout_flags must be a list of non-empty strings")
        normalized.append({
            "source_section_id": str(record["source_section_id"]),
            "source_section_hash": str(record["source_section_hash"]), "source_text": text,
            "heading": str(record["heading"]), "stable_identity": identity,
            "provenance_hash": str(record["provenance_hash"]), "text_hash": text_hash,
            "page_start": record.get("page_start"), "page_end": record.get("page_end"),
                "printed_page": record.get("printed_page"),
                "native_word_count": record.get("native_word_count"),
                "native_word_digest": record.get("native_word_digest"),
                "native_word_anchors": record.get("native_word_anchors"),
                "layout_flags": sorted(set(flags)),
            })
    raw_blocks = run.get("trusted_blocks", [])
    raw_quarantine = run.get("trusted_quarantine", [])
    if (
        not isinstance(raw_blocks, list)
        or any(not isinstance(item, Mapping) for item in raw_blocks)
        or not isinstance(raw_quarantine, list)
        or any(not isinstance(item, Mapping) for item in raw_quarantine)
    ):
        raise ValueError("trusted structural and quarantine records must be lists of objects")
    section_ids = {str(record["source_section_id"]) for record in normalized}
    blocks_by_section: dict[str, list[dict[str, object]]] = {}
    for raw in raw_blocks:
        source_section_id = str(raw.get("source_section_id", ""))
        anchors = raw.get("native_word_anchors")
        text = raw.get("source_text")
        if (
            source_section_id not in section_ids
            or raw.get("kind") not in {"heading", "body", "sidebar", "table"}
            or not isinstance(raw.get("block_ordinal"), int)
            or not isinstance(raw.get("physical_page"), int)
            or not isinstance(text, str)
            or hashlib.sha256(text.encode("utf-8")).hexdigest() != raw.get("text_hash")
            or not isinstance(anchors, list)
            or not anchors
            or any(not _is_sha256(anchor) for anchor in anchors)
            or raw.get("anchor_digest")
            != _anchor_digest("section-block-native-word-anchors-v1", anchors)
        ):
            raise ValueError("trusted parser run contains an invalid structural block")
        blocks_by_section.setdefault(source_section_id, []).append(dict(raw))
    for source_section_id, block_rows in blocks_by_section.items():
        block_rows.sort(key=lambda item: int(item["block_ordinal"]))
        if [int(item["block_ordinal"]) for item in block_rows] != list(range(len(block_rows))):
            raise ValueError("trusted structural block ordinals are not contiguous")
        section = next(
            item for item in normalized if item["source_section_id"] == source_section_id
        )
        block_anchors = [
            str(anchor) for item in block_rows for anchor in item["native_word_anchors"]
        ]
        if block_anchors != section["native_word_anchors"]:
            raise ValueError("trusted structural blocks do not match section anchor order")
    quarantine_records: list[dict[str, object]] = []
    for raw in raw_quarantine:
        anchors = raw.get("native_word_anchors")
        text = raw.get("source_text")
        if (
            not isinstance(raw.get("quarantine_id"), str)
            or not str(raw["quarantine_id"])
            or raw.get("reason") not in {
                "repeated-furniture", "page-number", "contents-index", "credits-legal",
                "unresolved-table", "unbound-layout", "heading-artifact",
                "unresolved-continuation", "unresolved-layout",
                "layout-order-conflict", "oversize-block",
            }
            or not isinstance(raw.get("physical_page"), int)
            or not isinstance(text, str)
            or hashlib.sha256(text.encode("utf-8")).hexdigest() != raw.get("text_hash")
            or not isinstance(anchors, list)
            or not anchors
            or any(not _is_sha256(anchor) for anchor in anchors)
            or len(set(anchors)) != len(anchors)
            or raw.get("native_word_count") != len(anchors)
            or raw.get("native_word_digest")
            != _anchor_digest("quarantine-native-word-anchors-v1", anchors)
        ):
            raise ValueError("trusted parser run contains an invalid quarantine record")
        quarantine_records.append(dict(raw))
    output_digest = _parser_output_digest(normalized)
    if output_digest != run["parser_output_digest"]:
        raise ValueError("parser_output_digest does not match canonical section records")
    manifest = run.get("complete_manifest")
    complete = 0
    manifest_version: str | None = None
    manifest_digest: str | None = None
    coverage_digest: str | None = None
    if manifest is not None:
        if not isinstance(manifest, Mapping):
            raise ValueError("complete_manifest must be an object")
        manifest_version = manifest.get("version") if isinstance(manifest.get("version"), str) else None
        declared_count = manifest.get("declared_section_count")
        manifest_digest = manifest.get("output_digest") if isinstance(manifest.get("output_digest"), str) else None
        coverage_digest = manifest.get("native_word_coverage_digest") if isinstance(
            manifest.get("native_word_coverage_digest"), str
        ) else None
        if (
            not manifest_version
            or not isinstance(declared_count, int)
            or declared_count != len(normalized)
            or manifest_digest != output_digest
            or not _is_sha256(coverage_digest)
        ):
            raise ValueError("complete_manifest does not match the staged parser records")
        seen_anchors: set[str] = set()
        for record in normalized:
            anchors = record["native_word_anchors"]
            if (
                not isinstance(anchors, list)
                or not anchors
                or any(not _is_sha256(anchor) for anchor in anchors)
                or len(set(anchors)) != len(anchors)
                or seen_anchors.intersection(anchors)
            ):
                raise ValueError("complete_manifest requires unique per-section native-word anchors")
            seen_anchors.update(anchors)
            expected_digest = _anchor_digest("section-native-word-anchors-v1", anchors)
            if record["native_word_count"] != len(anchors) or record["native_word_digest"] != expected_digest:
                raise ValueError("per-section native-word coverage digest does not match anchors")
        for item in quarantine_records:
            anchors = item["native_word_anchors"]
            if seen_anchors.intersection(anchors):
                raise ValueError("quarantine native-word anchors overlap active sections")
            seen_anchors.update(anchors)
        if coverage_digest != _native_word_coverage_digest(normalized, quarantine_records):
            raise ValueError("native-word coverage digest does not match section coverage records")
        complete = 1
    conn = _connect(workspace)
    try:
        _migrate_review_workspace(conn)
        conn.execute("BEGIN IMMEDIATE")
        _require_unambiguous_targets(conn)
        now = int(time.time())
        asset_id = "asset:" + _digest(product, fingerprint)
        supplied_asset_provenance = run.get("asset_provenance_hash")
        asset_provenance = str(supplied_asset_provenance or _digest("asset", product, fingerprint))
        existing_asset = conn.execute(
            "SELECT * FROM source_assets WHERE product_code=? AND source_fingerprint=?",
            (product, fingerprint),
        ).fetchone()
        trusted_inventory = bundle.inventory
        inventory_digest = _anchor_digest("asset-native-word-inventory-v1", trusted_inventory.anchors)
        ignored_digest = _ignored_anchor_digest(trusted_inventory.ignored_anchors)
        inventory_manifest_digest = _digest(
            "source-asset-native-inventory-v1", "native-words-v1", "1", inventory_digest,
            str(len(trusted_inventory.anchors)), "constrained-private-native-v1", ignored_digest,
        )
        if existing_asset is None:
            conn.execute(
                """INSERT INTO source_assets
                (asset_id, product_code, source_fingerprint, source_content_fingerprint, provenance_hash, created_at,
                 inventory_profile, native_inventory_version, native_word_anchor_digest,
                 native_word_anchor_count, ignored_anchor_policy, ignored_anchor_digest,
                 inventory_manifest_digest, inventory_bound_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (asset_id, product, fingerprint, fingerprint, asset_provenance, now,
                 "native-words-v1", "1", inventory_digest, len(trusted_inventory.anchors),
                 "constrained-private-native-v1", ignored_digest, inventory_manifest_digest, now),
            )
            existing_asset = conn.execute("SELECT * FROM source_assets WHERE asset_id=?", (asset_id,)).fetchone()
        elif existing_asset["inventory_manifest_digest"] is None:
            conn.execute(
                """UPDATE source_assets SET inventory_profile=?, native_inventory_version=?,
                native_word_anchor_digest=?, native_word_anchor_count=?, ignored_anchor_policy=?,
                ignored_anchor_digest=?, inventory_manifest_digest=?, inventory_bound_at=? WHERE asset_id=?""",
                ("native-words-v1", "1", inventory_digest, len(trusted_inventory.anchors),
                 "constrained-private-native-v1", ignored_digest, inventory_manifest_digest, now, asset_id),
            )
            existing_asset = conn.execute("SELECT * FROM source_assets WHERE asset_id=?", (asset_id,)).fetchone()
        elif (
            existing_asset["inventory_manifest_digest"] is not None
            and (
                existing_asset["inventory_manifest_digest"] != inventory_manifest_digest
                or existing_asset["native_word_anchor_digest"] != inventory_digest
                or int(existing_asset["native_word_anchor_count"] or 0) != len(trusted_inventory.anchors)
                or existing_asset["ignored_anchor_digest"] != ignored_digest
            )
        ):
            raise ValueError("source asset native inventory is immutable once bound")
        if (
            existing_asset is not None
            and supplied_asset_provenance is not None
            and str(existing_asset["provenance_hash"]) != asset_provenance
        ):
            # Older staging mixed parser-layout evidence into an asset-level
            # provenance hash. The immutable native inventory above is the
            # actual source identity, so normalize this derived field only
            # after that complete inventory has matched exactly.
            conn.execute(
                "UPDATE source_assets SET provenance_hash=? WHERE asset_id=?",
                (asset_provenance, asset_id),
            )
            existing_asset = conn.execute(
                "SELECT * FROM source_assets WHERE asset_id=?", (asset_id,)
            ).fetchone()
        source_inventory_digest: str | None = None
        if complete:
            if existing_asset is None or not _is_sha256(existing_asset["inventory_manifest_digest"]):
                raise ValueError(
                    "complete parser runs require an independently bound source-asset native inventory"
                )
            ignored = manifest.get("ignored_anchors") if isinstance(manifest, Mapping) else None
            policy = manifest.get("ignored_anchor_policy") if isinstance(manifest, Mapping) else None
            if not isinstance(ignored, list) or policy != existing_asset["ignored_anchor_policy"]:
                raise ValueError("complete_manifest must use the bound ignored-anchor policy")
            ignored_hashes: list[str] = []
            for item in ignored:
                if (
                    not isinstance(item, Mapping)
                    or not _is_sha256(item.get("anchor_hash"))
                    or not isinstance(item.get("reason"), str)
                    or str(item["reason"]) not in _TRUSTED_IGNORED_ANCHOR_REASONS
                ):
                    raise ValueError("ignored native-word anchors require a hash and constrained furniture reason")
                ignored_hashes.append(str(item["anchor_hash"]))
            section_anchors = [
                str(anchor) for record in normalized for anchor in record["native_word_anchors"]
            ]
            quarantine_anchors = [
                str(anchor)
                for record in quarantine_records
                for anchor in record["native_word_anchors"]
            ]
            assigned_anchors = section_anchors + quarantine_anchors
            if (
                len(set(ignored_hashes)) != len(ignored_hashes)
                or len(set(assigned_anchors)) != len(assigned_anchors)
                or set(ignored_hashes).intersection(assigned_anchors)
                or _ignored_anchor_digest(ignored) != existing_asset["ignored_anchor_digest"]
                or len(assigned_anchors) + len(ignored_hashes) != existing_asset["native_word_anchor_count"]
                or _anchor_digest("asset-native-word-inventory-v1", assigned_anchors + ignored_hashes)
                != existing_asset["native_word_anchor_digest"]
            ):
                raise ValueError(
                    "parser-run coverage does not exactly match the bound source native inventory"
                )
            source_inventory_digest = str(existing_asset["inventory_manifest_digest"])
        run_id = "parser-run:" + _digest(product, fingerprint, str(run["parser_output_digest"]))
        # V3 keyed source sections by (product, content_fingerprint,
        # source_section_id).  Preserve that historical table/foreign-key
        # shape while allowing several parser outputs for one physical asset by
        # giving each output a derived revision key.  The parser-independent
        # fingerprint remains canonically on parser_runs/source_assets.
        run_revision = _digest("parser-run-revision-v1", fingerprint, str(run["parser_output_digest"]))
        exists = conn.execute("SELECT 1 FROM parser_runs WHERE parser_run_id=?", (run_id,)).fetchone()
        if exists is not None:
            raise ValueError("parser run is already staged")
        section_commitments = [_section_commitment(item) for item in normalized]
        bundle_commitment = _digest(
            "trusted-run-commitment-v1", str(run["bundle_seal"]),
            str(run["trusted_parser_output_digest"]), str(inventory_manifest_digest),
            _canonical_json(sorted(section_commitments)),
            _ignored_anchor_digest(trusted_inventory.ignored_anchors),
        )
        conn.execute(
            """INSERT INTO parser_runs
            (parser_run_id, asset_id, product_code, source_fingerprint, parser_version,
             parser_output_digest, state, review_enabled, complete, created_at,
             manifest_version, manifest_digest, declared_section_count, native_word_coverage_digest,
             source_inventory_digest, origin, bundle_seal, bundle_parser_output_digest, bundle_commitment)
            VALUES (?, ?, ?, ?, ?, ?, 'staged', 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, asset_id, product, fingerprint, str(run["parser_version"]), str(run["parser_output_digest"]),
             complete, now, manifest_version, manifest_digest, len(normalized) if complete else None,
             coverage_digest, source_inventory_digest, _TRUSTED_RUN_ORIGIN,
             str(run["bundle_seal"]), str(run["trusted_parser_output_digest"]), bundle_commitment),
        )
        conn.execute(
            """INSERT OR IGNORE INTO source_revisions
            (product_code, content_fingerprint, license, era, parser_version,
             source_schema_version, printing_revision)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (product, run_revision, str(run["license"]), str(run["era"]), str(run["parser_version"]),
             str(run.get("source_schema_version")) if run.get("source_schema_version") is not None else None,
             str(run.get("printing_revision") or f"content-{fingerprint[:16]}")),
        )
        # Compare only against the currently active run for this product.  A
        # duplicate stable anchor there is ambiguous and intentionally queues
        # the section instead of attempting fuzzy reuse.
        active_run = conn.execute(
            "SELECT parser_run_id FROM parser_runs WHERE product_code=? AND state='active' AND review_enabled=1",
            (product,),
        ).fetchone()
        old_by_identity: dict[str, list[sqlite3.Row]] = {}
        if active_run is not None:
            for old in conn.execute(
                """SELECT s.* FROM source_sections AS s WHERE s.parser_run_id=?""",
                (active_run["parser_run_id"],),
            ):
                old_by_identity.setdefault(str(old["stable_identity"]), []).append(old)
        if complete:
            declared_removed = manifest.get("removed_stable_identities") if isinstance(manifest, Mapping) else None
            if (
                declared_removed is not None
                and (
                    not isinstance(declared_removed, list)
                    or any(not isinstance(identity, str) or not identity for identity in declared_removed)
                    or len(set(declared_removed)) != len(declared_removed)
                    or set(declared_removed) != (set(old_by_identity) - seen_identity)
                )
            ):
                raise ValueError("complete_manifest must explicitly account for every removed stable identity")
        counts = {
            "added": 0,
            "changed": 0,
            "unchanged": 0,
            "ambiguous": 0,
            "removed": 0,
            "reused": 0,
            "screening_reused": 0,
        }
        exact_section_mapping: dict[str, str] = {}
        next_shard_ordinal = int(conn.execute(
            """SELECT COALESCE(MAX(shard_ordinal), -1) + 1 FROM review_shards
            WHERE product_code=? AND content_fingerprint=?""", (product, run_revision)
        ).fetchone()[0])
        shard_id: int | None = None
        for ordinal, item in enumerate(sorted(normalized, key=lambda value: str(value["source_section_id"]))):
            if ordinal % shard_size == 0:
                cursor = conn.execute(
                    """INSERT INTO review_shards
                    (product_code, content_fingerprint, shard_ordinal, section_count, parser_run_id)
                    VALUES (?, ?, ?, ?, ?)""",
                    (product, run_revision, next_shard_ordinal + (ordinal // shard_size),
                     min(shard_size, len(normalized) - ordinal), run_id),
                )
                shard_id = int(cursor.lastrowid)
            assert shard_id is not None
            section_key = _digest(product, run_id, str(item["source_section_id"]), str(item["source_section_hash"]))
            conn.execute(
                """INSERT INTO source_sections
                (section_key, product_code, content_fingerprint, source_section_id, source_section_hash,
                 page_start, page_end, printed_page, heading, source_text, shard_id,
                 parser_run_id, stable_identity, provenance_hash, text_hash, layout_flags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (section_key, product, run_revision, item["source_section_id"], item["source_section_hash"],
                 item["page_start"], item["page_end"], item["printed_page"], item["heading"], item["source_text"],
                 shard_id, run_id, item["stable_identity"], item["provenance_hash"], item["text_hash"],
                 _canonical_json(sorted(str(flag) for flag in item.get("layout_flags", [])))),
            )
            inserted = conn.execute("SELECT * FROM source_sections WHERE section_key=?", (section_key,)).fetchone()
            prior = old_by_identity.get(str(item["stable_identity"]), [])
            reused_from: str | None = None
            if not prior:
                counts["added"] += 1
            elif len(prior) != 1:
                counts["ambiguous"] += 1
            else:
                old = prior[0]
                if (
                    str(old["text_hash"]) == item["text_hash"]
                    and old["page_start"] == item["page_start"]
                    and old["page_end"] == item["page_end"]
                    and str(old["printed_page"] or "") == str(item["printed_page"] or "")
                    and str(old["heading"]) == item["heading"]
                    and _canonical_json(sorted(json.loads(str(old["layout_flags"] or "[]"))))
                    == _canonical_json(sorted(item["layout_flags"]))
                ):
                    counts["unchanged"] += 1
                    exact_section_mapping[str(old["section_key"])] = section_key
                    if _clone_resolution(conn, old_section_key=str(old["section_key"]), new_section=inserted, now=now):
                        reused_from = str(old["section_key"])
                        counts["reused"] += 1
                else:
                    counts["changed"] += 1
            conn.execute(
                """INSERT INTO parser_run_sections
                (parser_run_id, section_key, stable_identity, provenance_hash, text_hash, section_commitment, reused_from_section_key)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (run_id, section_key, item["stable_identity"], item["provenance_hash"], item["text_hash"],
                 _section_commitment(item), reused_from),
            )
            conn.executemany(
                "INSERT INTO parser_section_anchors(parser_run_id, section_key, anchor_hash) VALUES (?, ?, ?)",
                [(run_id, section_key, str(anchor)) for anchor in item["native_word_anchors"]],
            )
            conn.executemany(
                """INSERT INTO parser_section_blocks
                   (parser_run_id, section_key, block_ordinal, kind, physical_page,
                    source_text, text_hash, table_json, anchor_digest)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        run_id, section_key, block["block_ordinal"], block["kind"],
                        block["physical_page"], block["source_text"], block["text_hash"],
                        block.get("table_json"), block["anchor_digest"],
                    )
                    for block in blocks_by_section.get(str(item["source_section_id"]), [])
                ],
            )
            conn.executemany(
                """INSERT INTO parser_section_block_anchors
                   (parser_run_id, section_key, block_ordinal, anchor_ordinal, anchor_hash)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (run_id, section_key, block["block_ordinal"], anchor_ordinal, str(anchor))
                    for block in blocks_by_section.get(str(item["source_section_id"]), [])
                    for anchor_ordinal, anchor in enumerate(block["native_word_anchors"])
                ],
            )
        for item in quarantine_records:
            conn.execute(
                """INSERT INTO parser_quarantine
                   (parser_run_id, quarantine_id, product_code, reason, physical_page,
                    source_text, text_hash, anchor_count, anchor_digest)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, item["quarantine_id"], product, item["reason"],
                    item["physical_page"], item["source_text"], item["text_hash"],
                    item["native_word_count"], item["native_word_digest"],
                ),
            )
            conn.executemany(
                """INSERT INTO parser_quarantine_anchors
                   (parser_run_id, quarantine_id, anchor_hash) VALUES (?, ?, ?)""",
                [
                    (run_id, item["quarantine_id"], str(anchor))
                    for anchor in item["native_word_anchors"]
                ],
            )
        counts["screening_reused"] = _clone_screening_resolutions(
            conn,
            parser_run_id=run_id,
            old_to_new=exact_section_mapping,
            now=now,
        )
        conn.executemany(
            "INSERT INTO parser_ignored_anchors(parser_run_id, anchor_hash, reason) VALUES (?, ?, ?)",
            [(run_id, str(item["anchor_hash"]), str(item["reason"])) for item in trusted_inventory.ignored_anchors],
        )
        if active_run is not None:
            counts["removed"] = sum(
                1 for identity in old_by_identity if identity not in seen_identity
            )
        conn.commit()
        return {"parser_run_id": run_id, "state": "staged", "complete": bool(complete), **counts}
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _validate_trusted_run_commitments(
    conn: sqlite3.Connection,
    target: sqlite3.Row,
    *,
    require_stored_coverage_digest: bool = True,
) -> str:
    """Recompute opaque direct-PDF commitments from the private workspace.

    The v8 migration uses the same full validation path before replacing its
    formerly order-sensitive *derived* coverage digest.  Every other staged
    value remains a trust input and must already match exactly.
    """
    if target["origin"] != _TRUSTED_RUN_ORIGIN or target["manifest_version"] != _TRUSTED_MANIFEST_VERSION:
        raise ValueError("parser run is not a publishable direct-PDF run")
    if (
        not _is_sha256(target["bundle_seal"])
        or not _is_sha256(target["bundle_parser_output_digest"])
        or not _is_sha256(target["bundle_commitment"])
        or not _is_sha256(target["native_word_coverage_digest"])
    ):
        raise ValueError("trusted parser run has no sealed bundle commitment")
    asset = conn.execute("SELECT * FROM source_assets WHERE asset_id=?", (target["asset_id"],)).fetchone()
    if asset is None:
        raise ValueError("trusted parser run has no immutable source inventory")
    _validate_trusted_source_inventory(asset)
    rows = conn.execute(
        """SELECT s.*, membership.section_commitment FROM parser_run_sections AS membership
        JOIN source_sections AS s ON s.section_key=membership.section_key
        WHERE membership.parser_run_id=? AND membership.membership_state='present'
        ORDER BY s.stable_identity, s.source_section_id""",
        (target["parser_run_id"],),
    ).fetchall()
    membership_count = int(conn.execute(
        "SELECT COUNT(*) FROM parser_run_sections WHERE parser_run_id=?",
        (target["parser_run_id"],),
    ).fetchone()[0])
    if membership_count != len(rows):
        raise ValueError("trusted staged parser run has non-present section membership")
    commitments: list[str] = []
    anchors: list[str] = []
    records: list[dict[str, object]] = []
    for row in rows:
        if hashlib.sha256(str(row["source_text"]).encode("utf-8")).hexdigest() != row["text_hash"]:
            raise ValueError("trusted parser source text no longer matches staged data")
        _validate_page_provenance(
            row["source_section_id"], row["page_start"], row["page_end"]
        )
        section_anchors = [str(item[0]) for item in conn.execute(
            "SELECT anchor_hash FROM parser_section_anchors WHERE parser_run_id=? AND section_key=? ORDER BY anchor_hash",
            (target["parser_run_id"], row["section_key"]),
        )]
        block_rows = conn.execute(
            """SELECT * FROM parser_section_blocks
               WHERE parser_run_id=? AND section_key=? ORDER BY block_ordinal""",
            (target["parser_run_id"], row["section_key"]),
        ).fetchall()
        block_anchor_union: list[str] = []
        for expected_ordinal, block in enumerate(block_rows):
            if int(block["block_ordinal"]) != expected_ordinal or hashlib.sha256(
                str(block["source_text"]).encode("utf-8")
            ).hexdigest() != block["text_hash"]:
                raise ValueError("trusted parser structural block is corrupted")
            block_anchors = [str(item[0]) for item in conn.execute(
                """SELECT anchor_hash FROM parser_section_block_anchors
                   WHERE parser_run_id=? AND section_key=? AND block_ordinal=?
                   ORDER BY anchor_ordinal""",
                (target["parser_run_id"], row["section_key"], expected_ordinal),
            )]
            if (
                not block_anchors
                or _anchor_digest("section-block-native-word-anchors-v1", block_anchors)
                != block["anchor_digest"]
            ):
                raise ValueError("trusted parser structural block anchors are corrupted")
            if block["table_json"] is not None:
                try:
                    table = json.loads(str(block["table_json"]))
                except json.JSONDecodeError as exc:
                    raise ValueError("trusted parser table structure is corrupted") from exc
                if block["kind"] != "table" or not isinstance(table, list):
                    raise ValueError("trusted parser table structure is invalid")
            block_anchor_union.extend(block_anchors)
        if block_rows and sorted(block_anchor_union) != sorted(section_anchors):
            raise ValueError("trusted parser structural blocks do not cover their section")
        try:
            flags = json.loads(str(row["layout_flags"] or "[]"))
        except json.JSONDecodeError as exc:
            raise ValueError("trusted parser run layout flags are corrupted") from exc
        record = {
            "source_section_id": row["source_section_id"], "source_section_hash": row["source_section_hash"],
            "text_hash": row["text_hash"], "heading": row["heading"], "stable_identity": row["stable_identity"],
            "provenance_hash": row["provenance_hash"], "page_start": row["page_start"], "page_end": row["page_end"],
            "printed_page": row["printed_page"], "layout_flags": flags,
            "native_word_count": len(section_anchors),
            "native_word_digest": _anchor_digest(
                "section-native-word-anchors-v1", section_anchors
            ),
            "native_word_anchors": section_anchors,
        }
        commitment = _section_commitment(record)
        if commitment != row["section_commitment"]:
            raise ValueError("trusted parser section commitment no longer matches staged data")
        records.append(record)
        commitments.append(commitment)
        anchors.extend(section_anchors)
    parser_output_digest = _parser_output_digest(records)
    quarantine_records: list[dict[str, object]] = []
    for item in conn.execute(
        """SELECT * FROM parser_quarantine WHERE parser_run_id=?
           ORDER BY quarantine_id""",
        (target["parser_run_id"],),
    ):
        quarantine_anchors = [str(row[0]) for row in conn.execute(
            """SELECT anchor_hash FROM parser_quarantine_anchors
               WHERE parser_run_id=? AND quarantine_id=? ORDER BY anchor_hash""",
            (target["parser_run_id"], item["quarantine_id"]),
        )]
        if (
            hashlib.sha256(str(item["source_text"]).encode("utf-8")).hexdigest()
            != item["text_hash"]
            or len(quarantine_anchors) != int(item["anchor_count"])
            or _anchor_digest("quarantine-native-word-anchors-v1", quarantine_anchors)
            != item["anchor_digest"]
        ):
            raise ValueError("trusted parser quarantine record is corrupted")
        quarantine_records.append({
            "quarantine_id": item["quarantine_id"],
            "reason": item["reason"],
            "physical_page": item["physical_page"],
            "native_word_count": len(quarantine_anchors),
            "native_word_digest": item["anchor_digest"],
            "native_word_anchors": quarantine_anchors,
        })
        anchors.extend(quarantine_anchors)
    coverage_digest = _native_word_coverage_digest(records, quarantine_records)
    if (
        parser_output_digest != target["parser_output_digest"]
        or parser_output_digest != target["manifest_digest"]
        or (
            require_stored_coverage_digest
            and coverage_digest != target["native_word_coverage_digest"]
        )
        or len(records) != int(target["declared_section_count"])
    ):
        raise ValueError("trusted parser run metadata no longer matches staged data")
    ignored = [dict(row) for row in conn.execute(
        "SELECT anchor_hash, reason FROM parser_ignored_anchors WHERE parser_run_id=? ORDER BY anchor_hash",
        (target["parser_run_id"],),
    )]
    if len(set(anchors)) != len(anchors) or set(anchors).intersection(item["anchor_hash"] for item in ignored):
        raise ValueError("trusted parser coverage has duplicate anchors")
    if (
        len(anchors) + len(ignored) != asset["native_word_anchor_count"]
        or _anchor_digest("asset-native-word-inventory-v1", anchors + [str(item["anchor_hash"]) for item in ignored])
        != asset["native_word_anchor_digest"]
        or _ignored_anchor_digest(ignored) != asset["ignored_anchor_digest"]
    ):
        raise ValueError("trusted parser coverage no longer matches the source inventory")
    expected = _digest(
        "trusted-run-commitment-v1", str(target["bundle_seal"]), str(target["bundle_parser_output_digest"]),
        str(asset["inventory_manifest_digest"]), _canonical_json(sorted(commitments)),
        _ignored_anchor_digest(ignored),
    )
    if expected != target["bundle_commitment"]:
        raise ValueError("trusted parser run commitment no longer matches staged data")
    return coverage_digest


def _validate_trusted_source_inventory(asset: sqlite3.Row) -> None:
    """Verify the immutable, direct-PDF inventory contract stored on an asset."""
    count = asset["native_word_anchor_count"]
    if (
        asset["inventory_profile"] != "native-words-v1"
        or asset["native_inventory_version"] != "1"
        or asset["ignored_anchor_policy"] != "constrained-private-native-v1"
        or not _is_sha256(asset["native_word_anchor_digest"])
        or not _is_sha256(asset["ignored_anchor_digest"])
        or not _is_sha256(asset["inventory_manifest_digest"])
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
    ):
        raise ValueError("trusted parser run has no immutable source inventory")
    expected = _digest(
        "source-asset-native-inventory-v1",
        str(asset["inventory_profile"]),
        str(asset["native_inventory_version"]),
        str(asset["native_word_anchor_digest"]),
        str(count),
        str(asset["ignored_anchor_policy"]),
        str(asset["ignored_anchor_digest"]),
    )
    if expected != asset["inventory_manifest_digest"]:
        raise ValueError("trusted parser source inventory no longer matches its manifest")


def _repair_staged_native_coverage_digests(conn: sqlite3.Connection) -> None:
    """Repair only v7's order-sensitive derived coverage digest.

    This function is called inside the workspace migration's ``BEGIN
    IMMEDIATE`` transaction.  It deliberately selects no active, retired,
    incomplete, legacy, or review-enabled run, and validates all durable trust
    inputs before its one-column update.
    """
    targets = conn.execute(
        """SELECT * FROM parser_runs
        WHERE state='staged' AND review_enabled=0 AND complete=1 AND origin=?
        ORDER BY parser_run_id""",
        (_TRUSTED_RUN_ORIGIN,),
    ).fetchall()
    for target in targets:
        coverage_digest = _validate_trusted_run_commitments(
            conn,
            target,
            require_stored_coverage_digest=False,
        )
        if coverage_digest != target["native_word_coverage_digest"]:
            conn.execute(
                "UPDATE parser_runs SET native_word_coverage_digest=? WHERE parser_run_id=?",
                (coverage_digest, target["parser_run_id"]),
            )


def activate_parser_run(workspace: Path | str, parser_run_id: str) -> dict[str, object]:
    """Atomically promote a complete staged run when no work lease is live."""
    if not parser_run_id:
        raise ValueError("parser_run_id is required")
    now = int(time.time())
    conn = _connect(workspace)
    try:
        _migrate_review_workspace(conn)
        conn.execute("BEGIN IMMEDIATE")
        target = conn.execute("SELECT * FROM parser_runs WHERE parser_run_id=?", (parser_run_id,)).fetchone()
        if (
            target is None
            or target["state"] != "staged"
            or not int(target["complete"])
            or not isinstance(target["manifest_version"], str)
            or not _is_sha256(target["manifest_digest"])
            or not isinstance(target["declared_section_count"], int)
            or not _is_sha256(target["native_word_coverage_digest"])
            or not _is_sha256(target["source_inventory_digest"])
        ):
            raise ValueError(
                "only a complete staged parser run with a versioned manifest and native-word coverage may be activated"
            )
        _validate_trusted_run_commitments(conn, target)
        staged_count = int(conn.execute(
            "SELECT COUNT(*) FROM parser_run_sections WHERE parser_run_id=? AND membership_state='present'",
            (parser_run_id,),
        ).fetchone()[0])
        if staged_count != int(target["declared_section_count"]):
            raise ValueError("staged parser-run membership no longer matches its complete manifest")
        asset = conn.execute(
            "SELECT inventory_manifest_digest FROM source_assets WHERE asset_id=?", (target["asset_id"],)
        ).fetchone()
        if asset is None or asset["inventory_manifest_digest"] != target["source_inventory_digest"]:
            raise ValueError("source-asset native inventory binding no longer matches the staged parser run")
        live = conn.execute(
            """SELECT 1 FROM review_shards AS sh JOIN parser_runs AS p ON p.parser_run_id=sh.parser_run_id
            WHERE p.product_code=? AND sh.claimant IS NOT NULL AND sh.lease_expires_at >= ? LIMIT 1""",
            (target["product_code"], now),
        ).fetchone()
        live_review = conn.execute(
            """SELECT 1 FROM review_claims AS rc JOIN candidates AS c ON c.candidate_id=rc.candidate_id
            JOIN source_sections AS s ON s.section_key=c.section_key JOIN parser_runs AS p ON p.parser_run_id=s.parser_run_id
            WHERE p.product_code=? AND rc.lease_expires_at >= ? LIMIT 1""",
            (target["product_code"], now),
        ).fetchone()
        live_screen = conn.execute(
            """SELECT 1 FROM draft_screening_claims AS claim
               JOIN parser_runs AS p ON p.parser_run_id=claim.parser_run_id
               WHERE p.product_code=? AND claim.lease_expires_at >= ? LIMIT 1""",
            (target["product_code"], now),
        ).fetchone()
        if live is not None or live_review is not None or live_screen is not None:
            raise PermissionError("cannot activate parser run while a product work lease is live")
        conn.execute(
            """UPDATE parser_runs SET state='retired', review_enabled=0
            WHERE product_code=? AND state='active'""", (target["product_code"],),
        )
        conn.execute(
            """UPDATE parser_runs SET state='active', review_enabled=1, activated_at=?
            WHERE parser_run_id=?""", (now, parser_run_id),
        )
        conn.execute(
            """INSERT OR IGNORE INTO review_product_scope
               (product_code, enabled, reason, updated_at)
               VALUES (?, 1, 'enabled', ?)""",
            (target["product_code"], now),
        )
        conn.execute(
            """UPDATE parser_run_sections SET membership_state='retired'
            WHERE parser_run_id IN (SELECT parser_run_id FROM parser_runs
                                    WHERE product_code=? AND state='retired')""",
            (target["product_code"],),
        )
        conn.commit()
        return {"parser_run_id": parser_run_id, "product_code": target["product_code"], "state": "active"}
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def set_review_target(workspace: Path | str, parser_run_id: str) -> dict[str, object]:
    """Compatibility name for the atomic active review-target switch."""
    return activate_parser_run(workspace, parser_run_id)


def _notices(notices_path: Path | str) -> dict[str, dict[str, str]]:
    raw = json.loads(Path(notices_path).read_text(encoding="utf-8"))
    records: dict[str, dict[str, str]] = {}
    if isinstance(raw, Mapping):
        items = raw.items()
    elif isinstance(raw, list):
        items = ((item.get("key"), item) for item in raw if isinstance(item, Mapping))
    else:
        raise ValueError("notices JSON must be an object or an array")
    for key, value in items:
        key = _validate_public_scalar(key, field="notice key")
        if isinstance(value, str):
            record = {"license": key, "text": value}
        elif isinstance(value, Mapping):
            record = {"license": value.get("license", key), "text": value.get("text", "")}
        else:
            raise ValueError(f"notice {key!r} has invalid content")
        license_name = _validate_public_scalar(record["license"], field="notice license")
        if license_name not in {"OGL", "ORC"}:
            raise ValueError(f"notice {key!r} has unsupported license")
        text = _validate_public_scalar(record["text"], field="notice text")
        records[key] = {"license": license_name, "text": text}
    return records


def _approved_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    sections = conn.execute(
        """SELECT * FROM source_sections AS s WHERE """ + _review_target_sql("s")
        + " ORDER BY product_code, content_fingerprint, source_section_id"
    ).fetchall()
    approved: list[sqlite3.Row] = []
    for section in sections:
        terminal = conn.execute(
            """SELECT 1 WHERE EXISTS (
                 SELECT 1 FROM duplicate_group_members AS member
                 WHERE member.section_key=? AND member.source_ordinal>0
               ) OR EXISTS (
                 SELECT 1 FROM foundry_coverage_confirmations AS coverage
                 JOIN metadata AS active ON active.key='active_foundry_snapshot'
                    AND active.value=coverage.snapshot_digest
                 WHERE coverage.section_key=?
               )""",
            (section["section_key"], section["section_key"]),
        ).fetchone()
        if terminal is not None:
            continue
        candidates = conn.execute(
            "SELECT * FROM candidates WHERE section_key=? ORDER BY candidate_ordinal",
            (section["section_key"],),
        ).fetchall()
        if not candidates:
            raise ValueError(f"unreviewed source section: {section['source_section_id']}")
        public_candidates: list[sqlite3.Row] = []
        for candidate in candidates:
            _validate_candidate_commitment(candidate)
            _validate_candidate_layout(candidate, section)
            if candidate["source_section_hash"] != section["source_section_hash"]:
                raise ValueError("candidate source hash no longer matches its source section")
            reviews = _active_valid_reviews(conn, candidate)
            verdicts = {str(review["verdict"]) for review in reviews}
            if len(verdicts) > 1:
                raise ValueError(f"ambiguous review verdicts: {candidate['candidate_id']}")
            if "APPROVE" in verdicts:
                if (candidate["decision"] not in PUBLIC_DECISIONS or not candidate["candidate_text"]
                        or not candidate["public_heading"]):
                    raise ValueError(f"invalid approved candidate: {candidate['candidate_id']}")
                policies = {str(review["policy_version"]) for review in reviews}
                if len(policies) != 1:
                    raise ValueError(f"ambiguous approval policy: {candidate['candidate_id']}")
                public_candidates.append(candidate)
        if len(public_candidates) > 1:
            raise ValueError(f"ambiguous multiple approvals: {section['source_section_id']}")
        latest = candidates[-1]
        latest_verdicts = {str(row["verdict"]) for row in _active_valid_reviews(conn, latest)}
        if not latest_verdicts:
            raise ValueError(f"unreviewed candidate: {latest['candidate_id']}")
        if latest_verdicts == {"REVISE"}:
            raise ValueError(f"rework remains pending: {section['source_section_id']}")
        if latest_verdicts == {"APPROVE"}:
            if (latest["decision"] not in PUBLIC_DECISIONS or not latest["candidate_text"]
                    or not latest["public_heading"]):
                raise ValueError(f"invalid approved candidate: {latest['candidate_id']}")
            approved.append(latest)
            continue
        if latest["decision"] not in {"EXCLUDE", "UNCERTAIN"} or latest_verdicts != {"REJECT"}:
            raise ValueError(f"unresolved source section: {section['source_section_id']}")
    return approved


def build_public_corpus(
    workspace: Path | str,
    output: Path | str,
    notices_path: Path | str,
) -> dict[str, int]:
    """Build a compact v3 public base with scoped many-source provenance."""
    notices = _notices(notices_path)
    _ensure_workspace_migrated(workspace)
    review = _connect(workspace, readonly=True)
    try:
        _require_unambiguous_targets(review)
        scope_rows = _review_scope_rows(review)
        covered_products = [
            str(row["product_code"]) for row in scope_rows if int(row["enabled"])
        ]
        if not covered_products:
            raise ValueError("public build requires a non-empty semantic product scope")
        scope_manifest = [
            {"product_code": product, "state": "enabled"}
            for product in covered_products
        ]
        scope_digest = _digest(REVIEW_SCOPE_VERSION, _canonical_json(scope_manifest))
        untrusted = review.execute(
            """SELECT 1 FROM parser_runs WHERE state='active' AND review_enabled=1
               AND (origin != ? OR complete != 1 OR manifest_version != ?) LIMIT 1""",
            (_TRUSTED_RUN_ORIGIN, _TRUSTED_MANIFEST_VERSION),
        ).fetchone()
        if untrusted is not None:
            raise ValueError("public build requires a reviewed complete direct-PDF parser run")
        for target in review.execute(
            "SELECT * FROM parser_runs WHERE state='active' AND review_enabled=1"
        ):
            _validate_trusted_run_commitments(review, target)

        approved = _approved_rows(review)
        catalog_order = {code: ordinal for ordinal, code in enumerate(PRODUCT_CATALOG)}
        prepared: list[dict[str, object]] = []
        revisions: dict[tuple[str, str], sqlite3.Row] = {}
        licenses: set[str] = set()
        for candidate in approved:
            primary = review.execute(
                """SELECT s.*, r.license, r.era, r.source_schema_version,
                          r.printing_revision, p.parser_version AS parser_run_version,
                          a.source_content_fingerprint AS canonical_content_fingerprint
                     FROM source_sections AS s
                     JOIN source_revisions AS r USING(product_code, content_fingerprint)
                     JOIN parser_runs AS p ON p.parser_run_id=s.parser_run_id
                     JOIN source_assets AS a ON a.asset_id=p.asset_id
                    WHERE s.section_key=?""",
                (candidate["section_key"],),
            ).fetchone()
            assert primary is not None
            approvals = [
                row for row in _active_valid_reviews(review, candidate)
                if row["verdict"] == "APPROVE"
            ]
            if not approvals:
                raise ValueError("approved candidate has no trusted approval")
            policy_versions = {str(row["policy_version"]) for row in approvals}
            if len(policy_versions) != 1:
                raise ValueError("approved candidate has ambiguous policy provenance")
            text = str(candidate["candidate_text"])
            heading = _validate_public_scalar(candidate["public_heading"], field="candidate heading")
            _validate_public_text(text, field="candidate text")
            method = _validate_public_scalar(candidate["extraction_method"], field="extraction method")
            group = review.execute(
                "SELECT group_id FROM duplicate_group_members WHERE section_key=?",
                (candidate["section_key"],),
            ).fetchone()
            if group is None:
                source_rows = [primary]
            else:
                source_rows = review.execute(
                    """SELECT s.*, r.license, r.era, r.source_schema_version,
                              r.printing_revision, p.parser_version AS parser_run_version,
                              a.source_content_fingerprint AS canonical_content_fingerprint
                         FROM duplicate_group_members AS member
                         JOIN source_sections AS s ON s.section_key=member.section_key
                         JOIN source_revisions AS r USING(product_code, content_fingerprint)
                         JOIN parser_runs AS p ON p.parser_run_id=s.parser_run_id
                         JOIN source_assets AS a ON a.asset_id=p.asset_id
                        WHERE member.group_id=?
                          AND """ + _semantic_scope_sql("s.product_code") +
                        """ ORDER BY member.source_ordinal""",
                    (group["group_id"],),
                ).fetchall()
            prepared.append({
                "candidate": candidate, "primary": primary, "sources": source_rows,
                "heading": heading, "text": text, "method": method,
                "policy_version": next(iter(policy_versions)),
                "public_group": duplicate_identity(
                    heading=heading, text=text, license_name=primary["license"], era=primary["era"],
                ),
            })

        public_groups: dict[str, list[dict[str, object]]] = {}
        for item in prepared:
            public_groups.setdefault(str(item["public_group"]), []).append(item)
        rules: list[dict[str, object]] = []
        rule_sources: list[dict[str, object]] = []
        for _group_id, items in sorted(public_groups.items()):
            items.sort(key=lambda item: (
                catalog_order.get(str(item["primary"]["product_code"]), 999),
                int(item["primary"]["page_start"] or 0),
                str(item["primary"]["stable_identity"]),
            ))
            canonical = items[0]
            primary = canonical["primary"]
            policy_versions = {str(item["policy_version"]) for item in items}
            if len(policy_versions) != 1:
                raise ValueError("deduplicated public rule has inconsistent policy versions")
            public_id = "licensed:" + _digest(
                "licensed-rule-v2", str(primary["stable_identity"])
            )
            text = str(canonical["text"])
            license_name = str(primary["license"])
            era = str(primary["era"])
            licenses.add(license_name)
            rules.append({
                "public_id": public_id,
                "heading": canonical["heading"],
                "text": text,
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "license": license_name,
                "era": era,
                "method": canonical["method"],
                "policy_version": next(iter(policy_versions)),
            })
            sources_by_key: dict[str, sqlite3.Row] = {}
            for item in items:
                for source in item["sources"]:
                    sources_by_key[str(source["section_key"])] = source
            sources = sorted(sources_by_key.values(), key=lambda source: (
                catalog_order.get(str(source["product_code"]), 999),
                int(source["page_start"] or 0), str(source["stable_identity"]),
            ))
            for ordinal, source in enumerate(sources):
                for field, value in (
                    ("product", source["product_code"]),
                    ("source section ID", source["source_section_id"]),
                    ("source section hash", source["text_hash"]),
                    ("content fingerprint", source["canonical_content_fingerprint"]),
                    ("parser version", source["parser_run_version"]),
                    ("printing revision", source["printing_revision"]),
                ):
                    _validate_public_scalar(value, field=field)
                if not _PUBLIC_SOURCE_ID_RE.fullmatch(str(source["source_section_id"])):
                    raise ValueError("public source section ID has unsafe structure")
                _validate_page_provenance(
                    source["source_section_id"], source["page_start"], source["page_end"]
                )
                if source["printed_page"] is not None:
                    _validate_public_scalar(source["printed_page"], field="printed page")
                fingerprint = str(source["canonical_content_fingerprint"])
                key = (str(source["product_code"]), fingerprint)
                revisions[key] = source
                rule_sources.append({
                    "public_id": public_id, "ordinal": ordinal,
                    "product": key[0], "fingerprint": fingerprint,
                    "source_id": source["source_section_id"], "source_hash": source["text_hash"],
                    "page_start": source["page_start"], "page_end": source["page_end"],
                    "printed_page": source["printed_page"],
                    "parser_version": source["parser_run_version"],
                    "printing_revision": source["printing_revision"],
                    "notice_key": source["license"],
                })

        active_snapshot = review.execute(
            "SELECT value FROM metadata WHERE key='active_foundry_snapshot'"
        ).fetchone()
        requirements: list[dict[str, object]] = []
        foundry_release = "none"
        snapshot_digest = "none"
        if active_snapshot is not None:
            snapshot_digest = str(active_snapshot[0])
            snapshot_row = review.execute(
                "SELECT pf2e_release FROM foundry_snapshots WHERE snapshot_digest=?",
                (snapshot_digest,),
            ).fetchone()
            if snapshot_row is None:
                raise ValueError("active Foundry snapshot metadata is missing")
            foundry_release = str(snapshot_row["pf2e_release"])
            requirements = [dict(row) for row in review.execute(
                """SELECT DISTINCT snapshot.foundry_id, snapshot.source_hash,
                          snapshot.normalized_hash, snapshot.publication_title,
                          snapshot.license, snapshot.era
                     FROM foundry_coverage_confirmations AS confirmation
                     JOIN source_sections AS section
                       ON section.section_key=confirmation.section_key,
                          json_each(confirmation.foundry_ids_json) AS selected
                     JOIN foundry_snapshot_rows AS snapshot
                       ON snapshot.snapshot_digest=confirmation.snapshot_digest
                      AND snapshot.foundry_id=selected.value
                    WHERE confirmation.snapshot_digest=?
                      AND """ + _semantic_scope_sql("section.product_code") +
                    """ ORDER BY snapshot.foundry_id""",
                (snapshot_digest,),
            )]
    finally:
        review.close()

    missing_notices = sorted(license_name for license_name in licenses if license_name not in notices)
    if missing_notices:
        raise ValueError("required notice keys are missing: " + ", ".join(missing_notices))
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged = output_path.with_name(f".{output_path.name}.staging-{os.getpid()}")
    staged.unlink(missing_ok=True)
    conn = sqlite3.connect(staged)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE source_revisions (
                product_code TEXT NOT NULL, content_fingerprint TEXT NOT NULL,
                license TEXT NOT NULL, era TEXT NOT NULL, parser_version TEXT NOT NULL,
                source_schema_version TEXT, printing_revision TEXT NOT NULL,
                PRIMARY KEY(product_code, content_fingerprint));
            CREATE TABLE notices (
                notice_key TEXT PRIMARY KEY, license TEXT NOT NULL, text TEXT NOT NULL);
            CREATE TABLE licensed_rules (
                public_id TEXT PRIMARY KEY, heading TEXT NOT NULL, text TEXT NOT NULL,
                content_hash TEXT NOT NULL, license TEXT NOT NULL, era TEXT NOT NULL,
                extraction_method TEXT NOT NULL, policy_version TEXT NOT NULL,
                notice_key TEXT NOT NULL REFERENCES notices(notice_key));
            CREATE TABLE licensed_rule_sources (
                public_id TEXT NOT NULL REFERENCES licensed_rules(public_id),
                source_ordinal INTEGER NOT NULL, product_code TEXT NOT NULL,
                content_fingerprint TEXT NOT NULL, source_section_id TEXT NOT NULL,
                source_section_hash TEXT NOT NULL, page_start INTEGER NOT NULL,
                page_end INTEGER NOT NULL, printed_page TEXT, parser_version TEXT NOT NULL,
                printing_revision TEXT NOT NULL, notice_key TEXT NOT NULL REFERENCES notices(notice_key),
                PRIMARY KEY(public_id, source_ordinal),
                UNIQUE(public_id, product_code, content_fingerprint, source_section_id),
                FOREIGN KEY(product_code, content_fingerprint)
                    REFERENCES source_revisions(product_code, content_fingerprint));
            CREATE TABLE required_foundry_rows (
                foundry_id TEXT PRIMARY KEY, source_hash TEXT NOT NULL,
                normalized_hash TEXT NOT NULL, publication_title TEXT NOT NULL,
                license TEXT NOT NULL, era TEXT NOT NULL);
            CREATE INDEX licensed_rule_sources_by_product
                ON licensed_rule_sources(product_code, content_fingerprint);
            """
        )
        conn.executemany("INSERT INTO metadata VALUES (?, ?)", [
            ("public_schema_version", str(PUBLIC_SCHEMA_VERSION)),
            ("content_scope", "licensed-core-reviewed"),
            ("policy_version", LICENSED_CORE_POLICY_VERSION),
            ("policy_digest", licensed_policy_digest()),
            ("normalizer_version", NORMALIZER_VERSION),
            ("review_scope_version", REVIEW_SCOPE_VERSION),
            ("covered_products", _canonical_json(covered_products)),
            ("review_scope_digest", scope_digest),
            ("foundry_release", foundry_release),
            ("foundry_snapshot_digest", snapshot_digest),
        ])
        for key in sorted(notices):
            conn.execute("INSERT INTO notices VALUES (?, ?, ?)", (
                key, notices[key]["license"], notices[key]["text"],
            ))
        for key, revision in sorted(revisions.items()):
            conn.execute("INSERT INTO source_revisions VALUES (?, ?, ?, ?, ?, ?, ?)", (
                key[0], key[1], revision["license"], revision["era"],
                revision["parser_run_version"], revision["source_schema_version"],
                revision["printing_revision"],
            ))
        for row in sorted(rules, key=lambda item: str(item["public_id"])):
            conn.execute("INSERT INTO licensed_rules VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                row["public_id"], row["heading"], row["text"], row["content_hash"],
                row["license"], row["era"], row["method"], row["policy_version"], row["license"],
            ))
        for row in sorted(rule_sources, key=lambda item: (str(item["public_id"]), int(item["ordinal"]))):
            conn.execute("INSERT INTO licensed_rule_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                row["public_id"], row["ordinal"], row["product"], row["fingerprint"],
                row["source_id"], row["source_hash"], row["page_start"], row["page_end"],
                row["printed_page"], row["parser_version"], row["printing_revision"], row["notice_key"],
            ))
        for row in requirements:
            conn.execute("INSERT INTO required_foundry_rows VALUES (?, ?, ?, ?, ?, ?)", (
                row["foundry_id"], row["source_hash"], row["normalized_hash"],
                row["publication_title"], row["license"], row["era"],
            ))
        conn.commit()
        conn.execute("VACUUM")
    except BaseException:
        conn.close()
        staged.unlink(missing_ok=True)
        raise
    else:
        conn.close()
    os.replace(staged, output_path)
    return {
        "sections": len(rules), "sources": len(rule_sources),
        "foundry_requirements": len(requirements), "revisions": len(revisions),
        "notices": len(notices), "covered_products": len(covered_products),
        "review_scope_digest": scope_digest,
    }


__all__ = [
    "POLICY_DECISIONS",
    "PUBLIC_DECISIONS",
    "PUBLIC_SCHEMA_VERSION",
    "REVIEW_SCHEMA_VERSION",
    "REVIEW_SCOPE_VERSION",
    "REVIEW_VERDICTS",
    "activate_parser_run",
    "build_public_corpus",
    "claim_review",
    "claim_shard",
    "claim_draft_screening_batch",
    "draft_screening_status",
    "foundry_coverage_evidence",
    "invalidate_reviews",
    "initialize_trusted_workspace",
    "initialize_workspace",
    "next_draft_screening_record",
    "prepare_deterministic_review",
    "read_claimed_shard",
    "read_claimed_review",
    "read_draft_screening_record",
    "review_product_scope",
    "reclaim_interrupted_shard",
    "release_draft_screening_batch",
    "reopen_draft_screening",
    "set_review_target",
    "set_review_product_scope",
    "stage_trusted_native_pdf",
    "stage_trusted_native_pdf_with_approved_stitches",
    "step_draft_screening",
    "submit_candidate",
    "submit_draft_screening_decision",
    "submit_review",
    "workspace_status",
]
