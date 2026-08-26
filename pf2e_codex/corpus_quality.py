"""Content-free quality audits for private licensed-corpus workspaces.

The audit deliberately reads a review database through SQLite's read-only
URI.  It reports counts and digests only: source text, paths, anchors,
watermarks, and other private values never cross the API boundary.

The runner can use :func:`audit_workspace` for the current active parser runs
and :func:`compare_quality` to evaluate a candidate against a saved baseline.
This module has no dependency on the review runner and does not mutate the
workspace.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path

QUALITY_AUDIT_VERSION = "corpus-quality-v1"

# These are deliberately part of the audit contract.  A future parser may add
# a new internal flag without causing that arbitrary value to be returned to a
# caller or logged by a CLI wrapper.
_ORDER_CONFLICT_FLAG = "layout-order-conflict"
_UNCLASSIFIED_FLAG = "unclassified-native-coverage"
# ``table-ambiguous`` is V5's exact-native-text recovery path. It is active
# review work and is forced through mixed extraction plus independent review;
# unlike the older ``layout-model-table`` flag, it does not represent missing
# or unresolved source coverage.
_UNRESOLVED_TABLE_FLAGS = {"layout-model-table"}
_RULE_BEARING_QUARANTINE_REASONS = frozenset({
    "unresolved-table",
    "unbound-layout",
    "heading-artifact",
    "unresolved-continuation",
    "unresolved-layout",
    "layout-order-conflict",
    "oversize-block",
    "other",
})
_QUARANTINE_REASONS = frozenset(
    {
        "repeated-furniture",
        "page-number",
        "contents-index",
        "credits-legal",
        "unresolved-table",
        "unbound-layout",
        "heading-artifact",
        "unresolved-continuation",
        "unresolved-layout",
        "layout-order-conflict",
        "oversize-block",
        # Existing native-export reasons are safe to aggregate too.
        "printed-page-number-v1",
        "repeated-margin-furniture-v1",
        "watermark-email-span-v1",
        "watermark-identity-row-v1",
    }
)
_LENGTH_BUCKETS = (
    "under-40",
    "40-79",
    "80-499",
    "500-1999",
    "2000-4999",
    "5000-9999",
    "10000-plus",
)
_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_WINDOWS_PATH_RE = re.compile(r"(?i)(?:^|\s)[A-Z]:[\\/]")
_PRIVATE_MARKERS = (".local-corpus", "source_sha256", "source_path", "file://")
_PRODUCT_RE = re.compile(r"^PZO[0-9]+$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_GATES = {
    # V3 reports affected sections while V4 quarantines one bounded record per
    # contradictory page. Keep a strong aggregate reduction requirement, but
    # do not pretend those historical units are identical. Quarantined-anchor
    # ratios independently bound the amount of affected source material.
    "layout_order_conflicts_overall": 0.85,
    "layout_order_conflicts_per_product": 0.75,
    "short_fragments_overall": 0.80,
    "short_fragments_per_product": 0.50,
    "sentence_like_headings_overall": 0.75,
}
_QUALITY_PROBES: Mapping[str, tuple[tuple[str, ...], ...]] = {
    # Each outer tuple is an OR; every term within one inner tuple must occur
    # in the same section. These are public query labels, never emitted source.
    "afflictions": (("affliction",),),
    "difficulty-classes": (("difficulty class",),),
    "dying-recovery": (("dying", "recovery check"),),
    "encounter-building": (
        ("building encounters",),
        ("encounter building",),
        ("encounter budget",),
    ),
    "exploration": (("exploration",),),
    "fireball": (("fireball",),),
    "line-of-effect": (("line of effect",),),
}


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(*parts: object) -> str:
    return hashlib.sha256(
        "\n".join(_canonical(part) for part in parts).encode("utf-8")
    ).hexdigest()


def _safe_product_code(value: object) -> str:
    code = str(value or "")
    return code if _PRODUCT_RE.fullmatch(code) else "unknown"


def _safe_run_id(value: object) -> str:
    run_id = str(value or "")
    return run_id if _RUN_ID_RE.fullmatch(run_id) else "[REDACTED]"


def _has_private_value(value: object) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.casefold()
    return bool(
        _EMAIL_RE.search(value)
        or _WINDOWS_PATH_RE.search(value)
        or any(marker in lowered for marker in _PRIVATE_MARKERS)
        or "\\users\\" in lowered
        or "/home/" in lowered
        or "/users/" in lowered
    )


def _connect_readonly(workspace: Path | str) -> sqlite3.Connection:
    path = Path(workspace).expanduser().resolve()
    if not path.is_file():
        # Do not echo a caller's local path through a CLI error or audit log.
        raise FileNotFoundError("review workspace does not exist")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _json_flags(value: object) -> tuple[set[str], bool]:
    if value is None or value == "":
        return set(), False
    try:
        decoded = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return set(), True
    if not isinstance(decoded, list) or any(not isinstance(item, str) for item in decoded):
        return set(), True
    return set(decoded), False


def _numeric_only_heading(value: str) -> bool:
    # Punctuation and whitespace around a page number are not meaningful
    # heading content, but a normal one-letter heading remains valid.
    compact = re.sub(r"[\s\W_]", "", value, flags=re.UNICODE)
    return bool(compact) and compact.isdecimal()


def _sentence_like_heading(value: str) -> bool:
    text = " ".join(value.split())
    if not text:
        return False
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    if len(text) > 80 or len(words) >= 16:
        return True
    return len(words) >= 8 and bool(re.search(r"[.!?;:]$", text))


def _length_bucket(length: int) -> str:
    if length < 40:
        return "under-40"
    if length < 80:
        return "40-79"
    if length < 500:
        return "80-499"
    if length < 2000:
        return "500-1999"
    if length < 5000:
        return "2000-4999"
    if length < 10000:
        return "5000-9999"
    return "10000-plus"


def _matched_probes(heading: str, source_text: str) -> tuple[str, ...]:
    haystack = " ".join(f"{heading} {source_text}".casefold().split())
    return tuple(
        probe
        for probe, alternatives in _QUALITY_PROBES.items()
        if any(all(term in haystack for term in required) for required in alternatives)
    )


def _select_runs(
    conn: sqlite3.Connection,
    parser_run_ids: Mapping[str, str] | None,
) -> list[sqlite3.Row]:
    if "parser_runs" not in _tables(conn):
        raise ValueError("review workspace is missing parser_runs")
    if parser_run_ids is not None:
        selected: list[sqlite3.Row] = []
        for product_code, run_id in sorted(parser_run_ids.items()):
            row = conn.execute(
                "SELECT parser_run_id, product_code, asset_id FROM parser_runs WHERE parser_run_id=?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"requested parser run does not exist: {_safe_run_id(run_id)}")
            if _safe_product_code(row["product_code"]) != _safe_product_code(product_code):
                raise ValueError("requested parser run product does not match its selector")
            selected.append(row)
        return selected

    rows = conn.execute(
        """SELECT parser_run_id, product_code, asset_id, activated_at, created_at
           FROM parser_runs
           WHERE state='active'
           ORDER BY product_code,
                    COALESCE(activated_at, 0) DESC,
                    COALESCE(created_at, 0) DESC,
                    parser_run_id DESC"""
    ).fetchall()
    selected_by_product: dict[str, sqlite3.Row] = {}
    for row in rows:
        product = _safe_product_code(row["product_code"])
        selected_by_product.setdefault(product, row)
    return [selected_by_product[key] for key in sorted(selected_by_product)]


def _section_rows(conn: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    tables = _tables(conn)
    if "source_sections" not in tables:
        raise ValueError("review workspace is missing source_sections")
    columns = _columns(conn, "source_sections")
    # parser_run_sections is authoritative when available. A retired membership
    # means the whole parser run is historical, not that its section vanished;
    # explicit baseline comparisons must therefore continue to see those rows.
    if "parser_run_sections" in tables:
        return conn.execute(
            """SELECT sections.*
                FROM source_sections AS sections
                JOIN parser_run_sections AS membership
                  ON membership.section_key=sections.section_key
                WHERE membership.parser_run_id=?
                ORDER BY sections.page_start, sections.source_section_id, sections.section_key""",
            (run_id,),
        ).fetchall()
    if "parser_run_id" in columns:
        return conn.execute(
            """SELECT * FROM source_sections
               WHERE parser_run_id=?
               ORDER BY page_start, source_section_id, section_key""",
            (run_id,),
        ).fetchall()
    return []


def _asset_anchor_count(
    conn: sqlite3.Connection, run: sqlite3.Row, tables: set[str]
) -> int | None:
    if "source_assets" not in tables or not run["asset_id"]:
        return None
    columns = _columns(conn, "source_assets")
    if "native_word_anchor_count" not in columns:
        return None
    row = conn.execute(
        "SELECT native_word_anchor_count FROM source_assets WHERE asset_id=?",
        (run["asset_id"],),
    ).fetchone()
    value = row[0] if row is not None else None
    return int(value) if isinstance(value, int) and value >= 0 else None


def _anchor_stats(
    conn: sqlite3.Connection, run_id: str, expected: int | None, tables: set[str]
) -> tuple[int | None, int | None, int | None, int | None, int | None, float | None]:
    if (
        "parser_section_anchors" not in tables
        and "parser_quarantine_anchors" not in tables
        and "parser_ignored_anchors" not in tables
    ):
        return None, None, None, None, None, None
    assigned_raw = assigned_distinct = ignored_raw = ignored_distinct = overlap = 0
    assigned_sources = [
        table for table in ("parser_section_anchors", "parser_quarantine_anchors")
        if table in tables
    ]
    if assigned_sources:
        union = " UNION ALL ".join(
            f"SELECT anchor_hash FROM {table} WHERE parser_run_id=?"
            for table in assigned_sources
        )
        row = conn.execute(
            f"""SELECT COUNT(*) AS raw, COUNT(DISTINCT anchor_hash) AS distinct_count
                FROM ({union})""",
            (run_id,) * len(assigned_sources),
        ).fetchone()
        assigned_raw, assigned_distinct = int(row["raw"]), int(row["distinct_count"])
    if "parser_ignored_anchors" in tables:
        row = conn.execute(
            """SELECT COUNT(*) AS raw, COUNT(DISTINCT anchor_hash) AS distinct_count
               FROM parser_ignored_anchors WHERE parser_run_id=?""",
            (run_id,),
        ).fetchone()
        ignored_raw, ignored_distinct = int(row["raw"]), int(row["distinct_count"])
    if assigned_sources and "parser_ignored_anchors" in tables:
        union = " UNION ".join(
            f"SELECT anchor_hash FROM {table} WHERE parser_run_id=?"
            for table in assigned_sources
        )
        overlap = int(
            conn.execute(
                f"""SELECT COUNT(*) FROM ({union}) AS assigned
                    JOIN parser_ignored_anchors AS ignored
                      ON ignored.anchor_hash=assigned.anchor_hash
                     AND ignored.parser_run_id=?""",
                (run_id,) * len(assigned_sources) + (run_id,),
            ).fetchone()[0]
        )
    duplicate_count = (assigned_raw - assigned_distinct) + (ignored_raw - ignored_distinct) + overlap
    covered = assigned_distinct + ignored_distinct - overlap
    missing = max(0, expected - covered) if expected is not None else None
    extra = max(0, covered - expected) if expected is not None else None
    ratio = covered / expected if expected not in (None, 0) else (1.0 if expected == 0 and covered == 0 else None)
    return assigned_distinct, ignored_distinct, missing, extra, duplicate_count, ratio


def _quarantine_stats(
    conn: sqlite3.Connection,
    run_id: str,
    product_code: str,
    tables: set[str],
) -> tuple[int, int, dict[str, int], int]:
    if "parser_quarantine" not in tables:
        return 0, 0, {}, 0
    columns = _columns(conn, "parser_quarantine")
    filters: list[str] = []
    params: list[object] = []
    if "parser_run_id" in columns:
        filters.append("parser_run_id=?")
        params.append(run_id)
    elif "product_code" not in columns:
        return 0, 0, {}, 0
    if "product_code" in columns:
        filters.append("product_code=?")
        params.append(product_code)
    select = ["*" if "reason" in columns else "1 AS _row"]
    rows = conn.execute(
        f"SELECT {', '.join(select)} FROM parser_quarantine WHERE {' AND '.join(filters) or '1=1'}",
        params,
    ).fetchall()
    reasons: Counter[str] = Counter()
    anchor_values: set[str] = set()
    privacy_count = 0
    text_columns = [column for column in ("source_text", "text", "heading") if column in columns]
    reason_column = "reason" if "reason" in columns else None
    for row in rows:
        reason = str(row[reason_column] or "") if reason_column else ""
        reasons[reason if reason in _QUARANTINE_REASONS else "other"] += 1
        if any(_has_private_value(row[column]) for column in text_columns):
            privacy_count += 1
    if "parser_quarantine_anchors" in tables:
        anchor_values = {
            str(row[0]) for row in conn.execute(
                "SELECT anchor_hash FROM parser_quarantine_anchors WHERE parser_run_id=?",
                (run_id,),
            )
        }
    elif "anchor_hash" in columns:
        anchor_values = {
            str(row["anchor_hash"]) for row in rows if row["anchor_hash"] is not None
        }
    return len(rows), len(anchor_values), dict(sorted(reasons.items())), privacy_count


@dataclass(frozen=True)
class ProductQuality:
    """Content-free quality metrics for one selected parser run."""

    product_code: str
    parser_run_id: str
    section_count: int
    quarantine_count: int
    quarantined_anchor_count: int
    quarantine_by_reason: Mapping[str, int]
    quarantine_anchor_ratio: float | None
    expected_anchor_count: int | None
    assigned_anchor_count: int | None
    ignored_anchor_count: int | None
    missing_anchor_count: int | None
    extra_anchor_count: int | None
    duplicate_anchor_count: int | None
    anchor_coverage_ratio: float | None
    empty_heading_count: int
    numeric_only_heading_count: int
    sentence_like_heading_count: int
    heading_defect_count: int
    layout_order_conflict_count: int
    layout_metadata_error_count: int
    unclassified_count: int
    unresolved_table_count: int
    short_under_40_count: int
    short_under_80_count: int
    native_fallback_section_count: int
    native_fallback_short_under_40_count: int
    oversize_over_5000_count: int
    oversize_over_10000_count: int
    length_buckets: Mapping[str, int]
    privacy_violation_count: int
    digest: str
    probe_hits: Mapping[str, Mapping[str, int]] = dataclass_field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "product_code": self.product_code,
            "parser_run_id": self.parser_run_id,
            "section_count": self.section_count,
            "quarantine_count": self.quarantine_count,
            "quarantined_anchor_count": self.quarantined_anchor_count,
            "quarantine_by_reason": dict(self.quarantine_by_reason),
            "quarantine_anchor_ratio": self.quarantine_anchor_ratio,
            "expected_anchor_count": self.expected_anchor_count,
            "assigned_anchor_count": self.assigned_anchor_count,
            "ignored_anchor_count": self.ignored_anchor_count,
            "missing_anchor_count": self.missing_anchor_count,
            "extra_anchor_count": self.extra_anchor_count,
            "duplicate_anchor_count": self.duplicate_anchor_count,
            "anchor_coverage_ratio": self.anchor_coverage_ratio,
            "empty_heading_count": self.empty_heading_count,
            "numeric_only_heading_count": self.numeric_only_heading_count,
            "sentence_like_heading_count": self.sentence_like_heading_count,
            "heading_defect_count": self.heading_defect_count,
            "layout_order_conflict_count": self.layout_order_conflict_count,
            "layout_metadata_error_count": self.layout_metadata_error_count,
            "unclassified_count": self.unclassified_count,
            "unresolved_table_count": self.unresolved_table_count,
            "short_under_40_count": self.short_under_40_count,
            "short_under_80_count": self.short_under_80_count,
            "native_fallback_section_count": self.native_fallback_section_count,
            "native_fallback_short_under_40_count": (
                self.native_fallback_short_under_40_count
            ),
            "oversize_over_5000_count": self.oversize_over_5000_count,
            "oversize_over_10000_count": self.oversize_over_10000_count,
            "length_buckets": dict(self.length_buckets),
            "privacy_violation_count": self.privacy_violation_count,
            "probe_hits": {
                probe: dict(values) for probe, values in sorted(self.probe_hits.items())
            },
            "digest": self.digest,
        }


@dataclass(frozen=True)
class QualityReport:
    """A deterministic, content-free audit report."""

    audit_version: str
    selected_runs: Mapping[str, str]
    products: tuple[ProductQuality, ...]
    digest: str

    def as_dict(self) -> dict[str, object]:
        totals = _aggregate_products(self.products)
        return {
            "audit_version": self.audit_version,
            "selected_runs": dict(self.selected_runs),
            "products": [product.as_dict() for product in self.products],
            "totals": totals,
            "validation": validate_quality(self),
            "digest": self.digest,
        }


@dataclass(frozen=True)
class QualityComparison:
    """Baseline/candidate gate results without private source material."""

    baseline_digest: str
    candidate_digest: str
    passed: bool
    gates: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "audit_version": QUALITY_AUDIT_VERSION,
            "baseline_digest": self.baseline_digest,
            "candidate_digest": self.candidate_digest,
            "passed": self.passed,
            "gates": self.gates,
        }


def _aggregate_products(products: Iterable[ProductQuality]) -> dict[str, object]:
    rows = tuple(products)
    sum_fields = (
        "section_count", "quarantine_count", "quarantined_anchor_count",
        "empty_heading_count", "numeric_only_heading_count", "sentence_like_heading_count",
        "heading_defect_count", "layout_order_conflict_count", "layout_metadata_error_count",
        "unclassified_count", "short_under_40_count", "short_under_80_count",
        "native_fallback_section_count", "native_fallback_short_under_40_count",
        "unresolved_table_count",
        "oversize_over_5000_count", "oversize_over_10000_count", "privacy_violation_count",
    )
    result: dict[str, object] = {field: sum(getattr(row, field) for row in rows) for field in sum_fields}
    result["length_buckets"] = {
        bucket: sum(row.length_buckets.get(bucket, 0) for row in rows)
        for bucket in _LENGTH_BUCKETS
    }
    anchor_fields = ("expected_anchor_count", "assigned_anchor_count", "ignored_anchor_count")
    for field in anchor_fields:
        values = [getattr(row, field) for row in rows]
        result[field] = sum(value for value in values if value is not None) if all(value is not None for value in values) else None
    for field in ("missing_anchor_count", "extra_anchor_count", "duplicate_anchor_count"):
        values = [getattr(row, field) for row in rows]
        result[field] = sum(value for value in values if value is not None) if all(value is not None for value in values) else None
    expected = result["expected_anchor_count"]
    assigned = result["assigned_anchor_count"]
    ignored = result["ignored_anchor_count"]
    result["anchor_coverage_ratio"] = (
        (assigned + ignored) / expected if isinstance(expected, int) and expected > 0 and isinstance(assigned, int) and isinstance(ignored, int)
        else (1.0 if expected == 0 and assigned == 0 and ignored == 0 else None)
    )
    quarantined = result["quarantined_anchor_count"]
    result["quarantine_anchor_ratio"] = (
        quarantined / expected
        if isinstance(expected, int) and expected > 0 and isinstance(quarantined, int)
        else (0.0 if expected == 0 and quarantined == 0 else None)
    )
    result["probe_hits"] = {
        probe: {
            "matched": sum(row.probe_hits.get(probe, {}).get("matched", 0) for row in rows),
            "coherent": sum(row.probe_hits.get(probe, {}).get("coherent", 0) for row in rows),
        }
        for probe in sorted(_QUALITY_PROBES)
    }
    return result


def _audit_run(conn: sqlite3.Connection, run: sqlite3.Row, tables: set[str]) -> ProductQuality:
    product_code = _safe_product_code(run["product_code"])
    run_id = _safe_run_id(run["parser_run_id"])
    rows = _section_rows(conn, str(run["parser_run_id"]))
    quarantine_count, quarantined_anchor_count, quarantine_by_reason, quarantine_privacy = _quarantine_stats(
        conn, str(run["parser_run_id"]), product_code, tables
    )
    expected = _asset_anchor_count(conn, run, tables)
    assigned, ignored, missing, extra, duplicate, coverage = _anchor_stats(
        conn, str(run["parser_run_id"]), expected, tables
    )
    length_buckets = dict.fromkeys(_LENGTH_BUCKETS, 0)
    empty = numeric = sentence = heading_defects = metadata_errors = 0
    order_conflict = quarantine_by_reason.get("layout-order-conflict", 0)
    unclassified = unresolved_table = short40 = short80 = 0
    fallback_sections = fallback_short40 = over5000 = over10000 = privacy = 0
    probe_hits: dict[str, dict[str, int]] = {
        probe: {"matched": 0, "coherent": 0} for probe in sorted(_QUALITY_PROBES)
    }
    signatures: list[dict[str, object]] = []
    for row in rows:
        heading = str(row["heading"] or "") if "heading" in row.keys() else ""
        source_text = str(row["source_text"] or "") if "source_text" in row.keys() else ""
        flags, malformed = _json_flags(row["layout_flags"] if "layout_flags" in row.keys() else None)
        metadata_errors += int(malformed)
        empty_defect = not heading.strip()
        numeric_defect = _numeric_only_heading(heading)
        sentence_defect = _sentence_like_heading(heading)
        empty += int(empty_defect)
        numeric += int(numeric_defect)
        sentence += int(sentence_defect)
        heading_defects += int(empty_defect or numeric_defect or sentence_defect)
        order_conflict += int(_ORDER_CONFLICT_FLAG in flags)
        unclassified += int(
            _UNCLASSIFIED_FLAG in flags or heading.casefold().startswith("unclassified native text")
        )
        unresolved_table += int(bool(flags.intersection(_UNRESOLVED_TABLE_FLAGS)))
        length = len(source_text)
        length_buckets[_length_bucket(length)] += 1
        native_fallback = "native-layout-fallback" in flags
        fallback_sections += int(native_fallback)
        fallback_short40 += int(native_fallback and length < 40)
        short40 += int(not native_fallback and length < 40)
        short80 += int(not native_fallback and length < 80)
        over5000 += int(length >= 5000)
        over10000 += int(length >= 10000)
        privacy += int(_has_private_value(heading) or _has_private_value(source_text))
        coherent = (
            80 <= length < 10000
            and not empty_defect
            and not numeric_defect
            and not sentence_defect
            and not malformed
            and _ORDER_CONFLICT_FLAG not in flags
            and _UNCLASSIFIED_FLAG not in flags
            and not native_fallback
            and not flags.intersection(_UNRESOLVED_TABLE_FLAGS)
            and not _has_private_value(heading)
            and not _has_private_value(source_text)
        )
        for probe in _matched_probes(heading, source_text):
            probe_hits[probe]["matched"] += 1
            probe_hits[probe]["coherent"] += int(coherent)
        signatures.append(
            {
                "section_hash": str(row["source_section_hash"] or "") if "source_section_hash" in row.keys() else _digest(source_text),
                "text_length": length,
                "heading_digest": _digest("heading", " ".join(heading.split())),
                "page_start": row["page_start"] if "page_start" in row.keys() else None,
                "page_end": row["page_end"] if "page_end" in row.keys() else None,
                "flags": sorted(flags),
            }
        )
    privacy += int(quarantine_privacy)
    signatures.sort(key=lambda value: _canonical(value))
    digest = _digest(
        QUALITY_AUDIT_VERSION,
        product_code,
        _canonical(signatures),
        _canonical({"quarantine": quarantine_by_reason, "anchors": [expected, assigned, ignored, missing, extra, duplicate]}),
    )
    return ProductQuality(
        product_code=product_code,
        parser_run_id=run_id,
        section_count=len(rows),
        quarantine_count=quarantine_count,
        quarantined_anchor_count=quarantined_anchor_count,
        quarantine_by_reason=quarantine_by_reason,
        quarantine_anchor_ratio=(
            quarantined_anchor_count / expected
            if expected not in (None, 0)
            else (0.0 if expected == 0 and quarantined_anchor_count == 0 else None)
        ),
        expected_anchor_count=expected,
        assigned_anchor_count=assigned,
        ignored_anchor_count=ignored,
        missing_anchor_count=missing,
        extra_anchor_count=extra,
        duplicate_anchor_count=duplicate,
        anchor_coverage_ratio=coverage,
        empty_heading_count=empty,
        numeric_only_heading_count=numeric,
        sentence_like_heading_count=sentence,
        heading_defect_count=heading_defects,
        layout_order_conflict_count=order_conflict,
        layout_metadata_error_count=metadata_errors,
        unclassified_count=unclassified,
        unresolved_table_count=unresolved_table,
        short_under_40_count=short40,
        short_under_80_count=short80,
        native_fallback_section_count=fallback_sections,
        native_fallback_short_under_40_count=fallback_short40,
        oversize_over_5000_count=over5000,
        oversize_over_10000_count=over10000,
        length_buckets=length_buckets,
        privacy_violation_count=privacy,
        digest=digest,
        probe_hits=probe_hits,
    )


def audit_workspace(
    workspace: Path | str,
    *,
    parser_run_ids: Mapping[str, str] | None = None,
) -> QualityReport:
    """Audit selected parser runs from a workspace without mutating it.

    By default, the newest active parser run per product is selected.  A
    caller comparing historical runs can pass ``{product_code: run_id}``.
    """

    conn = _connect_readonly(workspace)
    try:
        tables = _tables(conn)
        runs = _select_runs(conn, parser_run_ids)
        products = tuple(sorted((_audit_run(conn, run, tables) for run in runs), key=lambda item: item.product_code))
        selected_runs = {item.product_code: item.parser_run_id for item in products}
        digest = _digest(
            QUALITY_AUDIT_VERSION,
            _canonical(
                [
                    {"product_code": item.product_code, "digest": item.digest}
                    for item in products
                ]
            ),
        )
        return QualityReport(QUALITY_AUDIT_VERSION, selected_runs, products, digest)
    finally:
        conn.close()


def _metric(row: ProductQuality, name: str) -> int:
    return int(getattr(row, name))


def _reduction(baseline: int, candidate: int) -> float:
    if baseline == 0:
        return 1.0 if candidate == 0 else 0.0
    return (baseline - candidate) / baseline


def _gate(baseline: int, candidate: int, required_reduction: float) -> dict[str, object]:
    reduction = _reduction(baseline, candidate)
    return {
        "baseline": baseline,
        "candidate": candidate,
        "required_reduction": required_reduction,
        "actual_reduction": reduction,
        "passed": reduction >= required_reduction,
    }


def validate_quality(report: QualityReport) -> dict[str, object]:
    """Apply absolute fail-closed gates to one candidate report."""
    totals = _aggregate_products(report.products)
    checks = {
        "has_selected_products": bool(report.products),
        "anchor_coverage": all(
            item.expected_anchor_count is not None
            and item.missing_anchor_count == 0
            and item.extra_anchor_count == 0
            and item.duplicate_anchor_count == 0
            for item in report.products
        ),
        "privacy": all(item.privacy_violation_count == 0 for item in report.products),
        "layout_metadata": all(
            item.layout_metadata_error_count == 0 for item in report.products
        ),
        "active_structure": all(
            item.empty_heading_count == 0
            and item.numeric_only_heading_count == 0
            and item.unclassified_count == 0
            and item.unresolved_table_count == 0
            and item.oversize_over_10000_count == 0
            for item in report.products
        ),
        "quarantine_bound": (
            all(
                item.quarantine_anchor_ratio is not None
                and item.quarantine_anchor_ratio <= 0.25
                for item in report.products
            )
            and isinstance(totals["quarantine_anchor_ratio"], float)
            and totals["quarantine_anchor_ratio"] <= 0.20
        ),
        "quality_probes": all(
            int(values["matched"]) > 0 and int(values["coherent"]) > 0
            for values in totals["probe_hits"].values()
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def compare_quality(baseline: QualityReport, candidate: QualityReport) -> QualityComparison:
    """Compare reports using the parser-quality acceptance gates.

    Layout conflicts and suspicious fragments must meet both overall and
    per-product reductions. Sentence-like headings must improve overall and
    may not regress for any product. Anchor/privacy hard checks fail closed.
    """

    baseline_totals = _aggregate_products(baseline.products)
    candidate_totals = _aggregate_products(candidate.products)
    metric_specs = (
        ("layout_order_conflicts", "layout_order_conflict_count", _GATES["layout_order_conflicts_overall"], _GATES["layout_order_conflicts_per_product"]),
        ("short_fragments", "short_under_40_count", _GATES["short_fragments_overall"], _GATES["short_fragments_per_product"]),
        ("sentence_like_headings", "sentence_like_heading_count", _GATES["sentence_like_headings_overall"], 0.0),
    )
    gates: dict[str, object] = {}
    passed = True
    for name, field, overall_gate, product_gate in metric_specs:
        overall = _gate(int(baseline_totals[field]), int(candidate_totals[field]), overall_gate)
        gates[f"{name}_overall"] = overall
        passed &= bool(overall["passed"])
        per_product: dict[str, object] = {}
        baseline_by_product = {row.product_code: row for row in baseline.products}
        candidate_by_product = {row.product_code: row for row in candidate.products}
        for product in sorted(set(baseline_by_product) | set(candidate_by_product)):
            before = _metric(baseline_by_product[product], field) if product in baseline_by_product else 0
            after = _metric(candidate_by_product[product], field) if product in candidate_by_product else 0
            if name == "sentence_like_headings":
                result = {
                    "baseline": before,
                    "candidate": after,
                    "required_reduction": "no-regression",
                    "actual_reduction": _reduction(before, after),
                    "passed": after <= before,
                }
            else:
                result = _gate(before, after, product_gate)
            per_product[product] = result
            passed &= bool(result["passed"])
        gates[f"{name}_per_product"] = per_product

    candidate_validation = validate_quality(candidate)
    hard_checks = {
        "product_set_unchanged": set(baseline.selected_runs) == set(candidate.selected_runs),
        **{
            f"candidate_{name}": bool(value)
            for name, value in candidate_validation["checks"].items()
        },
    }
    gates["hard_checks"] = hard_checks
    passed &= all(hard_checks.values())
    return QualityComparison(baseline.digest, candidate.digest, passed, gates)


def compare_repair_quality(
    baseline: QualityReport, candidate: QualityReport
) -> QualityComparison:
    """Compare V5 repair output with V4 using conserved structural units.

    V4 quarantines rule-bearing anchors while V5 makes their exact native text
    active with review flags. Comparing active flags or fragments alone would
    therefore call recovered content a regression. Bound each candidate metric
    by the corresponding V4 active metric plus the V4 quarantine records that
    could legitimately move into it, and require all rule-bearing quarantine to
    disappear from V5.
    """
    baseline_by_product = {row.product_code: row for row in baseline.products}
    candidate_by_product = {row.product_code: row for row in candidate.products}
    products = sorted(set(baseline_by_product) | set(candidate_by_product))
    gates: dict[str, object] = {}
    passed = True

    def record_gate(name: str, product: str, before: object, after: object, ok: bool) -> None:
        nonlocal passed
        group = gates.setdefault(name, {})
        assert isinstance(group, dict)
        group[product] = {"baseline": before, "candidate": after, "passed": ok}
        passed &= ok

    for product in products:
        before = baseline_by_product.get(product)
        after = candidate_by_product.get(product)
        if before is None or after is None:
            for name in (
                "anchor_inventory", "rule_bearing_quarantine", "quarantine_ratio",
                "section_recovery", "layout_conflicts", "sentence_headings",
                "short_fragments",
            ):
                record_gate(name, product, before is not None, after is not None, False)
            continue
        rule_bearing = sum(
            int(before.quarantine_by_reason.get(reason, 0))
            for reason in _RULE_BEARING_QUARANTINE_REASONS
        )
        remaining_rule_bearing = sum(
            int(after.quarantine_by_reason.get(reason, 0))
            for reason in _RULE_BEARING_QUARANTINE_REASONS
        )
        record_gate(
            "anchor_inventory", product, before.expected_anchor_count,
            after.expected_anchor_count,
            before.expected_anchor_count == after.expected_anchor_count,
        )
        record_gate(
            "rule_bearing_quarantine", product, rule_bearing,
            remaining_rule_bearing, remaining_rule_bearing == 0,
        )
        ratio_ok = (
            isinstance(before.quarantine_anchor_ratio, float)
            and isinstance(after.quarantine_anchor_ratio, float)
            and after.quarantine_anchor_ratio <= before.quarantine_anchor_ratio
        )
        record_gate(
            "quarantine_ratio", product, before.quarantine_anchor_ratio,
            after.quarantine_anchor_ratio, ratio_ok,
        )
        record_gate(
            "section_recovery", product, before.section_count, after.section_count,
            before.section_count <= after.section_count <= before.section_count + rule_bearing,
        )
        conflict_limit = (
            before.layout_order_conflict_count
            + int(before.quarantine_by_reason.get("layout-order-conflict", 0))
        )
        record_gate(
            "layout_conflicts", product, conflict_limit,
            after.layout_order_conflict_count,
            after.layout_order_conflict_count <= conflict_limit,
        )
        sentence_limit = (
            before.sentence_like_heading_count
            + int(before.quarantine_by_reason.get("heading-artifact", 0))
        )
        record_gate(
            "sentence_headings", product, sentence_limit,
            after.sentence_like_heading_count,
            after.sentence_like_heading_count <= sentence_limit,
        )
        short_limit = before.short_under_40_count + rule_bearing
        record_gate(
            "short_fragments", product, short_limit, after.short_under_40_count,
            after.short_under_40_count <= short_limit,
        )

    candidate_validation = validate_quality(candidate)
    hard_checks = {
        "product_set_unchanged": set(baseline.selected_runs) == set(candidate.selected_runs),
        **{
            f"candidate_{name}": bool(value)
            for name, value in candidate_validation["checks"].items()
        },
    }
    gates["hard_checks"] = hard_checks
    passed &= all(hard_checks.values())
    return QualityComparison(baseline.digest, candidate.digest, passed, gates)


def compare_workspaces(
    baseline_workspace: Path | str,
    candidate_workspace: Path | str,
    *,
    baseline_parser_run_ids: Mapping[str, str] | None = None,
    candidate_parser_run_ids: Mapping[str, str] | None = None,
) -> QualityComparison:
    """Audit and compare two workspaces without writing either one."""

    baseline = audit_workspace(baseline_workspace, parser_run_ids=baseline_parser_run_ids)
    candidate = audit_workspace(candidate_workspace, parser_run_ids=candidate_parser_run_ids)
    return compare_quality(baseline, candidate)


__all__ = [
    "QUALITY_AUDIT_VERSION",
    "ProductQuality",
    "QualityComparison",
    "QualityReport",
    "audit_workspace",
    "compare_quality",
    "compare_repair_quality",
    "compare_workspaces",
    "validate_quality",
]
