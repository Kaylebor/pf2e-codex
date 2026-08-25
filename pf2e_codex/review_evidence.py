"""Bounded, read-only evidence service for licensed-corpus workers.

The executable consumes a supervisor-created claim context.  It never accepts
database paths or arbitrary SQL from the worker and never writes either SQLite
database.  All returned records are deterministic and content-bearing reads
are restricted to the IDs pre-authorized in that context.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_AON_HOSTS = {"2e.aonprd.com", "aonprd.com", "www.aonprd.com"}
_WORD_RE = re.compile(r"[a-z0-9]{2,}")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{Path(path).expanduser().resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA trusted_schema=OFF")
    return conn


def load_context(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("invalid evidence context")
    for key in ("workspace", "allowed_ids", "neighbor_ids"):
        if key not in payload:
            raise ValueError(f"evidence context is missing {key}")
    if not isinstance(payload["allowed_ids"], list) or not isinstance(payload["neighbor_ids"], list):
        raise ValueError("evidence context ID sets must be lists")
    return payload


def _authorized(context: dict[str, Any], section_id: str) -> None:
    if section_id not in set(context["allowed_ids"]):
        raise PermissionError("section ID is outside the current claimed batch")


def _section_row(conn: sqlite3.Connection, section_id: str) -> sqlite3.Row:
    row = conn.execute(
        """SELECT s.section_key AS id, s.source_section_id, s.heading, s.source_text,
                  s.page_start, s.page_end, s.printed_page, s.layout_flags,
                  s.product_code, r.era AS rules_era, r.license
           FROM source_sections AS s
           JOIN source_revisions AS r USING(product_code, content_fingerprint)
           JOIN parser_runs AS p ON p.parser_run_id=s.parser_run_id
           WHERE s.section_key=? AND p.state='active' AND p.review_enabled=1""",
        (section_id,),
    ).fetchone()
    if row is None:
        raise ValueError("unknown active section ID")
    return row


def section(context: dict[str, Any], section_id: str) -> dict[str, Any]:
    _authorized(context, section_id)
    with _connect(str(context["workspace"])) as conn:
        row = _section_row(conn, section_id)
    return {
        "id": row["id"],
        "source_section_id": row["source_section_id"],
        "product_code": row["product_code"],
        "rules_era": row["rules_era"],
        "license": row["license"],
        "heading": row["heading"],
        "text": row["source_text"],
        "page_start": row["page_start"],
        "page_end": row["page_end"],
        "printed_page": row["printed_page"],
        "layout_flags": json.loads(str(row["layout_flags"] or "[]")),
    }


def neighbors(context: dict[str, Any], section_id: str) -> dict[str, Any]:
    _authorized(context, section_id)
    permitted = set(context["neighbor_ids"])
    with _connect(str(context["workspace"])) as conn:
        target = _section_row(conn, section_id)
        rows = conn.execute(
            """SELECT s.section_key AS id, s.heading, s.source_text, s.page_start,
                      s.page_end, s.printed_page, s.layout_flags
               FROM source_sections AS s
               JOIN parser_runs AS p ON p.parser_run_id=s.parser_run_id
               WHERE p.parser_run_id=(SELECT parser_run_id FROM source_sections WHERE section_key=?)
                 AND p.state='active' AND p.review_enabled=1
               ORDER BY COALESCE(s.page_start, 2147483647), s.source_section_id, s.section_key""",
            (section_id,),
        ).fetchall()
    position = next(index for index, row in enumerate(rows) if row["id"] == target["id"])
    selected = [
        row for row in rows[max(0, position - 2) : position + 3]
        if row["id"] != section_id and row["id"] in permitted
    ]
    return {
        "id": section_id,
        "neighbors": [
            {
                "id": row["id"],
                "heading": row["heading"],
                "text": row["source_text"],
                "page_start": row["page_start"],
                "page_end": row["page_end"],
                "printed_page": row["printed_page"],
                "layout_flags": json.loads(str(row["layout_flags"] or "[]")),
            }
            for row in selected
        ],
    }


def _validate_clean_foundry(conn: sqlite3.Connection, database: str) -> None:
    tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"_meta", "chunks"}.issubset(tables):
        raise ValueError("Foundry evidence database has an unsupported schema")
    scope = conn.execute("SELECT value FROM _meta WHERE key='distribution_scope'").fetchone()
    if scope is None or str(scope[0]) != "redistributable":
        raise ValueError("Foundry evidence must use a validated redistributable database")
    leaked = conn.execute(
        "SELECT 1 FROM chunks WHERE origin NOT IN ('foundry','licensed-core') LIMIT 1"
    ).fetchone()
    if leaked is not None:
        raise ValueError("Foundry evidence database contains private or unknown rows")
    from .distribution import audit_database_slot

    audit_database_slot(database, "clean")


def _terms(value: str) -> list[str]:
    return sorted(set(_WORD_RE.findall(value.casefold())))[:12]


def foundry(context: dict[str, Any], section_id: str, *, limit: int = 5) -> dict[str, Any]:
    _authorized(context, section_id)
    if not 1 <= limit <= 10:
        raise ValueError("Foundry evidence limit must be between 1 and 10")
    database = context.get("foundry_database")
    if not isinstance(database, str) or not database:
        return {"id": section_id, "status": "unavailable", "results": []}
    with _connect(str(context["workspace"])) as review:
        source = _section_row(review, section_id)
    with _connect(database) as conn:
        _validate_clean_foundry(conn, database)
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(chunks)")}
        fields = [name for name in ("id", "name", "type", "pack", "text", "license", "remaster", "publication_title") if name in columns]
        exact = conn.execute(
            f"SELECT {','.join(fields)} FROM chunks WHERE origin='foundry' AND lower(name)=lower(?) ORDER BY id LIMIT ?",
            (source["heading"], limit),
        ).fetchall()
        rows = list(exact)
        if len(rows) < limit and "fts_chunks" in {
            str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }:
            terms = _terms(f"{source['heading']} {source['source_text']}")
            if terms:
                query = " OR ".join(f'"{term}"' for term in terms)
                lexical = conn.execute(
                    f"""SELECT {','.join('c.' + field for field in fields)}
                        FROM fts_chunks AS f JOIN chunks AS c ON c.rowid=f.rowid
                        WHERE fts_chunks MATCH ? AND c.origin='foundry'
                        ORDER BY bm25(fts_chunks, 10.0, 1.0), c.id LIMIT ?""",
                    (query, limit * 2),
                ).fetchall()
                known = {str(row["id"]) for row in rows}
                rows.extend(row for row in lexical if str(row["id"]) not in known)
        results = [{field: row[field] for field in fields} for row in rows[:limit]]
    return {"id": section_id, "status": "ok", "results": results}


def stitches(context: dict[str, Any], section_id: str) -> dict[str, Any]:
    _authorized(context, section_id)
    with _connect(str(context["workspace"])) as conn:
        rows = conn.execute(
            """SELECT candidate_id, section_keys, evidence_json
               FROM stitch_candidates
               WHERE parser_run_id=(SELECT parser_run_id FROM source_sections WHERE section_key=?)
               ORDER BY candidate_id""",
            (section_id,),
        ).fetchall()
    values = []
    for row in rows:
        section_ids = json.loads(str(row["section_keys"]))
        if section_id in section_ids:
            values.append({
                "candidate_id": row["candidate_id"],
                "section_ids": section_ids,
                "evidence": json.loads(str(row["evidence_json"])),
            })
    return {"id": section_id, "candidates": values}


def aon_cache(context: dict[str, Any], section_id: str) -> dict[str, Any]:
    _authorized(context, section_id)
    with _connect(str(context["workspace"])) as conn:
        source = _section_row(conn, section_id)
        normalized = " ".join(_terms(str(source["heading"])))
        row = conn.execute(
            "SELECT status, results_json, checked_at FROM aon_cache WHERE normalized_query=?",
            (normalized,),
        ).fetchone()
    if row is None:
        return {"id": section_id, "status": "not-cached", "results": []}
    results = json.loads(str(row["results_json"]))
    for result in results:
        parsed = urlparse(str(result.get("url", "")))
        if parsed.scheme != "https" or parsed.hostname not in _AON_HOSTS or set(result) - {"title", "url"}:
            raise ValueError("AON cache contains invalid provenance")
    return {"id": section_id, "status": row["status"], "results": results, "checked_at": row["checked_at"]}


def execute(context: dict[str, Any], command: str, section_id: str | None, *, limit: int = 5) -> object:
    if command == "help":
        return {
            "commands": {
                "section": "claimed section text and provenance",
                "neighbors": "up to two pre-authorized adjacent sections each way",
                "foundry": "bounded clean Foundry exact-name and FTS evidence",
                "stitches": "deterministic adjacent stitch candidates",
                "aon-cache": "cached AON title and URL provenance only",
            }
        }
    if not section_id:
        raise ValueError(f"{command} requires a section ID")
    handlers = {
        "section": section,
        "neighbors": neighbors,
        "stitches": stitches,
        "aon-cache": aon_cache,
    }
    if command == "foundry":
        return foundry(context, section_id, limit=limit)
    handler = handlers.get(command)
    if handler is None:
        raise ValueError("unknown evidence command")
    return handler(context, section_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("command", choices=("help", "section", "neighbors", "foundry", "stitches", "aon-cache"))
    parser.add_argument("section_id", nargs="?")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    print(_canonical(execute(load_context(args.context), args.command, args.section_id, limit=args.limit)))


if __name__ == "__main__":
    main()
