"""sqlite-vec database management and search."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

import sqlite_vec  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from .model_manager import ModelManager


def load_vec_extension(conn) -> None:
    """Load sqlite-vec, then leave extension loading disabled."""
    conn.enable_load_extension(True)
    try:
        conn.load_extension(sqlite_vec.loadable_path())
    finally:
        conn.enable_load_extension(False)


def init_db(db_path: Path, dim: int) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    load_vec_extension(conn)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            name TEXT,
            type TEXT,
            pack TEXT,
            slug TEXT,
            level INTEGER,
            traits TEXT,
            text TEXT,
            raw_rules_count INTEGER,
            source_hash TEXT,
            license TEXT DEFAULT 'NONE',
            remaster INTEGER DEFAULT NULL,
            translations TEXT DEFAULT NULL
        )
    """)
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            id TEXT PRIMARY KEY,
            embedding float[{dim}]
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
            name,
            text,
            content='chunks',
            content_rowid='rowid'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS refs (
            source_id TEXT,
            target_uuid TEXT,
            target_name TEXT,
            context TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_refs_source ON refs(source_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_refs_target ON refs(target_uuid)
    """)
    ensure_ambiguous_ref_targets(conn)
    conn.commit()
    conn.close()


def rebuild_fts(conn) -> None:
    """Rebuild the FTS5 index from the chunks table."""
    conn.execute("INSERT INTO fts_chunks(fts_chunks) VALUES ('rebuild')")


def ensure_ambiguous_ref_targets(conn) -> None:
    """Record bare IDs that have ever been ambiguous across packs.

    Incremental updates retain these tombstones even after one duplicate is
    removed. A full rebuild may clear and recompute the table because all
    references then come from one consistent snapshot.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ambiguous_ref_targets (
            bare_id TEXT PRIMARY KEY
        )
    """)
    conn.execute("""
        INSERT OR IGNORE INTO ambiguous_ref_targets (bare_id)
        SELECT substr(id, instr(id, ':') + 1)
        FROM chunks
        WHERE id <> ''
        GROUP BY substr(id, instr(id, ':') + 1)
        HAVING COUNT(*) > 1
    """)


def _current_ambiguous_ref_targets(conn) -> set[str]:
    """Return duplicate bare IDs present in the current chunks snapshot."""
    rows = conn.execute("""
        SELECT substr(id, instr(id, ':') + 1)
        FROM chunks
        WHERE id <> ''
        GROUP BY substr(id, instr(id, ':') + 1)
        HAVING COUNT(*) > 1
    """).fetchall()
    return {bare_id for (bare_id,) in rows}


def vec_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _normalize_id(raw: str) -> str:
    """Accept internal 'pack:id', Foundry UUIDs, or bare slugs; return best guess."""
    if not raw:
        return raw
    if ".Item." in raw:
        return raw.rsplit(".", 1)[-1]
    return raw


def _rrf_fuse(
    semantic_results: list[tuple[str, dict]],
    fts_results: list[tuple[str, dict]],
    k: int = 60,
    top_k: int = 5,
    weight_semantic: float = 0.85,
) -> list[dict]:
    """Reciprocal Rank Fusion of semantic and FTS result lists with weighting.

    weight_semantic: weight for semantic scores (FTS gets 1 - weight_semantic).
    Lower = more emphasis on exact text matching.
    """
    weight_fts = 1.0 - weight_semantic
    scores: dict[str, float] = {}
    details: dict[str, dict] = {}

    for rank, (cid, result) in enumerate(semantic_results, start=1):
        scores[cid] = scores.get(cid, 0.0) + weight_semantic / (k + rank)
        details[cid] = result

    for rank, (cid, result) in enumerate(fts_results, start=1):
        scores[cid] = scores.get(cid, 0.0) + weight_fts / (k + rank)
        if cid not in details:
            details[cid] = result

    # Sort by RRF score descending
    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [
        {**details[cid], "rrf_score": round(scores[cid], 4)}
        for cid in sorted_ids[:top_k]
    ]


def _apply_ref_weight(results: list[dict], ref_weight: float, top_k: int) -> list[dict]:
    """Blend relevance and incoming-reference scores, then return top results."""
    weight = max(0.0, min(float(ref_weight), 1.0))
    if not results or weight <= 0:
        return results[:top_k]

    base_scores: list[float] = []
    for result in results:
        if result.get("rerank_score") is not None:
            score = float(result["rerank_score"])
        elif result.get("rrf_score") is not None:
            score = float(result["rrf_score"])
        else:
            # sqlite-vec distance is lower-is-better, so invert its ordering.
            score = -float(result.get("distance", 0.0))
        base_scores.append(score)

    low, high = min(base_scores), max(base_scores)
    span = high - low
    max_refs = max((len(r.get("incoming_refs", [])) for r in results), default=0)

    for result, base in zip(results, base_scores, strict=True):
        relevance = (base - low) / span if span else 1.0
        ref_score = len(result.get("incoming_refs", [])) / max_refs if max_refs else 0.0
        adjusted = (1.0 - weight) * relevance + weight * ref_score
        result["ref_score"] = round(ref_score, 4)
        result["adjusted_score"] = round(adjusted, 4)

    results.sort(
        key=lambda result: (result["adjusted_score"], result["ref_score"]),
        reverse=True,
    )
    return results[:top_k]


class SearchIndex:
    """Search index backed by sqlite-vec with optional FTS5 hybrid blending.

    Model inference is delegated to a ModelManager — SearchIndex never creates
    ONNX providers or sessions directly.
    """

    def __init__(self, db_path: Path | str, manager: ModelManager):
        import sqlite3
        import threading as _threading

        self.db_path = Path(db_path)
        self._manager: ModelManager = manager
        self._conn: sqlite3.Connection | None = None
        self._conn_ro: sqlite3.Connection | None = None
        self._fts_ready: bool = False
        self._ambiguous_tombstones: set[str] = set()
        self._db_lock = _threading.Lock()

    def _ensure_loaded(self) -> None:
        with self._db_lock:
            if self._conn is not None:
                return
            import sqlite3

            if not self.db_path.exists():
                # Auto-download pre-built DB from GitHub Releases
                from .config import DEFAULT_RELEASE
                from .config import _model_safe_name
                import urllib.request as _req
                db_name = f"pf2e_{_model_safe_name(self._manager.model_name)}.db"
                release = DEFAULT_RELEASE
                url = f"https://github.com/Kaylebor/pf2e-codex/releases/download/{release}/{db_name}"
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"Downloading pre-computed DB ({db_name})...", end=" ", flush=True)
                try:
                    _req.urlretrieve(url, self.db_path)
                    size_mb = self.db_path.stat().st_size / 1024**2
                    print(f"{size_mb:.0f}MB")
                except Exception:
                    raise FileNotFoundError(
                        f"Database not found: {self.db_path}. "
                        f"Auto-download failed.\n"
                        f"Run 'pf2e-codex embed' to build from scratch"
                    )
            try:
                self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
                self._conn_ro = sqlite3.connect(
                    f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False
                )
                load_vec_extension(self._conn_ro)
                load_vec_extension(self._conn)
                row = self._conn_ro.execute(
                    "SELECT value FROM _meta WHERE key = 'embedding_model'"
                ).fetchone()
                db_model = row[0] if row else None
                if db_model and db_model != self._manager.model_name:
                    print(f"Warning: DB model {db_model} != config model {self._manager.model_name}")
                self._conn.execute("""
                    CREATE TABLE IF NOT EXISTS refs (
                        source_id TEXT,
                        target_uuid TEXT,
                        target_name TEXT,
                        context TEXT
                    )
                """)
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_refs_source ON refs(source_id)")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_refs_target ON refs(target_uuid)")
                tombstone_table_available = False
                try:
                    ensure_ambiguous_ref_targets(self._conn)
                    tombstone_table_available = True
                except sqlite3.OperationalError as exc:
                    if "readonly" not in str(exc).lower():
                        raise
                    # Published/prebuilt DBs may be intentionally read-only.
                    # Keep a current-snapshot fallback in memory; writable
                    # indexes persist the durable tombstone table instead.
                    self._ambiguous_tombstones = _current_ambiguous_ref_targets(self._conn_ro)
                self._conn.commit()
                if tombstone_table_available:
                    self._ambiguous_tombstones = {
                        bare_id for (bare_id,) in self._conn_ro.execute(
                            "SELECT bare_id FROM ambiguous_ref_targets"
                        ).fetchall()
                    }
            except Exception:
                if self._conn_ro is not None:
                    self._conn_ro.close()
                    self._conn_ro = None
                if self._conn is not None:
                    self._conn.close()
                    self._conn = None
                raise

    def _ensure_fts(self) -> None:
        with self._db_lock:
            if self._fts_ready:
                return
            row = self._conn_ro.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='fts_chunks'"
            ).fetchone()
            if not row:
                print("Creating FTS5 index...")
                self._conn.execute("""
                    CREATE VIRTUAL TABLE fts_chunks USING fts5(
                        name,
                        text,
                        content='chunks',
                        content_rowid='rowid'
                    )
                """)
                self._conn.execute("INSERT INTO fts_chunks(fts_chunks) VALUES ('rebuild')")
                self._conn.commit()
                print("FTS5 index ready")
            self._fts_ready = True

    def _encode(self, text: str) -> list[float]:
        return self._manager.embed_query(text)

    def search(
        self, query: str, top_k: int = 5, hybrid: bool = True,
        license: str | None = None, content_type: str | None = None,
        pack: str | None = None, remaster: bool | None = None,
        rerank: bool = False, rerank_candidates: int = 50,
        ref_weight: float = 0.0,
    ) -> list[dict]:
        self._ensure_loaded()
        import sys as _sys
        _sys.stderr.write(f"[search] search() called (hash={id(self):x})\n")

        # Build WHERE clauses for filters
        where_clauses = ["1=1"]
        params: list[str] = []
        if license:
            where_clauses.append("chunks.license = ?")
            params.append(license)
        if content_type:
            where_clauses.append("chunks.type = ?")
            params.append(content_type)
        if pack:
            where_clauses.append("chunks.pack = ?")
            params.append(pack)
        if remaster is not None:
            if remaster:
                where_clauses.append("chunks.remaster = 1")
            else:
                # remaster=False or NULL → both count as legacy
                where_clauses.append("(chunks.remaster = 0 OR chunks.remaster IS NULL)")
        where = " AND ".join(where_clauses)

        # 1. Semantic search (filtered)
        emb = self._encode(query)
        q_blob = vec_blob(emb)
        semantic_raw = self._conn_ro.execute(f"""
            SELECT chunks.id, chunks.name, chunks.type, chunks.pack, chunks.text, chunks.license, chunks.remaster, distance
            FROM vec_chunks
            JOIN chunks ON vec_chunks.id = chunks.id
            WHERE vec_chunks.embedding MATCH vec_f32(?)
              AND k = ? AND {where}
            ORDER BY distance
        """, [q_blob, rerank_candidates] + params).fetchall()

        semantic_results = [
            (r[0], {
                "id": r[0], "name": r[1], "type": r[2], "pack": r[3],
                "text": r[4], "license": r[5], "remaster": bool(r[6]) if r[6] is not None else None,
                "distance": r[7],
            })
            for r in semantic_raw
        ]

        # 2. FTS5 full-text search (replaces name bag-of-words LIKE)
        fts_results: list[tuple[str, dict]] = []
        if hybrid:
            words = [w.strip().lower() for w in query.split() if len(w.strip()) > 2]
            if words:
                try:
                    self._ensure_fts()
                    fts_query = " AND ".join(f'"{w}"' for w in words)
                    fts_raw = self._conn_ro.execute(f"""
                        SELECT c.id, c.name, c.type, c.pack, c.text, c.license, c.remaster, rank
                        FROM fts_chunks
                        JOIN chunks c ON c.rowid = fts_chunks.rowid
                        WHERE fts_chunks MATCH ? AND {where}
                        ORDER BY rank
                        LIMIT ?
                    """, [fts_query] + params + [rerank_candidates]).fetchall()
                    for r in fts_raw:
                        fts_results.append((r[0], {
                            "id": r[0], "name": r[1], "type": r[2], "pack": r[3],
                            "text": r[4], "license": r[5],
                            "remaster": bool(r[6]) if r[6] is not None else None,
                            "distance": r[7],
                        }))
                except Exception:
                    pass

        # 3. Build results — hybrid: semantic embeddings + FTS5 full-text
        # Reference weighting needs the larger candidate pool even when the
        # cross-encoder reranker is disabled; otherwise highly referenced
        # entries outside the initial top_k can never be boosted into view.
        rrf_top_k = rerank_candidates if (rerank or ref_weight > 0) else top_k
        if not hybrid:
            results = [r for _, r in semantic_results[:rrf_top_k]]
        else:
            results = _rrf_fuse(semantic_results, fts_results, top_k=rrf_top_k)

        # 4. Enrich with refs, legacy names, confidence
        self._enrich_results(results)

        # Optional second-stage cross-encoder reranking. Keep all candidates
        # when reference weighting is enabled so a highly referenced result
        # cannot be discarded before its boost is applied.
        if rerank and len(results) > 1:
            try:
                rerank_top_k = len(results) if ref_weight > 0 else top_k
                results = self._manager.rerank(query, results, top_k=rerank_top_k)
            except Exception as e:
                print(f"Reranker failed: {e}")

        if ref_weight > 0:
            results = _apply_ref_weight(results, ref_weight, top_k=top_k)
        else:
            results = results[:top_k]

        return results

    def _enrich_results(self, results: list[dict]) -> None:
        """Add refs, legacy_name, and confidence to search results in-place."""
        if not results:
            return

        # Batch-fetch outgoing refs (what this entry references)
        ids = [r["id"] for r in results]
        placeholders = ",".join("?" * len(ids))
        refs_rows = self._conn_ro.execute(f"""
            SELECT source_id, target_name, target_uuid FROM refs
            WHERE source_id IN ({placeholders})
        """, ids).fetchall()
        refs_by_source: dict[str, list[dict]] = {}
        for src, name, uuid in refs_rows:
            refs_by_source.setdefault(src, []).append({"name": name, "id": uuid})

        # Batch-fetch incoming refs by stable target UUID/internal ID. Names
        # are mutable (and can be duplicated), so they must not determine
        # reference counts or merge unrelated entries.
        target_ids: list[str] = []
        target_aliases: dict[str, list[str]] = {}
        bare_ids: list[str] = []
        for result in results:
            internal_id = result["id"]
            bare_id = internal_id.rsplit(":", 1)[-1]
            target_ids.append(internal_id)
            bare_ids.append(bare_id)

        # Older indexes stored bare Foundry IDs in refs. Those IDs are not
        # globally unique across packs, so only retain a bare alias when the
        # chunks table proves it has one owner. Recovering ambiguous legacy
        # refs requires a reindex/schema migration; guessing would apply a
        # wrong incoming-reference boost to one or more duplicate entries.
        bare_ids = list(dict.fromkeys(bare_ids))
        bare_placeholders = ",".join("?" * len(bare_ids))
        chunk_rows = self._conn_ro.execute(f"""
            SELECT id FROM chunks
            WHERE substr(id, instr(id, ':') + 1) IN ({bare_placeholders})
        """, bare_ids).fetchall()
        bare_counts: dict[str, int] = {}
        for (chunk_id,) in chunk_rows:
            bare_counts[chunk_id.rsplit(":", 1)[-1]] = bare_counts.get(
                chunk_id.rsplit(":", 1)[-1], 0
            ) + 1
        import sqlite3
        try:
            tombstone_rows = self._conn_ro.execute(f"""
                SELECT bare_id FROM ambiguous_ref_targets
                WHERE bare_id IN ({bare_placeholders})
            """, bare_ids).fetchall()
            ambiguous_tombstones = {bare_id for (bare_id,) in tombstone_rows}
        except sqlite3.OperationalError:
            # Read-only prebuilt DBs cannot be migrated in place; loading
            # computes current duplicate ownership into this memory fallback.
            ambiguous_tombstones = set(getattr(self, "_ambiguous_tombstones", set()))

        for result in results:
            internal_id = result["id"]
            bare_id = internal_id.rsplit(":", 1)[-1]
            if bare_counts.get(bare_id) == 1 and bare_id not in ambiguous_tombstones:
                aliases = [internal_id, bare_id] if internal_id != bare_id else [bare_id]
            elif ":" in internal_id:
                # An exact pack-qualified target remains safe even when its
                # legacy bare counterpart is ambiguous.
                aliases = [internal_id]
            else:
                aliases = []
            target_aliases[internal_id] = aliases
            target_ids.extend(alias for alias in aliases if alias not in target_ids)
        target_ids = list(dict.fromkeys(target_ids))
        target_placeholders = ",".join("?" * len(target_ids))
        incoming_rows = self._conn_ro.execute(f"""
            SELECT target_uuid, source_id FROM refs
            WHERE target_uuid IN ({target_placeholders})
        """, target_ids).fetchall()
        incoming_by_target: dict[str, list[dict]] = {}
        for target, source in incoming_rows:
            incoming_by_target.setdefault(target, []).append({"id": source})

        for r in results:
            r["refs"] = refs_by_source.get(r["id"], [])
            incoming: list[dict] = []
            seen_sources: set[str] = set()
            for alias in target_aliases[r["id"]]:
                for ref in incoming_by_target.get(alias, []):
                    if ref["id"] not in seen_sources:
                        incoming.append(ref)
                        seen_sources.add(ref["id"])
            r["incoming_refs"] = incoming

            # NONE license → OGL (missing metadata, but pre-ORC content)
            if r.get("license") in ("NONE", None, ""):
                r["license"] = "OGL"
            # Extract legacy name from alias pattern: "X (formerly Y)"
            name = r.get("name", "")
            if " (formerly " in name:
                r["legacy_name"] = name.split(" (formerly ", 1)[1].rstrip(")")
            else:
                r["legacy_name"] = None

            # Confidence from score
            score = r.get("rrf_score") or r.get("distance")
            if score is None:
                r["confidence"] = "high"  # exact fetch
            elif r.get("rrf_score") is not None:
                # RRF: higher is better
                if score > 0.015:
                    r["confidence"] = "high"
                elif score > 0.008:
                    r["confidence"] = "medium"
                else:
                    r["confidence"] = "low"
            else:
                # Semantic: distance is available but we use rrf_score when hybrid
                r["confidence"] = "medium" if (score or 0) < 0.5 else "low"
    def rules_explain(self, topic: str, top_k: int = 3,
                      license: str | None = None, content_type: str | None = None,
                      remaster: bool | None = None) -> list[dict]:
        """Search with boosted journal pages and conditions for core rules.

        Uses query rewriting: prepends "pf2e rule for" to the topic so the
        semantic embedding lands closer to rule text than to generic entries.
        """
        self._ensure_loaded()

        where_clauses = ["1=1"]
        params: list[str] = []
        if license:
            where_clauses.append("chunks.license = ?")
            params.append(license)
        if content_type:
            where_clauses.append("chunks.type = ?")
            params.append(content_type)
        if remaster is not None:
            if remaster:
                where_clauses.append("chunks.remaster = 1")
            else:
                where_clauses.append("(chunks.remaster = 0 OR chunks.remaster IS NULL)")
        where = " AND ".join(where_clauses)

        # Rewrite query: "flanking" → "pf2e flanking" to help the
        # semantic embedding land near actual rule/condition text
        # regardless of query type (condition name, skill, general question).
        # Smart rewrite: single-word queries are likely condition names
        # ("blinded", "flanking") — append "condition" to land near the
        # right condition page. Multi-word queries ("treat wounds",
        # "craft magic items") keep the original for general search.
        search_topic = f"{topic} condition" if " " not in topic.strip() else topic
        emb = self._encode(search_topic)
        q_blob = vec_blob(emb)
        candidate_k = max(top_k, 50)
        results = self._conn_ro.execute(f"""
            SELECT chunks.id, chunks.name, chunks.type, chunks.pack, chunks.text, chunks.license, chunks.remaster, distance
            FROM vec_chunks
            JOIN chunks ON vec_chunks.id = chunks.id
            WHERE vec_chunks.embedding MATCH vec_f32(?)
              AND k = ? AND {where}
            ORDER BY distance
        """, [q_blob, candidate_k] + params).fetchall()
        scored = []
        for r in results:
            ctype = r[2]
            distance = r[7]
            boost = 0.0
            if ctype == "journal_page":
                boost = 0.15
            elif ctype == "condition":
                boost = 0.05
            scored.append((distance - boost, r))
        scored.sort(key=lambda x: x[0])
        return [
            {"id": r[0], "name": r[1], "type": r[2], "pack": r[3], "text": r[4],
             "license": r[5], "remaster": bool(r[6]) if r[6] is not None else None,
             "distance": r[7]}
            for _, r in scored[:top_k]
        ]

    def catalog(self) -> dict:
        """Return the structure of the database: types, licenses, remaster, packs, and counts."""
        self._ensure_loaded()

        # Content type breakdown
        types = self._conn_ro.execute(
            "SELECT type, COUNT(*) FROM chunks GROUP BY type ORDER BY COUNT(*) DESC"
        ).fetchall()

        # License breakdown
        licenses = self._conn_ro.execute(
            "SELECT license, COUNT(*) FROM chunks GROUP BY license ORDER BY COUNT(*) DESC"
        ).fetchall()

        # Remaster breakdown (NULL grouped with legacy)
        remaster_counts = self._conn_ro.execute(
            "SELECT CASE WHEN remaster = 1 THEN 'remaster' ELSE 'legacy' END as label, COUNT(*) FROM chunks GROUP BY label ORDER BY COUNT(*) DESC"
        ).fetchall()

        # Pack breakdown (top 20)
        packs = self._conn_ro.execute(
            "SELECT pack, COUNT(*) FROM chunks GROUP BY pack ORDER BY COUNT(*) DESC LIMIT 20"
        ).fetchall()

        # Total chunks
        total = self._conn_ro.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

        # Refs count
        refs_count = self._conn_ro.execute("SELECT COUNT(*) FROM refs").fetchone()[0]

        return {
            "total_chunks": total,
            "total_references": refs_count,
            "types": {r[0]: r[1] for r in types},
            "licenses": {r[0]: r[1] for r in licenses},
            "remaster": {r[0]: r[1] for r in remaster_counts},
            "packs": {r[0]: r[1] for r in packs},
        }

    def fetch_by_id(self, entry_id: str) -> dict | None:
        """Fetch a single chunk by its internal ID, Foundry UUID, slug, or name."""
        self._ensure_loaded()
        normalized = _normalize_id(entry_id)

        # 1. Exact ID match (pack:id or bare Foundry _id)
        for sql, param in [
            ("SELECT id, name, type, pack, text FROM chunks WHERE id = ?", (normalized,)),
            ("SELECT id, name, type, pack, text FROM chunks WHERE id LIKE ?", (f"%:{normalized}",)),
        ]:
            row = self._conn_ro.execute(sql, param).fetchone()
            if row:
                return {
                    "id": row[0], "name": row[1], "type": row[2],
                    "pack": row[3], "text": row[4],
                }

        # 2. Slug match (e.g. "fury-instinct")
        row = self._conn_ro.execute(
            "SELECT id, name, type, pack, text FROM chunks WHERE slug = ?",
            (normalized,),
        ).fetchone()
        if row:
            return {
                "id": row[0], "name": row[1], "type": row[2],
                "pack": row[3], "text": row[4],
            }

        # 3. Exact name match (case-insensitive)
        row = self._conn_ro.execute(
            "SELECT id, name, type, pack, text FROM chunks WHERE LOWER(name) = LOWER(?)",
            (normalized,),
        ).fetchone()
        if row:
            return {
                "id": row[0], "name": row[1], "type": row[2],
                "pack": row[3], "text": row[4],
            }

        return None

    def related(self, entry_id: str, direction: str = "both", limit: int = 20) -> dict:
        """Find entries related by cross-references.

        Args:
            entry_id: ID, slug, name, or UUID of the entry.
            direction: "outgoing" (what X references), "incoming" (what references X),
                       or "both".
            limit: Max results per direction.

        Returns:
            {"outgoing": [...], "incoming": [...]} where each item is
            {id, name, type, pack, context}.
        """
        self._ensure_loaded()
        normalized = _normalize_id(entry_id)

        result: dict[str, list[dict]] = {"outgoing": [], "incoming": []}

        # Resolve entry_id to an internal chunk ID for outgoing queries
        chunk = self.fetch_by_id(entry_id)
        source_id = chunk["id"] if chunk else normalized

        if direction in ("outgoing", "both"):
            rows = self._conn_ro.execute("""
                SELECT chunks.id, chunks.name, chunks.type, chunks.pack, refs.context
                FROM refs
                JOIN chunks ON refs.target_uuid = chunks.id OR chunks.id LIKE '%:' || refs.target_uuid
                WHERE refs.source_id = ?
                GROUP BY chunks.id
                LIMIT ?
            """, (source_id, limit)).fetchall()
            result["outgoing"] = [
                {"id": r[0], "name": r[1], "type": r[2], "pack": r[3], "context": r[4]}
                for r in rows
            ]

        if direction in ("incoming", "both"):
            # Find the bare UUID of this entry for incoming lookups
            bare_uuid = normalized
            if chunk:
                # Extract bare UUID from pack:id format
                if ":" in chunk["id"]:
                    bare_uuid = chunk["id"].rsplit(":", 1)[-1]
                else:
                    bare_uuid = chunk["id"]
            rows = self._conn_ro.execute("""
                SELECT chunks.id, chunks.name, chunks.type, chunks.pack, refs.context
                FROM refs
                JOIN chunks ON refs.source_id = chunks.id
                WHERE refs.target_uuid = ?
                LIMIT ?
            """, (bare_uuid, limit)).fetchall()
            result["incoming"] = [
                {"id": r[0], "name": r[1], "type": r[2], "pack": r[3], "context": r[4]}
                for r in rows
            ]

        return result

    def status(self) -> dict:
        self._ensure_loaded()
        meta = {}
        for row in self._conn_ro.execute("SELECT key, value FROM _meta"):
            meta[row[0]] = row[1]
        meta["actual_chunks"] = self._conn_ro.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        meta["db_path"] = str(self.db_path)
        return meta

    def close(self) -> None:
        with self._db_lock:
            if self._conn_ro:
                self._conn_ro.close()
                self._conn_ro = None
            if self._conn:
                self._conn.close()
                self._conn = None
            self._fts_ready = False
