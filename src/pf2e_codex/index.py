"""sqlite-vec database management and search."""

from __future__ import annotations

import struct
import time
from pathlib import Path

import sqlite_vec  # type: ignore[import-untyped]

from .embeddings import EmbeddingProvider, get_provider


def load_vec_extension(conn) -> None:
    conn.enable_load_extension(True)
    conn.load_extension(sqlite_vec.loadable_path())


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
            license TEXT DEFAULT 'NONE'
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
    conn.commit()
    conn.close()


def rebuild_fts(conn) -> None:
    """Rebuild the FTS5 index from the chunks table."""
    conn.execute("INSERT INTO fts_chunks(fts_chunks) VALUES ('rebuild')")


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


class SearchIndex:
    """Search index backed by sqlite-vec with optional FTS5 hybrid blending."""

    def __init__(self, db_path: Path | str, model_name: str, provider: str = "auto", onnx_provider: str | None = None):
        import sqlite3

        self.db_path = Path(db_path)
        self.model_name = model_name
        self._provider_type = provider
        self._onnx_provider = onnx_provider
        self._provider: EmbeddingProvider | None = None
        self._dim: int | None = None
        self._conn: sqlite3.Connection | None = None
        self._fts_ready: bool = False

    def _ensure_loaded(self) -> None:
        if self._conn is not None:
            return
        import sqlite3

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        self._conn = sqlite3.connect(str(self.db_path))
        load_vec_extension(self._conn)
        row = self._conn.execute(
            "SELECT value FROM _meta WHERE key = 'embedding_model'"
        ).fetchone()
        db_model = row[0] if row else None
        if db_model and db_model != self.model_name:
            print(f"Warning: DB model {db_model} != config model {self.model_name}")
        row = self._conn.execute(
            "SELECT value FROM _meta WHERE key = 'embedding_dim'"
        ).fetchone()
        self._dim = int(row[0]) if row else 384
        # Lazy-create refs table for DBs built before this feature
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

    def _ensure_fts(self) -> None:
        if self._fts_ready:
            return
        row = self._conn.execute(
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

    @property
    def provider(self) -> EmbeddingProvider:
        if self._provider is None:
            self._provider = get_provider(
                self.model_name,
                provider=self._provider_type,
                onnx_provider=self._onnx_provider,
            )
        return self._provider

    def _encode(self, text: str) -> list[float]:
        return self.provider.embed_query(text)

    def search(self, query: str, top_k: int = 5, hybrid: bool = True) -> list[dict]:
        self._ensure_loaded()

        # Semantic search
        emb = self._encode(query)
        q_blob = vec_blob(emb)
        semantic_raw = self._conn.execute("""
            SELECT chunks.id, chunks.name, chunks.type, chunks.pack, chunks.text, chunks.license, distance
            FROM vec_chunks
            JOIN chunks ON vec_chunks.id = chunks.id
            WHERE vec_chunks.embedding MATCH vec_f32(?)
              AND k = ?
            ORDER BY distance
        """, (q_blob, top_k * 3)).fetchall()

        semantic_results = [
            (r[0], {
                "id": r[0], "name": r[1], "type": r[2], "pack": r[3],
                "text": r[4], "license": r[5], "distance": r[6],
            })
            for r in semantic_raw
        ]

        if not hybrid:
            return [r for _, r in semantic_results[:top_k]]

        # Name bag-of-words search — match on content words only, skip
        # generic type labels like 'spell', 'feat', 'action'.
        _STOP_WORDS = frozenset({
            'spell', 'feat', 'action', 'effect', 'item', 'weapon', 'armor',
            'equipment', 'condition', 'ability', 'feature', 'class',
            'ancestry', 'background', 'heritage', 'deity', 'hazard',
            'vehicle', 'familiar', 'npc', 'creature', 'monster', 'ritual',
        })
        like_results: list[tuple[str, dict]] = []
        try:
            words = [
                w.strip().lower() for w in query.split()
                if len(w.strip()) > 2 and w.strip().lower() not in _STOP_WORDS
            ]
            if words:
                like_conditions = " OR ".join(["LOWER(name) LIKE ?" for _ in words])
                like_params = [f"%{w}%" for w in words]
                like_raw = self._conn.execute(f"""
                    SELECT id, name, type, pack, text, license
                    FROM chunks
                    WHERE {like_conditions}
                    LIMIT ?
                """, like_params + [top_k * 3]).fetchall()
                # Score by number of matching words in name
                scored: dict[str, tuple[tuple, int]] = {}
                for r in like_raw:
                    cid = r[0]
                    match_count = sum(1 for w in words if w in r[1].lower())
                    if match_count > 0:
                        existing = scored.get(cid)
                        if not existing or existing[1] < match_count:
                            scored[cid] = (r, match_count)
                sorted_entries = sorted(scored.values(), key=lambda x: -x[1])
                for r, _mc in sorted_entries[:top_k * 3]:
                    like_results.append((r[0], {
                        "id": r[0], "name": r[1], "type": r[2], "pack": r[3],
                        "text": r[4], "license": r[5], "distance": None,
                    }))
        except Exception:
            pass

        return _rrf_fuse(semantic_results, like_results, top_k=top_k)

    def rules_explain(self, topic: str, top_k: int = 3) -> list[dict]:
        """Search with boosted journal pages and conditions for core rules."""
        self._ensure_loaded()
        emb = self._encode(topic)
        q_blob = vec_blob(emb)
        results = self._conn.execute("""
            SELECT chunks.id, chunks.name, chunks.type, chunks.pack, chunks.text, chunks.license, distance
            FROM vec_chunks
            JOIN chunks ON vec_chunks.id = chunks.id
            WHERE vec_chunks.embedding MATCH vec_f32(?)
              AND k = ?
            ORDER BY distance
        """, (q_blob, top_k * 3)).fetchall()
        scored = []
        for r in results:
            ctype = r[2]
            distance = r[5]
            boost = 0.0
            if ctype == "journal_page":
                boost = 0.15
            elif ctype == "condition":
                boost = 0.05
            scored.append((distance - boost, r))
        scored.sort(key=lambda x: x[0])
        return [
            {"id": r[0], "name": r[1], "type": r[2], "pack": r[3], "text": r[4], "license": r[5], "distance": r[6]}
            for _, r in scored[:top_k]
        ]

    def fetch_by_id(self, entry_id: str) -> dict | None:
        """Fetch a single chunk by its internal ID, Foundry UUID, slug, or name."""
        self._ensure_loaded()
        normalized = _normalize_id(entry_id)

        # 1. Exact ID match (pack:id or bare Foundry _id)
        for sql, param in [
            ("SELECT id, name, type, pack, text FROM chunks WHERE id = ?", (normalized,)),
            ("SELECT id, name, type, pack, text FROM chunks WHERE id LIKE ?", (f"%:{normalized}",)),
        ]:
            row = self._conn.execute(sql, param).fetchone()
            if row:
                return {
                    "id": row[0], "name": row[1], "type": row[2],
                    "pack": row[3], "text": row[4],
                }

        # 2. Slug match (e.g. "fury-instinct")
        row = self._conn.execute(
            "SELECT id, name, type, pack, text FROM chunks WHERE slug = ?",
            (normalized,),
        ).fetchone()
        if row:
            return {
                "id": row[0], "name": row[1], "type": row[2],
                "pack": row[3], "text": row[4],
            }

        # 3. Exact name match (case-insensitive)
        row = self._conn.execute(
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
            rows = self._conn.execute("""
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
            rows = self._conn.execute("""
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
        for row in self._conn.execute("SELECT key, value FROM _meta"):
            meta[row[0]] = row[1]
        meta["actual_chunks"] = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        meta["db_path"] = str(self.db_path)
        return meta

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
