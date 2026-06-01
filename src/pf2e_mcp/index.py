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
            raw_rules_count INTEGER
        )
    """)
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            id TEXT PRIMARY KEY,
            embedding float[{dim}]
        )
    """)
    conn.commit()
    conn.close()


def vec_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


class SearchIndex:
    """Search index backed by sqlite-vec."""

    def __init__(self, db_path: Path, model_name: str):
        import sqlite3

        self.db_path = db_path
        self.model_name = model_name
        self._provider: EmbeddingProvider | None = None
        self._dim: int | None = None
        self._conn: sqlite3.Connection | None = None

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

    @property
    def provider(self) -> EmbeddingProvider:
        if self._provider is None:
            self._provider = get_provider(self.model_name)
        return self._provider

    def _encode(self, text: str) -> list[float]:
        return self.provider.embed_query(text)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        self._ensure_loaded()
        emb = self._encode(query)
        q_blob = vec_blob(emb)
        results = self._conn.execute("""
            SELECT chunks.id, chunks.name, chunks.type, chunks.pack, chunks.text, distance
            FROM vec_chunks
            JOIN chunks ON vec_chunks.id = chunks.id
            WHERE vec_chunks.embedding MATCH vec_f32(?)
              AND k = ?
            ORDER BY distance
        """, (q_blob, top_k)).fetchall()
        return [
            {"id": r[0], "name": r[1], "type": r[2], "pack": r[3], "text": r[4], "distance": r[5]}
            for r in results
        ]

    def lookup(self, name: str) -> list[dict]:
        self._ensure_loaded()
        results = self._conn.execute("""
            SELECT id, name, type, pack, text
            FROM chunks
            WHERE LOWER(name) = LOWER(?)
            ORDER BY type, pack
            LIMIT 20
        """, (name,)).fetchall()
        return [{"id": r[0], "name": r[1], "type": r[2], "pack": r[3], "text": r[4]} for r in results]

    def rules_explain(self, topic: str, top_k: int = 3) -> list[dict]:
        self._ensure_loaded()
        emb = self._encode(topic)
        q_blob = vec_blob(emb)
        results = self._conn.execute("""
            SELECT chunks.id, chunks.name, chunks.type, chunks.pack, chunks.text, distance
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
            {"id": r[0], "name": r[1], "type": r[2], "pack": r[3], "text": r[4], "distance": r[5]}
            for _, r in scored[:top_k]
        ]

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
