"""Orchestration: fetch → extract → chunk → embed → index."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .chunker import ChunkBuilder, UUIDResolver
from .config import Settings
from .embeddings import get_provider
from .fetcher import extract_all_packs, extract_lang, get_cached_zip
from .index import init_db, load_vec_extension, vec_blob


def build_chunks(settings: Settings) -> list[dict[str, Any]]:
    """Fetch data and build enriched chunks."""
    zip_path = get_cached_zip(settings)
    cache_extract = settings.cache_dir / f"extract-{settings.release}"

    print("Loading all packs...")
    all_entries = extract_all_packs(zip_path, cache_extract)
    print(f"Loaded {len(all_entries)} packs")

    _localizer = extract_lang(zip_path, cache_extract)
    resolver = UUIDResolver(all_entries)
    builder = ChunkBuilder(resolver)

    chunks = []
    for pack_name, entries in all_entries.items():
        for entry in entries:
            for chunk in builder.build_all(entry, pack_name):
                chunks.append(chunk)

    types = Counter(c["type"] for c in chunks)
    print(f"Generated {len(chunks)} chunks")
    print("Chunk types:", dict(types))
    print(f"Total text chars: {sum(len(c['text']) for c in chunks):,}")
    print(f"Avg chunk size: {sum(len(c['text']) for c in chunks) / len(chunks):.0f} chars")
    return chunks


def save_chunks(chunks: list[dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps(chunks, indent=2))
    print(f"Wrote {len(chunks)} chunks -> {path}")


def embed_and_index(chunks: list[dict[str, Any]], settings: Settings, rebuild: bool = False) -> None:
    """Embed chunks and store in sqlite-vec."""
    provider = get_provider(settings.model)
    dim = provider.dim
    print(f"Embedding model: {settings.model} (dim={dim})")

    db_path = settings.db
    init_db(db_path, dim)

    import sqlite3

    conn = sqlite3.connect(str(db_path))
    load_vec_extension(conn)

    row = conn.execute("SELECT value FROM _meta WHERE key = 'embedding_model'").fetchone()
    if row and row[0] != settings.model:
        print(f"Model mismatch: DB has {row[0]}, using {settings.model}")
        if not rebuild:
            print("Pass --rebuild to replace")
            conn.close()
            return

    if rebuild:
        conn.execute("DROP TABLE IF EXISTS vec_chunks")
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM _meta")
        conn.execute(f"""
            CREATE VIRTUAL TABLE vec_chunks USING vec0(
                id TEXT PRIMARY KEY,
                embedding float[{dim}]
            )
        """)
        conn.commit()
        print("Cleared existing index")

    print("Embedding...")
    start = time.time()
    embeddings = provider.embed([c["text"] for c in chunks])
    print(f"Embedded {len(chunks)} chunks in {time.time() - start:.1f}s")

    print("Inserting into database...")
    start = time.time()
    for chunk, emb in zip(chunks, embeddings):
        conn.execute("""
            INSERT OR REPLACE INTO chunks (id, name, type, pack, slug, level, traits, text, raw_rules_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk["id"], chunk["name"], chunk["type"], chunk["pack"],
            chunk.get("slug", ""),
            chunk.get("level") if chunk.get("level") is not None else None,
            json.dumps(chunk.get("traits", [])),
            chunk["text"], chunk["raw_rules_count"],
        ))
        conn.execute(
            "INSERT INTO vec_chunks (id, embedding) VALUES (?, vec_f32(?))",
            (chunk["id"], vec_blob(emb)),
        )
    conn.commit()
    print(f"Inserted {len(chunks)} chunks in {time.time() - start:.1f}s")

    for k, v in [
        ("embedding_model", settings.model),
        ("embedding_dim", str(dim)),
        ("total_chunks", str(len(chunks))),
        ("index_date", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
    ]:
        conn.execute("INSERT OR REPLACE INTO _meta (key, value) VALUES (?, ?)", (k, v))
    conn.commit()
    conn.close()
    print(f"Index: {db_path}")


def index_all(settings: Settings, rebuild: bool = False) -> None:
    """Full pipeline: fetch → build chunks → embed → index."""
    chunks = build_chunks(settings)
    embed_and_index(chunks, settings, rebuild=rebuild)
