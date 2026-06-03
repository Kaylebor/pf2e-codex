"""Orchestration: fetch → extract → chunk → embed → index."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .chunker import ChunkBuilder, UUIDResolver, entry_hash
from .config import Settings
from .embeddings import get_provider
from .fetcher import extract_all_packs, extract_lang, get_cached_zip
from .index import init_db, load_vec_extension, rebuild_fts, vec_blob


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
    provider = get_provider(
        settings.model,
        provider=settings.provider,
        onnx_provider=settings.onnx_provider,
    )
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
        conn.execute("DROP TABLE IF EXISTS fts_chunks")
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM refs")
        conn.execute("DELETE FROM _meta")
        conn.execute(f"""
            CREATE VIRTUAL TABLE vec_chunks USING vec0(
                id TEXT PRIMARY KEY,
                embedding float[{dim}]
            )
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE fts_chunks USING fts5(
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
            INSERT OR REPLACE INTO chunks (id, name, type, pack, slug, level, traits, text, raw_rules_count, source_hash, license, remaster)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk["id"], chunk["name"], chunk["type"], chunk["pack"],
            chunk.get("slug", ""),
            chunk.get("level") if chunk.get("level") is not None else None,
            json.dumps(chunk.get("traits", [])),
            chunk["text"], chunk["raw_rules_count"],
            chunk.get("source_hash"),
            chunk.get("license", "NONE"),
            1 if chunk.get("remaster") else (0 if chunk.get("remaster") is not None else None),
        ))
        conn.execute(
            "INSERT INTO vec_chunks (id, embedding) VALUES (?, vec_f32(?))",
            (chunk["id"], vec_blob(emb)),
        )
        # Insert cross-references
        for ref in chunk.get("refs", []):
            conn.execute("""
                INSERT OR IGNORE INTO refs (source_id, target_uuid, target_name, context)
                VALUES (?, ?, ?, ?)
            """, (chunk["id"], ref["uuid"], ref["name"], ref.get("context", "")[:200]))
    conn.commit()
    print(f"Inserted {len(chunks)} chunks and refs in {time.time() - start:.1f}s")

    print("Building FTS5 index...")
    start = time.time()
    rebuild_fts(conn)
    conn.commit()
    print(f"FTS5 index built in {time.time() - start:.1f}s")

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


def update_index(settings: Settings) -> None:
    """Incremental update: diff against indexed release, only re-process changed entries."""
    import sqlite3

    from .embeddings import get_provider
    from .fetcher import extract_all_packs, get_cached_zip
    from .index import init_db, load_vec_extension, rebuild_fts, vec_blob

    # Ensure DB exists
    dim = get_provider(settings.model, provider=settings.provider, onnx_provider=settings.onnx_provider).dim
    init_db(settings.db, dim)

    conn = sqlite3.connect(str(settings.db))
    load_vec_extension(conn)

    # Get current indexed release
    current_release = conn.execute(
        "SELECT value FROM _meta WHERE key = 'pf2e_release'"
    ).fetchone()
    current_release = current_release[0] if current_release else None

    # Lazy-add source_hash column for existing DBs
    try:
        conn.execute("ALTER TABLE chunks ADD COLUMN source_hash TEXT")
    except Exception:
        pass  # already exists
    try:
        conn.execute("ALTER TABLE chunks ADD COLUMN license TEXT DEFAULT 'NONE'")
    except Exception:
        pass  # already exists
    try:
        conn.execute("ALTER TABLE chunks ADD COLUMN remaster INTEGER DEFAULT NULL")
    except Exception:
        pass  # already exists

    if current_release == settings.release:
        print(f"Already indexed version {settings.release}")
        conn.close()
        return

    print(f"Current: {current_release}, Target: {settings.release}")

    # Load existing hashes into memory for fast lookup
    existing_hashes: dict[str, str] = {}
    for row in conn.execute("SELECT id, source_hash FROM chunks"):
        # Strip page suffix from journal IDs for entry-level lookup
        eid = row[0].rsplit("_page_", 1)[0] if "_page_" in row[0] else row[0]
        existing_hashes.setdefault(eid, row[1] or "")

    # Fetch new release
    zip_path = get_cached_zip(settings)
    cache_extract = settings.cache_dir / f"extract-{settings.release}"
    all_entries = extract_all_packs(zip_path, cache_extract)

    # Diff: find new/changed entries
    resolver = UUIDResolver(all_entries)
    builder = ChunkBuilder(resolver)

    changed: list[dict] = []
    unchanged = 0
    all_new_ids: set[str] = set()

    for pack_name, entries in all_entries.items():
        for entry in entries:
            entry_id = entry.get("_id", "")
            packed_id = f"{pack_name}:{entry_id}"
            all_new_ids.add(packed_id)
            h = entry_hash(entry)
            existing = existing_hashes.get(packed_id, "")
            if existing and existing == h:
                unchanged += 1
                continue
            # New or changed — rebuild chunks
            for chunk in builder.build_all(entry, pack_name):
                changed.append(chunk)

    orphan_ids = set(existing_hashes.keys()) - all_new_ids

    print(f"Unchanged: {unchanged}, Changed/New: {len(changed)} entries ({len(changed)} chunks), Orphaned (removed): {len(orphan_ids)}")

    if not changed:
        print("Nothing to update")
        conn.close()
        return

    # Embed and index only changed chunks
    provider = get_provider(settings.model, provider=settings.provider, onnx_provider=settings.onnx_provider)
    texts = [c["text"] for c in changed]
    print(f"Embedding {len(changed)} changed chunks...")
    import time
    start = time.time()
    embeddings = provider.embed(texts)
    print(f"Embedded in {time.time() - start:.1f}s")

    print("Updating database...")
    start = time.time()
    for chunk, emb in zip(changed, embeddings):
        # Delete old chunks for this entry (all pages)
        base_id = chunk["id"].split("_page_")[0] if "_page_" in chunk["id"] else chunk["id"]
        conn.execute("DELETE FROM vec_chunks WHERE id LIKE ?", (f"{base_id}%",))
        conn.execute("DELETE FROM chunks WHERE id LIKE ?", (f"{base_id}%",))
        conn.execute("DELETE FROM refs WHERE source_id LIKE ?", (f"{base_id}%",))

        # Insert new chunk
        conn.execute("""
            INSERT OR REPLACE INTO chunks (id, name, type, pack, slug, level, traits, text, raw_rules_count, source_hash, license, remaster)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk["id"], chunk["name"], chunk["type"], chunk["pack"],
            chunk.get("slug", ""),
            chunk.get("level") if chunk.get("level") is not None else None,
            json.dumps(chunk.get("traits", [])),
            chunk["text"], chunk["raw_rules_count"],
            chunk.get("source_hash"),
            chunk.get("license", "NONE"),
            1 if chunk.get("remaster") else (0 if chunk.get("remaster") is not None else None),
        ))
        conn.execute(
            "INSERT INTO vec_chunks (id, embedding) VALUES (?, vec_f32(?))",
            (chunk["id"], vec_blob(emb)),
        )
        for ref in chunk.get("refs", []):
            conn.execute(
                "INSERT OR IGNORE INTO refs (source_id, target_uuid, target_name, context) VALUES (?, ?, ?, ?)",
                (chunk["id"], ref["uuid"], ref["name"], ref.get("context", "")[:200]),
            )

    # Remove orphaned entries (deleted from source)
    for oid in orphan_ids:
        conn.execute("DELETE FROM vec_chunks WHERE id LIKE ?", (f"{oid}%",))
        conn.execute("DELETE FROM chunks WHERE id LIKE ?", (f"{oid}%",))
        conn.execute("DELETE FROM refs WHERE source_id LIKE ?", (f"{oid}%",))

    # Update meta
    conn.execute("INSERT OR REPLACE INTO _meta (key, value) VALUES (?, ?)",
        ("pf2e_release", settings.release))
    conn.commit()

    print(f"Updated in {time.time() - start:.1f}s")
    print(f"  Changed: {len(changed)} chunks")
    print(f"  Removed: {len(orphan_ids)} entries")
    print(f"  DB: {settings.db}")
    conn.close()
