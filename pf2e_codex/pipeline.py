"""Orchestration: fetch → extract → chunk → embed → index."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .chunker import ChunkBuilder, UUIDResolver, entry_hash
from .config import Settings
from .embeddings import EmbeddingProvider, get_provider
from .fetcher import extract_all_packs, get_cached_zip
from .index import (
    ensure_ambiguous_ref_targets,
    init_db,
    load_vec_extension,
    rebuild_fts,
    vec_blob,
)


def _like_escape(value: str) -> str:
    """Escape LIKE metacharacters for exact page-prefix matching."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _entry_page_pattern(base_id: str) -> str:
    """Return an escaped LIKE pattern matching only an entry's page chunks."""
    return f"{_like_escape(base_id)}\\_page\\_%"


def _delete_entry_rows(conn: Any, base_id: str) -> None:
    """Delete a base chunk and its journal pages without wildcard overreach."""
    page_pattern = _entry_page_pattern(base_id)
    for table, column in (("vec_chunks", "id"), ("chunks", "id"), ("refs", "source_id")):
        conn.execute(
            f"DELETE FROM {table} WHERE {column} = ? OR {column} LIKE ? ESCAPE '\\'",
            (base_id, page_pattern),
        )


def build_chunks(settings: Settings) -> list[dict[str, Any]]:
    """Fetch data and build enriched chunks."""
    zip_path = get_cached_zip(settings)
    cache_extract = settings.cache_dir / f"extract-{settings.release}"

    print("Loading all packs...")
    all_entries = extract_all_packs(zip_path, cache_extract)
    print(f"Loaded {len(all_entries)} packs")

    # Merge translations for configured languages
    if settings.languages:
        for lang in settings.languages:
            if lang == "en":
                continue
            print(f"Loading {lang} translations...")
            try:
                from .translate import (  # noqa: PLC0415
                    build_pack_map, fetch_translations, merge_entries,
                )
                translations = fetch_translations(settings.cache_dir, lang=lang)
                pack_map = build_pack_map(translations, set(all_entries.keys()))
                print(f"  Matched {len(pack_map)}/{len(translations)} packs")
                for en_name, es_name in pack_map.items():
                    if en_name in all_entries and es_name in translations:
                        merge_entries(all_entries[en_name], translations[es_name])
            except Exception as e:
                print(f"  Failed to load {lang} translations: {e}")
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


def embed_and_index(chunks: list[dict[str, Any]], settings: Settings, rebuild: bool = False,
                    provider: EmbeddingProvider | None = None) -> None:
    """Embed chunks and store in sqlite-vec."""
    if provider is None:
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
        conn.execute("DELETE FROM ambiguous_ref_targets")
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
            INSERT OR REPLACE INTO chunks (id, name, type, pack, slug, level, traits, text, raw_rules_count, source_hash, license, remaster, translations)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk["id"], chunk["name"], chunk["type"], chunk["pack"],
            chunk.get("slug", ""),
            chunk.get("level") if chunk.get("level") is not None else None,
            json.dumps(chunk.get("traits", [])),
            chunk["text"], chunk["raw_rules_count"],
            chunk.get("source_hash"),
            chunk.get("license", "NONE"),
            1 if chunk.get("remaster") else (0 if chunk.get("remaster") is not None else None),
            json.dumps(chunk.get("translations")) if chunk.get("translations") else None,
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
    # Mark duplicate IDs introduced by this inserted snapshot. Incremental
    # updates use INSERT OR IGNORE so prior ambiguity tombstones persist.
    ensure_ambiguous_ref_targets(conn)
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
        ("pf2e_release", settings.release),
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


def update_index(settings: Settings, _provider: EmbeddingProvider | None = None) -> None:
    """Incremental update: diff against indexed release, only re-process changed entries."""
    import sqlite3

    from .embeddings import get_provider as gp
    from .fetcher import extract_all_packs, get_cached_zip
    from .index import init_db, load_vec_extension, rebuild_fts, vec_blob

    # Ensure DB exists
    if _provider is None:
        _provider = gp(settings.model, provider=settings.provider, onnx_provider=settings.onnx_provider)
    dim = _provider.dim
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
    try:
        conn.execute("ALTER TABLE chunks ADD COLUMN translations TEXT DEFAULT NULL")
    except Exception:
        pass  # already exists
    # Persist duplicate-ID tombstones before any incremental deletions. The
    # markers intentionally survive ordinary updates to prevent a former
    # sibling's legacy bare refs from boosting the remaining target.
    ensure_ambiguous_ref_targets(conn)
    conn.commit()

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
    # Track changed source entries independently of emitted chunks. A valid
    # source entry can now produce zero chunks; its old rows still need to be
    # removed and the release advanced atomically.
    changed_entry_ids: set[str] = set()
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
            changed_entry_ids.add(packed_id)
            # New or changed — rebuild chunks
            for chunk in builder.build_all(entry, pack_name):
                changed.append(chunk)

    orphan_ids = set(existing_hashes.keys()) - all_new_ids

    print(
        f"Unchanged: {unchanged}, Changed/New: {len(changed_entry_ids)} entries "
        f"({len(changed)} chunks), Orphaned (removed): {len(orphan_ids)}"
    )

    # Embed only changed chunks. Deletions and metadata updates still need to
    # run when a release contains no changed entries.
    embeddings: list[list[float]] = []
    if changed:
        texts = [c["text"] for c in changed]
        print(f"Embedding {len(changed)} changed chunks...")
        start = time.time()
        embeddings = _provider.embed(texts)
        print(f"Embedded in {time.time() - start:.1f}s")

    print("Updating database...")
    start = time.time()
    try:
        conn.execute("BEGIN")

        embedded_by_entry: dict[str, list[tuple[dict, list[float]]]] = {}
        for chunk, emb in zip(changed, embeddings, strict=True):
            base_id = chunk["id"].split("_page_", 1)[0]
            embedded_by_entry.setdefault(base_id, []).append((chunk, emb))

        # Delete each changed entry once, then insert all of its replacement
        # chunks. This preserves journal pages when an entry has many pages.
        for base_id in changed_entry_ids:
            entry_chunks = embedded_by_entry.get(base_id, [])
            _delete_entry_rows(conn, base_id)

            for chunk, emb in entry_chunks:
                conn.execute("""
                    INSERT OR REPLACE INTO chunks (id, name, type, pack, slug, level, traits, text, raw_rules_count, source_hash, license, remaster, translations)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk["id"], chunk["name"], chunk["type"], chunk["pack"],
                    chunk.get("slug", ""),
                    chunk.get("level") if chunk.get("level") is not None else None,
                    json.dumps(chunk.get("traits", [])),
                    chunk["text"], chunk["raw_rules_count"],
                    chunk.get("source_hash"),
                    chunk.get("license", "NONE"),
                    1 if chunk.get("remaster") else (0 if chunk.get("remaster") is not None else None),
                    json.dumps(chunk.get("translations")) if chunk.get("translations") else None,
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

        # Remove orphaned entries (deleted from source) and their outgoing
        # references. Legacy bare-target incoming refs are retained because
        # they may belong to another pack with the same ID.
        for oid in orphan_ids:
            _delete_entry_rows(conn, oid)

        if changed_entry_ids or orphan_ids:
            # External-content FTS5 indexes do not automatically follow direct
            # writes to chunks, so rebuild after every data mutation.
            rebuild_fts(conn)

        # Mark duplicates introduced by replacement chunks; INSERT OR IGNORE
        # preserves tombstones recorded before orphan/deletion processing.
        ensure_ambiguous_ref_targets(conn)

        actual = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.execute("INSERT OR REPLACE INTO _meta (key, value) VALUES (?, ?)",
            ("total_chunks", str(actual)))
        conn.execute("INSERT OR REPLACE INTO _meta (key, value) VALUES (?, ?)",
            ("pf2e_release", settings.release))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print(f"Updated in {time.time() - start:.1f}s")
    print(f"  Changed: {len(changed)} chunks")
    print(f"  Removed: {len(orphan_ids)} entries")
    print(f"  DB: {settings.db}")


def embed_all_models(
    settings: Settings,
    models: list[str],
    concurrency: int = 1,
    update: bool = False,
    rebuild: bool = False,
) -> dict[str, bool]:
    """Build chunks once, then export sequentially, embed in parallel.

    ONNX export modifies global state (GLOBALS.in_onnx_export) — must
    be sequential. Embedding is pure inference — can be parallel.

    Args:
        settings: Base settings (data_dir, release, etc.).
        models: List of model names to embed.
        concurrency: Max parallel embedding jobs.
        update: If True, run incremental update on existing DBs instead of skipping.

    Returns {model_name: success} dict.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .config import Settings as S
    from .models import get_model_info

    # — Shared phase: fetch + chunk once —
    print(f"Building chunks from {settings.release}...")
    chunks = build_chunks(settings)
    print(f"Generated {len(chunks)} chunks\n")

    results: dict[str, bool] = {}
    pending: list[str] = []  # models needing fresh embed
    upd_pending: list[str] = []  # models needing incremental update

    for model in models:
        info = get_model_info(model)
        name = info.name if info else model
        db = settings.data_dir / f"pf2e_{model.replace('/', '--')}.db"
        if db.exists():
            if rebuild:
                pending.append(model)
            elif update:
                upd_pending.append(model)
            else:
                print(f"[skip] {name} — DB exists")
                results[model] = True
        else:
            pending.append(model)

    if not pending and not upd_pending:
        print("All models already indexed.")
        return results

    export_pending = list(set(pending) | set(upd_pending))
    providers: dict[str, EmbeddingProvider] = {}

    # — Phase 1: Sequential ONNX export + compile —
    # Only models without a cached ONNX export need this.
    # Cached models skip to Phase 2 (compile happens there, in parallel).
    from .embeddings import _onnx_cache_dir as _ocd
    need_export = [m for m in export_pending if not (_ocd(m) / "model.onnx").exists()]
    export_ok: set[str] = set(export_pending)  # assume all OK, mark failures

    if need_export:
        print(f"Phase 1: ONNX export + compile ({len(need_export)} models, sequential)\n")
        for model in need_export:
            model_settings = S(
                model=model,
                data_dir=str(settings.data_dir),
                release=settings.release,
                provider=settings.provider,
                onnx_provider=settings.onnx_provider,
            )
            try:
                from .embeddings import get_provider as gp
                prov = gp(model_settings.model, provider=model_settings.provider,
                          onnx_provider=model_settings.onnx_provider)
                providers[model] = prov
                print(f"[export] {model} OK")
            except Exception as e:
                print(f"[export] {model} FAIL: {e}")
                export_ok.discard(model)
                results[model] = False

    if skip_export := [m for m in export_pending if m not in need_export]:
        print(f"Phase 1: {len(skip_export)} models already exported (skipped)\n")

    # — Phase 2: Parallel embedding/update —
    embed_pending = [m for m in pending if m in export_ok]
    upd_ok = [m for m in upd_pending if m in export_ok]

    if not embed_pending and not upd_ok:
        print("\nNo models to embed.")
        return results

    def _embed_one(model: str) -> tuple[str, bool]:
        ms = S(
            model=model,
            data_dir=str(settings.data_dir),
            release=settings.release,
            provider=settings.provider,
            onnx_provider=settings.onnx_provider,
        )
        try:
            embed_and_index(chunks, ms, rebuild=rebuild, provider=providers.get(model))
            return (model, True)
        except Exception as e:
            print(f"[FAIL] {model}: {e}")
            return (model, False)

    def _update_one(model: str) -> tuple[str, bool]:
        ms = S(
            model=model,
            data_dir=str(settings.data_dir),
            release=settings.release,
            provider=settings.provider,
            onnx_provider=settings.onnx_provider,
        )
        try:
            update_index(ms, _provider=providers.get(model))
            return (model, True)
        except Exception as e:
            print(f"[FAIL] {model}: {e}")
            return (model, False)

    if embed_pending:
        print(f"\nPhase 2a: Fresh embed ({len(embed_pending)} models, concurrency={concurrency})\n")
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_embed_one, m): m for m in embed_pending}
            for future in as_completed(futures):
                model, ok = future.result()
                results[model] = ok
                status = "[done]" if ok else "[FAIL]"
                print(f"{status}  {model}")

    if upd_ok:
        print(f"\nPhase 2b: Incremental update ({len(upd_ok)} models, concurrency={concurrency})\n")
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_update_one, m): m for m in upd_ok}
            for future in as_completed(futures):
                model, ok = future.result()
                results[model] = ok
                status = "[done]" if ok else "[FAIL]"
                print(f"{status}  {model}")

    print()
    failed = sum(1 for v in results.values() if not v)
    if failed:
        print(f"{failed} model(s) failed.")
    else:
        print("All models embedded successfully.")

    return results
