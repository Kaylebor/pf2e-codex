"""Orchestration: fetch → extract → chunk → embed → index."""

from __future__ import annotations

import json
import os
import time
import uuid
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
    migrate_db,
    rebuild_fts,
    vec_blob,
)


def _corpus_scope_value(settings: Settings) -> str:
    """Return the configured seed scope as a stable metadata string."""
    scope = getattr(settings, "corpus_scope", "redistributable")
    return str(getattr(scope, "value", scope))


def _require_seed_slot(settings: Settings) -> None:
    """Prevent a seed policy from ever writing the opposite physical slot."""
    resolved = getattr(settings, "resolved_database_scope", None)
    if resolved is None:
        return
    actual = str(getattr(resolved, "value", resolved))
    expected = "local" if _corpus_scope_value(settings) == "local-full" else "clean"
    if actual != expected:
        raise RuntimeError(
            f"seed scope {_corpus_scope_value(settings)!r} requires the {expected} "
            f"database slot, not {actual}"
        )


def _like_escape(value: str) -> str:
    """Escape LIKE metacharacters for exact page-prefix matching."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _entry_page_pattern(base_id: str) -> str:
    """Return an escaped LIKE pattern matching only an entry's page chunks."""
    return f"{_like_escape(base_id)}\\_page\\_%"


def _delete_entry_rows(conn: Any, base_id: str, *, origin: str | None = None) -> None:
    """Delete a base chunk and its journal pages without wildcard overreach."""
    page_pattern = _entry_page_pattern(base_id)
    clause = "(id = ? OR id LIKE ? ESCAPE '\\')"
    params: list[str] = [base_id, page_pattern]
    if origin is not None:
        clause += " AND origin = ?"
        params.append(origin)
    # References must be removed before their owning chunks, and vector rows
    # use the owning chunk IDs so similarly named corpus rows cannot be swept.
    conn.execute(
        f"DELETE FROM refs WHERE source_id IN (SELECT id FROM chunks WHERE {clause})",
        params,
    )
    conn.execute(
        f"DELETE FROM vec_chunks WHERE id IN (SELECT id FROM chunks WHERE {clause})",
        params,
    )
    conn.execute(f"DELETE FROM chunks WHERE {clause}", params)


def _foundry_source(settings: Settings) -> dict[str, Any]:
    return {
        "source_id": f"foundry:{settings.release}",
        "source": "foundry",
        "product": "FoundryVTT PF2E",
        "revision": settings.release,
        "parser": "foundry-json",
        "license": "mixed",
        "era": "mixed",
        "provenance": {"release": settings.release},
    }


def _source_for_chunk(
    chunk: dict[str, Any], settings: Settings, *, force_origin: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve a chunk's source contract into a source-table record."""
    origin = force_origin or chunk.get("origin") or "foundry"
    source_value = chunk.get("source")
    metadata = dict(chunk.get("source_metadata") or {})
    if isinstance(source_value, dict):
        metadata = {**source_value, **metadata}

    if origin == "foundry":
        foundry = _foundry_source(settings)
        metadata = {**foundry, **metadata}
        # A Foundry update must always point to its release source, never a
        # source identifier supplied by an unrelated parser.
        if force_origin == "foundry":
            metadata["source_id"] = foundry["source_id"]
    else:
        metadata.setdefault("source", source_value if isinstance(source_value, str) else origin)

    source_id = chunk.get("source_id") or metadata.get("source_id") or metadata.get("id")
    if origin == "foundry" and force_origin == "foundry":
        source_id = _foundry_source(settings)["source_id"]
    if not source_id:
        raise ValueError(f"{origin} chunk {chunk.get('id', '<unknown>')} is missing source_id")

    metadata["source_id"] = source_id
    metadata.setdefault("source", origin)
    for field in ("product", "revision", "parser", "license", "era", "provenance"):
        if field not in metadata and field in chunk:
            metadata[field] = chunk[field]
    return origin, metadata


def _upsert_source(conn: Any, source: dict[str, Any]) -> None:
    provenance = source.get("provenance")
    if provenance is not None and not isinstance(provenance, str):
        provenance = json.dumps(provenance, sort_keys=True)
    conn.execute("""
        INSERT INTO sources
            (source_id, source, product, revision, parser, license, era, provenance)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id) DO UPDATE SET
            source = excluded.source,
            product = COALESCE(excluded.product, sources.product),
            revision = COALESCE(excluded.revision, sources.revision),
            parser = COALESCE(excluded.parser, sources.parser),
            license = COALESCE(excluded.license, sources.license),
            era = COALESCE(excluded.era, sources.era),
            provenance = COALESCE(excluded.provenance, sources.provenance)
    """, (
        source["source_id"], source["source"], source.get("product"),
        source.get("revision"), source.get("parser"), source.get("license"),
        source.get("era"), provenance,
    ))


def _insert_chunk(
    conn: Any, chunk: dict[str, Any], embedding: list[float], settings: Settings,
    *, force_origin: str | None = None,
) -> None:
    """Persist one section and its source metadata using the shared chunk contract."""
    origin, source = _source_for_chunk(chunk, settings, force_origin=force_origin)
    _upsert_source(conn, source)
    conn.execute("""
        INSERT OR REPLACE INTO chunks (
            id, name, type, pack, slug, level, traits, text, raw_rules_count,
            source_hash, license, remaster, translations, origin, source_id,
            source_page_start, source_page_end, printed_page, section_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        chunk["id"], chunk["name"], chunk["type"], chunk["pack"],
        chunk.get("slug", ""),
        chunk.get("level") if chunk.get("level") is not None else None,
        json.dumps(chunk.get("traits", [])), chunk["text"], chunk["raw_rules_count"],
        chunk.get("source_hash"), chunk.get("license", "NONE"),
        1 if chunk.get("remaster") else (0 if chunk.get("remaster") is not None else None),
        json.dumps(chunk.get("translations")) if chunk.get("translations") else None,
        origin, source["source_id"], chunk.get("source_page_start"),
        chunk.get("source_page_end"), chunk.get("printed_page"), chunk.get("section_hash"),
    ))
    conn.execute(
        "INSERT OR REPLACE INTO vec_chunks (id, embedding) VALUES (?, vec_f32(?))",
        (chunk["id"], vec_blob(embedding)),
    )
    for ref in chunk.get("refs", []):
        conn.execute("""
            INSERT OR IGNORE INTO refs (source_id, target_uuid, target_name, context)
            VALUES (?, ?, ?, ?)
        """, (chunk["id"], ref["uuid"], ref["name"], ref.get("context", "")[:200]))


def _staging_db_path(db_path: Path) -> Path:
    """Return a unique sibling path so replacement is atomic on one filesystem."""
    return db_path.with_name(f".{db_path.name}.staging-{uuid.uuid4().hex}")


def _validate_staged_db(db_path: Path, expected_chunks: int) -> None:
    """Check the minimum invariants required before replacing a live index."""
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        load_vec_extension(conn)
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("staged database integrity check failed")
        version = conn.execute(
            "SELECT value FROM _meta WHERE key = 'schema_version'"
        ).fetchone()
        if not version or int(version[0]) < 2:
            raise RuntimeError("staged database schema migration was not recorded")
        actual_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        actual_vectors = conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
        actual_fts = conn.execute("SELECT COUNT(*) FROM fts_chunks").fetchone()[0]
        source_count = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        scope_row = conn.execute(
            "SELECT value FROM _meta WHERE key = 'distribution_scope'"
        ).fetchone()
        scope = str(scope_row[0]) if scope_row else None
        non_foundry_count = conn.execute(
            "SELECT COUNT(*) FROM chunks "
            "WHERE origin IS NULL OR origin <> 'foundry'"
        ).fetchone()[0]
        unknown_origin_count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE origin IS NULL "
            "OR origin NOT IN ('foundry', 'corpus')"
        ).fetchone()[0]
        if (
            actual_chunks != expected_chunks
            or actual_vectors != expected_chunks
            or actual_fts != expected_chunks
        ):
            raise RuntimeError("staged database chunk/vector/FTS counts do not match")
        if expected_chunks and source_count == 0:
            raise RuntimeError("staged database contains no provenance sources")
        if scope not in ("redistributable", "local-full"):
            raise RuntimeError("staged database has no valid distribution scope")
        if scope == "redistributable" and non_foundry_count:
            raise RuntimeError(
                "redistributable staging database contains non-Foundry or unowned rows"
            )
        if scope == "local-full" and unknown_origin_count:
            raise RuntimeError("local staging database contains unknown row ownership")
    finally:
        conn.close()


def build_chunks(settings: Settings) -> list[dict[str, Any]]:
    """Fetch Foundry data and build enriched Foundry and local-corpus chunks."""
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
                    build_pack_map,
                    fetch_translations,
                    merge_entries,
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
                # Foundry content owns its rows even when a later parser uses
                # the same database for local book sections.
                chunks.append({
                    **chunk,
                    "origin": "foundry",
                    "source_id": f"foundry:{settings.release}",
                })

    types = Counter(c["type"] for c in chunks)
    print(f"Generated {len(chunks)} chunks")
    print("Chunk types:", dict(types))
    print(f"Total text chars: {sum(len(c['text']) for c in chunks):,}")
    print(f"Avg chunk size: {sum(len(c['text']) for c in chunks) / len(chunks):.0f} chars")
    corpus_chunks = build_corpus_chunks(settings)
    if corpus_chunks:
        chunks.extend(corpus_chunks)
        print(f"Added {len(corpus_chunks)} local corpus sections")
    return chunks


def build_corpus_chunks(settings: Settings) -> list[dict[str, Any]]:
    """Discover, export, and parse configured user-owned Paizo sources."""
    if _corpus_scope_value(settings) != "local-full":
        return []
    if not getattr(settings, "corpus_auto_discover", True):
        return []
    root = getattr(settings, "effective_corpus_dir", None)
    if root is None:
        return []
    root = Path(root)
    source_root = root / "sources"
    if not source_root.is_dir():
        if getattr(settings, "corpus_dir", None) is not None:
            raise FileNotFoundError(f"configured corpus source directory does not exist: {source_root}")
        return []

    from .corpus import discover_sources, parse_exports, prepare_exports, select_revisions

    sources = discover_sources(
        source_root,
        include=getattr(settings, "corpus_include", ()),
        exclude=getattr(settings, "corpus_exclude", ()),
    )
    revisions = select_revisions(
        sources,
        prefer=getattr(settings, "corpus_prefer", {}),
        state_root=root,
    )
    if not revisions:
        return []
    prepared = prepare_exports(root, revisions)
    chunks = parse_exports(prepared)
    ids = [chunk["id"] for chunk in chunks]
    duplicates = [value for value, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(
            "corpus parser produced duplicate stable IDs: " + ", ".join(duplicates[:5])
        )
    return chunks


def save_chunks(chunks: list[dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps(chunks, indent=2))
    print(f"Wrote {len(chunks)} chunks -> {path}")


def embed_and_index(
    chunks: list[dict[str, Any]],
    settings: Settings,
    rebuild: bool = False,
    provider: EmbeddingProvider | None = None,
    *,
    activate: bool = True,
) -> Path:
    """Embed chunks and store in sqlite-vec."""
    _require_seed_slot(settings)
    db_path = settings.db
    if rebuild and activate:
        daemon_path = Path(getattr(settings, "data_dir", db_path.parent)) / "server.json"
        if daemon_path.exists():
            raise RuntimeError(
                "Refusing rebuild while a registered daemon may hold the live database. "
                "Stop the daemon and retry."
            )
    if provider is None:
        provider = get_provider(
            settings.model,
            provider=settings.provider,
            onnx_provider=settings.onnx_provider,
        )
    dim = provider.dim
    print(f"Embedding model: {settings.model} (dim={dim})")

    target_db = _staging_db_path(db_path) if rebuild else db_path
    import sqlite3
    try:
        init_db(target_db, dim)
        conn = sqlite3.connect(str(target_db))
        load_vec_extension(conn)
        migrate_db(conn)
        conn.commit()
    except Exception:
        if "conn" in locals():
            conn.close()
        if rebuild and target_db.exists():
            target_db.unlink()
        raise

    row = conn.execute("SELECT value FROM _meta WHERE key = 'embedding_model'").fetchone()
    if row and row[0] != settings.model:
        print(f"Model mismatch: DB has {row[0]}, using {settings.model}")
        if not rebuild:
            print("Pass --rebuild to replace")
            conn.close()
            return db_path

    print("Embedding...")
    start = time.time()
    try:
        embeddings = provider.embed([c["text"] for c in chunks])
    except Exception:
        conn.close()
        if rebuild and target_db.exists():
            target_db.unlink()
        raise
    print(f"Embedded {len(chunks)} chunks in {time.time() - start:.1f}s")

    print("Inserting into database...")
    start = time.time()
    try:
        conn.execute("BEGIN")
        for chunk, emb in zip(chunks, embeddings, strict=True):
            _insert_chunk(conn, chunk, emb, settings)
        # Mark duplicate IDs introduced by this inserted snapshot. Incremental
        # updates use INSERT OR IGNORE so prior ambiguity tombstones persist.
        ensure_ambiguous_ref_targets(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        if rebuild and target_db.exists():
            target_db.unlink()
        raise
    print(f"Inserted {len(chunks)} chunks and refs in {time.time() - start:.1f}s")

    try:
        print("Building FTS5 index...")
        start = time.time()
        rebuild_fts(conn)
        conn.commit()
        print(f"FTS5 index built in {time.time() - start:.1f}s")

        for k, v in [
            ("embedding_model", settings.model),
            ("embedding_dim", str(dim)),
            ("total_chunks", str(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])),
            ("pf2e_release", settings.release),
            ("index_date", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
            ("distribution_scope", _corpus_scope_value(settings)),
        ]:
            conn.execute("INSERT OR REPLACE INTO _meta (key, value) VALUES (?, ?)", (k, v))
        conn.commit()
        conn.close()
    except Exception:
        conn.rollback()
        conn.close()
        if rebuild and target_db.exists():
            target_db.unlink()
        raise
    if rebuild:
        try:
            _validate_staged_db(target_db, len(chunks))
            if activate:
                os.replace(target_db, db_path)
            else:
                print(f"Validated staging index: {target_db}")
                return target_db
        except Exception:
            if target_db.exists():
                target_db.unlink()
            raise
    print(f"Index: {db_path}")
    return db_path


def activate_staged_index(staged_db: Path, settings: Settings) -> Path:
    """Revalidate and atomically activate an explicitly built sibling index."""
    staged_db = staged_db.resolve()
    db_path = settings.db.resolve()
    expected_prefix = f".{db_path.name}.staging-"
    if staged_db.parent != db_path.parent or not staged_db.name.startswith(expected_prefix):
        raise ValueError("staged database must be a generated sibling of the live database")
    daemon_path = settings.data_dir / "server.json"
    if daemon_path.exists():
        raise RuntimeError(
            "Refusing database activation while a registered daemon may hold the live database. "
            "Stop the daemon and retry."
        )

    import sqlite3

    conn = sqlite3.connect(str(staged_db))
    try:
        expected = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    finally:
        conn.close()
    _validate_staged_db(staged_db, expected)
    os.replace(staged_db, db_path)
    return db_path


def index_all(settings: Settings, rebuild: bool = False) -> None:
    """Full pipeline: fetch → build chunks → embed → index."""
    chunks = build_chunks(settings)
    # This is a complete Foundry + corpus snapshot. Always build a validated
    # sibling DB so stale rows cannot survive and a late failure cannot mutate
    # the live index. Keep ``rebuild`` for API/CLI compatibility.
    embed_and_index(chunks, settings, rebuild=True)


def sync_corpus_index(
    settings: Settings,
    chunks: list[dict[str, Any]] | None = None,
    *,
    provider: EmbeddingProvider | None = None,
) -> dict[str, int]:
    """Atomically refresh corpus-owned rows, embedding only changed sections."""
    import sqlite3

    _require_seed_slot(settings)

    daemon_path = settings.data_dir / "server.json"
    if daemon_path.exists():
        raise RuntimeError(
            "Refusing corpus mutation while a registered daemon may hold the database. "
            "Stop the daemon and retry."
        )
    if not settings.db.is_file():
        raise FileNotFoundError(
            f"index does not exist: {settings.db}; run a full index rebuild first"
        )
    if chunks is None:
        chunks = build_corpus_chunks(settings)

    ids = [chunk["id"] for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("corpus sync requires unique stable section IDs")
    for chunk in chunks:
        if chunk.get("origin") != "corpus" or not chunk.get("source_id"):
            raise ValueError(f"invalid corpus chunk ownership contract: {chunk.get('id')}")

    conn = sqlite3.connect(str(settings.db))
    try:
        load_vec_extension(conn)
        migrate_db(conn)
        conn.commit()
        row = conn.execute(
            "SELECT value FROM _meta WHERE key = 'embedding_model'"
        ).fetchone()
        if not row or row[0] != settings.model:
            raise RuntimeError("database embedding model does not match corpus sync settings")

        existing = {
            row[0]: (row[1] or "", row[2])
            for row in conn.execute(
                "SELECT id, section_hash, source_id FROM chunks WHERE origin = 'corpus'"
            )
        }
        incoming = {chunk["id"]: chunk for chunk in chunks}
        changed = [
            chunk for chunk in chunks
            if chunk["id"] not in existing
            or existing[chunk["id"]][0] != chunk.get("section_hash", "")
        ]
        removed_ids = set(existing) - set(incoming)
        unchanged = len(chunks) - len(changed)

        embeddings: list[list[float]] = []
        if changed:
            if provider is None:
                provider = get_provider(
                    settings.model,
                    provider=settings.provider,
                    onnx_provider=settings.onnx_provider,
                )
            embeddings = provider.embed([chunk["text"] for chunk in changed])

        conn.execute("BEGIN")
        for chunk_id in removed_ids | {chunk["id"] for chunk in changed}:
            conn.execute("DELETE FROM refs WHERE source_id = ?", (chunk_id,))
            conn.execute("DELETE FROM vec_chunks WHERE id = ?", (chunk_id,))
            conn.execute("DELETE FROM chunks WHERE id = ? AND origin = 'corpus'", (chunk_id,))
        for chunk, embedding in zip(changed, embeddings, strict=True):
            _insert_chunk(conn, chunk, embedding, settings)

        changed_ids = {chunk["id"] for chunk in changed}
        for chunk in chunks:
            if chunk["id"] in changed_ids:
                continue
            # Non-text parser/provenance corrections (for example a better
            # printed-page extraction) must not require re-embedding.
            conn.execute("""
                UPDATE chunks SET
                    source_page_start = ?, source_page_end = ?, printed_page = ?,
                    license = ?, remaster = ?, source_hash = ?, section_hash = ?
                WHERE id = ? AND origin = 'corpus'
            """, (
                chunk.get("source_page_start"), chunk.get("source_page_end"),
                chunk.get("printed_page"), chunk.get("license", "NONE"),
                1 if chunk.get("remaster") else (
                    0 if chunk.get("remaster") is not None else None
                ),
                chunk.get("source_hash"), chunk.get("section_hash"), chunk["id"],
            ))

        # Metadata such as a new local raw hash may change while normalized
        # rules and embeddings remain identical.
        seen_sources: set[str] = set()
        for chunk in chunks:
            _origin, source = _source_for_chunk(chunk, settings)
            if source["source_id"] not in seen_sources:
                _upsert_source(conn, source)
                seen_sources.add(source["source_id"])
        if seen_sources:
            placeholders = ",".join("?" for _ in seen_sources)
            conn.execute(
                f"DELETE FROM sources WHERE source <> 'foundry' "
                f"AND source_id NOT IN ({placeholders})",
                tuple(sorted(seen_sources)),
            )
        else:
            conn.execute("DELETE FROM sources WHERE source <> 'foundry'")

        rebuild_fts(conn)
        corpus_count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE origin = 'corpus'"
        ).fetchone()[0]
        corpus_vectors = conn.execute(
            "SELECT COUNT(*) FROM vec_chunks "
            "WHERE id IN (SELECT id FROM chunks WHERE origin = 'corpus')"
        ).fetchone()[0]
        if corpus_count != len(chunks) or corpus_vectors != len(chunks):
            raise RuntimeError("corpus validation failed: chunk/vector counts do not match")
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("corpus validation failed: database integrity check")
        conn.execute(
            "INSERT OR REPLACE INTO _meta (key, value) VALUES (?, ?)",
            ("total_chunks", str(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])),
        )
        conn.execute(
            "INSERT OR REPLACE INTO _meta (key, value) VALUES (?, ?)",
            ("corpus_sync_date", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        )
        conn.execute(
            "INSERT OR REPLACE INTO _meta (key, value) VALUES (?, ?)",
            ("distribution_scope", "local-full"),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return {
        "active": len(chunks),
        "changed": len(changed),
        "removed": len(removed_ids),
        "unchanged": unchanged,
    }


def update_index(settings: Settings, _provider: EmbeddingProvider | None = None) -> None:
    """Incremental update: diff against indexed release, only re-process changed entries."""
    import sqlite3

    from .embeddings import get_provider as gp
    from .fetcher import extract_all_packs, get_cached_zip
    from .index import init_db, load_vec_extension, migrate_db, rebuild_fts

    _require_seed_slot(settings)

    # Ensure DB exists
    if _provider is None:
        _provider = gp(settings.model, provider=settings.provider, onnx_provider=settings.onnx_provider)
    dim = _provider.dim
    init_db(settings.db, dim)

    conn = sqlite3.connect(str(settings.db))
    load_vec_extension(conn)
    migrate_db(conn)

    # Get current indexed release
    current_release = conn.execute(
        "SELECT value FROM _meta WHERE key = 'pf2e_release'"
    ).fetchone()
    current_release = current_release[0] if current_release else None

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
    for row in conn.execute(
        "SELECT id, source_hash FROM chunks WHERE origin = 'foundry'"
    ):
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
            _delete_entry_rows(conn, base_id, origin="foundry")

            for chunk, emb in entry_chunks:
                _insert_chunk(conn, chunk, emb, settings, force_origin="foundry")

        # Remove orphaned entries (deleted from source) and their outgoing
        # references. Legacy bare-target incoming refs are retained because
        # they may belong to another pack with the same ID.
        for oid in orphan_ids:
            _delete_entry_rows(conn, oid, origin="foundry")

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
        db = settings.model_copy(update={"model": model}).db
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
            model_settings = settings.model_copy(update={"model": model})
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
        ms = settings.model_copy(update={"model": model})
        try:
            # ``pending`` contains only fresh or explicitly rebuilt complete
            # snapshots; incremental updates use _update_one instead.
            embed_and_index(chunks, ms, rebuild=True, provider=providers.get(model))
            return (model, True)
        except Exception as e:
            print(f"[FAIL] {model}: {e}")
            return (model, False)

    def _update_one(model: str) -> tuple[str, bool]:
        ms = settings.model_copy(update={"model": model})
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
