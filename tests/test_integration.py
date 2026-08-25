"""Integration tests — search, get, related against local DBs.

Tests are opt-in because they load real ONNX models and query an existing database.
Run them through ``make test-integration``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

DATA_DIR = Path(os.environ.get("PF2E_DATA_DIR", Path.home() / ".local" / "share" / "pf2e-codex"))
DEFAULT_MODEL = "Snowflake/snowflake-arctic-embed-xs"
MODEL_SAFE = DEFAULT_MODEL.replace("/", "--")
DB_PATH = DATA_DIR / f"pf2e_{MODEL_SAFE}.db"
RUN_INTEGRATION = os.environ.get("PF2E_RUN_INTEGRATION") == "1"


def _search() -> object | None:
    """Return a SearchIndex if the DB + onnxruntime are available, else None."""
    if not DB_PATH.exists():
        return None
    try:
        import onnxruntime
        onnxruntime.get_available_providers()  # verify it works
    except Exception:
        return None
    from pf2e_codex.config import Settings
    from pf2e_codex.index import SearchIndex
    from pf2e_codex.model_manager import ModelManager
    s = Settings(model=DEFAULT_MODEL, data_dir=str(DATA_DIR))
    manager = ModelManager(s.model, s.reranker_model, onnx_provider=s.query_provider)
    manager.start()
    return SearchIndex(s.db, manager)


needs_db = pytest.mark.skipif(
    not RUN_INTEGRATION or not DB_PATH.exists(),
    reason="set PF2E_RUN_INTEGRATION=1 and provide an existing database",
)


@pytest.fixture(scope="module")
def search_index():
    """Load the production inference stack once for this opt-in test module."""
    index = _search()
    if index is None:
        pytest.skip("onnxruntime is unavailable")
    try:
        yield index
    finally:
        index.close()


class TestSearch:
    @needs_db
    def test_search_returns_results(self, search_index):
        results = search_index.search("fireball", top_k=3)
        assert len(results) > 0
        assert any("fireball" in r["name"].lower() for r in results)

    @needs_db
    def test_search_license_filter(self, search_index):
        results = search_index.search("fireball", top_k=5, license="ORC")
        for r in results:
            assert r.get("license") == "ORC"

    @needs_db
    def test_search_remaster_filter(self, search_index):
        results = search_index.search("fireball", top_k=5, remaster=True)
        for r in results:
            assert r.get("remaster") is True

    @needs_db
    def test_search_content_type_filter(self, search_index):
        results = search_index.search("fireball", top_k=5, content_type="spell")
        for r in results:
            assert r.get("type") == "spell"


class TestGetEntry:
    @needs_db
    def test_get_by_slug(self, search_index):
        result = search_index.fetch_by_id("off-guard")
        assert result is not None
        assert "Off-Guard" in result.get("name", "")

    @needs_db
    def test_get_returns_rules(self, search_index):
        result = search_index.fetch_by_id("fury-instinct")
        if result:
            assert "name" in result
            assert "text" in result

    @needs_db
    def test_get_nonexistent_returns_none(self, search_index):
        result = search_index.fetch_by_id("nonexistent-entry-12345")
        assert result is None


class TestRelated:
    @needs_db
    def test_related_returns_refs(self, search_index):
        results = search_index.related("off-guard", direction="incoming")
        assert isinstance(results, dict)
        assert "incoming" in results
        assert len(results["incoming"]) > 0

    @needs_db
    def test_related_bidirectional(self, search_index):
        outgoing = search_index.related("off-guard", direction="outgoing")
        incoming = search_index.related("off-guard", direction="incoming")
        combined = search_index.related("off-guard", direction="both")
        assert len(combined["incoming"]) >= len(incoming["incoming"])
        assert len(combined["outgoing"]) >= len(outgoing["outgoing"])


class TestCatalog:
    @needs_db
    def test_catalog_types(self, search_index):
        cat = search_index.catalog()
        assert "types" in cat
        assert len(cat["types"]) > 0

    @needs_db
    def test_catalog_packs(self, search_index):
        cat = search_index.catalog()
        assert "packs" in cat
        assert len(cat["packs"]) > 0

    @needs_db
    def test_catalog_contains_total_chunks(self, search_index):
        cat = search_index.catalog()
        assert "total_chunks" in cat
        assert int(cat["total_chunks"]) > 0
