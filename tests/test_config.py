"""Configuration coverage for optional local corpus discovery."""

import pytest
from pydantic import ValidationError

from pf2e_codex.config import CorpusScope, DatabaseScope, Settings, get_settings


def test_explicit_corpus_dir_is_resolved(tmp_path):
    settings = Settings(corpus_dir=tmp_path)

    assert settings.effective_corpus_dir == tmp_path.resolve()


def test_local_corpus_is_discovered_only_when_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    assert settings.effective_corpus_dir is None

    (tmp_path / ".local-corpus").mkdir()
    assert settings.effective_corpus_dir == (tmp_path / ".local-corpus").resolve()


def test_corpus_selection_defaults_are_not_shared():
    first = Settings()
    second = Settings()

    first.corpus_include.append("PZO12001")
    first.corpus_prefer["PZO12001"] = "new.pdf"

    assert second.corpus_include == []
    assert second.corpus_prefer == {}


def test_corpus_scope_defaults_to_redistributable_and_validates_values():
    assert Settings().corpus_scope is CorpusScope.REDISTRIBUTABLE
    assert Settings(corpus_scope="local-full").corpus_scope is CorpusScope.LOCAL_FULL

    with pytest.raises(ValidationError):
        Settings(corpus_scope="maybe")


def test_database_scope_auto_prefers_private_slot(tmp_path):
    settings = Settings(data_dir=tmp_path, model="org/model")

    assert settings.clean_db == tmp_path / "pf2e_org--model.db"
    assert settings.local_db == tmp_path / "pf2e_org--model.local.db"
    assert settings.resolved_database_scope is DatabaseScope.CLEAN
    assert settings.db == settings.clean_db

    settings.local_db.touch()
    assert settings.resolved_database_scope is DatabaseScope.LOCAL
    assert settings.db == settings.local_db
    assert settings.model_copy(
        update={"database_scope": DatabaseScope.CLEAN}
    ).db == settings.clean_db
    local_seed = Settings(
        data_dir=tmp_path,
        model="other/model",
        corpus_scope=CorpusScope.LOCAL_FULL,
    )
    assert local_seed.db == local_seed.local_db


def test_structured_environment_settings_are_json_decoded(monkeypatch):
    monkeypatch.setenv("PF2E_CORPUS_INCLUDE", '["PZO12001", "PZO12002"]')
    monkeypatch.setenv("PF2E_CORPUS_PREFER", '{"PZO12001": "new.pdf"}')
    monkeypatch.setenv("PF2E_LANGUAGES", '["en", "es"]')

    settings = get_settings()

    assert settings.corpus_include == ["PZO12001", "PZO12002"]
    assert settings.corpus_prefer == {"PZO12001": "new.pdf"}
    assert settings.languages == ["en", "es"]
