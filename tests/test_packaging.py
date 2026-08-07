import sqlite3
import tomllib
import urllib.request
from pathlib import Path

from typer.testing import CliRunner

from pf2e_codex import pipeline
from pf2e_codex.cli import app

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_arch_package_bundles_corpus_extra_for_isolated_launcher():
    pkgbuild = (REPO_ROOT / "PKGBUILD").read_text()

    assert "project['optional-dependencies']['corpus']" in pkgbuild
    assert "exec /usr/bin/python3 -S -m pf2e_codex.cli" in pkgbuild


def test_core_dependencies_do_not_force_cpu_onnxruntime():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]

    assert not any(
        dependency.split("[", 1)[0].split(">", 1)[0] == "onnxruntime"
        for dependency in project["dependencies"]
    )
    assert "onnxruntime>=1.20" in project["optional-dependencies"]["cpu"]


def test_amd_dev_setup_installs_only_official_migraphx_runtime():
    makefile = (REPO_ROOT / "Makefile").read_text()

    assert "$(UV) sync --group dev --extra corpus" in makefile
    assert "pip install --no-deps 'onnxruntime-migraphx>=1.25'" in makefile
    assert "repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1" in makefile
    assert "MIGraphXExecutionProvider" in makefile


def test_release_script_requires_redistributable_audit():
    script = (REPO_ROOT / "scripts" / "release-dbs.sh").read_text()

    assert "--corpus-scope redistributable" in script
    assert 'audit-db "$db_path" --strict' in script
    assert '--expected-release "$PF2E_RELEASE"' in script
    assert '--expected-model "$model"' in script
    assert '--data-dir "$RELEASE_DIR"' in script
    assert '"$PF2E_CODEX_BIN" embed' in script
    assert '"$PF2E_CODEX_BIN" audit-db' in script
    assert "mv ~/.local/share" not in script
    assert "--latest" not in script


def test_embed_command_exits_nonzero_when_any_model_fails(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "embed_all_models",
        lambda *_args, **_kwargs: {"test-model": False},
    )

    result = CliRunner().invoke(app, ["embed", "--models", "test-model"])

    assert result.exit_code == 1
    assert "Embedding failed for: test-model" in result.output


def test_pull_rejects_artifact_for_wrong_release_and_leaves_no_database(
    tmp_path: Path, monkeypatch,
):
    source = tmp_path / "source.db"
    conn = sqlite3.connect(source)
    conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, origin TEXT)")
    conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO chunks VALUES ('foundry:one', 'foundry')")
    conn.executemany(
        "INSERT INTO _meta VALUES (?, ?)",
        [
            ("distribution_scope", "redistributable"),
            ("pf2e_release", "pf2e-old"),
            ("embedding_model", "test-model"),
        ],
    )
    conn.commit()
    conn.close()

    def fake_download(_url: str, target: str | Path):
        Path(target).write_bytes(source.read_bytes())
        return str(target), None

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_download)
    result = CliRunner().invoke(
        app,
        [
            "pull",
            "--model",
            "test-model",
            "--release",
            "pf2e-8.4.0",
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code == 1
    assert "database release" in result.output
    assert not (tmp_path / "data" / "pf2e_test-model.db").exists()


def test_pull_replaces_stale_existing_artifact_atomically(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    destination = data_dir / "pf2e_test-model.db"

    def write_database(path: Path, release: str) -> None:
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, origin TEXT)")
        conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO chunks VALUES ('foundry:one', 'foundry')")
        conn.executemany(
            "INSERT INTO _meta VALUES (?, ?)",
            [
                ("distribution_scope", "redistributable"),
                ("pf2e_release", release),
                ("embedding_model", "test-model"),
            ],
        )
        conn.commit()
        conn.close()

    write_database(destination, "pf2e-old")
    replacement = tmp_path / "replacement.db"
    write_database(replacement, "pf2e-8.4.0")

    def fake_download(_url: str, target: str | Path):
        Path(target).write_bytes(replacement.read_bytes())
        return str(target), None

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_download)
    result = CliRunner().invoke(
        app,
        [
            "pull",
            "--model",
            "test-model",
            "--release",
            "pf2e-8.4.0",
            "--data-dir",
            str(data_dir),
        ],
    )

    assert result.exit_code == 0
    conn = sqlite3.connect(destination)
    assert conn.execute(
        "SELECT value FROM _meta WHERE key = 'pf2e_release'"
    ).fetchone() == ("pf2e-8.4.0",)
    conn.close()
