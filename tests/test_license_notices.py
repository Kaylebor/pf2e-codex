"""Legal-notice assembly tests using synthetic native-export geometry."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _notice_module():
    script = Path(__file__).parents[1] / "scripts" / "build-licensed-notices.py"
    spec = importlib.util.spec_from_file_location("build_licensed_notices", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _word(text: str, x0: float, top: float) -> dict[str, object]:
    return {"text": text, "x0": x0, "top": top}


def _monster_export(path: Path, *, watermark: bool = False) -> Path:
    words = [
        _word("ORC", 40, 500),
        _word("Notice", 75, 500),
        _word("Attribution:", 40, 515),
        _word("Pathfinder", 40, 530),
        _word("Monster", 100, 530),
        _word("Core", 160, 530),
        _word("Reserved", 330, 500),
        _word("Material:", 390, 500),
        _word("Expressly", 330, 515),
        _word("Designated", 390, 515),
        _word("Licensed", 465, 515),
        _word("Material:", 525, 515),
    ]
    if watermark:
        words.append(_word("reader@example.com", 330, 530))
    path.write_text(json.dumps({"schema_version": 1, "pages": [{"number": 373, "words": words}]}))
    return path


def _project_notice(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "orc_notice": (
                    "pf2e-codex ORC Notice\nAttribution: Example.\n"
                    "Reserved Material: Example.\n"
                    "Expressly Designated Licensed Material: Example."
                ),
                "ogl_designation": "The functional rules are Open Game Content.",
                "ogl_section_15": "pf2e-codex, Copyright 2026 Example.",
            }
        )
    )
    return path


def test_build_notices_combines_verified_inputs_without_private_provenance(tmp_path: Path):
    module = _notice_module()
    orc = tmp_path / "orc.txt"
    foundry_orc = tmp_path / "ORCLicense.md"
    foundry_ogl = tmp_path / "OpenGameLicense.md"
    orc.write_text("ORC LICENSE FINAL\nTerms")
    foundry_orc.write_text("1. ORC NOTICE\nPathfinder Player Core")
    foundry_ogl.write_text("OPEN GAME LICENSE Version 1.0a\n15. COPYRIGHT NOTICE")

    notices = module.build_notices(
        orc,
        foundry_orc,
        foundry_ogl,
        _monster_export(tmp_path / "monster.json"),
        _project_notice(tmp_path / "project.json"),
        verify_pinned=False,
    )

    assert set(notices) == {"OGL", "ORC"}
    assert "Pathfinder Monster Core" in notices["ORC"]["text"]
    assert ".local-corpus" not in json.dumps(notices)


def test_monster_notice_rejects_watermark_like_email(tmp_path: Path):
    module = _notice_module()

    with pytest.raises(ValueError, match="watermark PII"):
        module._monster_core_notice(_monster_export(tmp_path / "monster.json", watermark=True))
