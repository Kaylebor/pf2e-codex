#!/usr/bin/env python3
"""Build exact OGL/ORC notice JSON from verified public and native-text inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

_PINNED_INPUT_SHA256 = {
    "orc_license": "553f1d3b57467bad0d854bbc215d299834d000e8374db167692eca622dab89af",
    "foundry_orc": "1e67784676bb566472d857b65e66126bc8b7259039f1d5f944f1fb7f8ee882ff",
    "foundry_ogl": "a134a6a467427a1efa37026f958d53f432af78d1d4b4c4aff02f29f69a2bda31",
}
_MONSTER_NOTICE_SHA256 = "f452afd6f3260fea1c6879561f999a918355308882781a33748df0e1c1ec3661"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_text(path: Path) -> str:
    return "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()).strip()


def _monster_core_notice(export_path: Path) -> str:
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    pages = payload.get("pages")
    if payload.get("schema_version") != 1 or not isinstance(pages, list):
        raise ValueError("Monster Core notice input is not a native exporter v1 artifact")
    page = next((item for item in pages if item.get("number") == 373), None)
    if not isinstance(page, dict) or not isinstance(page.get("words"), list):
        raise ValueError("Monster Core physical page 373 is missing")

    columns: list[list[dict[str, Any]]] = [[], []]
    for word in page["words"]:
        if not isinstance(word, dict):
            continue
        top = float(word.get("top", 0))
        x0 = float(word.get("x0", 0))
        if top < 495:
            continue
        columns[0 if x0 < 310 else 1].append(word)

    rendered_columns: list[str] = []
    for words in columns:
        lines: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for word in sorted(words, key=lambda item: (float(item["top"]), float(item["x0"]))):
            top = float(word["top"])
            line_top = next((value for value in lines if abs(value - top) <= 2.0), top)
            lines[line_top].append(word)
        rendered_columns.append(
            "\n".join(
                " ".join(str(word["text"]) for word in sorted(lines[top], key=lambda item: float(item["x0"])))
                for top in sorted(lines)
            )
        )
    notice = re.sub(r"[ \t]+([,:;.])", r"\1", "\n".join(rendered_columns)).strip()
    required = (
        "ORC Notice",
        "Attribution:",
        "Pathfinder Monster Core",
        "Reserved Material:",
        "Expressly Designated Licensed Material:",
    )
    missing = [value for value in required if value not in notice]
    if missing:
        raise ValueError("reconstructed Monster Core notice is missing: " + ", ".join(missing))
    if "@" in notice:
        raise ValueError("reconstructed notice contains possible watermark PII")
    return notice


def _project_notice(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("project notice must be a JSON object")
    required = ("orc_notice", "ogl_designation", "ogl_section_15")
    values = {key: str(payload.get(key, "")).strip() for key in required}
    if any(not values[key] for key in required):
        raise ValueError("project notice is missing required OGL/ORC text")
    orc_terms = (
        "pf2e-codex",
        "Attribution:",
        "Reserved Material:",
        "Expressly Designated Licensed Material:",
    )
    if any(term not in values["orc_notice"] for term in orc_terms):
        raise ValueError("project ORC notice is missing required downstream statements")
    if "Open Game Content" not in values["ogl_designation"]:
        raise ValueError("project OGL designation must identify its Open Game Content")
    if "pf2e-codex" not in values["ogl_section_15"] or "Copyright" not in values["ogl_section_15"]:
        raise ValueError("project OGL Section 15 entry is incomplete")
    encoded = json.dumps(values).casefold()
    if any(marker in encoded for marker in (".local-corpus", "/home/", "source_sha256")):
        raise ValueError("project notice contains private local provenance")
    return values


def build_notices(
    orc_license: Path,
    foundry_orc: Path,
    foundry_ogl: Path,
    monster_export: Path,
    project_notice: Path,
    *,
    verify_pinned: bool = True,
) -> dict[str, dict[str, str]]:
    """Return the two complete notice records consumed by the public builder."""
    if verify_pinned:
        inputs = {
            "orc_license": orc_license,
            "foundry_orc": foundry_orc,
            "foundry_ogl": foundry_ogl,
        }
        mismatches = [
            key for key, path in inputs.items()
            if _sha256(path) != _PINNED_INPUT_SHA256[key]
        ]
        if mismatches:
            raise ValueError("unpinned legal-notice inputs: " + ", ".join(mismatches))
    orc_terms = _normalized_text(orc_license)
    orc_attribution = _normalized_text(foundry_orc)
    ogl = _normalized_text(foundry_ogl)
    if "ORC LICENSE FINAL" not in orc_terms:
        raise ValueError("official ORC license text is missing its title")
    if "1. ORC NOTICE" not in orc_attribution:
        raise ValueError("Foundry ORC notice is invalid")
    if "OPEN GAME LICENSE Version 1.0a" not in ogl or "15. COPYRIGHT NOTICE" not in ogl:
        raise ValueError("Foundry Open Game License notice is invalid")
    monster_notice = _monster_core_notice(monster_export)
    if verify_pinned and hashlib.sha256(monster_notice.encode("utf-8")).hexdigest() != _MONSTER_NOTICE_SHA256:
        raise ValueError("reconstructed Monster Core notice does not match the reviewed text")
    project = _project_notice(project_notice)
    orc = "\n\n".join(
        (
            orc_terms,
            orc_attribution,
            "Pathfinder Monster Core ORC notice:\n" + monster_notice,
            "pf2e-codex downstream ORC notice:\n" + project["orc_notice"],
        )
    )
    ogl = "\n\n".join(
        (
            ogl,
            "pf2e-codex Open Game Content designation:\n" + project["ogl_designation"],
            "pf2e-codex Section 15 addition:\n" + project["ogl_section_15"],
        )
    )
    return {
        "OGL": {"license": "OGL", "text": ogl},
        "ORC": {"license": "ORC", "text": orc},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orc-license", type=Path, required=True)
    parser.add_argument("--foundry-orc", type=Path, required=True)
    parser.add_argument("--foundry-ogl", type=Path, required=True)
    parser.add_argument("--monster-export", type=Path, required=True)
    parser.add_argument("--project-notice", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    notices = build_notices(
        args.orc_license,
        args.foundry_orc,
        args.foundry_ogl,
        args.monster_export,
        args.project_notice,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(notices, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: {"chars": len(record["text"])} for key, record in notices.items()},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
