#!/usr/bin/env python3
"""Deterministic supervisor for the private licensed-core review workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pf2e_codex.pdf_layout import DEFAULT_LAYOUT_MODEL_DIR  # noqa: E402
from pf2e_codex.review_runner import (  # noqa: E402
    build_base,
    maintainer_item_evidence,
    prepare_workspace,
    promote_base,
    resolve_maintainer_item,
    run_queues,
    runner_status,
    verify_workspace,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="stage and activate a fresh trusted five-PDF workspace")
    prepare.add_argument("workspace", type=Path)
    prepare.add_argument("sources", type=Path)
    prepare.add_argument("--parser-version", default="paizo-native-v3", choices=("paizo-native-v3",))
    prepare.add_argument("--shard-size", type=int, default=32)
    prepare.add_argument("--layout-model", type=Path, default=DEFAULT_LAYOUT_MODEL_DIR)
    prepare.add_argument(
        "--layout-provider",
        choices=("auto", "migraphx", "rocm", "cuda", "cpu"),
        default="auto",
    )

    status = commands.add_parser("status", help="show content-free queue and session counts")
    status.add_argument("workspace", type=Path)

    run = commands.add_parser("run", help="drain all queues or one selected queue")
    run.add_argument("workspace", type=Path)
    run.add_argument("--queue", default="all")
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--foundry-database", type=Path)
    run.add_argument("--sources", type=Path)
    run.add_argument(
        "--pilot", action="store_true",
        help="run at most one selected-queue batch per product; screening waits for layout",
    )

    verify = commands.add_parser("verify", help="validate parser, workflow, privacy, and readiness")
    verify.add_argument("workspace", type=Path)
    verify.add_argument("--complete", action="store_true")

    resolve = commands.add_parser(
        "resolve-maintainer",
        help="explicitly resolve one independent stitch disagreement",
    )
    resolve.add_argument("workspace", type=Path)
    resolve.add_argument("maintenance_id")
    resolve.add_argument("decision", choices=("merge", "no-merge"))

    inspect = commands.add_parser(
        "inspect-maintainer",
        help="show bounded evidence for one maintainer item",
    )
    inspect.add_argument("workspace", type=Path)
    inspect.add_argument("maintenance_id")
    inspect.add_argument(
        "--include-text", action="store_true",
        help="print private source text for the item's two or three sections",
    )

    build = commands.add_parser("build-base", help="build an ignored audited model-independent base")
    build.add_argument("workspace", type=Path)
    build.add_argument("output", type=Path)
    build.add_argument("notices", type=Path)

    promote = commands.add_parser("promote-base", help="explicitly replace the tracked projection")
    promote.add_argument("staged", type=Path)
    promote.add_argument("tracked", type=Path)

    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_workspace(
            args.workspace, args.sources, parser_version=args.parser_version,
            shard_size=args.shard_size, layout_model_dir=args.layout_model,
            layout_provider=args.layout_provider,
        )
    elif args.command == "status":
        result = runner_status(args.workspace)
    elif args.command == "run":
        result = run_queues(
            args.workspace, queue=args.queue, concurrency=args.concurrency,
            foundry_database=args.foundry_database, sources_root=args.sources,
            pilot=args.pilot,
        )
    elif args.command == "verify":
        result = verify_workspace(args.workspace, require_complete=args.complete)
    elif args.command == "resolve-maintainer":
        result = resolve_maintainer_item(
            args.workspace,
            args.maintenance_id,
            args.decision,
        )
    elif args.command == "inspect-maintainer":
        result = maintainer_item_evidence(
            args.workspace,
            args.maintenance_id,
            include_text=args.include_text,
        )
    elif args.command == "build-base":
        result = build_base(args.workspace, args.output, args.notices)
    else:
        result = promote_base(args.staged, args.tracked)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    if args.command == "verify" and not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
