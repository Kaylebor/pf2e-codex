#!/usr/bin/env python3
"""Deterministic supervisor for the private licensed-core review workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pf2e_codex.licensed_coverage import build_foundry_evidence_database  # noqa: E402
from pf2e_codex.pdf_layout import DEFAULT_LAYOUT_MODEL_DIR  # noqa: E402
from pf2e_codex.review_runner import (  # noqa: E402
    build_base,
    compare_workspace_quality,
    maintainer_item_evidence,
    prepare_review_data,
    prepare_workspace,
    preview_screen_batches,
    promote_base,
    quality_workspace,
    reopen_screening,
    resolve_maintainer_item,
    run_queues,
    runner_status,
    set_review_product_scope,
    verify_workspace,
)


def _run_selection(values: list[str]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for value in values:
        product, separator, run_id = value.partition("=")
        if not separator or not product.startswith("PZO") or not run_id or product in selected:
            raise argparse.ArgumentTypeError(
                "parser runs must be unique PRODUCT=RUN selections"
            )
        selected[product] = run_id
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="stage and activate a fresh trusted five-PDF workspace")
    prepare.add_argument("workspace", type=Path)
    prepare.add_argument("sources", type=Path)
    prepare.add_argument("--shard-size", type=int, default=32)
    prepare.add_argument("--layout-model", type=Path, default=DEFAULT_LAYOUT_MODEL_DIR)
    prepare.add_argument(
        "--layout-provider",
        choices=("auto", "migraphx", "rocm", "cuda", "cpu"),
        default="auto",
    )

    foundry = commands.add_parser(
        "prepare-foundry", help="build a vector-free strictly filtered Foundry evidence database"
    )
    foundry.add_argument("archive", type=Path)
    foundry.add_argument("output", type=Path)
    foundry.add_argument("--release", required=True)

    status = commands.add_parser("status", help="show content-free queue and session counts")
    status.add_argument("workspace", type=Path)

    scope = commands.add_parser(
        "set-scope", help="persist the products enabled for semantic review"
    )
    scope.add_argument("workspace", type=Path)
    scope.add_argument(
        "--include", action="append", required=True, metavar="PZO_CODE",
        help="enable one active product; repeat for every enabled product",
    )
    scope.add_argument(
        "--held-reason", choices=("legacy-study", "maintainer-hold"),
        default="maintainer-hold",
    )

    review_data = commands.add_parser(
        "prepare-review",
        help="prepare deterministic duplicates and Foundry evidence without workers",
    )
    review_data.add_argument("workspace", type=Path)
    review_data.add_argument("--foundry-database", type=Path, required=True)

    preview = commands.add_parser(
        "preview", help="read-only preview of exact worker batch envelopes"
    )
    preview.add_argument("workspace", type=Path)
    preview.add_argument("--queue", choices=("screen",), default="screen")
    preview.add_argument("--foundry-database", type=Path, required=True)

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
    verify.add_argument("--foundry-database", type=Path)

    quality = commands.add_parser("quality", help="report content-free parser quality metrics")
    quality.add_argument("workspace", type=Path)
    quality.add_argument(
        "--run", action="append", default=[], metavar="PRODUCT=RUN",
        help="audit an explicit parser run instead of the active run; repeat per product",
    )

    compare = commands.add_parser(
        "compare-quality", help="compare complete baseline and candidate parser-run selections"
    )
    compare.add_argument("workspace", type=Path)
    compare.add_argument("--baseline-run", action="append", required=True, metavar="PRODUCT=RUN")
    compare.add_argument("--candidate-run", action="append", required=True, metavar="PRODUCT=RUN")

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

    reopen = commands.add_parser(
        "reopen-screening",
        help="append an explicit maintainer screening reopen event",
    )
    reopen.add_argument("workspace", type=Path)
    reopen.add_argument("section_key")
    reopen.add_argument(
        "--reason",
        choices=("parser-quality", "scope-correction", "maintainer-review"),
        required=True,
    )
    reopen.add_argument("--maintainer", default="maintainer")

    build = commands.add_parser("build-base", help="build an ignored audited model-independent base")
    build.add_argument("workspace", type=Path)
    build.add_argument("output", type=Path)
    build.add_argument("notices", type=Path)
    build.add_argument("--foundry-database", type=Path, required=True)

    promote = commands.add_parser("promote-base", help="explicitly replace the tracked projection")
    promote.add_argument("staged", type=Path)
    promote.add_argument("tracked", type=Path)

    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_workspace(
            args.workspace, args.sources,
            shard_size=args.shard_size, layout_model_dir=args.layout_model,
            layout_provider=args.layout_provider,
        )
    elif args.command == "prepare-foundry":
        result = build_foundry_evidence_database(
            args.archive, args.output, release=args.release
        )
    elif args.command == "status":
        result = runner_status(args.workspace)
    elif args.command == "set-scope":
        result = set_review_product_scope(
            args.workspace, args.include, held_reason=args.held_reason
        )
    elif args.command == "prepare-review":
        result = prepare_review_data(args.workspace, args.foundry_database)
    elif args.command == "preview":
        result = preview_screen_batches(args.workspace, args.foundry_database)
    elif args.command == "run":
        result = run_queues(
            args.workspace, queue=args.queue, concurrency=args.concurrency,
            foundry_database=args.foundry_database, sources_root=args.sources,
            pilot=args.pilot,
        )
    elif args.command == "verify":
        result = verify_workspace(
            args.workspace, require_complete=args.complete,
            foundry_database=args.foundry_database,
        )
    elif args.command == "quality":
        selection = _run_selection(args.run)
        result = quality_workspace(
            args.workspace, parser_run_ids=selection or None
        )
    elif args.command == "compare-quality":
        result = compare_workspace_quality(
            args.workspace,
            baseline_parser_run_ids=_run_selection(args.baseline_run),
            candidate_parser_run_ids=_run_selection(args.candidate_run),
        )
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
    elif args.command == "reopen-screening":
        result = reopen_screening(
            args.workspace,
            args.section_key,
            reason=args.reason,
            maintainer=args.maintainer,
        )
    elif args.command == "build-base":
        result = build_base(
            args.workspace, args.output, args.notices, args.foundry_database
        )
    else:
        result = promote_base(args.staged, args.tracked)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    if args.command == "verify" and not result["ok"]:
        raise SystemExit(1)
    if args.command == "compare-quality" and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
