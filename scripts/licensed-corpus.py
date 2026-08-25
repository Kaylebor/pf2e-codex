#!/usr/bin/env python3
"""JSON-oriented CLI for the private licensed-core review workspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pf2e_codex.licensed_corpus import (
    activate_parser_run,
    build_public_corpus,
    claim_draft_screening_batch,
    claim_review,
    claim_shard,
    draft_screening_status,
    initialize_workspace,
    invalidate_reviews,
    next_draft_screening_record,
    read_claimed_review,
    read_claimed_shard,
    read_draft_screening_record,
    release_draft_screening_batch,
    stage_trusted_native_pdf,
    step_draft_screening,
    submit_candidate,
    submit_draft_screening_decision,
    submit_review,
    workspace_status,
)


def _json_inputs(value: str) -> list[dict[str, Any]]:
    stripped = value.lstrip()
    if stripped.startswith(("{", "[")):
        payload = json.loads(value)
    else:
        path = Path(value)
        try:
            is_file = path.is_file()
        except OSError:
            is_file = False
        if is_file and path.suffix == ".jsonl":
            payload = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            payload = json.loads(path.read_text(encoding="utf-8")) if is_file else json.loads(value)
    values = payload if isinstance(payload, list) else [payload]
    if not all(isinstance(item, dict) for item in values):
        raise ValueError("JSON input must be an object, array of objects, or JSONL objects")
    return values


def _emit(value: object) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _review_id_inputs(value: str) -> list[str]:
    """Read an exact, non-content review-ID selection from inline JSON or a file."""
    stripped = value.lstrip()
    if stripped.startswith("["):
        payload = json.loads(value)
    else:
        path = Path(value)
        try:
            is_file = path.is_file()
        except OSError:
            is_file = False
        payload = json.loads(path.read_text(encoding="utf-8")) if is_file else json.loads(value)
    if not isinstance(payload, list) or not payload or any(
        not isinstance(review_id, str) or not review_id for review_id in payload
    ):
        raise ValueError("review IDs must be a non-empty JSON array of strings")
    if len(set(payload)) != len(payload):
        raise ValueError("review IDs must not contain duplicates")
    return payload


def _select_private_record(
    records: list[dict[str, object]], index: int | None
) -> list[dict[str, object]] | dict[str, object]:
    """Limit private CLI output to one record when a worker requests it."""
    if index is None:
        return records
    if index < 0 or index >= len(records):
        raise ValueError(
            f"private record index {index} is outside 0..{max(len(records) - 1, 0)}"
        )
    return records[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="create a private review workspace")
    init.add_argument("workspace", type=Path)
    init.add_argument("source_database", type=Path)
    init.add_argument("--shard-size", type=int, default=100)

    status = commands.add_parser("status", help="show aggregate, non-content workspace state")
    status.add_argument("workspace", type=Path)

    screen_status = commands.add_parser(
        "screen-status", help="show aggregate state for the private quad-state first-pass draft"
    )
    screen_status.add_argument("workspace", type=Path)

    screen_claim = commands.add_parser(
        "screen-claim", help="claim one private first-pass batch without reading its content"
    )
    screen_claim.add_argument("workspace", type=Path)
    screen_claim.add_argument("claimant")
    screen_claim.add_argument("--product-code")
    screen_claim.add_argument("--lease-seconds", type=int, default=3600)
    screen_claim.add_argument("--shard-id", type=int, help="claim a specific available batch")
    screen_claim.add_argument(
        "--queue", choices=("unprocessed", "deferred"), default="unprocessed"
    )

    screen_read = commands.add_parser(
        "screen-read", help="read exactly one record from a claimed private first-pass batch"
    )
    screen_read.add_argument("workspace", type=Path)
    screen_read.add_argument("shard_id", type=int)
    screen_read.add_argument("claimant")
    screen_read.add_argument("index", type=int)

    screen_next = commands.add_parser(
        "screen-next", help="read the next pending record from a claimed private batch"
    )
    screen_next.add_argument("workspace", type=Path)
    screen_next.add_argument("shard_id", type=int)
    screen_next.add_argument("claimant")
    screen_next.add_argument("--after-index", type=int, default=-1)

    screen_decide = commands.add_parser(
        "screen-decide", help="idempotently add, reject, or defer one private parsed section"
    )
    screen_decide.add_argument("workspace", type=Path)
    screen_decide.add_argument("shard_id", type=int)
    screen_decide.add_argument("claimant")
    screen_decide.add_argument("index", type=int)
    screen_decide.add_argument("decision", choices=("add", "reject", "defer"))
    screen_decide.add_argument(
        "--defer-reason",
        choices=("layout", "scope", "complex-rule", "insufficient-context"),
    )

    screen_step = commands.add_parser(
        "screen-step",
        help="persist one decision and return the next eligible private record",
    )
    screen_step.add_argument("workspace", type=Path)
    screen_step.add_argument("shard_id", type=int)
    screen_step.add_argument("claimant")
    screen_step.add_argument("index", type=int)
    screen_step.add_argument("decision", choices=("add", "reject", "defer"))
    screen_step.add_argument(
        "--defer-reason",
        choices=("layout", "scope", "complex-rule", "insufficient-context"),
    )

    screen_release = commands.add_parser(
        "screen-release", help="release a private first-pass batch while preserving decisions"
    )
    screen_release.add_argument("workspace", type=Path)
    screen_release.add_argument("shard_id", type=int)
    screen_release.add_argument("claimant")

    stage = commands.add_parser(
        "stage-native-pdf",
        help="read one PDF in-process, verify native coverage, and stage a complete private run",
    )
    stage.add_argument("workspace", type=Path)
    stage.add_argument("pdf", type=Path, help="user-owned Paizo PDF; cached JSON is not accepted")
    stage.add_argument("product_code", help="known PZO product profile, e.g. PZO12001")
    stage.add_argument(
        "--parser-version",
        default="paizo-native-v1",
        choices=("paizo-native-v1", "paizo-native-v2", "paizo-native-v3"),
    )
    stage.add_argument("--shard-size", type=int, default=100)

    activate = commands.add_parser("activate-run", help="atomically switch a product to a staged parser run")
    activate.add_argument("workspace", type=Path)
    activate.add_argument("parser_run_id")

    claim = commands.add_parser(
        "claim", help="atomically claim one pristine or fully reviewed revision shard without reading it"
    )
    claim.add_argument("workspace", type=Path)
    claim.add_argument("claimant")
    claim.add_argument("--lease-seconds", type=int, default=3600)
    claim.add_argument("--shard-id", type=int, help="claim a specific available shard")

    read = commands.add_parser("read", help="explicitly read a worker's currently claimed private shard")
    read.add_argument("workspace", type=Path)
    read.add_argument("shard_id", type=int)
    read.add_argument("claimant")
    read.add_argument(
        "--index",
        type=int,
        help="emit only one zero-based section to minimize private-text exposure",
    )

    submit = commands.add_parser("submit", help="submit one candidate JSON object or JSON file")
    submit.add_argument("workspace", type=Path)
    submit.add_argument("submission")

    review = commands.add_parser("review", help="submit one independent review JSON object or file")
    review.add_argument("workspace", type=Path)
    review.add_argument("review")

    review_claim = commands.add_parser("claim-review", help="atomically claim one candidate for review")
    review_claim.add_argument("workspace", type=Path)
    review_claim.add_argument("reviewer")
    review_claim.add_argument("--lease-seconds", type=int, default=3600)

    review_read = commands.add_parser("read-review", help="explicitly read a currently claimed review")
    review_read.add_argument("workspace", type=Path)
    review_read.add_argument("candidate_id")
    review_read.add_argument("reviewer")

    invalidate = commands.add_parser(
        "invalidate-reviews",
        help="append an audit invalidation for an exact active reviewer-owned review-ID set",
    )
    invalidate.add_argument("workspace", type=Path)
    invalidate.add_argument("reviewer", help="reviewer who owns every selected review ID")
    invalidate.add_argument("invalidated_by", help="actor authorizing the invalidation")
    invalidate.add_argument("reason", help="non-empty audit reason")
    invalidate.add_argument(
        "review_ids",
        help="non-empty JSON array or JSON file containing exactly the review IDs to invalidate",
    )

    build = commands.add_parser("build", help="build a public DB from approved candidates")
    build.add_argument("workspace", type=Path)
    build.add_argument("output", type=Path)
    build.add_argument("notices", type=Path)

    args = parser.parse_args()
    if args.command == "init":
        result = initialize_workspace(args.workspace, args.source_database, shard_size=args.shard_size)
    elif args.command == "status":
        result = workspace_status(args.workspace)
    elif args.command == "screen-status":
        result = draft_screening_status(args.workspace)
    elif args.command == "screen-claim":
        result = claim_draft_screening_batch(
            args.workspace,
            args.claimant,
            product_code=args.product_code,
            lease_seconds=args.lease_seconds,
            preferred_shard_id=args.shard_id,
            queue=args.queue,
        )
    elif args.command == "screen-read":
        result = read_draft_screening_record(
            args.workspace, args.shard_id, args.claimant, args.index
        )
    elif args.command == "screen-next":
        result = next_draft_screening_record(
            args.workspace,
            args.shard_id,
            args.claimant,
            after_index=args.after_index,
        )
    elif args.command == "screen-decide":
        result = submit_draft_screening_decision(
            args.workspace,
            args.shard_id,
            args.claimant,
            args.index,
            args.decision,
            defer_reason=args.defer_reason,
        )
    elif args.command == "screen-step":
        result = step_draft_screening(
            args.workspace,
            args.shard_id,
            args.claimant,
            args.index,
            args.decision,
            defer_reason=args.defer_reason,
        )
    elif args.command == "screen-release":
        result = release_draft_screening_batch(
            args.workspace, args.shard_id, args.claimant
        )
    elif args.command == "stage-native-pdf":
        result = stage_trusted_native_pdf(
            args.workspace, args.pdf, product_code=args.product_code,
            parser_version=args.parser_version, shard_size=args.shard_size,
        )
    elif args.command == "activate-run":
        result = activate_parser_run(args.workspace, args.parser_run_id)
    elif args.command == "claim":
        result = claim_shard(
            args.workspace,
            args.claimant,
            lease_seconds=args.lease_seconds,
            preferred_shard_id=args.shard_id,
        )
    elif args.command == "read":
        result = _select_private_record(
            read_claimed_shard(args.workspace, args.shard_id, args.claimant),
            args.index,
        )
    elif args.command == "submit":
        inputs = _json_inputs(args.submission)
        values = [submit_candidate(args.workspace, item) for item in inputs]
        result = values[0] if len(values) == 1 else values
    elif args.command == "review":
        inputs = _json_inputs(args.review)
        values = [submit_review(args.workspace, item) for item in inputs]
        result = values[0] if len(values) == 1 else values
    elif args.command == "claim-review":
        result = claim_review(args.workspace, args.reviewer, lease_seconds=args.lease_seconds)
    elif args.command == "read-review":
        result = read_claimed_review(args.workspace, args.candidate_id, args.reviewer)
    elif args.command == "invalidate-reviews":
        result = invalidate_reviews(
            args.workspace,
            args.reviewer,
            _review_id_inputs(args.review_ids),
            invalidated_by=args.invalidated_by,
            reason=args.reason,
        )
    else:
        result = build_public_corpus(args.workspace, args.output, args.notices)
    _emit(result)


if __name__ == "__main__":
    main()
