#!/usr/bin/env python3
"""Merge subagent-generated training triplets into a clean dataset.

Usage:
  uv run scripts/merge-training-data.py [--input-dir training_data/raw] [--output training_data/dataset.jsonl]

Schema per JSONL line:
  {"query": "...", "pos": "...", "neg": "..."}

- Valid lines: query, pos, neg are non-empty strings.
- Invalid lines: flagged to a `.errors` file alongside the raw source.
- Fully valid files: deleted after merge (data is in dataset.jsonl).
- Partially invalid files: kept for manual review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def validate_line(line: dict, line_num: int) -> list[str]:
    """Return a list of validation errors (empty = valid)."""
    errors = []
    for field in ("query", "pos", "neg"):
        val = line.get(field)
        if not isinstance(val, str) or not val.strip():
            errors.append(f"  line {line_num}: '{field}' missing or empty")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge subagent training data")
    parser.add_argument("--input-dir", default="training_data/raw", help="Raw JSONL input directory")
    parser.add_argument("--output", default="training_data/dataset.jsonl", help="Merged output path")
    args = parser.parse_args()

    raw_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_dir.is_dir():
        print(f"No input directory: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    raw_files = sorted(raw_dir.glob("*.jsonl"))
    if not raw_files:
        print(f"No .jsonl files in {raw_dir}")
        return

    total_valid = 0
    total_invalid = 0

    for fpath in raw_files:
        lines = fpath.read_text().strip().splitlines()
        valid_lines: list[str] = []
        error_lines: list[str] = []
        has_errors = False

        for i, line_text in enumerate(lines, start=1):
            line_text = line_text.strip()
            if not line_text:
                continue
            try:
                data = json.loads(line_text)
            except json.JSONDecodeError as e:
                error_lines.append(f"  line {i}: JSON parse error: {e}")
                has_errors = True
                continue

            errors = validate_line(data, i)
            if errors:
                error_lines.extend(errors)
                has_errors = True
            else:
                valid_lines.append(json.dumps(data, sort_keys=True))

        # Write valid lines to dataset
        if valid_lines:
            with open(output_path, "a") as out:
                for line in valid_lines:
                    out.write(line + "\n")
            total_valid += len(valid_lines)

        # Handle errors
        if has_errors:
            total_invalid += len(lines) - len(valid_lines)
            # Write error report alongside the source file
            error_path = fpath.with_suffix(".errors")
            error_path.write_text(
                f"{len(valid_lines)} valid, {len(lines) - len(valid_lines)} invalid\n"
                + "\n".join(error_lines) + "\n"
            )
            print(f"⚠  {fpath.name}: {len(valid_lines)} valid, {len(lines) - len(valid_lines)} invalid (errors written to {error_path.name})")
        else:
            # Fully valid: delete raw file
            fpath.unlink()
            print(f"✓  {fpath.name}: {len(valid_lines)} valid — deleted")

    print(f"\nDone. {total_valid} total entries in {output_path}")
    if total_invalid:
        print(f"{total_invalid} entries flagged for review in {raw_dir}/*.errors")
    else:
        print("No errors. All raw files consumed.")


if __name__ == "__main__":
    main()
