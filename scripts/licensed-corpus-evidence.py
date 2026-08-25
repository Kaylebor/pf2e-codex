#!/usr/bin/env python3
"""Read-only evidence entry point for deterministic licensed-corpus workers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pf2e_codex.review_evidence import main  # noqa: E402

if __name__ == "__main__":
    main()
