"""PF2E Codex — Pathfinder 2E rules knowledge base with MCP, CLI, and SDK interfaces."""

from __future__ import annotations

# Pre-load bundled protobuf 34 before onnxruntime imports (see _preload_onnx.py)
from ._preload_onnx import _preloaded  # noqa: F401

__version__ = "0.1.0"
