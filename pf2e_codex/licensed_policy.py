"""Versioned selection policy for the reviewed licensed-core projection."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Any

LICENSED_CORE_POLICY_VERSION = "mechanics-v1"
LICENSED_CORE_POLICY_RESOURCE = "data/licensed_core_policy_v1.json"


def load_licensed_policy(path: Path | str | None = None) -> dict[str, Any]:
    """Load and minimally validate the exact tracked review policy."""
    if path is None:
        resource = resources.files("pf2e_codex").joinpath(LICENSED_CORE_POLICY_RESOURCE)
        raw = resource.read_text(encoding="utf-8")
    else:
        raw = Path(path).expanduser().resolve().read_text(encoding="utf-8")
    policy = json.loads(raw)
    if not isinstance(policy, dict):
        raise ValueError("licensed-core policy must be a JSON object")
    if policy.get("policy_version") != LICENSED_CORE_POLICY_VERSION:
        raise ValueError("licensed-core policy has an unsupported version")
    for key in ("scope", "public_decisions", "nonpublic_decisions", "required_review"):
        if not policy.get(key):
            raise ValueError(f"licensed-core policy is missing {key}")
    return policy


def licensed_policy_digest(policy: dict[str, Any] | None = None) -> str:
    """Return a stable digest of the policy that governed public approvals."""
    value = policy if policy is not None else load_licensed_policy()
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "LICENSED_CORE_POLICY_RESOURCE",
    "LICENSED_CORE_POLICY_VERSION",
    "licensed_policy_digest",
    "load_licensed_policy",
]
