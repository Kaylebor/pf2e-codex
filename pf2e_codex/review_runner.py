"""Deterministic supervisor for the private licensed-core review workflow.

Scheduling, leases, evidence preparation, exact-ID validation, persistence,
and state transitions live here. Codex processes receive bounded private text
and return semantic judgments only; they never write the workspace directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urljoin, urlparse
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from .corpus import (
    PAIZO_NATIVE_PARSER_V4,
    PAIZO_NATIVE_PARSER_V5,
    PRODUCT_CATALOG,
    CorpusSource,
    discover_sources,
    select_revisions,
)
from .corpus_quality import (
    audit_workspace,
    compare_quality,
    compare_repair_quality,
    validate_quality,
)
from .licensed_core import licensed_core_digest, load_licensed_core
from .licensed_corpus import (
    REVIEW_SCHEMA_VERSION,
    REVIEW_SCOPE_VERSION,
    _ensure_workspace_migrated,
    _foundry_coverage_evidence_for_snapshot,
    _release_shard_if_complete,
    _review_scope_rows,
    _semantic_scope_sql,
    _unauthorized_foundry_confirmation_count,
    activate_parser_run,
    build_public_corpus,
    claim_draft_screening_batch,
    claim_review,
    claim_shard,
    draft_screening_status,
    foundry_coverage_evidence,
    initialize_trusted_workspace,
    prepare_deterministic_review,
    read_claimed_review,
    read_claimed_shard,
    reclaim_interrupted_shard,
    release_draft_screening_batch,
    reopen_draft_screening,
    review_product_scope,
    set_review_product_scope,
    stage_trusted_native_pdf,
    stage_trusted_native_pdf_with_approved_stitches,
    submit_candidate,
    submit_draft_screening_decision,
    submit_review,
    workspace_status,
)
from .licensed_coverage import NORMALIZER_VERSION, load_clean_foundry
from .licensed_policy import LICENSED_CORE_POLICY_VERSION, licensed_policy_digest

RUNNER_VERSION = "licensed-corpus-runner-v4"
PROMPT_VERSION = "licensed-review-v3"
LOCAL_QWEN_QUEUE_MODEL = "local-qwen-gate"
EXPECTED_PRODUCTS = ("PZO2101", "PZO12001", "PZO12002", "PZO12003", "PZO12004")
MODEL_BY_QUEUE = {
    "stitch-select": "gpt-5.6-luna",
    "stitch-confirm": "gpt-5.6-terra",
    "screen": LOCAL_QWEN_QUEUE_MODEL,
    "coverage-confirm": "gpt-5.6-sol",
    "classify": "gpt-5.6-luna",
    "extract": "gpt-5.6-terra",
    "review": "gpt-5.6-luna",
    "review-mixed": "gpt-5.6-terra",
    "rework-terra": "gpt-5.6-terra",
    "rework-sol": "gpt-5.6-sol",
}
MODEL_CAPS = {
    LOCAL_QWEN_QUEUE_MODEL: 1,
    "gpt-5.6-luna": 2,
    "gpt-5.6-terra": 2,
    "gpt-5.6-sol": 1,
}
MAX_BATCH_RECORDS = 32
MAX_BATCH_BYTES = 64 * 1024
MAX_LOCAL_RESPONSE_BYTES = 1024 * 1024
MAX_RETRIES = 3
SESSION_BATCH_LIMIT = 4
SESSION_EVIDENCE_LIMIT = 256 * 1024
LEASE_SECONDS = 3600
STITCH_HEURISTIC_VERSION = "adjacent-fragment-v4"
_AON_ROOT = "https://2e.aonprd.com/"
_AON_HOSTS = {"2e.aonprd.com", "aonprd.com", "www.aonprd.com"}
_TERM_RE = re.compile(r"[a-z0-9]{2,}")
_APPROVED_STITCH_PREDICATE = """
    EXISTS (SELECT 1 FROM stitch_votes AS selector
            WHERE selector.candidate_id=c.candidate_id
              AND selector.role='selector' AND selector.decision='merge')
    AND (
      EXISTS (SELECT 1 FROM stitch_votes AS confirmer
              WHERE confirmer.candidate_id=c.candidate_id
                AND confirmer.role='confirmer' AND confirmer.decision='merge')
      OR EXISTS (SELECT 1 FROM runner_maintenance AS maintenance
                 WHERE maintenance.queue_name='stitch'
                   AND maintenance.subject_id=c.candidate_id
                   AND maintenance.reason='independent-disagreement'
                   AND maintenance.resolved_at IS NOT NULL
                   AND maintenance.resolution='merge')
    )
"""


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(*values: object) -> str:
    return hashlib.sha256("\n".join(str(value) for value in values).encode("utf-8")).hexdigest()


@contextmanager
def _connect(path: Path | str, *, readonly: bool = False) -> Iterable[sqlite3.Connection]:
    resolved = Path(path).expanduser().resolve()
    if readonly:
        conn = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True, timeout=30)
        conn.execute("PRAGMA query_only=ON")
    else:
        conn = sqlite3.connect(resolved, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
    finally:
        conn.close()


def _object_schema(item_properties: dict[str, object], required: Sequence[str]) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": item_properties,
                    "required": list(required),
                },
            }
        },
        "required": ["results"],
    }


SCHEMAS: dict[str, dict[str, object]] = {
    "coverage-gate": _object_schema(
        {
            "id": {"type": "string"},
            "input_status": {
                "type": "string",
                "enum": ["valid", "needs-layout", "insufficient-context"],
            },
            "coverage": {
                "type": "string",
                "enum": ["covered", "additional-mechanics", "uncertain", "not-applicable"],
            },
            "issue_tags": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "heading-body-mismatch",
                        "adjacent-contamination",
                        "dangling-fragment",
                        "page-number-stub",
                        "table-order",
                        "missing-context",
                        "numeric-role-mismatch",
                        "condition-rank-mismatch",
                        "partial-mechanics",
                        "other-structural",
                    ],
                },
                "maxItems": 6,
                "uniqueItems": True,
            },
            "foundry_ids": {
                "type": "array", "items": {"type": "string"},
                "maxItems": 3, "uniqueItems": True,
            },
        },
        ("id", "input_status", "coverage", "issue_tags", "foundry_ids"),
    ),
    "classify": _object_schema(
        {
            "id": {"type": "string"},
            "decision": {"type": "string", "enum": ["PUBLIC_AS_IS", "MIXED_NEEDS_EXTRACTION", "EXCLUDE", "UNCERTAIN"]},
            "reason_tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 8},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        ("id", "decision", "reason_tags", "confidence"),
    ),
    "extract": _object_schema(
        {
            "id": {"type": "string"},
            "heading": {"type": "string", "minLength": 1, "maxLength": 240},
            "text": {"type": "string", "minLength": 1},
            "reason_tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 8},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        ("id", "heading", "text", "reason_tags", "confidence"),
    ),
    "review": _object_schema(
        {
            "id": {"type": "string"},
            "verdict": {"type": "string", "enum": ["APPROVE", "REJECT", "REVISE"]},
            "reason_tags": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        },
        ("id", "verdict", "reason_tags"),
    ),
    "stitch": _object_schema(
        {
            "id": {"type": "string"},
            "decision": {"type": "string", "enum": ["merge", "no-merge"]},
            "reason": {"type": "string", "minLength": 1, "maxLength": 120},
        },
        ("id", "decision", "reason"),
    ),
}


class ExactIDError(ValueError):
    """A worker response did not contain exactly the submitted record IDs."""


class ResultSchemaError(ValueError):
    """A worker response violated the queue's bounded result schema."""


def validate_exact_results(payload: object, expected_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Validate the strict result envelope and exact submitted ID multiset."""
    if not isinstance(payload, dict) or set(payload) != {"results"} or not isinstance(payload["results"], list):
        raise ExactIDError("worker result must contain only a results array")
    results = payload["results"]
    ids = [item.get("id") if isinstance(item, dict) else None for item in results]
    if any(not isinstance(item, dict) for item in results):
        raise ExactIDError("every worker result must be an object")
    if len(ids) != len(set(ids)):
        raise ExactIDError("worker result contains duplicate IDs")
    if set(ids) != set(expected_ids) or len(ids) != len(expected_ids):
        raise ExactIDError("worker result IDs do not exactly match the submitted batch")
    by_id = {str(item["id"]): item for item in results}
    return [by_id[value] for value in expected_ids]


def _matches_type(value: object, expected: str) -> bool:
    return {
        "string": isinstance(value, str),
        "null": value is None,
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "array": isinstance(value, list),
    }[expected]


def validate_result_schema(results: Sequence[Mapping[str, Any]], schema: Mapping[str, object]) -> None:
    """Validate the bounded JSON-Schema subset used by every worker queue."""
    item_schema = schema["properties"]["results"]["items"]  # type: ignore[index]
    properties = item_schema["properties"]
    required = set(item_schema["required"])
    for result in results:
        if set(result) != set(properties) or not required.issubset(result):
            raise ResultSchemaError("worker result fields do not exactly match the output schema")
        for name, rule in properties.items():
            value = result[name]
            expected_type = rule.get("type")
            types = [expected_type] if isinstance(expected_type, str) else list(expected_type)
            if not any(_matches_type(value, item_type) for item_type in types):
                raise ResultSchemaError(f"worker result {name} has the wrong type")
            if "enum" in rule and value not in rule["enum"]:
                raise ResultSchemaError(f"worker result {name} is outside its enum")
            if isinstance(value, str):
                if len(value) < int(rule.get("minLength", 0)) or len(value) > int(rule.get("maxLength", 2**31)):
                    raise ResultSchemaError(f"worker result {name} has invalid length")
            if isinstance(value, list):
                if len(value) < int(rule.get("minItems", 0)) or len(value) > int(rule.get("maxItems", 2**31)):
                    raise ResultSchemaError(f"worker result {name} has invalid item count")
                item_type = rule.get("items", {}).get("type")
                if item_type and any(not _matches_type(item, item_type) for item in value):
                    raise ResultSchemaError(f"worker result {name} has invalid items")
                item_enum = rule.get("items", {}).get("enum")
                if item_enum and any(item not in item_enum for item in value):
                    raise ResultSchemaError(f"worker result {name} contains an unbounded item")
                if rule.get("uniqueItems") and len(value) != len(set(value)):
                    raise ResultSchemaError(f"worker result {name} contains duplicate items")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if value < rule.get("minimum", value) or value > rule.get("maximum", value):
                    raise ResultSchemaError(f"worker result {name} is outside its bounds")


def pack_batches(records: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Pack deterministic microbatches by both record count and JSON byte size."""
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    size = 2
    for record in records:
        encoded = len(_canonical(record).encode("utf-8")) + (1 if current else 0)
        if encoded + 2 > MAX_BATCH_BYTES:
            raise ValueError("one evidence record exceeds the 64 KiB batch limit")
        if current and (len(current) >= MAX_BATCH_RECORDS or size + encoded > MAX_BATCH_BYTES):
            batches.append(current)
            current = []
            size = 2
        current.append(record)
        size += encoded
    if current:
        batches.append(current)
    return batches


def _prompt_batches(queue: str, records: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for record in records:
        proposed = [*current, record]
        if len(proposed) > MAX_BATCH_RECORDS or len(_worker_prompt(queue, proposed).encode("utf-8")) > MAX_BATCH_BYTES:
            if not current:
                raise ValueError("one evidence record exceeds the 64 KiB worker limit")
            batches.append(current)
            current = [record]
            if len(_worker_prompt(queue, current).encode("utf-8")) > MAX_BATCH_BYTES:
                raise ValueError("one evidence record exceeds the 64 KiB worker limit")
        else:
            current = proposed
    if current:
        batches.append(current)
    return batches


def _run_packed(
    workspace: Path,
    *,
    queue: str,
    slot: int,
    records: Sequence[dict[str, Any]],
    foundry_db: Path | None,
    executor: CodexExecutor,
    schema_key: str | None = None,
) -> list[dict[str, Any]]:
    return [
        result
        for batch in _prompt_batches(queue, records)
        for result in run_codex_batch(
            workspace, queue=queue, slot=slot, records=batch,
            foundry_db=foundry_db, executor=executor, schema_key=schema_key,
        )
    ]


@dataclass(frozen=True)
class CodexResult:
    payload: dict[str, Any]
    thread_id: str | None
    usage: dict[str, Any]
    result_hash: str


class _CodexProcessError(RuntimeError):
    """Sanitized Codex CLI failure classification without captured output."""

    def __init__(self, category: str, *, retryable: bool) -> None:
        super().__init__(category)
        self.category = category
        self.retryable = retryable


def _hosted_output_schema(value: object) -> object:
    """Strip hosted-unsupported keywords while retaining local validation."""
    if isinstance(value, dict):
        return {
            str(key): _hosted_output_schema(item)
            for key, item in value.items()
            if key != "uniqueItems"
        }
    if isinstance(value, list):
        return [_hosted_output_schema(item) for item in value]
    return value


class CodexExecutor:
    """Schema-constrained noninteractive Codex process adapter."""

    def __init__(self, binary: str = "codex", *, timeout: int = 1800):
        self.binary = binary
        self.timeout = timeout
        self._version: str | None = None

    @property
    def version(self) -> str:
        if self._version is None:
            result = subprocess.run(
                [self.binary, "--version"], capture_output=True, text=True, timeout=30, check=True
            )
            self._version = result.stdout.strip()
        return self._version

    def execute(
        self,
        *,
        model: str,
        prompt: str,
        schema: Mapping[str, object],
        workdir: Path,
        thread_id: str | None,
    ) -> CodexResult:
        schema_path = workdir / "output-schema.json"
        output_path = workdir / "last-message.json"
        schema_path.write_text(_canonical(_hosted_output_schema(schema)), encoding="utf-8")
        common = [
            "--model", model,
            "--ignore-user-config",
            "--skip-git-repo-check",
            "--json",
            "--output-schema", str(schema_path),
            "-o", str(output_path),
        ]
        if thread_id:
            command = [self.binary, "--sandbox", "read-only", "exec", "resume", *common, thread_id, "-"]
        else:
            command = [self.binary, "--sandbox", "read-only", "exec", *common, "-"]
        env = dict(os.environ)
        project_root = str(Path(__file__).parents[1])
        env["PYTHONPATH"] = project_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        completed = subprocess.run(
            command,
            input=prompt,
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if completed.returncode != 0:
            diagnostic = f"{completed.stdout}\n{completed.stderr}".casefold()
            if "hit your usage limit" in diagnostic or "usage limit for" in diagnostic:
                raise _CodexProcessError("model-usage-limit", retryable=False)
            if "authentication" in diagnostic or "not logged in" in diagnostic:
                raise _CodexProcessError("authentication-required", retryable=False)
            if "invalid_json_schema" in diagnostic:
                raise _CodexProcessError("invalid-json-schema", retryable=False)
            raise _CodexProcessError(
                f"codex-exit-{completed.returncode}", retryable=True
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        events: list[dict[str, Any]] = []
        for line in completed.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        found_thread = thread_id
        usage: dict[str, Any] = {}
        for event in events:
            if event.get("type") == "thread.started" and isinstance(event.get("thread_id"), str):
                found_thread = str(event["thread_id"])
            candidate_usage = event.get("usage")
            if isinstance(candidate_usage, dict):
                usage = candidate_usage
        encoded = _canonical(payload)
        return CodexResult(payload, found_thread, usage, hashlib.sha256(encoded.encode()).hexdigest())


class LocalQwenExecutor:
    """Bounded llama.cpp OpenAI-compatible adapter for local coverage triage."""

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8081/v1/chat/completions",
        *,
        model: str = "qwen3.8-27b-q4-xl",
        timeout: int = 600,
        _opener: object | None = None,
    ) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Qwen endpoint must be an HTTP(S) chat-completions URL")
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Qwen endpoint must be loopback-local to protect private corpus text")
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        # Never let environment HTTP_PROXY settings relay purchased text.
        self._opener = _opener or build_opener(ProxyHandler({}))

    @property
    def version(self) -> str:
        # The endpoint is intentionally omitted from persisted audit metadata.
        return f"llama.cpp-openai:{self.model}"

    def execute(
        self,
        *,
        model: str,
        prompt: str,
        schema: Mapping[str, object],
        workdir: Path,
        thread_id: str | None,
    ) -> CodexResult:
        del workdir, thread_id
        if model != LOCAL_QWEN_QUEUE_MODEL:
            raise ValueError("local Qwen executor received a non-local queue model")
        request = Request(
            self.endpoint,
            data=_canonical(
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "min_p": 0,
                    "presence_penalty": 1.5,
                    "repeat_penalty": 1.0,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "max_tokens": 4096,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "licensed_corpus_results",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:  # type: ignore[attr-defined]
                body = response.read(MAX_LOCAL_RESPONSE_BYTES + 1)
                if len(body) > MAX_LOCAL_RESPONSE_BYTES:
                    raise ResultSchemaError("local Qwen response exceeds the bounded limit")
                envelope = json.loads(body.decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            raise _CodexProcessError("local-qwen-unavailable", retryable=True) from exc
        try:
            content = envelope["choices"][0]["message"]["content"]
            payload = content if isinstance(content, dict) else json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ResultSchemaError("local Qwen returned a malformed response envelope") from exc
        if not isinstance(payload, dict):
            raise ResultSchemaError("local Qwen result must be a JSON object")
        encoded = _canonical(payload)
        usage = envelope.get("usage") if isinstance(envelope.get("usage"), dict) else {}
        return CodexResult(
            payload=payload,
            thread_id=None,
            usage=usage,
            result_hash=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        )


class RoutedExecutor:
    """Route local Qwen work locally and hosted review work through Codex CLI."""

    def __init__(self, local: LocalQwenExecutor, hosted: CodexExecutor | None = None) -> None:
        self.local = local
        self.hosted = hosted or CodexExecutor()

    def version_for(self, model: str) -> str:
        return self.local.version if model == LOCAL_QWEN_QUEUE_MODEL else self.hosted.version

    @property
    def version(self) -> str:
        return self.hosted.version

    def execute(self, *, model: str, **kwargs: object) -> CodexResult:
        target = self.local if model == LOCAL_QWEN_QUEUE_MODEL else self.hosted
        return target.execute(model=model, **kwargs)


def _executor_version(executor: object, model: str) -> str:
    resolver = getattr(executor, "version_for", None)
    return str(resolver(model)) if callable(resolver) else str(executor.version)  # type: ignore[attr-defined]


def _schema_digest(schema: Mapping[str, object]) -> str:
    return _digest(_canonical(schema))


def _prompt_digest(queue: str, model: str) -> str:
    return _digest(RUNNER_VERSION, PROMPT_VERSION, queue, model)


def _session(
    workspace: Path,
    queue: str,
    slot: int,
    model: str,
    cli_version: str,
    schema: Mapping[str, object],
) -> dict[str, Any]:
    now = int(time.time())
    prompt_digest = _prompt_digest(queue, model)
    schema_digest = _schema_digest(schema)
    policy_digest = licensed_policy_digest()
    role = "reviewer" if queue.startswith("review") else "producer"
    with _connect(workspace) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM runner_sessions WHERE queue_name=? AND slot=?", (queue, slot)
        ).fetchone()
        rotate = row is not None and (
            row["model"] != model
            or row["cli_version"] != cli_version
            or row["prompt_digest"] != prompt_digest
            or row["schema_digest"] != schema_digest
            or row["policy_digest"] != policy_digest
            or int(row["completed_batches"]) >= SESSION_BATCH_LIMIT
            or int(row["submitted_evidence_bytes"]) >= SESSION_EVIDENCE_LIMIT
        )
        if row is None or rotate:
            conn.execute(
                """INSERT INTO runner_sessions
                   (queue_name, slot, role, model, cli_version, thread_id, prompt_digest, schema_digest,
                    policy_digest, completed_batches, submitted_evidence_bytes, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, 0, 0, ?, ?)
                   ON CONFLICT(queue_name, slot) DO UPDATE SET
                     role=excluded.role, model=excluded.model, cli_version=excluded.cli_version, thread_id=NULL,
                     prompt_digest=excluded.prompt_digest, schema_digest=excluded.schema_digest,
                     policy_digest=excluded.policy_digest, completed_batches=0,
                     submitted_evidence_bytes=0, created_at=excluded.created_at,
                     updated_at=excluded.updated_at""",
                (queue, slot, role, model, cli_version, prompt_digest, schema_digest, policy_digest, now, now),
            )
            row = conn.execute(
                "SELECT * FROM runner_sessions WHERE queue_name=? AND slot=?", (queue, slot)
            ).fetchone()
        conn.commit()
        assert row is not None
        return dict(row)


def _finish_session(workspace: Path, queue: str, slot: int, result: CodexResult, evidence_bytes: int) -> None:
    with _connect(workspace) as conn:
        conn.execute(
            """UPDATE runner_sessions SET thread_id=?, completed_batches=completed_batches+1,
               submitted_evidence_bytes=submitted_evidence_bytes+?, updated_at=?
               WHERE queue_name=? AND slot=?""",
            (result.thread_id, evidence_bytes, int(time.time()), queue, slot),
        )
        conn.commit()


def _record_attempt(
    workspace: Path,
    *,
    queue: str,
    batch_key: str,
    slot: int,
    model: str,
    cli_version: str,
    thread_id: str | None,
    attempt: int,
    input_digest: str,
    status: str,
    result: CodexResult | None = None,
    error_kind: str | None = None,
) -> None:
    now = int(time.time())
    attempt_id = "attempt:" + _digest(queue, batch_key, attempt)
    with _connect(workspace) as conn:
        conn.execute(
            """INSERT INTO runner_attempts
               (attempt_id, queue_name, batch_key, slot, model, thread_id, attempt,
                cli_version, input_digest, result_digest, status, exit_code, usage_json,
                started_at, completed_at, error_kind)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(queue_name, batch_key, attempt) DO UPDATE SET
                 thread_id=excluded.thread_id, result_digest=excluded.result_digest,
                 status=excluded.status, exit_code=excluded.exit_code,
                 usage_json=excluded.usage_json, completed_at=excluded.completed_at,
                 error_kind=excluded.error_kind""",
            (
                attempt_id, queue, batch_key, slot, model, thread_id, attempt, cli_version,
                input_digest, result.result_hash if result else None, status,
                0 if result else None, _canonical(result.usage) if result else None,
                now, now if status != "running" else None, error_kind,
            ),
        )
        conn.commit()


def _evidence_context(workspace: Path, workdir: Path, ids: Sequence[str], foundry_db: Path | None) -> Path:
    with _connect(workspace, readonly=True) as conn:
        placeholders = ",".join("?" for _ in ids)
        runs = conn.execute(
            f"SELECT DISTINCT parser_run_id FROM source_sections WHERE section_key IN ({placeholders})",
            tuple(ids),
        ).fetchall() if ids else []
        neighbor_ids: set[str] = set()
        for run in runs:
            ordered = [
                str(row[0]) for row in conn.execute(
                    """SELECT section_key FROM source_sections WHERE parser_run_id=?
                       ORDER BY COALESCE(page_start, 2147483647), source_section_id, section_key""",
                    (run[0],),
                )
            ]
            positions = {value: index for index, value in enumerate(ordered)}
            for section_id in ids:
                if section_id in positions:
                    position = positions[section_id]
                    neighbor_ids.update(ordered[max(0, position - 2) : position + 3])
    context = {
        "version": 1,
        "workspace": str(workspace.resolve()),
        "foundry_database": str(foundry_db.resolve()) if foundry_db else None,
        "allowed_ids": list(ids),
        "neighbor_ids": sorted(neighbor_ids - set(ids)),
    }
    path = workdir / "claim.json"
    path.write_text(_canonical(context), encoding="utf-8")
    path.chmod(0o600)
    return path


def _worker_prompt(queue: str, records: Sequence[dict[str, Any]]) -> str:
    policies = {
        "screen": "First validate the isolated PDF section. A fragment, heading/body mismatch, adjacent-rule contamination, or page-number stub is needs-layout, never covered merely because Foundry has a complete rule. For valid input, compare every mechanic, including repeated numbers by their mechanical role. Return covered only with the supplied Foundry IDs whose union is complete; return additional-mechanics when the PDF adds anything; use uncertain when the compact packet cannot decide. This triage never authorizes suppression.",
        "coverage-confirm": "Independently judge the structurally valid PDF section against the supplied Foundry candidates. Do not rely on or infer any prior model vote. A fragment or structural defect is needs-layout. Return covered only when the selected supplied Foundry rows jointly preserve every mechanic, including condition ranks and repeated numbers by role. Otherwise return additional-mechanics or uncertain.",
        "classify": "Classify under mechanics-v1. PUBLIC_AS_IS means the whole supplied section is functional public mechanics; MIXED_NEEDS_EXTRACTION means mechanics must be reconstructed; exclusions and uncertainty produce no prose.",
        "extract": "Write concise mechanics-only text under mechanics-v1. Do not copy narrative, setting prose, examples, art captions, trademarks, or attribution. Preserve complete functional conditions and outcomes.",
        "review": "Independently review the candidate against the source and mechanics-v1. APPROVE only public text; REJECT only confirms EXCLUDE/UNCERTAIN; otherwise REVISE.",
        "review-mixed": "Independently review the mechanics-only reconstruction against the source. APPROVE if complete and safely scoped; otherwise REVISE.",
        "rework-terra": "Produce a corrected terminal classification. Include concise mechanics-only text only when mixed extraction is required.",
        "rework-sol": "Final rework: produce a corrected terminal classification with mechanics-only text for mixed extraction. If genuinely unresolved, choose UNCERTAIN.",
        "stitch-select": "For each proposed adjacent group, choose merge only when the full sections are one fragmented logical rule; otherwise no-merge.",
        "stitch-confirm": "Independently confirm each proposed merge. Merge only when the exact full adjacent group is necessary and supported.",
    }
    return (
        f"You are a bounded {queue} worker. {policies[queue]}\n"
        "Return only the schema-constrained result for every supplied ID exactly once. "
        "You may inspect local evidence with `python -m pf2e_codex.review_evidence --context claim.json help`; "
        "that executable is read-only and accepts only this claim. Do not use network access.\n"
        "Records:\n" + _canonical(list(records))
    )


def run_codex_batch(
    workspace: Path,
    *,
    queue: str,
    slot: int,
    records: Sequence[dict[str, Any]],
    foundry_db: Path | None,
    executor: CodexExecutor,
    schema_key: str | None = None,
) -> list[dict[str, Any]]:
    schema_name = schema_key or ("stitch" if queue.startswith("stitch") else "review" if queue.startswith("review") else queue)
    schema = SCHEMAS[schema_name]
    model = MODEL_BY_QUEUE[queue]
    cli_version = _executor_version(executor, model)
    expected_ids = [str(record["id"]) for record in records]
    prompt = _worker_prompt(queue, records)
    input_digest = _digest(prompt)
    batch_key = _digest(queue, PROMPT_VERSION, model, input_digest, *expected_ids)
    evidence_bytes = len(prompt.encode("utf-8"))
    if len(records) > MAX_BATCH_RECORDS or evidence_bytes > MAX_BATCH_BYTES:
        raise ValueError("worker batch exceeds deterministic limits")
    temp_root = workspace.parent / ".licensed-runner-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    with _connect(workspace, readonly=True) as conn:
        prior = int(conn.execute(
            "SELECT COALESCE(MAX(attempt),0) FROM runner_attempts WHERE queue_name=? AND batch_key=?",
            (queue, batch_key),
        ).fetchone()[0])
    for retry in range(1, MAX_RETRIES + 1):
        attempt = prior + retry
        session = _session(workspace, queue, slot, model, cli_version, schema)
        _record_attempt(
            workspace, queue=queue, batch_key=batch_key, slot=slot, model=model,
            cli_version=cli_version,
            thread_id=session.get("thread_id"), attempt=attempt, input_digest=input_digest,
            status="running",
        )
        try:
            with tempfile.TemporaryDirectory(prefix="batch-", dir=temp_root) as raw_workdir:
                workdir = Path(raw_workdir)
                _evidence_context(workspace, workdir, expected_ids, foundry_db)
                result = executor.execute(
                    model=model, prompt=prompt, schema=schema, workdir=workdir,
                    thread_id=session.get("thread_id"),
                )
                values = validate_exact_results(result.payload, expected_ids)
                validate_result_schema(values, schema)
            _record_attempt(
                workspace, queue=queue, batch_key=batch_key, slot=slot, model=model,
                cli_version=cli_version,
                thread_id=result.thread_id, attempt=attempt, input_digest=input_digest,
                status="accepted", result=result,
            )
            _finish_session(workspace, queue, slot, result, evidence_bytes)
            return values
        except _CodexProcessError as exc:
            kind = "transport-failure"
            _record_attempt(
                workspace, queue=queue, batch_key=batch_key, slot=slot, model=model,
                cli_version=cli_version,
                thread_id=session.get("thread_id"), attempt=attempt,
                input_digest=input_digest, status=kind, error_kind=exc.category,
            )
            if not exc.retryable:
                raise RuntimeError(f"{queue} blocked by {exc.category}") from exc
        except (OSError, subprocess.SubprocessError, RuntimeError, json.JSONDecodeError) as exc:
            kind = "transport-failure"
            _record_attempt(
                workspace, queue=queue, batch_key=batch_key, slot=slot, model=model,
                cli_version=cli_version,
                thread_id=session.get("thread_id"), attempt=attempt, input_digest=input_digest,
                status=kind, error_kind=type(exc).__name__,
            )
        except ValueError as exc:
            kind = "schema-failure"
            _record_attempt(
                workspace, queue=queue, batch_key=batch_key, slot=slot, model=model,
                cli_version=cli_version,
                thread_id=session.get("thread_id"), attempt=attempt, input_digest=input_digest,
                status=kind, error_kind=type(exc).__name__,
            )
        if retry == MAX_RETRIES:
            raise RuntimeError(f"{queue} batch failed after {MAX_RETRIES} attempts")
    raise AssertionError("unreachable")


def _active_product_rows(workspace: Path) -> list[sqlite3.Row]:
    with _connect(workspace, readonly=True) as conn:
        return conn.execute(
            f"""SELECT p.*, r.era, r.license
               FROM parser_runs AS p
               JOIN source_revisions AS r ON r.product_code=p.product_code
                AND r.content_fingerprint=(SELECT content_fingerprint FROM source_sections
                    WHERE parser_run_id=p.parser_run_id LIMIT 1)
               WHERE p.state='active' AND p.review_enabled=1
                 AND {_semantic_scope_sql('p.product_code')}
               ORDER BY p.product_code"""
        ).fetchall()


def verify_workspace(
    workspace: Path | str,
    *,
    require_complete: bool = False,
    foundry_database: Path | str | None = None,
) -> dict[str, Any]:
    path = Path(workspace).expanduser().resolve()
    _ensure_workspace_migrated(path)
    with _connect(path, readonly=True) as conn:
        integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        schema = conn.execute("SELECT value FROM metadata WHERE key='review_schema_version'").fetchone()
        products = conn.execute(
            """SELECT p.product_code, r.era, r.license, p.parser_version, p.complete,
                      p.origin, COUNT(s.section_key) AS sections
               FROM parser_runs AS p
               JOIN source_sections AS s ON s.parser_run_id=p.parser_run_id
               JOIN source_revisions AS r USING(product_code, content_fingerprint)
               WHERE p.state='active' AND p.review_enabled=1
               GROUP BY p.parser_run_id ORDER BY p.product_code"""
        ).fetchall()
        live_claims = int(conn.execute(
            """SELECT (SELECT COUNT(*) FROM review_shards WHERE claimant IS NOT NULL AND lease_expires_at>=?)
                      +(SELECT COUNT(*) FROM review_claims WHERE lease_expires_at>=?)
                      +(SELECT COUNT(*) FROM draft_screening_claims WHERE lease_expires_at>=?)
                      +(SELECT COUNT(*) FROM stitch_claims WHERE lease_expires_at>=?)""",
            (int(time.time()),) * 4,
        ).fetchone()[0])
        maintenance = int(conn.execute(
            f"""SELECT COUNT(*) FROM runner_maintenance AS maintenance
                WHERE maintenance.resolved_at IS NULL AND (
                  EXISTS (SELECT 1 FROM source_sections AS section
                          WHERE section.section_key=maintenance.subject_id
                            AND {_semantic_scope_sql('section.product_code')})
                  OR EXISTS (SELECT 1 FROM candidates AS candidate
                             JOIN source_sections AS section
                               ON section.section_key=candidate.section_key
                             WHERE candidate.candidate_id=maintenance.subject_id
                               AND {_semantic_scope_sql('section.product_code')})
                  OR EXISTS (SELECT 1 FROM stitch_candidates AS stitch
                             WHERE stitch.candidate_id=maintenance.subject_id
                               AND {_semantic_scope_sql('stitch.product_code')})
                )"""
        ).fetchone()[0])
        unsafe_runner_metadata = 0
        for table in (
            "runner_sessions",
            "runner_attempts",
            "runner_maintenance",
            "runner_screen_rejections",
            "stitch_candidates",
            "stitch_votes",
            "stitch_claims",
            "aon_cache",
            "duplicate_groups",
            "duplicate_group_members",
            "foundry_snapshots",
            "foundry_snapshot_rows",
            "foundry_coverage_candidates",
            "foundry_coverage_confirmations",
            "foundry_coverage_votes",
            "review_product_scope",
        ):
            for row in conn.execute(f"SELECT * FROM {table}"):
                encoded = _canonical(dict(row)).casefold()
                if any(marker in encoded for marker in (".local-corpus", "/home/", "\\users\\", "file://")) or "@" in encoded:
                    unsafe_runner_metadata += 1
        unresolved_resolutions = int(conn.execute(
            f"""SELECT COUNT(*) FROM source_sections AS s
               JOIN parser_runs AS p ON p.parser_run_id=s.parser_run_id
               WHERE p.state='active' AND p.review_enabled=1
                 AND {_semantic_scope_sql('p.product_code')}
                 AND EXISTS (
                   SELECT 1 FROM draft_screening_current AS screen
                   WHERE screen.parser_run_id=s.parser_run_id
                     AND screen.section_key=s.section_key
                     AND screen.decision IN ('ADD','REJECT')
                 )
                 AND NOT EXISTS (
                 SELECT 1 FROM candidates AS c JOIN reviews AS r ON r.candidate_id=c.candidate_id
                 LEFT JOIN review_invalidations AS i ON i.review_id=r.review_id
                 WHERE c.section_key=s.section_key
                   AND c.candidate_ordinal=(SELECT MAX(c2.candidate_ordinal) FROM candidates c2 WHERE c2.section_key=s.section_key)
                   AND i.review_id IS NULL
                   AND ((c.decision IN ('PUBLIC_AS_IS','MIXED_NEEDS_EXTRACTION') AND r.verdict='APPROVE')
                     OR (c.decision IN ('EXCLUDE','UNCERTAIN') AND r.verdict='REJECT'))
               ) AND NOT EXISTS (
                 SELECT 1 FROM duplicate_group_members AS member
                 WHERE member.section_key=s.section_key AND member.source_ordinal>0
               ) AND NOT EXISTS (
                 SELECT 1 FROM foundry_coverage_confirmations AS coverage
                 JOIN metadata AS active ON active.key='active_foundry_snapshot'
                    AND active.value=coverage.snapshot_digest
                 WHERE coverage.section_key=s.section_key
               )"""
        ).fetchone()[0])
        intentional = ("page-number", "repeated-furniture", "contents-index", "credits-legal")
        placeholders = ",".join("?" for _ in intentional)
        global_unresolved_quarantine = int(conn.execute(
            f"""SELECT COUNT(*) FROM parser_quarantine AS q JOIN parser_runs AS p
                   ON p.parser_run_id=q.parser_run_id
                  WHERE p.state='active' AND p.review_enabled=1
                    AND q.reason NOT IN ({placeholders})""",
            intentional,
        ).fetchone()[0])
        unresolved_quarantine = int(conn.execute(
            f"""SELECT COUNT(*) FROM parser_quarantine AS q JOIN parser_runs AS p
                   ON p.parser_run_id=q.parser_run_id
                  WHERE p.state='active' AND p.review_enabled=1
                    AND {_semantic_scope_sql('p.product_code')}
                    AND q.reason NOT IN ({placeholders})""",
            intentional,
        ).fetchone()[0])
        active_snapshot = conn.execute(
            "SELECT value FROM metadata WHERE key='active_foundry_snapshot'"
        ).fetchone()
        unauthorized_coverage = _unauthorized_foundry_confirmation_count(conn)
    scope = review_product_scope(path)
    errors: list[str] = []
    if integrity != "ok":
        errors.append("sqlite-integrity")
    if schema is None or int(schema[0]) != REVIEW_SCHEMA_VERSION:
        errors.append("schema-version")
    if len(products) != len(EXPECTED_PRODUCTS) or {
        str(row["product_code"]) for row in products
    } != set(EXPECTED_PRODUCTS):
        errors.append("five-product-catalog")
    if {
        str(item["product_code"]) for item in scope["products"]
    } != {str(row["product_code"]) for row in products}:
        errors.append("product-scope-catalog")
    if unsafe_runner_metadata:
        errors.append("runner-metadata-privacy")
    if unauthorized_coverage:
        errors.append("unauthorized-foundry-coverage")
    for row in products:
        expected = PRODUCT_CATALOG[str(row["product_code"])]
        if row["era"] != expected.rules_era or row["license"] != expected.license:
            errors.append(f"catalog-provenance:{row['product_code']}")
        if int(row["complete"]) != 1 or row["origin"] != "trusted-direct-pdf-v1":
            errors.append(f"trusted-parser:{row['product_code']}")
    screen = draft_screening_status(path)
    unresolved = maintenance + sum(
        int(product["unprocessed"]) + int(product["deferred"])
        for product in screen["products"]
    )
    if require_complete:
        unresolved += unresolved_resolutions
        if unresolved_quarantine:
            errors.append("unresolved-rule-quarantine")
            unresolved += unresolved_quarantine
        if foundry_database is None:
            errors.append("foundry-database-required")
        else:
            snapshot = load_clean_foundry(foundry_database)
            if active_snapshot is None or str(active_snapshot[0]) != snapshot.digest:
                errors.append("stale-foundry-coverage")
        if live_claims:
            errors.append("live-claims")
        if unresolved:
            errors.append("unresolved-work")
    return {
        "ok": not errors,
        "errors": sorted(set(errors)),
        "products": [
            {
                "product_code": row["product_code"],
                "rules_era": row["era"],
                "license": row["license"],
                "parser_version": row["parser_version"],
                "sections": int(row["sections"]),
            }
            for row in products
        ],
        "semantic_scope": scope,
        "live_claims": live_claims,
        "maintainer_items": maintenance,
        "unresolved": unresolved,
        "unresolved_rule_quarantine": unresolved_quarantine,
        "global_unresolved_rule_quarantine": global_unresolved_quarantine,
    }


def quality_workspace(
    workspace: Path | str,
    *,
    parser_run_ids: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Return a deterministic content-free parser quality report."""
    path = Path(workspace).expanduser().resolve()
    _ensure_workspace_migrated(path)
    return audit_workspace(path, parser_run_ids=parser_run_ids).as_dict()


def compare_workspace_quality(
    workspace: Path | str,
    *,
    baseline_parser_run_ids: Mapping[str, str],
    candidate_parser_run_ids: Mapping[str, str],
) -> dict[str, object]:
    """Compare two complete product run selections inside one private workspace."""
    path = Path(workspace).expanduser().resolve()
    _ensure_workspace_migrated(path)
    baseline = audit_workspace(path, parser_run_ids=baseline_parser_run_ids)
    candidate = audit_workspace(path, parser_run_ids=candidate_parser_run_ids)
    return compare_quality(baseline, candidate).as_dict()


def _compact_retired_structural_evidence(workspace: Path) -> int:
    """Drop bulky retired parser geometry after carry-forward and comparison.

    Candidate, review, screening, stitch, source, and parser-run audit rows stay
    intact. Only native-anchor/block/quarantine material owned exclusively by
    retired runs is removed from the already validated sibling.
    """
    deleted = 0
    with _connect(workspace) as conn:
        conn.execute("BEGIN IMMEDIATE")
        retired = "SELECT parser_run_id FROM parser_runs WHERE state='retired'"
        for table in (
            "parser_section_block_anchors",
            "parser_section_blocks",
            "parser_quarantine_anchors",
            "parser_quarantine",
            "parser_section_anchors",
            "parser_ignored_anchors",
        ):
            deleted += conn.execute(
                f"DELETE FROM {table} WHERE parser_run_id IN ({retired})"
            ).rowcount
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("VACUUM")
    return deleted


def prepare_workspace(
    workspace: Path | str,
    sources_root: Path | str,
    *,
    parser_version: str = PAIZO_NATIVE_PARSER_V5,
    shard_size: int = MAX_BATCH_RECORDS,
    layout_model_dir: Path | str | None = None,
    layout_provider: str | None = None,
) -> dict[str, Any]:
    """Build a validated sibling while carrying forward exact reviewed work."""
    target = Path(workspace).expanduser().resolve()
    sources_path = Path(sources_root).expanduser().resolve()
    found = discover_sources(sources_path, include=EXPECTED_PRODUCTS)
    state_root = target.parent / ".licensed-runner-selection"
    selected = select_revisions(found, include=EXPECTED_PRODUCTS, state_root=state_root)
    by_product = {item.product.code: item for item in selected}
    if tuple(sorted(by_product)) != tuple(sorted(EXPECTED_PRODUCTS)):
        missing = sorted(set(EXPECTED_PRODUCTS) - set(by_product))
        raise ValueError("trusted source discovery is missing: " + ", ".join(missing))
    sibling = target.with_name(f".{target.name}.rebuild-{os.getpid()}")
    if sibling.exists():
        raise FileExistsError(f"staging workspace already exists: {sibling}")
    baseline_runs: dict[str, str] = {}
    if target.exists():
        source_conn = sqlite3.connect(f"file:{target}?mode=ro", uri=True, timeout=30)
        destination_conn = sqlite3.connect(sibling, timeout=30)
        try:
            source_conn.backup(destination_conn)
        finally:
            destination_conn.close()
            source_conn.close()
        _ensure_workspace_migrated(sibling)
        with _connect(sibling, readonly=True) as conn:
            baseline_runs = {
                str(row["product_code"]): str(row["parser_run_id"])
                for row in conn.execute(
                    """SELECT product_code, parser_run_id FROM parser_runs
                       WHERE state='active' AND review_enabled=1 ORDER BY product_code"""
                )
            }
    else:
        initialize_trusted_workspace(sibling)
    try:
        staged = []
        candidate_runs: dict[str, str] = {}
        layout_analyzer = None
        for product_code in EXPECTED_PRODUCTS:
            revision = by_product[product_code]
            if len(revision.sources) != 1 or not revision.sources[0].combined or revision.sources[0].member is not None:
                raise ValueError(f"{product_code} requires one selected combined PDF for trusted staging")
            layout_artifact: Path | None = None
            if layout_model_dir is not None:
                from .pdf_layout import LayoutAnalyzer, export_pdf_layout

                layout_artifact = _layout_artifact_path(sources_path, revision.sources[0])
                if not layout_artifact.is_file():
                    if layout_analyzer is None:
                        layout_analyzer = LayoutAnalyzer(
                            layout_model_dir,
                            force_provider=layout_provider,
                        )
                    export_pdf_layout(
                        revision.sources[0].path,
                        layout_artifact,
                        analyzer=layout_analyzer,
                    )
            result = stage_trusted_native_pdf(
                sibling, revision.sources[0].path, product_code=product_code,
                parser_version=parser_version, shard_size=shard_size,
                printing_revision=(
                    f"printing-{revision.sources[0].printing}"
                    if revision.sources[0].printing is not None else None
                ),
                layout_artifact=layout_artifact,
            )
            activate_parser_run(sibling, str(result["parser_run_id"]))
            candidate_runs[product_code] = str(result["parser_run_id"])
            with _connect(sibling, readonly=True) as conn:
                section_count = int(conn.execute(
                    "SELECT COUNT(*) FROM source_sections WHERE parser_run_id=?",
                    (result["parser_run_id"],),
                ).fetchone()[0])
            staged.append({"product_code": product_code, "sections": section_count})
        verification = verify_workspace(sibling)
        if not verification["ok"]:
            raise ValueError("fresh trusted workspace failed validation: " + ", ".join(verification["errors"]))
        quality = audit_workspace(sibling, parser_run_ids=candidate_runs)
        absolute_quality = validate_quality(quality)
        if not absolute_quality["passed"]:
            failed = sorted(
                name
                for name, passed in absolute_quality["checks"].items()
                if not passed
            )
            raise ValueError(
                "candidate parser failed the absolute quality gates: "
                + ", ".join(failed)
            )
        comparison: dict[str, object] | None = None
        if baseline_runs:
            if set(baseline_runs) != set(candidate_runs):
                raise ValueError("carry-forward workspace does not contain the complete five-product baseline")
            baseline_quality = audit_workspace(sibling, parser_run_ids=baseline_runs)
            compared = (
                compare_repair_quality(baseline_quality, quality)
                if parser_version == PAIZO_NATIVE_PARSER_V5
                else compare_quality(baseline_quality, quality)
            )
            comparison = compared.as_dict()
            if not compared.passed:
                failed = _failed_comparison_gates(comparison["gates"])
                raise ValueError(
                    "candidate parser failed the deterministic quality gates: "
                    + ", ".join(failed)
                )
        compacted_structural_rows = _compact_retired_structural_evidence(sibling)
        with _connect(sibling) as checkpoint_conn:
            checkpoint_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        journal_conn = sqlite3.connect(sibling, timeout=30)
        journal_conn.execute("PRAGMA journal_mode=DELETE")
        journal_conn.close()
        os.replace(sibling, target)
        for suffix in ("-wal", "-shm"):
            Path(str(target) + suffix).unlink(missing_ok=True)
        return {
            "workspace": target.name,
            "products": staged,
            "verification": verification,
            "quality": quality.as_dict(),
            "comparison": comparison,
            "carried_forward": bool(baseline_runs),
            "compacted_structural_rows": compacted_structural_rows,
        }
    except BaseException:
        sibling.unlink(missing_ok=True)
        Path(str(sibling) + "-wal").unlink(missing_ok=True)
        Path(str(sibling) + "-shm").unlink(missing_ok=True)
        raise


def _failed_comparison_gates(gates: Mapping[str, object]) -> list[str]:
    """Return content-free leaf paths for failed comparison gates."""
    failed: list[str] = []
    for name, value in sorted(gates.items()):
        if isinstance(value, Mapping):
            if "passed" in value:
                if value["passed"] is not True:
                    failed.append(name)
                continue
            for child, child_value in sorted(value.items()):
                if isinstance(child_value, Mapping) and "passed" in child_value:
                    if child_value["passed"] is not True:
                        failed.append(f"{name}.{child}")
                elif child_value is not True:
                    failed.append(f"{name}.{child}")
        elif value is not True:
            failed.append(name)
    return failed


def generate_stitch_candidates(workspace: Path | str) -> dict[str, int]:
    """Persist bounded deterministic adjacent two/three-section proposals."""
    path = Path(workspace).expanduser().resolve()
    _ensure_workspace_migrated(path)
    created = 0
    with _connect(path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM stitch_claims WHERE lease_expires_at<?", (int(time.time()),))
        conn.execute(
            """DELETE FROM stitch_candidates
               WHERE parser_run_id IN (
                 SELECT parser_run_id FROM parser_runs
                 WHERE state='active' AND review_enabled=1
               ) AND NOT EXISTS (
                 SELECT 1 FROM stitch_votes
                 WHERE stitch_votes.candidate_id=stitch_candidates.candidate_id
               ) AND NOT EXISTS (
                 SELECT 1 FROM stitch_claims
                 WHERE stitch_claims.candidate_id=stitch_candidates.candidate_id
               )"""
        )
        runs = conn.execute(
            "SELECT parser_run_id, product_code FROM parser_runs WHERE state='active' AND review_enabled=1 ORDER BY product_code"
        ).fetchall()
        for run in runs:
            sections = conn.execute(
                """SELECT s.section_key, s.source_section_id, s.page_start, s.page_end,
                          s.heading, s.source_text, s.layout_flags,
                          COUNT(a.anchor_hash) AS anchor_count
                   FROM source_sections AS s
                   LEFT JOIN parser_section_anchors AS a
                     ON a.parser_run_id=s.parser_run_id AND a.section_key=s.section_key
                   WHERE s.parser_run_id=?
                   GROUP BY s.section_key
                   ORDER BY COALESCE(s.page_start, 2147483647), s.source_section_id, s.section_key""",
                (run["parser_run_id"],),
            ).fetchall()
            explicit_stitch_flags = {"continuation", "fragment", "orphan-heading"}
            for width in (2, 3):
                for offset in range(0, len(sections) - width + 1):
                    group = sections[offset : offset + width]
                    section_flags = [
                        set(json.loads(str(row["layout_flags"] or "[]")))
                        for row in group
                    ]
                    flags = {
                        flag for values in section_flags for flag in values
                    }
                    if "stitched-adjacent-v1" in flags:
                        continue
                    anchor_counts = [int(row["anchor_count"]) for row in group]
                    if any(count < 1 for count in anchor_counts):
                        continue
                    continuous = all(
                        int(group[index + 1]["page_start"] or 0) - int(group[index]["page_end"] or 0) in {0, 1}
                        for index in range(len(group) - 1)
                    )
                    incomplete = any(
                        not str(row["source_text"]).rstrip().endswith((".", "!", "?", ":", ";", ")", "]"))
                        for row in group[:-1]
                    )
                    same_heading = len({str(row["heading"]).casefold().strip() for row in group}) == 1
                    crosses_page = any(
                        int(group[index + 1]["page_start"] or 0)
                        > int(group[index]["page_end"] or 0)
                        for index in range(len(group) - 1)
                    )
                    middle_fragment = width == 3 and bool(
                        section_flags[1].intersection(explicit_stitch_flags)
                    )
                    edge_fragment = width == 2 and (
                        (
                            offset == 0
                            and bool(
                                section_flags[0].intersection(explicit_stitch_flags)
                            )
                        )
                        or (
                            offset + width == len(sections)
                            and bool(
                                section_flags[-1].intersection(explicit_stitch_flags)
                            )
                        )
                    )
                    shared_layout_regions = set.intersection(
                        *[
                            {
                                flag
                                for flag in values
                                if flag.startswith("layout-region-split:")
                            }
                            for values in section_flags
                        ]
                    )
                    layout_corroborates_fragment = bool(shared_layout_regions) and same_heading
                    explicit_fragment = any(
                        values.intersection(explicit_stitch_flags)
                        for values in section_flags
                    )
                    eligible = (
                        layout_corroborates_fragment
                        or edge_fragment
                        or (crosses_page and incomplete and explicit_fragment)
                        if width == 2
                        else layout_corroborates_fragment or middle_fragment
                    )
                    if not continuous or not eligible:
                        continue
                    keys = [str(row["section_key"]) for row in group]
                    candidate_id = "stitch:" + _digest(run["parser_run_id"], *keys)
                    evidence = {
                        "width": width,
                        "offset": offset,
                        "heuristic_version": STITCH_HEURISTIC_VERSION,
                        "page_continuity": continuous,
                        "crosses_page": crosses_page,
                        "incomplete_terminal": incomplete,
                        "same_heading": same_heading,
                        "shared_layout_region": bool(shared_layout_regions),
                        "layout_flags": sorted(flags),
                        "anchor_counts": anchor_counts,
                    }
                    before = conn.total_changes
                    conn.execute(
                        """INSERT OR IGNORE INTO stitch_candidates
                           (candidate_id, parser_run_id, product_code, section_keys, evidence_json, created_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (candidate_id, run["parser_run_id"], run["product_code"], _canonical(keys), _canonical(evidence), int(time.time())),
                    )
                    inserted = conn.total_changes > before
                    created += int(inserted)
                    if inserted:
                        prior = _prior_stitch_candidate(
                            conn,
                            candidate_id,
                            str(run["product_code"]),
                            _canonical(keys),
                        )
                        if prior is not None:
                            _reuse_stitch_judgment(
                                conn, candidate_id, str(prior["candidate_id"]),
                            )
            created -= _prune_overlapping_unreviewed_stitch_candidates(
                conn, str(run["parser_run_id"])
            )
        conn.commit()
    return {"created": created}


def _reuse_stitch_judgment(
    conn: sqlite3.Connection,
    candidate_id: str,
    prior_candidate_id: str,
) -> None:
    """Carry an exact unchanged vote set and explicit maintainer resolution."""
    conn.execute(
        """INSERT OR IGNORE INTO stitch_votes
           (candidate_id, role, worker, decision, reason, decided_at)
           SELECT ?, role, worker, decision, reason, decided_at
           FROM stitch_votes WHERE candidate_id=?""",
        (candidate_id, prior_candidate_id),
    )
    resolution = conn.execute(
        """SELECT resolved_at, resolution FROM runner_maintenance
           WHERE queue_name='stitch' AND subject_id=?
             AND reason='independent-disagreement' AND resolved_at IS NOT NULL""",
        (prior_candidate_id,),
    ).fetchone()
    if resolution is None:
        return
    conn.execute(
        """INSERT OR IGNORE INTO runner_maintenance
           (item_id, queue_name, subject_id, reason, created_at,
            resolved_at, resolution)
           VALUES (?, 'stitch', ?, 'independent-disagreement', ?, ?, ?)""",
        (
            "maintainer:" + _digest(candidate_id, "stitch-disagreement"),
            candidate_id,
            int(resolution["resolved_at"]),
            int(resolution["resolved_at"]),
            str(resolution["resolution"]),
        ),
    )


def _prior_stitch_candidate(
    conn: sqlite3.Connection,
    candidate_id: str,
    product_code: str,
    section_keys: str,
) -> sqlite3.Row | None:
    """Find the newest judgment for the exact ordered source-section set."""
    return conn.execute(
        """SELECT candidate_id FROM stitch_candidates
           WHERE candidate_id<>? AND product_code=? AND section_keys=?
           ORDER BY created_at DESC, candidate_id DESC LIMIT 1""",
        (candidate_id, product_code, section_keys),
    ).fetchone()


def _prune_overlapping_unreviewed_stitch_candidates(
    conn: sqlite3.Connection,
    parser_run_id: str,
) -> int:
    """Keep the smallest deterministic disjoint proposals for one parser run.

    Larger fragments still present after an approved two-section repair are
    rediscovered by the next fixed-point parser pass. This avoids asking
    independent workers to approve competing groups in the same round.
    """
    voted_rows = conn.execute(
        """SELECT c.section_keys FROM stitch_candidates AS c
           WHERE c.parser_run_id=? AND EXISTS (
             SELECT 1 FROM stitch_votes AS v WHERE v.candidate_id=c.candidate_id
           )""",
        (parser_run_id,),
    ).fetchall()
    occupied = {
        key
        for row in voted_rows
        for key in json.loads(str(row["section_keys"]))
    }
    rows = conn.execute(
        """SELECT c.candidate_id, c.section_keys
           FROM stitch_candidates AS c
           WHERE c.parser_run_id=? AND NOT EXISTS (
             SELECT 1 FROM stitch_votes AS v WHERE v.candidate_id=c.candidate_id
           ) AND NOT EXISTS (
             SELECT 1 FROM stitch_claims AS claim WHERE claim.candidate_id=c.candidate_id
           )
           ORDER BY json_extract(c.evidence_json, '$.width'),
                    json_extract(c.evidence_json, '$.offset'), c.candidate_id""",
        (parser_run_id,),
    ).fetchall()
    remove: list[str] = []
    for row in rows:
        keys = set(json.loads(str(row["section_keys"])))
        if occupied.intersection(keys):
            remove.append(str(row["candidate_id"]))
        else:
            occupied.update(keys)
    if remove:
        conn.executemany(
            "DELETE FROM stitch_candidates WHERE candidate_id=?",
            ((candidate_id,) for candidate_id in remove),
        )
    return len(remove)


def _normalized_query(value: str) -> str:
    return " ".join(sorted(set(_TERM_RE.findall(value.casefold())))[:12])


class _AONResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "a":
            return
        href = dict(attrs).get("href")
        if isinstance(href, str) and ".aspx?ID=" in href:
            self._href = href
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() != "a" or self._href is None:
            return
        url = urljoin(_AON_ROOT, self._href)
        parsed = urlparse(url)
        title = " ".join("".join(self._text).split())
        if parsed.scheme == "https" and parsed.hostname in _AON_HOSTS and title:
            record = {"title": title[:240], "url": url}
            if record not in self.results:
                self.results.append(record)
        self._href = None
        self._text = []


def refresh_aon_cache(
    workspace: Path | str,
    *,
    opener: Callable[..., Any] = urlopen,
    pause: Callable[[float], None] = time.sleep,
    interval_seconds: float = 1.0,
    limit: int | None = None,
) -> dict[str, int]:
    """Drain uncached scope deferrals through a rate-limited supervisor queue.

    Only normalized queries, status, titles, and AON URLs are persisted. A
    no-match or any transport/parsing failure is merely corroborative absence
    and cannot reject a section.
    """
    path = Path(workspace).expanduser().resolve()
    _ensure_workspace_migrated(path)
    with _connect(path, readonly=True) as conn:
        headings = [
            str(row[0]) for row in conn.execute(
                f"""SELECT DISTINCT s.heading FROM draft_screening_current AS d
                   JOIN source_sections AS s ON s.section_key=d.section_key
                   JOIN parser_runs AS p ON p.parser_run_id=d.parser_run_id
                   WHERE p.state='active' AND p.review_enabled=1
                     AND {_semantic_scope_sql('p.product_code')}
                     AND d.decision='DEFER' AND d.defer_reason='scope'
                   ORDER BY s.heading"""
            )
        ]
        cached = {
            str(row[0]) for row in conn.execute("SELECT normalized_query FROM aon_cache")
        }
    queries = []
    for heading in headings:
        normalized = _normalized_query(heading)
        if normalized and normalized not in cached and normalized not in {item[0] for item in queries}:
            queries.append((normalized, heading))
    if limit is not None:
        queries = queries[:limit]
    counts = {"match": 0, "no-match": 0, "inconclusive": 0, "cached": len(cached)}
    for index, (normalized, heading) in enumerate(queries):
        status = "inconclusive"
        results: list[dict[str, str]] = []
        try:
            request = Request(
                f"{_AON_ROOT}Search.aspx?q={quote_plus(heading)}",
                headers={"User-Agent": "pf2e-codex-licensed-review/1"},
            )
            with opener(request, timeout=30) as response:
                final_url = str(response.geturl())
                parsed_url = urlparse(final_url)
                if parsed_url.scheme != "https" or parsed_url.hostname not in _AON_HOSTS:
                    raise ValueError("AON search redirected outside the approved domain")
                body = response.read(2 * 1024 * 1024 + 1)
            if len(body) > 2 * 1024 * 1024:
                raise ValueError("AON search response exceeded the bounded body limit")
            parser = _AONResultParser()
            parser.feed(body.decode("utf-8", errors="replace"))
            results = sorted(parser.results, key=lambda item: (item["title"].casefold(), item["url"]))[:10]
            status = "match" if results else "no-match"
        except (HTTPError, URLError, OSError, ValueError):
            status = "inconclusive"
            results = []
        with _connect(path) as conn:
            conn.execute(
                """INSERT INTO aon_cache VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(query_digest) DO UPDATE SET
                     normalized_query=excluded.normalized_query, status=excluded.status,
                     results_json=excluded.results_json, checked_at=excluded.checked_at""",
                (_digest("aon-query-v1", normalized), normalized, status, _canonical(results), int(time.time())),
            )
            conn.commit()
        counts[status] += 1
        if index + 1 < len(queries) and interval_seconds > 0:
            pause(interval_seconds)
    return counts


def reopen_screening(
    workspace: Path | str,
    section_key: str,
    *,
    reason: str,
    maintainer: str = "maintainer",
) -> dict[str, object]:
    """Explicitly reopen one screening result for maintainer re-review."""
    return reopen_draft_screening(
        workspace,
        section_key,
        maintainer=maintainer,
        reason=reason,
    )


def runner_status(workspace: Path | str) -> dict[str, Any]:
    path = Path(workspace).expanduser().resolve()
    _ensure_workspace_migrated(path)
    base = workspace_status(path)
    screen = draft_screening_status(path)
    with _connect(path, readonly=True) as conn:
        sessions = [
            {
                "queue": row["queue_name"], "slot": int(row["slot"]), "model": row["model"],
                "has_thread": row["thread_id"] is not None,
                "completed_batches": int(row["completed_batches"]),
                "evidence_bytes": int(row["submitted_evidence_bytes"]),
            }
            for row in conn.execute(
                "SELECT * FROM runner_sessions ORDER BY queue_name, slot"
            )
        ]
        attempts = {
            str(row["status"]): int(row["count"])
            for row in conn.execute(
                "SELECT status, COUNT(*) AS count FROM runner_attempts GROUP BY status ORDER BY status"
            )
        }
        attempts_by_queue = [
            {
                "queue": str(row["queue_name"]),
                "model": str(row["model"]),
                "status": str(row["status"]),
                "error_kind": str(row["error_kind"] or "") or None,
                "attempts": int(row["attempts"]),
                "retries": int(row["retries"]),
            }
            for row in conn.execute(
                """SELECT queue_name, model, status, error_kind, COUNT(*) AS attempts,
                          SUM(CASE WHEN attempt > 1 THEN 1 ELSE 0 END) AS retries
                   FROM runner_attempts
                   GROUP BY queue_name, model, status, error_kind
                   ORDER BY queue_name, model, status, error_kind"""
            )
        ]
        stitches = {
            "candidates": int(conn.execute(
                f"""SELECT COUNT(*) FROM stitch_candidates AS c JOIN parser_runs AS p
                   ON p.parser_run_id=c.parser_run_id
                   WHERE p.state='active' AND p.review_enabled=1
                     AND {_semantic_scope_sql('p.product_code')}"""
            ).fetchone()[0]),
            "selector_pending": int(conn.execute(
                f"""SELECT COUNT(*) FROM stitch_candidates AS c JOIN parser_runs AS p
                   ON p.parser_run_id=c.parser_run_id
                   WHERE p.state='active' AND p.review_enabled=1
                     AND {_semantic_scope_sql('p.product_code')} AND NOT EXISTS
                   (SELECT 1 FROM stitch_votes WHERE candidate_id=c.candidate_id AND role='selector')"""
            ).fetchone()[0]),
            "confirmer_pending": int(conn.execute(
                f"""SELECT COUNT(*) FROM stitch_candidates AS c JOIN stitch_votes AS s
                   ON s.candidate_id=c.candidate_id AND s.role='selector' AND s.decision='merge'
                   JOIN parser_runs AS p ON p.parser_run_id=c.parser_run_id
                   WHERE p.state='active' AND p.review_enabled=1
                     AND {_semantic_scope_sql('p.product_code')} AND NOT EXISTS
                   (SELECT 1 FROM stitch_votes WHERE candidate_id=c.candidate_id AND role='confirmer')"""
            ).fetchone()[0]),
            "live_claims": int(conn.execute(
                "SELECT COUNT(*) FROM stitch_claims WHERE lease_expires_at>=?",
                (int(time.time()),),
            ).fetchone()[0]),
        }
        maintenance_predicate = f"""maintenance.resolved_at IS NULL AND (
            EXISTS (SELECT 1 FROM source_sections AS section
                    WHERE section.section_key=maintenance.subject_id
                      AND {_semantic_scope_sql('section.product_code')})
            OR EXISTS (SELECT 1 FROM candidates AS candidate
                       JOIN source_sections AS section
                         ON section.section_key=candidate.section_key
                       WHERE candidate.candidate_id=maintenance.subject_id
                         AND {_semantic_scope_sql('section.product_code')})
            OR EXISTS (SELECT 1 FROM stitch_candidates AS stitch
                       WHERE stitch.candidate_id=maintenance.subject_id
                         AND {_semantic_scope_sql('stitch.product_code')})
        )"""
        maintenance = int(conn.execute(
            "SELECT COUNT(*) FROM runner_maintenance AS maintenance WHERE "
            + maintenance_predicate
        ).fetchone()[0])
        maintenance_items = [
            {
                "maintenance_id": str(row["maintenance_id"]),
                "queue": str(row["queue_name"]),
                "subject_id": str(row["subject_id"]),
                "reason": str(row["reason"]),
            }
            for row in conn.execute(
                """SELECT item_id AS maintenance_id, queue_name, subject_id, reason
                   FROM runner_maintenance AS maintenance WHERE """
                + maintenance_predicate
                + """ ORDER BY queue_name, item_id"""
            )
        ]
        aon = {
            str(row["status"]): int(row["count"])
            for row in conn.execute(
                "SELECT status, COUNT(*) AS count FROM aon_cache GROUP BY status ORDER BY status"
            )
        }
        active_snapshot = conn.execute(
            "SELECT value FROM metadata WHERE key='active_foundry_snapshot'"
        ).fetchone()
        snapshot_digest = str(active_snapshot[0]) if active_snapshot else None
        coverage = {
            "normalizer_version": NORMALIZER_VERSION,
            "snapshot_digest": snapshot_digest,
            "canonical_sections": int(
                conn.execute("SELECT COUNT(*) FROM duplicate_groups").fetchone()[0]
            ),
            "shadow_duplicates": int(
                conn.execute(
                    "SELECT COUNT(*) FROM duplicate_group_members WHERE source_ordinal>0"
                ).fetchone()[0]
            ),
            "foundry_candidates": int(
                conn.execute(
                    "SELECT COUNT(*) FROM foundry_coverage_candidates WHERE snapshot_digest=?",
                    (snapshot_digest,),
                ).fetchone()[0]
            )
            if snapshot_digest
            else 0,
            "confirmed_foundry": int(
                conn.execute(
                    "SELECT COUNT(*) FROM foundry_coverage_confirmations WHERE snapshot_digest=?",
                    (snapshot_digest,),
                ).fetchone()[0]
            )
            if snapshot_digest
            else 0,
            "qwen_triage": int(
                conn.execute(
                    """SELECT COUNT(*) FROM foundry_coverage_votes
                    WHERE snapshot_digest=? AND role='qwen-triage'""",
                    (snapshot_digest,),
                ).fetchone()[0]
            )
            if snapshot_digest
            else 0,
            "sol_confirmations": int(
                conn.execute(
                    """SELECT COUNT(*) FROM foundry_coverage_votes
                    WHERE snapshot_digest=? AND role='sol-confirm'""",
                    (snapshot_digest,),
                ).fetchone()[0]
            )
            if snapshot_digest
            else 0,
            "stale_foundry": int(
                conn.execute(
                    "SELECT COUNT(*) FROM foundry_coverage_confirmations WHERE snapshot_digest<>?",
                    (snapshot_digest,),
                ).fetchone()[0]
            )
            if snapshot_digest
            else int(
                conn.execute("SELECT COUNT(*) FROM foundry_coverage_confirmations").fetchone()[0]
            ),
        }
        coverage["semantic_scope"] = {
            "canonical_sections": int(conn.execute(
                f"""SELECT COUNT(*) FROM duplicate_groups AS groups_
                    JOIN source_sections AS section
                      ON section.section_key=groups_.canonical_section_key
                    WHERE {_semantic_scope_sql('section.product_code')}"""
            ).fetchone()[0]),
            "shadow_duplicates": int(conn.execute(
                f"""SELECT COUNT(*) FROM duplicate_group_members AS member
                    JOIN source_sections AS section ON section.section_key=member.section_key
                    WHERE member.source_ordinal>0
                      AND {_semantic_scope_sql('section.product_code')}"""
            ).fetchone()[0]),
            "foundry_candidates": int(conn.execute(
                f"""SELECT COUNT(*) FROM foundry_coverage_candidates AS candidate
                    JOIN source_sections AS section ON section.section_key=candidate.section_key
                    WHERE candidate.snapshot_digest=?
                      AND {_semantic_scope_sql('section.product_code')}""",
                (snapshot_digest,),
            ).fetchone()[0]) if snapshot_digest else 0,
            "confirmed_foundry": int(conn.execute(
                f"""SELECT COUNT(*) FROM foundry_coverage_confirmations AS confirmation
                    JOIN source_sections AS section ON section.section_key=confirmation.section_key
                    WHERE confirmation.snapshot_digest=?
                      AND {_semantic_scope_sql('section.product_code')}""",
                (snapshot_digest,),
            ).fetchone()[0]) if snapshot_digest else 0,
        }
        intentional_quarantine = {
            "page-number", "repeated-furniture", "contents-index", "credits-legal"
        }
        quarantine_rows = conn.execute(
            """SELECT q.reason, COUNT(*) AS records, SUM(q.anchor_count) AS anchors
                 FROM parser_quarantine AS q JOIN parser_runs AS p
                   ON p.parser_run_id=q.parser_run_id
                WHERE p.state='active' AND p.review_enabled=1
                GROUP BY q.reason ORDER BY q.reason"""
        ).fetchall()
        scoped_quarantine_rows = conn.execute(
            f"""SELECT q.reason, COUNT(*) AS records, SUM(q.anchor_count) AS anchors
                 FROM parser_quarantine AS q JOIN parser_runs AS p
                   ON p.parser_run_id=q.parser_run_id
                WHERE p.state='active' AND p.review_enabled=1
                  AND {_semantic_scope_sql('p.product_code')}
                GROUP BY q.reason ORDER BY q.reason"""
        ).fetchall()
        quarantine = {
            "intentional_records": sum(
                int(row["records"]) for row in quarantine_rows
                if str(row["reason"]) in intentional_quarantine
            ),
            "unresolved_rule_records": sum(
                int(row["records"]) for row in quarantine_rows
                if str(row["reason"]) not in intentional_quarantine
            ),
            "unresolved_rule_anchors": sum(
                int(row["anchors"] or 0) for row in quarantine_rows
                if str(row["reason"]) not in intentional_quarantine
            ),
            "by_reason": {
                str(row["reason"]): {
                    "records": int(row["records"]), "anchors": int(row["anchors"] or 0)
                }
                for row in quarantine_rows
            },
            "semantic_scope": {
                "intentional_records": sum(
                    int(row["records"]) for row in scoped_quarantine_rows
                    if str(row["reason"]) in intentional_quarantine
                ),
                "unresolved_rule_records": sum(
                    int(row["records"]) for row in scoped_quarantine_rows
                    if str(row["reason"]) not in intentional_quarantine
                ),
                "unresolved_rule_anchors": sum(
                    int(row["anchors"] or 0) for row in scoped_quarantine_rows
                    if str(row["reason"]) not in intentional_quarantine
                ),
            },
        }
    return {
        "workspace_scope": "private-review-runner",
        "semantic_scope": review_product_scope(path),
        "review": base,
        "screening": screen,
        "stitches": stitches,
        "sessions": sessions,
        "attempts": attempts,
        "attempts_by_queue": attempts_by_queue,
        "aon_cache": aon,
        "coverage": coverage,
        "quarantine": quarantine,
        "needs_maintainer": maintenance,
        "maintainer_items": maintenance_items,
    }


def _candidate_records(workspace: Path, shard_id: int, worker: str) -> list[dict[str, Any]]:
    records = read_claimed_shard(workspace, shard_id, worker)
    with _connect(workspace, readonly=True) as conn:
        values = []
        for record in records:
            terminal = conn.execute(
                """SELECT 1
                     WHERE EXISTS (
                       SELECT 1 FROM duplicate_group_members AS member
                       WHERE member.section_key=? AND member.source_ordinal>0
                     ) OR EXISTS (
                       SELECT 1 FROM foundry_coverage_confirmations AS coverage
                       JOIN metadata AS active
                         ON active.key='active_foundry_snapshot'
                        AND active.value=coverage.snapshot_digest
                       WHERE coverage.section_key=?
                     )""",
                (record["section_key"], record["section_key"]),
            ).fetchone()
            if terminal is not None:
                continue
            exists = conn.execute("SELECT 1 FROM candidates WHERE section_key=?", (record["section_key"],)).fetchone()
            if exists is not None:
                continue
            screen = conn.execute(
                """SELECT d.decision, d.defer_reason, rejection.reason AS reject_reason
                   FROM draft_screening_current AS d
                   JOIN source_sections AS s ON s.section_key=d.section_key
                   LEFT JOIN runner_screen_rejections AS rejection
                     ON rejection.section_key=d.section_key
                   WHERE d.section_key=? AND d.parser_run_id=s.parser_run_id""",
                (record["section_key"],),
            ).fetchone()
            if screen is None or screen["decision"] == "DEFER":
                raise ValueError("candidate work requires a terminal screen decision")
            values.append({
                **record,
                "screen_decision": screen["decision"],
                "screen_reject_reason": screen["reject_reason"],
            })
        return values


def _screen_records_for_shard(
    workspace: Path,
    shard_id: int,
    foundry_db: Path,
    *,
    mode: str = "ordinary",
    prepared_coverage: Mapping[str, list[dict[str, object]]] | None = None,
) -> list[dict[str, Any]]:
    """Serialize exactly the eligible records from one screening shard."""
    if mode not in {"ordinary", "coverage-confirm"}:
        raise ValueError("unknown screening serialization mode")
    with _connect(workspace, readonly=True) as conn:
        all_rows = conn.execute(
            """SELECT s.section_key AS id, s.heading, s.source_text AS text,
                      s.page_start, s.page_end, s.layout_flags, r.era AS rules_era,
                      d.decision AS current_decision, d.defer_reason
               FROM source_sections AS s
               JOIN source_revisions AS r USING(product_code, content_fingerprint)
               LEFT JOIN draft_screening_current AS d
                 ON d.parser_run_id=s.parser_run_id AND d.section_key=s.section_key
               WHERE s.shard_id=?
               ORDER BY s.source_section_id""",
            (shard_id,),
        ).fetchall()
    records = [
        {
            **{key: value for key, value in dict(row).items() if key != "current_decision"},
            "_index": index,
            "layout_flags": json.loads(str(row["layout_flags"] or "[]")),
        }
        for index, row in enumerate(all_rows)
        if (
            row["current_decision"] is None
            if mode == "ordinary"
            else row["current_decision"] == "DEFER" and row["defer_reason"] == "complex-rule"
        )
    ]
    coverage = prepared_coverage
    if coverage is None:
        coverage = foundry_coverage_evidence(
            workspace, foundry_db, [str(record["id"]) for record in records]
        )
    for record in records:
        record["foundry_candidates"] = coverage.get(str(record["id"]), [])
    return records


def _validate_coverage_gate_result(result: Mapping[str, Any], record: Mapping[str, Any]) -> None:
    status = str(result["input_status"])
    coverage = str(result["coverage"])
    foundry_ids = [str(value) for value in result["foundry_ids"]]
    supplied = {str(item["id"]) for item in record.get("foundry_candidates", [])}
    if not set(foundry_ids).issubset(supplied):
        raise ValueError("coverage result selected an unsupplied Foundry candidate")
    if status != "valid":
        if coverage != "not-applicable" or foundry_ids:
            raise ValueError("invalid input must use not-applicable coverage without IDs")
        return
    if coverage == "not-applicable":
        raise ValueError("valid input requires a coverage judgment")
    if coverage == "covered" and not foundry_ids:
        raise ValueError("covered input requires at least one supplied Foundry ID")
    if coverage != "covered" and foundry_ids:
        raise ValueError("only covered input may select Foundry IDs")


def prepare_review_data(
    workspace: Path | str,
    foundry_database: Path | str,
) -> dict[str, object]:
    """Run only deterministic duplicate and Foundry evidence preparation."""
    path = Path(workspace).expanduser().resolve()
    foundry = Path(foundry_database).expanduser().resolve()
    return prepare_deterministic_review(path, foundry)


def preview_screen_batches(
    workspace: Path | str,
    foundry_database: Path | str,
) -> dict[str, object]:
    """Read-only preview of the exact compact local-Qwen gate envelopes."""
    path = Path(workspace).expanduser().resolve()
    foundry = Path(foundry_database).expanduser().resolve()
    snapshot = load_clean_foundry(foundry)
    with _connect(path, readonly=True) as conn:
        schema = conn.execute(
            "SELECT value FROM metadata WHERE key='review_schema_version'"
        ).fetchone()
        if schema is None or str(schema[0]) != str(REVIEW_SCHEMA_VERSION):
            raise ValueError("screen preview requires the current review schema")
        active_snapshot = conn.execute(
            "SELECT value FROM metadata WHERE key='active_foundry_snapshot'"
        ).fetchone()
        if active_snapshot is None or str(active_snapshot[0]) != snapshot.digest:
            raise ValueError("screen preview requires prepared current Foundry evidence")
        scope_rows = _review_scope_rows(conn)
        scope_products = [
            {
                "product_code": str(row["product_code"]),
                "state": "enabled" if int(row["enabled"]) else "held",
                "reason": str(row["reason"]),
            }
            for row in scope_rows
        ]
        scope = {
            "version": REVIEW_SCOPE_VERSION,
            "digest": _digest(REVIEW_SCOPE_VERSION, _canonical(scope_products)),
            "enabled_products": [
                str(row["product_code"]) for row in scope_rows if int(row["enabled"])
            ],
            "held_products": [
                str(row["product_code"]) for row in scope_rows if not int(row["enabled"])
            ],
            "products": scope_products,
        }
        now = int(time.time())
        live_claims = int(conn.execute(
            """SELECT
                (SELECT COUNT(*) FROM review_shards
                  WHERE claimant IS NOT NULL AND lease_expires_at>=?)
              + (SELECT COUNT(*) FROM review_claims WHERE lease_expires_at>=?)
              + (SELECT COUNT(*) FROM draft_screening_claims WHERE lease_expires_at>=?)
              + (SELECT COUNT(*) FROM stitch_claims WHERE lease_expires_at>=?)""",
            (now, now, now, now),
        ).fetchone()[0])
        pending_layout = int(conn.execute(
            f"""SELECT COUNT(*) FROM stitch_candidates AS candidate
                JOIN parser_runs AS run ON run.parser_run_id=candidate.parser_run_id
                WHERE run.state='active' AND run.review_enabled=1
                  AND {_semantic_scope_sql('run.product_code')}
                  AND NOT EXISTS (
                    SELECT 1 FROM stitch_votes AS vote
                    WHERE vote.candidate_id=candidate.candidate_id
                      AND vote.role='selector'
                  )"""
        ).fetchone()[0]) + int(conn.execute(
            f"""SELECT COUNT(*) FROM stitch_candidates AS candidate
                JOIN stitch_votes AS selector
                  ON selector.candidate_id=candidate.candidate_id
                 AND selector.role='selector' AND selector.decision='merge'
                JOIN parser_runs AS run ON run.parser_run_id=candidate.parser_run_id
                WHERE run.state='active' AND run.review_enabled=1
                  AND {_semantic_scope_sql('run.product_code')}
                  AND NOT EXISTS (
                    SELECT 1 FROM stitch_votes AS vote
                    WHERE vote.candidate_id=candidate.candidate_id
                      AND vote.role='confirmer'
                  )"""
        ).fetchone()[0])
        intentional = ("page-number", "repeated-furniture", "contents-index", "credits-legal")
        placeholders = ",".join("?" for _ in intentional)
        unresolved_quarantine = int(conn.execute(
            f"""SELECT COUNT(*) FROM parser_quarantine AS quarantine
                JOIN parser_runs AS run ON run.parser_run_id=quarantine.parser_run_id
                WHERE run.state='active' AND run.review_enabled=1
                  AND {_semantic_scope_sql('run.product_code')}
                  AND quarantine.reason NOT IN ({placeholders})""",
            intentional,
        ).fetchone()[0])
        maintainer_items = int(conn.execute(
            f"""SELECT COUNT(*) FROM runner_maintenance AS maintenance
                WHERE maintenance.resolved_at IS NULL AND (
                  EXISTS (SELECT 1 FROM source_sections AS section
                          WHERE section.section_key=maintenance.subject_id
                            AND {_semantic_scope_sql('section.product_code')})
                  OR EXISTS (SELECT 1 FROM candidates AS candidate
                             JOIN source_sections AS section
                               ON section.section_key=candidate.section_key
                             WHERE candidate.candidate_id=maintenance.subject_id
                               AND {_semantic_scope_sql('section.product_code')})
                  OR EXISTS (SELECT 1 FROM stitch_candidates AS stitch
                             WHERE stitch.candidate_id=maintenance.subject_id
                               AND {_semantic_scope_sql('stitch.product_code')})
                )"""
        ).fetchone()[0])
        shards = [
            (int(row["shard_id"]), str(row["product_code"]))
            for row in conn.execute(
                f"""SELECT shard.shard_id, run.product_code
                    FROM review_shards AS shard
                    JOIN parser_runs AS run ON run.parser_run_id=shard.parser_run_id
                    WHERE run.state='active' AND run.review_enabled=1
                      AND {_semantic_scope_sql('run.product_code')}
                      AND EXISTS (
                        SELECT 1 FROM source_sections AS section
                        WHERE section.shard_id=shard.shard_id
                          AND NOT EXISTS (
                            SELECT 1 FROM draft_screening_current AS current
                            WHERE current.parser_run_id=run.parser_run_id
                              AND current.section_key=section.section_key
                          )
                      )
                    ORDER BY run.product_code, shard.shard_ordinal, shard.shard_id"""
            )
        ]
        eligible_ids = [
            str(row[0])
            for row in conn.execute(
                f"""SELECT section.section_key
                    FROM source_sections AS section
                    JOIN parser_runs AS run ON run.parser_run_id=section.parser_run_id
                    WHERE run.state='active' AND run.review_enabled=1
                      AND {_semantic_scope_sql('run.product_code')}
                      AND NOT EXISTS (
                        SELECT 1 FROM draft_screening_current AS current
                        WHERE current.parser_run_id=run.parser_run_id
                          AND current.section_key=section.section_key
                      )
                    ORDER BY run.product_code, section.source_section_id"""
            )
        ]
    blockers = []
    if live_claims:
        blockers.append("live-claims")
    if pending_layout:
        blockers.append("layout-review-required")
    if unresolved_quarantine:
        blockers.append("unresolved-rule-quarantine")
    if maintainer_items:
        blockers.append("needs-maintainer")
    coverage = _foundry_coverage_evidence_for_snapshot(path, snapshot, eligible_ids)
    by_product: dict[str, dict[str, object]] = {
        product: {
            "product_code": product,
            "eligible_records": 0,
            "batches": 0,
            "evidence_bytes": 0,
            "min_batch_bytes": None,
            "max_batch_bytes": 0,
            "foundry_candidate_records": 0,
            "exact_foundry_candidate_records": 0,
            "lexical_foundry_candidate_records": 0,
            "batch_digests": [],
        }
        for product in scope["enabled_products"]
    }
    for shard_id, product in shards:
        records = _screen_records_for_shard(
            path, shard_id, foundry, prepared_coverage=coverage,
        )
        prompt_records = [
            {key: value for key, value in record.items() if key != "_index"}
            for record in records
        ]
        product_result = by_product[product]
        product_result["eligible_records"] = int(product_result["eligible_records"]) + len(records)
        for record in records:
            candidates = list(record.get("foundry_candidates", []))
            if candidates:
                product_result["foundry_candidate_records"] = int(
                    product_result["foundry_candidate_records"]
                ) + 1
                if any(bool(item.get("metrics", {}).get("exact_identity")) for item in candidates):
                    product_result["exact_foundry_candidate_records"] = int(
                        product_result["exact_foundry_candidate_records"]
                    ) + 1
                else:
                    product_result["lexical_foundry_candidate_records"] = int(
                        product_result["lexical_foundry_candidate_records"]
                    ) + 1
        for batch in _prompt_batches("screen", prompt_records):
            prompt = _worker_prompt("screen", batch)
            size = len(prompt.encode("utf-8"))
            product_result["batches"] = int(product_result["batches"]) + 1
            product_result["evidence_bytes"] = int(product_result["evidence_bytes"]) + size
            minimum = product_result["min_batch_bytes"]
            product_result["min_batch_bytes"] = size if minimum is None else min(int(minimum), size)
            product_result["max_batch_bytes"] = max(int(product_result["max_batch_bytes"]), size)
            product_result["batch_digests"].append(_digest("screen-preview-v1", prompt))
    products = []
    manifest_digests = []
    for product in sorted(by_product):
        result = by_product[product]
        batch_digests = list(result.pop("batch_digests"))
        result["batch_digest"] = _digest("screen-product-preview-v1", *batch_digests)
        manifest_digests.extend(batch_digests)
        products.append(result)
    return {
        "preview_version": "screen-batch-preview-v1",
        "ready": not blockers,
        "blockers": blockers,
        "model": MODEL_BY_QUEUE["screen"],
        "limits": {"records": MAX_BATCH_RECORDS, "bytes": MAX_BATCH_BYTES},
        "foundry_release": snapshot.release,
        "foundry_snapshot_digest": snapshot.digest,
        "scope": scope,
        "products": products,
        "eligible_records": sum(int(item["eligible_records"]) for item in products),
        "batches": sum(int(item["batches"]) for item in products),
        "evidence_digest": _digest("screen-preview-v1", *manifest_digests),
    }


def _process_screen(
    workspace: Path,
    slot: int,
    foundry_db: Path | None,
    executor: CodexExecutor,
    *,
    product_code: str | None = None,
    pilot: bool = False,
) -> bool:
    queue = "screen"
    worker = f"{queue}:{slot}" + (f":{product_code}" if product_code else "")
    claim = claim_draft_screening_batch(
        workspace, worker, lease_seconds=LEASE_SECONDS,
        product_code=product_code,
        queue="unprocessed",
    )
    if claim is None:
        return False
    shard_id = int(claim["shard_id"])
    if foundry_db is None:
        raise ValueError("screening requires --foundry-database")
    records = _screen_records_for_shard(
        workspace,
        shard_id,
        foundry_db,
    )
    try:
        if not records:
            release_draft_screening_batch(workspace, shard_id, worker)
            return False
        if pilot:
            selected_prompt = _prompt_batches(queue, [
                {key: value for key, value in record.items() if key != "_index"}
                for record in records
            ])[0]
            selected_ids = {str(record["id"]) for record in selected_prompt}
            records = [record for record in records if str(record["id"]) in selected_ids]
            prompt_records = selected_prompt
        else:
            prompt_records = [
                {key: value for key, value in record.items() if key != "_index"}
                for record in records
            ]
        results = _run_packed(
            workspace,
            queue=queue,
            slot=slot,
            records=prompt_records,
            foundry_db=foundry_db,
            executor=executor,
            schema_key="coverage-gate",
        )
        valid_result_ids: set[str] = set()
        for result in results:
            source_record = next(row for row in records if row["id"] == result["id"])
            try:
                _validate_coverage_gate_result(result, source_record)
            except ValueError:
                continue
            valid_result_ids.add(str(result["id"]))
        for result in results:
            source_record = next(row for row in records if row["id"] == result["id"])
            valid_result = str(result["id"]) in valid_result_ids
            if (
                valid_result
                and result["input_status"] == "valid"
                and result["coverage"] == "covered"
            ):
                decision, defer_reason = "defer", "complex-rule"
            else:
                decision, defer_reason = "add", None
            submit_draft_screening_decision(
                workspace, shard_id, worker,
                next(int(row["_index"]) for row in records if row["id"] == result["id"]),
                decision,
                defer_reason=defer_reason,
                coverage_vote=(
                    {
                        "role": "qwen-triage",
                        "input_status": result["input_status"],
                        "coverage": result["coverage"],
                        "issue_tags": result["issue_tags"],
                        "foundry_ids": result["foundry_ids"],
                        "prompt_version": PROMPT_VERSION,
                    }
                    if valid_result
                    else None
                ),
            )
        release_draft_screening_batch(workspace, shard_id, worker)
        return True
    except BaseException:
        release_draft_screening_batch(workspace, shard_id, worker)
        raise


def _process_coverage_confirm(
    workspace: Path,
    slot: int,
    foundry_db: Path | None,
    executor: CodexExecutor,
) -> bool:
    if foundry_db is None:
        raise ValueError("coverage confirmation requires --foundry-database")
    queue = "coverage-confirm"
    worker = f"{queue}:{slot}"
    with _connect(workspace, readonly=True) as conn:
        row = conn.execute(
            f"""SELECT section.shard_id
                  FROM source_sections AS section
                  JOIN parser_runs AS run ON run.parser_run_id=section.parser_run_id
                  JOIN draft_screening_current AS screen
                    ON screen.parser_run_id=section.parser_run_id
                   AND screen.section_key=section.section_key
                  JOIN metadata AS active ON active.key='active_foundry_snapshot'
                  JOIN foundry_coverage_votes AS vote
                    ON vote.section_key=section.section_key
                   AND vote.snapshot_digest=active.value
                   AND vote.role='qwen-triage'
                   AND vote.input_status='valid' AND vote.coverage='covered'
                 WHERE run.state='active' AND run.review_enabled=1
                   AND {_semantic_scope_sql("run.product_code")}
                   AND screen.decision='DEFER' AND screen.defer_reason='complex-rule'
                 ORDER BY run.product_code, section.shard_id LIMIT 1"""
        ).fetchone()
    if row is None:
        return False
    shard_id = int(row[0])
    claim = claim_draft_screening_batch(
        workspace,
        worker,
        lease_seconds=LEASE_SECONDS,
        preferred_shard_id=shard_id,
        queue="deferred",
    )
    if claim is None:
        return False
    try:
        all_records = _screen_records_for_shard(
            workspace,
            shard_id,
            foundry_db,
            mode="coverage-confirm",
        )
        with _connect(workspace, readonly=True) as conn:
            snapshot = conn.execute(
                "SELECT value FROM metadata WHERE key='active_foundry_snapshot'"
            ).fetchone()
            eligible = {
                str(item[0]): dict(item)
                for item in conn.execute(
                    """SELECT vote.section_key, vote.coverage AS qwen_coverage,
                              vote.foundry_ids_json AS qwen_foundry_ids
                         FROM foundry_coverage_votes AS vote
                         JOIN draft_screening_current AS screen
                           ON screen.section_key=vote.section_key
                        WHERE vote.snapshot_digest=? AND vote.role='qwen-triage'
                          AND vote.input_status='valid' AND vote.coverage='covered'
                          AND screen.decision='DEFER'
                          AND screen.defer_reason='complex-rule'""",
                    (snapshot[0],),
                )
            }
        records = [record for record in all_records if str(record["id"]) in eligible]
        if not records:
            release_draft_screening_batch(workspace, shard_id, worker)
            return False
        prompt_records = [
            {key: value for key, value in record.items() if key not in {"_index", "defer_reason"}}
            for record in records
        ]
        results = _run_packed(
            workspace,
            queue=queue,
            slot=slot,
            records=prompt_records,
            foundry_db=foundry_db,
            executor=executor,
            schema_key="coverage-gate",
        )
        valid_result_ids: set[str] = set()
        for result in results:
            source_record = next(record for record in records if record["id"] == result["id"])
            try:
                _validate_coverage_gate_result(result, source_record)
            except ValueError:
                continue
            valid_result_ids.add(str(result["id"]))
        for result in results:
            section_key = str(result["id"])
            source_record = next(record for record in records if record["id"] == section_key)
            qwen_coverage = str(eligible[section_key]["qwen_coverage"])
            qwen_foundry_ids = sorted(
                str(value)
                for value in json.loads(str(eligible[section_key]["qwen_foundry_ids"]))
            )
            suppress = (
                section_key in valid_result_ids
                and qwen_coverage == "covered"
                and result["input_status"] == "valid"
                and result["coverage"] == "covered"
                and sorted(str(value) for value in result["foundry_ids"])
                == qwen_foundry_ids
            )
            submit_draft_screening_decision(
                workspace,
                shard_id,
                worker,
                int(source_record["_index"]),
                "reject" if suppress else "add",
                reject_reason="duplicate" if suppress else None,
                foundry_ids=tuple(str(value) for value in result["foundry_ids"])
                if suppress
                else (),
                coverage_vote=(
                    {
                        "role": "sol-confirm",
                        "input_status": result["input_status"],
                        "coverage": result["coverage"],
                        "issue_tags": result["issue_tags"],
                        "foundry_ids": result["foundry_ids"],
                        "prompt_version": PROMPT_VERSION,
                    }
                    if section_key in valid_result_ids
                    else None
                ),
            )
        release_draft_screening_batch(workspace, shard_id, worker)
        return True
    except BaseException:
        release_draft_screening_batch(workspace, shard_id, worker)
        raise


def _submit_decision(
    workspace: Path,
    record: Mapping[str, Any],
    result: Mapping[str, Any],
    worker: str,
) -> None:
    decision = str(result["decision"])
    submission: dict[str, Any] = {
        "section_key": record["section_key"],
        "source_section_id": record["source_section_id"],
        "source_section_hash": record["source_section_hash"],
        "decision": decision,
        "worker": worker,
        "prompt_version": PROMPT_VERSION,
        "reason_tags": result.get("reason_tags") or ["deterministic-screen-exclusion"],
        "confidence": float(result.get("confidence", 1.0)),
    }
    if decision == "PUBLIC_AS_IS":
        submission.update({
            "candidate_text": record["source_text"],
            "public_heading": record["heading"],
            "extraction_method": "verbatim-reviewed-v1",
        })
    elif decision == "MIXED_NEEDS_EXTRACTION":
        submission.update({
            "candidate_text": result["text"],
            "public_heading": result["heading"],
            "extraction_method": "mechanics-v1",
            "reason_tags": sorted({*submission["reason_tags"], "layout-reviewed"}),
        })
    submit_candidate(workspace, submission)


def _process_candidates(workspace: Path, slot: int, foundry_db: Path | None, executor: CodexExecutor) -> bool:
    with _connect(workspace, readonly=True) as conn:
        unresolved_screen = conn.execute(
            f"""SELECT 1 FROM source_sections AS s
               JOIN parser_runs AS p ON p.parser_run_id=s.parser_run_id
               LEFT JOIN draft_screening_current AS d
                 ON d.parser_run_id=s.parser_run_id AND d.section_key=s.section_key
               WHERE p.state='active' AND p.review_enabled=1
                 AND {_semantic_scope_sql('p.product_code')}
                 AND (d.section_key IS NULL OR d.decision='DEFER') LIMIT 1"""
        ).fetchone()
    if unresolved_screen is not None:
        return False
    worker = f"producer:{slot}"
    with _connect(workspace) as conn:
        for row in conn.execute(
            "SELECT shard_id FROM review_shards WHERE claimant=?",
            (worker,),
        ):
            _release_shard_if_complete(conn, int(row["shard_id"]))
        conn.commit()
    claim = claim_shard(workspace, worker, lease_seconds=LEASE_SECONDS)
    if claim is None:
        with _connect(workspace, readonly=True) as conn:
            partial = conn.execute(
                f"""SELECT sh.shard_id FROM review_shards AS sh
                   JOIN parser_runs AS p ON p.parser_run_id=sh.parser_run_id
                   WHERE p.state='active' AND p.review_enabled=1
                     AND {_semantic_scope_sql('p.product_code')}
                     AND (sh.claimant IS NULL OR sh.lease_expires_at<?)
                     AND EXISTS(SELECT 1 FROM source_sections s JOIN candidates c ON c.section_key=s.section_key WHERE s.shard_id=sh.shard_id)
                     AND EXISTS(SELECT 1 FROM source_sections s WHERE s.shard_id=sh.shard_id AND NOT EXISTS(SELECT 1 FROM candidates c WHERE c.section_key=s.section_key))
                   ORDER BY sh.shard_id LIMIT 1""",
                (int(time.time()),),
            ).fetchone()
        if partial is None:
            return False
        claim = reclaim_interrupted_shard(workspace, int(partial[0]), worker, lease_seconds=LEASE_SECONDS)
    shard_id = int(claim["shard_id"])
    records = _candidate_records(workspace, shard_id, worker)
    if not records:
        with _connect(workspace) as conn:
            _release_shard_if_complete(conn, shard_id)
            conn.commit()
        return False
    if claim["claim_mode"] == "rework":
        with _connect(workspace, readonly=True) as conn:
            ordinal = max(
                int(conn.execute("SELECT COALESCE(MAX(candidate_ordinal),0) FROM candidates WHERE section_key=?", (record["section_key"],)).fetchone()[0])
                for record in records
            )
        if ordinal >= 3:
            with _connect(workspace) as conn:
                now = int(time.time())
                for record in records:
                    conn.execute(
                        "INSERT OR IGNORE INTO runner_maintenance VALUES (?, ?, ?, ?, ?, NULL, NULL)",
                        ("maintainer:" + _digest(record["section_key"], "rework-exhausted"), "rework", record["section_key"], "rework-exhausted", now),
                    )
                conn.commit()
            return False
        queue = "rework-terra" if ordinal == 1 else "rework-sol"
        prompt_records = [
            {"id": row["section_key"], "heading": row["heading"], "text": row["source_text"], "rules_era": PRODUCT_CATALOG[str(row["product_code"])].rules_era}
            for row in records
        ]
        results = _run_packed(
            workspace, queue=queue, slot=slot, records=prompt_records,
            foundry_db=foundry_db, executor=executor, schema_key="classify",
        )
    else:
        rejected = [record for record in records if record["screen_decision"] == "REJECT"]
        added = [record for record in records if record["screen_decision"] == "ADD"]
        results_by_id: dict[str, dict[str, Any]] = {
            str(record["section_key"]): {
                "id": record["section_key"], "decision": "EXCLUDE",
                "reason_tags": [
                    "deterministic-screen-exclusion",
                    str(record["screen_reject_reason"] or "legacy-unspecified-rejection"),
                ],
                "confidence": 1.0,
            }
            for record in rejected
        }
        if added:
            prompt_records = [
                {"id": row["section_key"], "heading": row["heading"], "text": row["source_text"], "rules_era": PRODUCT_CATALOG[str(row["product_code"])].rules_era}
                for row in added
            ]
            classified = _run_packed(
                workspace, queue="classify", slot=slot, records=prompt_records,
                foundry_db=foundry_db, executor=executor,
            )
            results_by_id.update({str(item["id"]): item for item in classified})
        results = [results_by_id[str(record["section_key"])] for record in records]
    mixed = [result for result in results if result["decision"] == "MIXED_NEEDS_EXTRACTION"]
    if mixed:
        by_id = {str(record["section_key"]): record for record in records}
        extraction_records = [
            {"id": item["id"], "heading": by_id[str(item["id"])]["heading"], "text": by_id[str(item["id"])]["source_text"], "rules_era": PRODUCT_CATALOG[str(by_id[str(item["id"])]["product_code"])].rules_era}
            for item in mixed
        ]
        extracted = _run_packed(
            workspace, queue="extract", slot=slot, records=extraction_records,
            foundry_db=foundry_db, executor=executor,
        )
        extracted_by_id = {str(item["id"]): item for item in extracted}
        results = [
            {**result, **extracted_by_id[str(result["id"])], "decision": "MIXED_NEEDS_EXTRACTION"}
            if result["decision"] == "MIXED_NEEDS_EXTRACTION" else result
            for result in results
        ]
    by_id = {str(record["section_key"]): record for record in records}
    for result in results:
        _submit_decision(workspace, by_id[str(result["id"])], result, worker)
    return True


def _process_review(workspace: Path, slot: int, foundry_db: Path | None, executor: CodexExecutor) -> bool:
    reviewer = f"reviewer:{slot}"
    claim = claim_review(workspace, reviewer, lease_seconds=LEASE_SECONDS)
    if claim is None:
        return False
    record = read_claimed_review(workspace, str(claim["candidate_id"]), reviewer)
    mixed = record["decision"] == "MIXED_NEEDS_EXTRACTION"
    queue = "review-mixed" if mixed else "review"
    prompt = [{
        "id": record["candidate_id"], "decision": record["decision"],
        "source_heading": record["heading"], "source_text": record["source_text"],
        "candidate_heading": record["public_heading"], "candidate_text": record["candidate_text"],
    }]
    result = run_codex_batch(
        workspace, queue=queue, slot=slot, records=prompt,
        foundry_db=foundry_db, executor=executor,
    )[0]
    verdict = str(result["verdict"])
    if record["decision"] in {"PUBLIC_AS_IS", "MIXED_NEEDS_EXTRACTION"} and verdict == "REJECT":
        verdict = "REVISE"
    if record["decision"] in {"EXCLUDE", "UNCERTAIN"} and verdict == "APPROVE":
        verdict = "REVISE"
    submit_review(
        workspace,
        {
            "candidate_id": record["candidate_id"], "reviewer": reviewer,
            "verdict": verdict, "policy_version": LICENSED_CORE_POLICY_VERSION,
            "reason_tags": result["reason_tags"],
        },
    )
    return True


def _stitch_records(
    workspace: Path,
    queue: str,
    worker: str,
    limit: int = MAX_BATCH_RECORDS,
    *,
    product_code: str | None = None,
) -> list[dict[str, Any]]:
    role = "selector" if queue == "stitch-select" else "confirmer"
    with _connect(workspace) as conn:
        now = int(time.time())
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM stitch_claims WHERE lease_expires_at<?", (now,))
        product_clause = " AND c.product_code=?" if product_code else ""
        parameters: tuple[object, ...] = (
            (role, product_code, limit) if product_code else (role, limit)
        )
        if queue == "stitch-select":
            rows = conn.execute(
                f"""SELECT c.* FROM stitch_candidates AS c
                   JOIN parser_runs AS p ON p.parser_run_id=c.parser_run_id
                   WHERE p.state='active' AND p.review_enabled=1
                     AND {_semantic_scope_sql('p.product_code')} AND NOT EXISTS
                    (SELECT 1 FROM stitch_votes WHERE candidate_id=c.candidate_id AND role='selector')
                   AND NOT EXISTS (SELECT 1 FROM stitch_claims
                     WHERE candidate_id=c.candidate_id AND role=?)
                   {product_clause}
                   ORDER BY c.candidate_id LIMIT ?""", parameters,
            ).fetchall()
        else:
            rows = conn.execute(
                f"""SELECT c.* FROM stitch_candidates AS c
                   JOIN parser_runs AS p ON p.parser_run_id=c.parser_run_id
                   JOIN stitch_votes AS v ON v.candidate_id=c.candidate_id
                     AND v.role='selector' AND v.decision='merge'
                   WHERE p.state='active' AND p.review_enabled=1
                     AND {_semantic_scope_sql('p.product_code')} AND NOT EXISTS
                    (SELECT 1 FROM stitch_votes WHERE candidate_id=c.candidate_id AND role='confirmer')
                   AND NOT EXISTS (SELECT 1 FROM stitch_claims
                     WHERE candidate_id=c.candidate_id AND role=?)
                   {product_clause}
                   ORDER BY c.candidate_id LIMIT ?""", parameters,
            ).fetchall()
        conn.executemany(
            "INSERT INTO stitch_claims VALUES (?, ?, ?, ?, ?)",
            (
                (row["candidate_id"], role, worker, now, now + LEASE_SECONDS)
                for row in rows
            ),
        )
        values = []
        for row in rows:
            keys = json.loads(str(row["section_keys"]))
            placeholders = ",".join("?" for _ in keys)
            sections = conn.execute(
                f"SELECT section_key, heading, source_text AS text, page_start, page_end, layout_flags FROM source_sections WHERE section_key IN ({placeholders})",
                keys,
            ).fetchall()
            by_key = {str(section["section_key"]): section for section in sections}
            values.append({
                "id": row["candidate_id"],
                "sections": [
                    {**dict(by_key[key]), "layout_flags": json.loads(str(by_key[key]["layout_flags"] or "[]"))}
                    for key in keys
                ],
                "deterministic_evidence": json.loads(str(row["evidence_json"])),
            })
        conn.commit()
        return values


def _release_stitch_claims(
    workspace: Path,
    *,
    role: str,
    worker: str,
    candidate_ids: Sequence[str],
) -> None:
    if not candidate_ids:
        return
    placeholders = ",".join("?" for _ in candidate_ids)
    with _connect(workspace) as conn:
        conn.execute(
            f"DELETE FROM stitch_claims WHERE role=? AND claimant=? AND candidate_id IN ({placeholders})",
            (role, worker, *candidate_ids),
        )
        conn.commit()


def _process_stitches(
    workspace: Path,
    queue: str,
    slot: int,
    executor: CodexExecutor,
    *,
    product_code: str | None = None,
    pilot: bool = False,
) -> bool:
    role = "selector" if queue == "stitch-select" else "confirmer"
    worker = f"{role}:{slot}" + (f":{product_code}" if product_code else "")
    records = _stitch_records(
        workspace,
        queue,
        worker,
        product_code=product_code,
    )
    if not records:
        return False
    if pilot:
        submitted = _prompt_batches(queue, records)[0]
        submitted_ids = {str(record["id"]) for record in submitted}
        _release_stitch_claims(
            workspace,
            role=role,
            worker=worker,
            candidate_ids=[
                str(record["id"])
                for record in records
                if str(record["id"]) not in submitted_ids
            ],
        )
        records = submitted
    candidate_ids = [str(record["id"]) for record in records]
    try:
        results = _run_packed(
            workspace, queue=queue, slot=slot, records=records,
            foundry_db=None, executor=executor,
        )
    except BaseException:
        _release_stitch_claims(
            workspace,
            role=role,
            worker=worker,
            candidate_ids=candidate_ids,
        )
        raise
    with _connect(workspace) as conn:
        conn.execute("BEGIN IMMEDIATE")
        for result in results:
            conn.execute(
                "INSERT INTO stitch_votes VALUES (?, ?, ?, ?, ?, ?)",
                (result["id"], role, worker, result["decision"], result["reason"], int(time.time())),
            )
        placeholders = ",".join("?" for _ in candidate_ids)
        conn.execute(
            f"DELETE FROM stitch_claims WHERE role=? AND claimant=? AND candidate_id IN ({placeholders})",
            (role, worker, *candidate_ids),
        )
        conn.commit()
    if queue == "stitch-confirm":
        _reconcile_stitch_maintenance(workspace)
    return True


def _drain(
    workspace: Path,
    queue: str,
    concurrency: int,
    foundry_db: Path | None,
    executor: CodexExecutor,
) -> int:
    model = MODEL_BY_QUEUE[queue]
    workers = max(1, min(concurrency, MODEL_CAPS[model]))
    processed = 0
    lock = threading.Lock()

    def loop(slot: int) -> None:
        nonlocal processed
        while True:
            if queue == "screen":
                did_work = _process_screen(workspace, slot, foundry_db, executor)
            elif queue == "coverage-confirm":
                did_work = _process_coverage_confirm(
                    workspace,
                    slot,
                    foundry_db,
                    executor,
                )
            elif queue == "classify":
                did_work = _process_candidates(workspace, slot, foundry_db, executor)
            elif queue == "review":
                did_work = _process_review(workspace, slot, foundry_db, executor)
            else:
                did_work = _process_stitches(workspace, queue, slot, executor)
            if not did_work:
                return
            with lock:
                processed += 1

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"licensed-{queue}") as pool:
        futures = [pool.submit(loop, slot) for slot in range(workers)]
        for future in futures:
            future.result()
    return processed


def _screening_pilot(
    workspace: Path,
    *,
    concurrency: int,
    foundry_db: Path | None,
    executor: CodexExecutor,
) -> dict[str, Any]:
    """Run exactly one compact local-Qwen gate batch for every enabled product."""
    workers = max(1, min(concurrency, MODEL_CAPS[MODEL_BY_QUEUE["screen"]]))
    pending = list(review_product_scope(workspace)["enabled_products"])
    pending_lock = threading.Lock()
    completed: list[str] = []
    completed_lock = threading.Lock()

    def loop(slot: int) -> None:
        while True:
            with pending_lock:
                if not pending:
                    return
                product_code = pending.pop(0)
            if _process_screen(
                workspace, slot, foundry_db, executor, product_code=product_code,
                pilot=True,
            ):
                with completed_lock:
                    completed.append(product_code)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="licensed-screen-pilot") as pool:
        futures = [pool.submit(loop, slot) for slot in range(workers)]
        for future in futures:
            future.result()
    return {
        "completed_products": sorted(completed),
        "completed_batches": len(completed),
        "status": runner_status(workspace),
    }


def _stitch_pilot(
    workspace: Path,
    *,
    queue: str,
    concurrency: int,
    executor: CodexExecutor,
) -> dict[str, Any]:
    """Run at most one stitch batch for every enabled product."""
    workers = max(1, min(concurrency, MODEL_CAPS[MODEL_BY_QUEUE[queue]]))
    pending = list(review_product_scope(workspace)["enabled_products"])
    pending_lock = threading.Lock()
    completed: list[str] = []
    completed_lock = threading.Lock()

    def loop(slot: int) -> None:
        while True:
            with pending_lock:
                if not pending:
                    return
                product_code = pending.pop(0)
            if _process_stitches(
                workspace, queue, slot, executor, product_code=product_code,
                pilot=True,
            ):
                with completed_lock:
                    completed.append(product_code)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"licensed-{queue}-pilot") as pool:
        futures = [pool.submit(loop, slot) for slot in range(workers)]
        for future in futures:
            future.result()
    _reconcile_stitch_maintenance(workspace)
    return {
        "completed_products": sorted(completed),
        "completed_batches": len(completed),
        "status": runner_status(workspace),
    }


def _selected_pdf_sources(sources_root: Path) -> dict[str, CorpusSource]:
    selected = select_revisions(
        discover_sources(sources_root, include=EXPECTED_PRODUCTS),
        include=EXPECTED_PRODUCTS,
        state_root=sources_root.parent / ".licensed-runner-selection",
    )
    result: dict[str, CorpusSource] = {}
    for revision in selected:
        if len(revision.sources) != 1 or revision.sources[0].member is not None or not revision.sources[0].combined:
            raise ValueError(f"{revision.product.code} needs a selected combined local PDF")
        result[revision.product.code] = revision.sources[0]
    return result


def _layout_artifact_path(sources_root: Path, source: CorpusSource) -> Path:
    """Return the ignored, source-bound layout artifact path without PII."""
    return sources_root.parent / "layout" / source.product.code / f"{source.source_sha256}.json"


def _reconcile_stitch_maintenance(workspace: Path) -> int:
    """Materialize active disagreement/overlap blockers immediately."""
    with _connect(workspace, readonly=True) as conn:
        approved_rows = conn.execute(
            f"""SELECT c.candidate_id, c.product_code, c.section_keys
               FROM stitch_candidates AS c
               JOIN parser_runs AS p ON p.parser_run_id=c.parser_run_id
               WHERE p.state='active' AND p.review_enabled=1
                 AND {_semantic_scope_sql('p.product_code')}
                 AND {_APPROVED_STITCH_PREDICATE}
               ORDER BY c.product_code, c.candidate_id"""
        ).fetchall()
        disagreements = conn.execute(
            f"""SELECT c.candidate_id FROM stitch_candidates c
               JOIN stitch_votes s ON s.candidate_id=c.candidate_id AND s.role='selector' AND s.decision='merge'
               JOIN stitch_votes t ON t.candidate_id=c.candidate_id AND t.role='confirmer' AND t.decision='no-merge'
               JOIN parser_runs p ON p.parser_run_id=c.parser_run_id
               WHERE p.state='active' AND p.review_enabled=1
                 AND {_semantic_scope_sql('p.product_code')}"""
        ).fetchall()
    seen: dict[str, set[str]] = {}
    overlaps: list[str] = []
    for row in approved_rows:
        keys = set(json.loads(str(row["section_keys"])))
        product_seen = seen.setdefault(str(row["product_code"]), set())
        if product_seen.intersection(keys):
            overlaps.append(str(row["candidate_id"]))
        product_seen.update(keys)
    with _connect(workspace) as conn:
        now = int(time.time())
        for candidate_id in overlaps:
            conn.execute(
                "INSERT OR IGNORE INTO runner_maintenance VALUES (?, 'stitch', ?, 'overlapping-approved-groups', ?, NULL, NULL)",
                ("maintainer:" + _digest(candidate_id, "stitch-overlap"), candidate_id, now),
            )
        for row in disagreements:
            conn.execute(
                "INSERT OR IGNORE INTO runner_maintenance VALUES (?, 'stitch', ?, 'independent-disagreement', ?, NULL, NULL)",
                ("maintainer:" + _digest(row[0], "stitch-disagreement"), row[0], now),
            )
        open_items = int(conn.execute(
            f"""SELECT COUNT(*) FROM runner_maintenance AS maintenance
                JOIN stitch_candidates AS stitch
                  ON stitch.candidate_id=maintenance.subject_id
                JOIN parser_runs AS run ON run.parser_run_id=stitch.parser_run_id
                WHERE maintenance.resolved_at IS NULL
                  AND run.state='active' AND run.review_enabled=1
                  AND {_semantic_scope_sql('run.product_code')}"""
        ).fetchone()[0])
        conn.commit()
    return open_items


def resolve_maintainer_item(
    workspace: Path | str,
    maintenance_id: str,
    decision: str,
) -> dict[str, str]:
    """Explicitly resolve one independent stitch disagreement."""
    if decision not in {"merge", "no-merge"}:
        raise ValueError("maintainer decision must be merge or no-merge")
    path = Path(workspace).expanduser().resolve()
    _ensure_workspace_migrated(path)
    with _connect(path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM runner_maintenance WHERE item_id=?",
            (maintenance_id,),
        ).fetchone()
        if row is None:
            raise KeyError("unknown maintainer item")
        if row["queue_name"] != "stitch" or row["reason"] != "independent-disagreement":
            raise ValueError("this maintainer item needs a specialized resolution")
        scoped = conn.execute(
            f"""SELECT 1 FROM stitch_candidates AS stitch
                JOIN parser_runs AS run ON run.parser_run_id=stitch.parser_run_id
                WHERE stitch.candidate_id=?
                  AND run.state='active' AND run.review_enabled=1
                  AND {_semantic_scope_sql('run.product_code')}""",
            (row["subject_id"],),
        ).fetchone()
        if scoped is None:
            raise ValueError("maintainer item is outside the semantic product scope")
        if row["resolved_at"] is not None:
            if row["resolution"] != decision:
                raise ValueError("maintainer item already has a different resolution")
        else:
            conn.execute(
                "UPDATE runner_maintenance SET resolved_at=?, resolution=? WHERE item_id=?",
                (int(time.time()), decision, maintenance_id),
            )
        conn.commit()
    _reconcile_stitch_maintenance(path)
    return {"maintenance_id": maintenance_id, "decision": decision}


def maintainer_item_evidence(
    workspace: Path | str,
    maintenance_id: str,
    *,
    include_text: bool = False,
) -> dict[str, Any]:
    """Read one bounded maintainer item without exposing arbitrary workspace rows."""
    path = Path(workspace).expanduser().resolve()
    _ensure_workspace_migrated(path)
    _reconcile_stitch_maintenance(path)
    with _connect(path, readonly=True) as conn:
        item = conn.execute(
            "SELECT * FROM runner_maintenance WHERE item_id=?",
            (maintenance_id,),
        ).fetchone()
        if item is None:
            raise KeyError("unknown maintainer item")
        if item["queue_name"] != "stitch" or item["reason"] != "independent-disagreement":
            raise ValueError("this maintainer item needs a specialized inspection command")
        candidate = conn.execute(
            f"""SELECT stitch.* FROM stitch_candidates AS stitch
                JOIN parser_runs AS run ON run.parser_run_id=stitch.parser_run_id
                WHERE stitch.candidate_id=?
                  AND run.state='active' AND run.review_enabled=1
                  AND {_semantic_scope_sql('run.product_code')}""",
            (item["subject_id"],),
        ).fetchone()
        if candidate is None:
            raise ValueError(
                "maintainer item is outside the semantic product scope or refers "
                "to a missing stitch candidate"
            )
        section_keys = json.loads(str(candidate["section_keys"]))
        if not isinstance(section_keys, list) or not 2 <= len(section_keys) <= 3:
            raise ValueError("maintainer stitch candidate has an invalid section set")
        sections: list[dict[str, Any]] = []
        for section_key in section_keys:
            section = conn.execute(
                """SELECT section_key, page_start, page_end, printed_page, heading,
                          source_text, layout_flags
                   FROM source_sections
                   WHERE section_key=? AND parser_run_id=?""",
                (section_key, candidate["parser_run_id"]),
            ).fetchone()
            if section is None:
                raise ValueError("maintainer stitch candidate refers to a missing section")
            record: dict[str, Any] = {
                "section_key": str(section["section_key"]),
                "page_start": section["page_start"],
                "page_end": section["page_end"],
                "printed_page": section["printed_page"],
                "heading": str(section["heading"]),
                "layout_flags": json.loads(str(section["layout_flags"])),
            }
            if include_text:
                record["source_text"] = str(section["source_text"])
            sections.append(record)
        votes = [
            {
                "role": str(row["role"]),
                "worker": str(row["worker"]),
                "decision": str(row["decision"]),
                "reason": str(row["reason"]),
            }
            for row in conn.execute(
                """SELECT role, worker, decision, reason FROM stitch_votes
                   WHERE candidate_id=? ORDER BY role""",
                (candidate["candidate_id"],),
            )
        ]
    return {
        "maintenance_id": str(item["item_id"]),
        "reason": str(item["reason"]),
        "resolved": item["resolved_at"] is not None,
        "resolution": item["resolution"],
        "candidate_id": str(candidate["candidate_id"]),
        "product_code": str(candidate["product_code"]),
        "votes": votes,
        "sections": sections,
        "private_text_included": include_text,
    }


def _apply_confirmed_stitches(workspace: Path, sources_root: Path) -> int:
    if _reconcile_stitch_maintenance(workspace):
        return 0
    with _connect(workspace, readonly=True) as conn:
        products = [
            (str(row[0]), str(row[1])) for row in conn.execute(
                f"""SELECT DISTINCT c.product_code, p.parser_version FROM stitch_candidates AS c
                   JOIN parser_runs AS p ON p.parser_run_id=c.parser_run_id
                   WHERE p.state='active' AND p.review_enabled=1
                     AND {_semantic_scope_sql('p.product_code')}
                     AND {_APPROVED_STITCH_PREDICATE}
                   ORDER BY c.product_code"""
            )
        ]
    if not products:
        return 0
    sources = _selected_pdf_sources(sources_root)
    for product, active_parser in products:
        source = sources[product]
        layout_artifact = (
            _layout_artifact_path(sources_root, source)
            if active_parser
            in {"paizo-native-v3+pp-doclayout-v3-v1", PAIZO_NATIVE_PARSER_V4, PAIZO_NATIVE_PARSER_V5}
            else None
        )
        if layout_artifact is not None and not layout_artifact.is_file():
            raise ValueError(f"{product} is missing its source-bound layout artifact")
        staged = stage_trusted_native_pdf_with_approved_stitches(
            workspace, source.path, product_code=product,
            parser_version=(
                "paizo-native-v3"
                if active_parser == "paizo-native-v3+pp-doclayout-v3-v1"
                else active_parser
            ),
            shard_size=MAX_BATCH_RECORDS,
            layout_artifact=layout_artifact,
        )
        activate_parser_run(workspace, str(staged["parser_run_id"]))
    return len(products)


def _run_layout_phase(
    workspace: Path,
    *,
    concurrency: int,
    foundry_database: Path | None,
    sources_root: Path | None,
    executor: CodexExecutor,
) -> tuple[dict[str, int], str | None]:
    """Drain layout review and repairs to a fixed point without screening."""
    completed = {"stitch-select": 0, "stitch-confirm": 0, "repaired_products": 0}
    while True:
        generate_stitch_candidates(workspace)
        for name in ("stitch-select", "stitch-confirm"):
            completed[name] += _drain(
                workspace, name, concurrency, foundry_database, executor,
            )
        if runner_status(workspace)["needs_maintainer"]:
            return completed, "needs-maintainer"
        if sources_root is None:
            with _connect(workspace, readonly=True) as conn:
                approved = conn.execute(
                    f"""SELECT 1 FROM stitch_candidates c
                       JOIN parser_runs p ON p.parser_run_id=c.parser_run_id
                       WHERE p.state='active' AND p.review_enabled=1
                         AND {_semantic_scope_sql('p.product_code')}
                         AND {_APPROVED_STITCH_PREDICATE} LIMIT 1"""
                ).fetchone()
            if approved is not None:
                raise ValueError("confirmed stitches require --sources for trusted PDF re-read")
            repaired = 0
        else:
            repaired = _apply_confirmed_stitches(workspace, sources_root)
            completed["repaired_products"] += repaired
        if repaired == 0:
            return completed, None


def run_queues(
    workspace: Path | str,
    *,
    queue: str = "all",
    concurrency: int = 4,
    foundry_database: Path | str | None = None,
    sources_root: Path | str | None = None,
    executor: CodexExecutor | None = None,
    pilot: bool = False,
    qwen_endpoint: str = "http://127.0.0.1:8081/v1/chat/completions",
    qwen_model: str = "qwen3.8-27b-q4-xl",
) -> dict[str, Any]:
    path = Path(workspace).expanduser().resolve()
    _ensure_workspace_migrated(path)
    if not 1 <= concurrency <= 4:
        raise ValueError("global concurrency must be between one and four")
    foundry = Path(foundry_database).expanduser().resolve() if foundry_database else None
    process = executor or RoutedExecutor(
        LocalQwenExecutor(qwen_endpoint, model=qwen_model), CodexExecutor()
    )
    completed: dict[str, int] = {}
    screen_queues = {"screen", "coverage-confirm"}
    if (queue == "all" or queue in screen_queues) and foundry is None:
        raise ValueError(f"{queue} requires --foundry-database")
    if pilot:
        generate_stitch_candidates(path)
        if queue in {"stitch-select", "stitch-confirm"}:
            return _stitch_pilot(
                path, queue=queue, concurrency=concurrency, executor=process,
            )
        if queue != "screen":
            raise ValueError("pilot mode supports stitch-select, stitch-confirm, or screen")
        layout = runner_status(path)
        pending_layout = int(layout["stitches"]["selector_pending"]) + int(
            layout["stitches"]["confirmer_pending"]
        )
        if pending_layout:
            return {
                "completed_batches": 0,
                "stopped": "layout-review-required",
                "pending_layout": pending_layout,
                "status": layout,
            }
        with _connect(path, readonly=True) as conn:
            approved = conn.execute(
                f"""SELECT 1 FROM stitch_candidates c
                   JOIN parser_runs p ON p.parser_run_id=c.parser_run_id
                   WHERE p.state='active' AND p.review_enabled=1
                     AND {_semantic_scope_sql('p.product_code')}
                     AND {_APPROVED_STITCH_PREDICATE} LIMIT 1"""
            ).fetchone()
        if approved is not None:
            if sources_root is None:
                raise ValueError("confirmed stitches require --sources for trusted PDF re-read")
            _apply_confirmed_stitches(path, Path(sources_root).expanduser().resolve())
            generate_stitch_candidates(path)
            layout = runner_status(path)
            pending_layout = int(layout["stitches"]["selector_pending"]) + int(
                layout["stitches"]["confirmer_pending"]
            )
            if pending_layout:
                return {
                    "completed_batches": 0,
                    "stopped": "layout-review-required",
                    "pending_layout": pending_layout,
                    "status": layout,
                }
        if runner_status(path)["needs_maintainer"]:
            return {"completed_batches": 0, "stopped": "needs-maintainer"}
        assert foundry is not None
        prepare_deterministic_review(path, foundry)
        return _screening_pilot(
            path, concurrency=concurrency, foundry_db=foundry, executor=process,
        )
    if queue in {"layout", "all"}:
        source_path = Path(sources_root).expanduser().resolve() if sources_root else None
        layout_completed, stopped = _run_layout_phase(
            path,
            concurrency=concurrency,
            foundry_database=foundry,
            sources_root=source_path,
            executor=process,
        )
        completed.update(layout_completed)
        if stopped is not None:
            return {"completed_batches": completed, "stopped": stopped, "status": runner_status(path)}
        if queue == "layout":
            return {"completed_batches": completed, "status": runner_status(path)}
    if queue == "all":
        assert foundry is not None
        prepare_deterministic_review(path, foundry)
        completed["screen"] = _drain(path, "screen", concurrency, foundry, process)
        completed["aon"] = sum(
            refresh_aon_cache(path).get(key, 0) for key in ("match", "no-match", "inconclusive")
        )
        for name in ("coverage-confirm", "classify", "review"):
            completed[name] = _drain(path, name, concurrency, foundry, process)
        # Reviews can create rework shards; iterate producer/reviewer queues to a fixed point.
        while True:
            candidates = _drain(path, "classify", concurrency, foundry, process)
            reviews = _drain(path, "review", concurrency, foundry, process)
            completed["classify"] += candidates
            completed["review"] += reviews
            if candidates == 0 and reviews == 0:
                break
        return {"completed_batches": completed, "status": runner_status(path)}
    if queue == "aon":
        return {"aon_cache": refresh_aon_cache(path), "status": runner_status(path)}
    aliases = {"extract", "review-mixed", "rework-terra", "rework-sol"}
    if queue in aliases:
        raise ValueError(f"{queue} is routed internally; run classify or review")
    if queue not in {
        "stitch-select",
        "stitch-confirm",
        "screen",
        "coverage-confirm",
        "classify",
        "review",
    }:
        raise ValueError("unknown runner queue")
    if queue in screen_queues:
        assert foundry is not None
        prepare_deterministic_review(path, foundry)
    completed[queue] = _drain(path, queue, concurrency, foundry, process)
    return {"completed_batches": completed, "status": runner_status(path)}


def build_base(
    workspace: Path | str,
    output: Path | str,
    notices: Path | str,
    foundry_database: Path | str,
) -> dict[str, Any]:
    verification = verify_workspace(
        workspace, require_complete=True, foundry_database=foundry_database
    )
    if not verification["ok"]:
        raise ValueError("base build is blocked: " + ", ".join(verification["errors"]))
    output_path = Path(output).expanduser().resolve()
    sibling = output_path.with_name(f".{output_path.name}.audited-{os.getpid()}")
    repeat = sibling.with_name(sibling.name + ".repeat")
    try:
        result = build_public_corpus(workspace, sibling, notices)
        bundle = load_licensed_core(sibling)
        manifest_digest = licensed_core_digest(bundle)
        build_public_corpus(workspace, repeat, notices)
        if hashlib.sha256(sibling.read_bytes()).hexdigest() != hashlib.sha256(repeat.read_bytes()).hexdigest():
            raise ValueError("licensed-core base repeat build is not deterministic")
        os.replace(sibling, output_path)
        return {**result, "manifest_digest": manifest_digest, "output": output_path.name}
    except BaseException:
        sibling.unlink(missing_ok=True)
        raise
    finally:
        repeat.unlink(missing_ok=True)


def promote_base(staged: Path | str, tracked: Path | str) -> dict[str, Any]:
    """Explicitly validate and atomically replace the tracked projection."""
    source = Path(staged).expanduser().resolve()
    target = Path(tracked).expanduser().resolve()
    bundle = load_licensed_core(source)
    manifest_digest = licensed_core_digest(bundle)
    target.parent.mkdir(parents=True, exist_ok=True)
    sibling = target.with_name(f".{target.name}.promote-{os.getpid()}")
    shutil.copyfile(source, sibling)
    os.replace(sibling, target)
    return {
        "sections": len(bundle.chunks),
        "revisions": len(bundle.source_revisions),
        "manifest_digest": manifest_digest,
        "target": target.name,
    }


__all__ = [
    "CodexExecutor",
    "CodexResult",
    "MODEL_BY_QUEUE",
    "RUNNER_VERSION",
    "SCHEMAS",
    "build_base",
    "compare_workspace_quality",
    "generate_stitch_candidates",
    "maintainer_item_evidence",
    "pack_batches",
    "prepare_review_data",
    "prepare_workspace",
    "preview_screen_batches",
    "promote_base",
    "quality_workspace",
    "reopen_screening",
    "refresh_aon_cache",
    "resolve_maintainer_item",
    "run_codex_batch",
    "run_queues",
    "runner_status",
    "set_review_product_scope",
    "validate_exact_results",
    "verify_workspace",
]
