"""Cross-encoder reranker for second-stage precision.

Takes the top-N candidates from hybrid (FTS5 + semantic) search and re-scores
them using a cross-encoder model that sees query and document jointly.
"""

from __future__ import annotations

import os
import sys as _sys
import time
from pathlib import Path
from typing import Any

from .embeddings import _onnx_cache_dir, _detect_onnx_provider, _has_onnx, _migraphx_cache_dir

_RERANKER_CACHE = Path.home() / ".cache" / "pf2e-codex" / "onnx" / "reranker"

# Pre-exported ONNX models (avoids needing optimum for export)
ONNX_RERANKER_MODELS = {
    "bge-reranker-v2-m3": {
        "repo": "onnx-community/bge-reranker-v2-m3-ONNX",
        "description": "Base cross-encoder (general domain)",
    },
}


class Reranker:
    """Cross-encoder reranker using onnxruntime.

    Loads a pre-exported ONNX model, batches query-document pairs, and returns
    relevance scores. Follows the same ONNX export + cache pattern as ONNXProvider.
    """

    def __init__(
        self,
        model_name: str = "bge-reranker-v2-m3",
        model_repo: str = "",
        force_provider: str | None = None,
    ):
        # Custom HF repo (fine-tuned) overrides built-in models
        if model_repo:
            self.model_name = model_repo
        else:
            entry = ONNX_RERANKER_MODELS.get(model_name)
            self.model_name = entry["repo"] if entry else model_name
        self._cache_dir = _RERANKER_CACHE / self.model_name.replace("/", "--")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._ensure_model()
        self._session = self._create_session(force_provider)
        self._tokenizer = self._load_tokenizer()

        # Test with a dummy pair to warm up
        self._warmup()

    def _ensure_model(self) -> None:
        """Download pre-exported ONNX model if not cached."""
        model_path = self._cache_dir / "model.onnx"
        quant_path = self._cache_dir / "model_quantized.onnx"
        if model_path.exists() or quant_path.exists():
            return

        print(f"Downloading reranker model {self.model_name}...")
        start = time.time()
        from huggingface_hub import hf_hub_download as _dl
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
        
        # Try model_quantized.onnx first (int8), fall back to model.onnx
        file_name = "model_quantized.onnx"
        try:
            _dl(repo_id=self.model_name, filename=file_name)
        except Exception:
            file_name = "model.onnx"
        
        model = ORTModelForSequenceClassification.from_pretrained(
            self.model_name, export=False, force_download=True, file_name=file_name
        )
        model.save_pretrained(self._cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, fix_mistral_regex=False)
        tokenizer.save_pretrained(self._cache_dir)
        print(f"Downloaded in {time.time() - start:.1f}s -> {self._cache_dir}")

    def _create_session(self, force_provider: str | None = None):
        """Create onnxruntime session (same provider detection as embeddings)."""
        import onnxruntime as ort

        model_path = self._cache_dir / "model.onnx"
        quant_path = self._cache_dir / "model_quantized.onnx"
        if quant_path.exists() and not model_path.exists():
            model_path = quant_path
        if not model_path.exists():
            # Check for quantized variant
            if quant_path.exists():
                model_path = quant_path
            else:
                raise RuntimeError(f"Model not found at {model_path} or {quant_path}")

        def _session(providers):
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            warmup_threads = os.environ.get("PF2E_WARMUP_THREADS")
            if warmup_threads:
                opts.intra_op_num_threads = int(warmup_threads)
            cache_dir = _migraphx_cache_dir()
            provider_opts = [{"migraphx_model_cache_dir": str(cache_dir)}]
            _sys.stderr.write(f"[reranker] Creating ONNX session: {self._model_repo} @ {','.join(providers)}\n")
            _sys.stderr.flush()
            return ort.InferenceSession(str(model_path), opts, providers=providers, provider_options=provider_opts)

        if force_provider and force_provider not in ("auto", ""):
            provider_map = {
                "migraphx": "MIGraphXExecutionProvider",
                "rocm": "ROCMExecutionProvider",
                "cuda": "CUDAExecutionProvider",
                "cpu": "CPUExecutionProvider",
            }
            mapped = provider_map.get(force_provider, force_provider)
            try:
                return _session([mapped])
            except Exception as e:
                raise RuntimeError(f"Provider '{force_provider}' unavailable: {e}")
        else:
            provider = _detect_onnx_provider()
            if not provider:
                raise RuntimeError("No ONNX execution provider available")
            try:
                return _session([provider])
            except Exception:
                return _session(["CPUExecutionProvider"])

    def _load_tokenizer(self):
        """Load tokenizer from cache or original model.

        fix_mistral_regex=False: workaround for transformers bug
        https://github.com/huggingface/transformers/issues/42591
        """
        from transformers import AutoTokenizer
        cache_path = str(self._cache_dir)
        try:
            return AutoTokenizer.from_pretrained(
                cache_path, local_files_only=True,
                fix_mistral_regex=False,
            )
        except Exception:
            return AutoTokenizer.from_pretrained(
                self.model_name,
                fix_mistral_regex=False,
            )

    def _warmup(self) -> None:
        """Run one inference to warm up MIGraphX compile."""
        _ = self.rerank("warmup", [{"text": "warmup document", "id": "_warmup"}], top_k=1)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Score each (query, candidate) pair, re-sort, return top_k.

        Args:
            query: The search query.
            candidates: List of result dicts, each must have at least ``text`` and ``id``.
            top_k: Number of results to return after reranking.

        Returns:
            Candidates sorted by cross-encoder relevance score, highest first.
            Each result gets a ``rerank_score`` field added.
        """
        if not candidates:
            return candidates

        import numpy as np

        texts = [c.get("text", "") for c in candidates]
        pair_inputs = self._tokenizer(
            [query] * len(texts),
            texts,
            padding=True,
            truncation="only_second",
            max_length=512,
            return_tensors="np",
        )

        outputs = self._session.run(None, dict(pair_inputs))[0]

        # Binary classification: logit[1] - logit[0] → relevance score
        if outputs.shape[1] >= 2:
            scores = outputs[:, 1] - outputs[:, 0]
        else:
            scores = outputs[:, 0]

        # Normalize to 0-1 via softmax
        exp_scores = np.exp(scores - scores.max())
        probs = exp_scores / exp_scores.sum()

        # Add scores and re-sort
        for i, cand in enumerate(candidates):
            cand["rerank_score"] = round(float(probs[i]), 4)

        candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return candidates[:top_k]
