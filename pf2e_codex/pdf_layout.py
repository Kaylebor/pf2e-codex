"""GPU-first document-layout evidence for native-text PDF parsing.

The PDF's selectable text remains authoritative. This module rasterizes pages
only to identify regions and reading order, then emits a separate private JSON
artifact whose coordinates can be bound back to native words.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LAYOUT_EXPORT_SCHEMA_VERSION = 1
LAYOUT_MODEL_SCHEMA_VERSION = 1
LAYOUT_EXTRACTOR_VERSION = 1
DEFAULT_LAYOUT_MODEL_DIR = (
    Path.home() / ".cache" / "pf2e-codex" / "layout" / "pp-doclayout-v3"
)


class PdfLayoutDependencyError(RuntimeError):
    """Raised when optional PDF rendering or ONNX dependencies are absent."""


@dataclass(frozen=True)
class LayoutRegion:
    """One model-detected region in logical reading order and PDF coordinates."""

    label: str
    score: float
    order: int
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class PdfLayoutSummary:
    """Content-free result of a completed private layout export."""

    output_path: Path
    source_pages: int
    exported_pages: int
    regions: int
    provider: str


@dataclass(frozen=True)
class BoundLayoutRegion:
    """A detected region bound only to opaque, non-watermark native anchors."""

    page: int
    label: str
    score: float
    order: int
    box: tuple[float, float, float, float]
    native_word_anchors: tuple[str, ...]


@dataclass(frozen=True)
class BoundNativeLayout:
    """Watermark-independent bridge between private native words and layout."""

    product_code: str
    regions: tuple[BoundLayoutRegion, ...]
    unbound_native_anchors: tuple[str, ...]
    selected_pages: tuple[int, ...]
    binding_digest: str

    @property
    def bound_anchor_count(self) -> int:
        return sum(len(region.native_word_anchors) for region in self.regions)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _require_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"layout artifact has invalid {field}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"layout artifact has non-finite {field}")
    return result


def validate_layout_model(model_dir: Path | str) -> dict[str, object]:
    """Validate a model artifact without importing Torch or Transformers."""
    root = Path(model_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("layout model manifest is missing or invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != LAYOUT_MODEL_SCHEMA_VERSION:
        raise ValueError("unsupported layout model schema")
    export = manifest.get("export")
    files = manifest.get("files")
    labels = manifest.get("labels")
    preprocessing = manifest.get("preprocessing")
    if not all(isinstance(item, dict) for item in (export, files, labels, preprocessing)):
        raise ValueError("layout model manifest is incomplete")
    if export.get("inputs") != {"page_pixels": [1, 3, 1040, 800]} or export.get(
        "outputs"
    ) != ["logits", "pred_boxes", "order_logits"]:
        raise ValueError("layout model graph contract is unsupported")
    if (
        preprocessing.get("canonical_page") != {"height": 1040, "width": 800}
        or preprocessing.get("render_width") != 800
        or preprocessing.get("model_size") != {"height": 800, "width": 800}
        or preprocessing.get("resize")
        != "bicubic-align-corners-false-antialias-false"
        or preprocessing.get("rescale_factor") != 1 / 255
    ):
        raise ValueError("layout model preprocessing contract is unsupported")
    for name in ("model.onnx", "model.onnx.data"):
        record = files.get(name)
        path = root / name
        if not isinstance(record, dict) or not path.is_file():
            raise ValueError(f"layout model is missing {name}")
        if record.get("size") != path.stat().st_size or record.get("sha256") != _sha256(path):
            raise ValueError(f"layout model file failed integrity validation: {name}")
    if not labels or any(not str(key).isdigit() or not isinstance(value, str) for key, value in labels.items()):
        raise ValueError("layout model labels are invalid")
    return manifest


def _sigmoid(value: Any) -> Any:
    import numpy as np

    clipped = np.clip(value, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _postprocess_layout(
    logits: Any,
    pred_boxes: Any,
    order_logits: Any,
    *,
    labels: Mapping[str, str],
    page_width: float,
    page_height: float,
    threshold: float,
) -> list[LayoutRegion]:
    """Reproduce the official box and reading-order postprocessing in NumPy."""
    import numpy as np

    if logits.shape[0] != 1 or pred_boxes.shape[:2] != logits.shape[:2]:
        raise ValueError("layout model returned incompatible detection tensors")
    queries = logits.shape[1]
    if order_logits.shape != (1, queries, queries):
        raise ValueError("layout model returned incompatible order tensor")

    order_scores = _sigmoid(order_logits)
    votes = np.triu(order_scores, k=1).sum(axis=1) + np.tril(
        1.0 - order_scores.transpose(0, 2, 1), k=-1
    ).sum(axis=1)
    pointers = np.argsort(votes, axis=1, kind="stable")
    order_seq = np.empty_like(pointers)
    np.put_along_axis(
        order_seq,
        pointers,
        np.broadcast_to(np.arange(queries, dtype=pointers.dtype), pointers.shape),
        axis=1,
    )

    probabilities = _sigmoid(logits).reshape(1, -1)
    ranked = np.argsort(-probabilities, axis=1, kind="stable")[:, :queries]
    scores = np.take_along_axis(probabilities, ranked, axis=1)[0]
    class_count = logits.shape[2]
    class_ids = (ranked % class_count)[0]
    query_ids = (ranked // class_count)[0]
    boxes = pred_boxes[0, query_ids]
    orders = order_seq[0, query_ids]

    centers = boxes[:, :2]
    dimensions = boxes[:, 2:]
    xyxy = np.concatenate((centers - dimensions / 2.0, centers + dimensions / 2.0), axis=1)
    xyxy *= np.array([page_width, page_height, page_width, page_height], dtype=np.float32)
    regions: list[LayoutRegion] = []
    for score, class_id, order, box in zip(scores, class_ids, orders, xyxy, strict=True):
        if float(score) < threshold:
            continue
        label = labels.get(str(int(class_id)))
        if label is None:
            raise ValueError("layout model returned an unknown class")
        regions.append(
            LayoutRegion(
                label=label,
                score=float(score),
                order=int(order),
                box=tuple(float(value) for value in box),
            )
        )
    regions.sort(key=lambda region: (region.order, region.label, region.box))
    return regions


class LayoutAnalyzer:
    """ONNX-only PP-DocLayoutV3 inference with GPU-first provider selection."""

    def __init__(
        self,
        model_dir: Path | str = DEFAULT_LAYOUT_MODEL_DIR,
        *,
        force_provider: str | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise PdfLayoutDependencyError("layout analysis requires ONNX Runtime") from exc
        from .embeddings import _detect_onnx_provider, _migraphx_cache_dir

        self.model_dir = Path(model_dir).expanduser().resolve()
        self.manifest = validate_layout_model(self.model_dir)
        provider_map = {
            "migraphx": "MIGraphXExecutionProvider",
            "rocm": "ROCMExecutionProvider",
            "cuda": "CUDAExecutionProvider",
            "cpu": "CPUExecutionProvider",
        }
        if force_provider and force_provider not in {"", "auto"}:
            provider = provider_map.get(force_provider, force_provider)
        else:
            provider = _detect_onnx_provider()
        if provider is None:
            raise RuntimeError("no ONNX execution provider is available for layout analysis")
        available = ort.get_available_providers()
        if provider not in available:
            raise RuntimeError(
                f"ONNX layout provider '{provider}' is unavailable; installed providers: "
                + ", ".join(available)
            )
        options = (
            [{"migraphx_model_cache_dir": str(_migraphx_cache_dir())}]
            if provider == "MIGraphXExecutionProvider"
            else [{}]
        )
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self.session = ort.InferenceSession(
                str(self.model_dir / "model.onnx"),
                session_options,
                providers=[provider],
                provider_options=options,
            )
        except Exception as exc:
            raise RuntimeError(
                f"ONNX layout provider '{provider}' failed; refusing silent CPU fallback"
            ) from exc
        if provider not in self.session.get_providers():
            raise RuntimeError(
                f"ONNX layout provider '{provider}' was not activated; refusing silent CPU fallback"
            )
        self.provider = provider
        self.labels = self.manifest["labels"]
        self.preprocessing = self.manifest["preprocessing"]

    def analyze(self, image: Any, *, page_width: float, page_height: float, threshold: float) -> list[LayoutRegion]:
        import numpy as np

        try:
            from PIL import Image
        except ImportError as exc:
            raise PdfLayoutDependencyError("layout analysis requires Pillow") from exc
        canonical = self.preprocessing["canonical_page"]
        width = int(canonical["width"])
        height = int(canonical["height"])
        rgb = image.convert("RGB")
        if rgb.width > width + 2 or rgb.height > height + 2:
            raise ValueError("PDF page raster exceeds the supported Paizo portrait layout")
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(rgb.crop((0, 0, min(rgb.width, width), min(rgb.height, height))), (0, 0))
        values = np.asarray(canvas, dtype=np.uint8).copy().transpose(2, 0, 1)[None, ...]
        logits, boxes, orders = self.session.run(
            ["logits", "pred_boxes", "order_logits"],
            {"page_pixels": values},
        )
        return _postprocess_layout(
            logits,
            boxes,
            orders,
            labels=self.labels,
            page_width=page_width,
            page_height=page_height,
            threshold=threshold,
        )


def validate_layout_export_payload(
    payload: Mapping[str, object],
    *,
    expected_source_sha256: str | None = None,
    expected_page_count: int | None = None,
) -> None:
    """Validate private layout evidence before binding it to native words."""
    if payload.get("schema_version") != LAYOUT_EXPORT_SCHEMA_VERSION:
        raise ValueError("unsupported PDF layout artifact schema")
    source = payload.get("source")
    extractor = payload.get("extractor")
    pages = payload.get("pages")
    if not isinstance(source, Mapping) or not isinstance(extractor, Mapping) or not isinstance(pages, Sequence):
        raise ValueError("PDF layout artifact is incomplete")
    source_hash = source.get("sha256")
    page_count = source.get("page_count")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("PDF layout artifact has invalid source hash")
    if expected_source_sha256 is not None and source_hash != expected_source_sha256:
        raise ValueError("PDF layout artifact does not match the native PDF")
    if isinstance(page_count, bool) or not isinstance(page_count, int) or page_count < 1:
        raise ValueError("PDF layout artifact has invalid page count")
    if expected_page_count is not None and page_count != expected_page_count:
        raise ValueError("PDF layout artifact page count does not match the native PDF")
    if extractor.get("name") != "pf2e-codex-pdf-layout" or extractor.get("profile_version") != LAYOUT_EXTRACTOR_VERSION:
        raise ValueError("PDF layout artifact extractor is unsupported")
    seen: set[int] = set()
    for page in pages:
        if not isinstance(page, Mapping):
            raise ValueError("PDF layout artifact page is invalid")
        number = page.get("number")
        regions = page.get("regions")
        if isinstance(number, bool) or not isinstance(number, int) or number < 1 or number > page_count or number in seen:
            raise ValueError("PDF layout artifact pages are invalid or duplicated")
        seen.add(number)
        if not isinstance(regions, Sequence):
            raise ValueError("PDF layout artifact regions are invalid")
        width = _require_number(page.get("width"), "page width")
        height = _require_number(page.get("height"), "page height")
        if width <= 0 or height <= 0:
            raise ValueError("PDF layout artifact page dimensions are invalid")
        for region in regions:
            if not isinstance(region, Mapping) or not isinstance(region.get("label"), str):
                raise ValueError("PDF layout artifact region is invalid")
            if isinstance(region.get("order"), bool) or not isinstance(region.get("order"), int):
                raise ValueError("PDF layout artifact region order is invalid")
            _require_number(region.get("score"), "region score")
            box = region.get("box")
            if not isinstance(box, Sequence) or len(box) != 4:
                raise ValueError("PDF layout artifact region box is invalid")
            for value in box:
                _require_number(value, "region box")


def _load_layout_payload(value: Path | str | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    try:
        payload = json.loads(Path(value).expanduser().resolve().read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("PDF layout artifact is missing or invalid") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("PDF layout artifact root must be an object")
    return payload


def bind_layout_to_native_export(
    artifact: Any,
    layout: Path | str | Mapping[str, object],
) -> BoundNativeLayout:
    """Bind layout regions to a trusted native export without retaining text."""
    from .pdf_export import _TRUSTED_PDF_ORIGIN, native_word_inventory

    if (
        getattr(artifact, "_verification_token", None) is not _TRUSTED_PDF_ORIGIN
        or not getattr(artifact, "pdf_verified", False)
    ):
        raise ValueError("layout binding requires a trusted direct-PDF native export")
    native_payload = artifact.payload
    native_source = native_payload.get("source")
    if not isinstance(native_source, Mapping):
        raise ValueError("trusted native export source is invalid")
    layout_payload = _load_layout_payload(layout)
    validate_layout_export_payload(
        layout_payload,
        expected_source_sha256=str(native_source.get("sha256")),
        expected_page_count=int(native_source.get("page_count", 0)),
    )
    inventory = native_word_inventory(native_payload, artifact.product_code, strict=True)
    ignored = set(inventory.ignored_anchor_reasons)
    native_pages = {
        int(page["number"]): page
        for page in native_payload["pages"]
        if isinstance(page, Mapping)
    }
    region_records: list[dict[str, object]] = []
    regions_by_page: dict[int, list[int]] = {}
    selected_pages: list[int] = []
    for page in layout_payload["pages"]:
        page_number = int(page["number"])
        native_page = native_pages.get(page_number)
        if native_page is None:
            raise ValueError("PDF layout artifact references an unknown native page")
        if abs(float(page["width"]) - float(native_page["width"])) > 0.01 or abs(
            float(page["height"]) - float(native_page["height"])
        ) > 0.01:
            raise ValueError("PDF layout artifact geometry does not match native PDF geometry")
        selected_pages.append(page_number)
        for region in page["regions"]:
            index = len(region_records)
            region_records.append(
                {
                    "page": page_number,
                    "label": str(region["label"]),
                    "score": float(region["score"]),
                    "order": int(region["order"]),
                    "box": tuple(float(value) for value in region["box"]),
                    "anchors": [],
                }
            )
            regions_by_page.setdefault(page_number, []).append(index)

    unbound: list[str] = []
    for page_number in selected_pages:
        page = native_pages[page_number]
        for ordinal, word in enumerate(page["words"]):
            anchor = inventory.word_anchors.get((page_number, ordinal))
            if anchor is None or anchor in ignored:
                continue
            center_x = (float(word["x0"]) + float(word["x1"])) / 2.0
            center_y = (float(word["top"]) + float(word["bottom"])) / 2.0
            candidates: list[tuple[float, float, int, int]] = []
            for index in regions_by_page.get(page_number, []):
                x0, y0, x1, y1 = region_records[index]["box"]
                if x0 <= center_x <= x1 and y0 <= center_y <= y1:
                    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
                    candidates.append(
                        (
                            area,
                            -float(region_records[index]["score"]),
                            int(region_records[index]["order"]),
                            index,
                        )
                    )
            if not candidates:
                unbound.append(anchor)
                continue
            chosen = min(candidates)[-1]
            region_records[chosen]["anchors"].append(anchor)

    bound = tuple(
        BoundLayoutRegion(
            page=int(record["page"]),
            label=str(record["label"]),
            score=float(record["score"]),
            order=int(record["order"]),
            box=tuple(record["box"]),
            native_word_anchors=tuple(record["anchors"]),
        )
        for record in region_records
    )
    digest_payload = {
        "version": "native-layout-binding-v1",
        "product": artifact.product_code,
        "regions": [
            {
                "page": region.page,
                "label": region.label,
                "order": region.order,
                "box": [round(value, 4) for value in region.box],
                "anchors": list(region.native_word_anchors),
            }
            for region in bound
        ],
        "unbound": sorted(unbound),
    }
    return BoundNativeLayout(
        product_code=artifact.product_code,
        regions=bound,
        unbound_native_anchors=tuple(sorted(unbound)),
        selected_pages=tuple(sorted(selected_pages)),
        binding_digest=_canonical_digest(digest_payload),
    )


def export_pdf_layout(
    source_path: Path | str,
    output_path: Path | str,
    *,
    model_dir: Path | str = DEFAULT_LAYOUT_MODEL_DIR,
    first_page: int = 1,
    last_page: int | None = None,
    threshold: float = 0.5,
    overwrite: bool = False,
    force_provider: str | None = None,
    analyzer: LayoutAnalyzer | None = None,
) -> PdfLayoutSummary:
    """Render PDF pages, infer layout on GPU, and atomically write private JSON."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("layout threshold must be between zero and one")
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    if first_page < 1 or (last_page is not None and last_page < first_page):
        raise ValueError("invalid PDF layout page range")
    try:
        import pdfplumber
        import pypdfium2
    except ImportError as exc:
        raise PdfLayoutDependencyError(
            "PDF layout export requires the optional 'corpus' dependencies"
        ) from exc

    if analyzer is None:
        analyzer = LayoutAnalyzer(model_dir, force_provider=force_provider)
    source_hash = _sha256(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with pdfplumber.open(source) as pdf:
            source_pages = len(pdf.pages)
            selected_last = source_pages if last_page is None else last_page
            if first_page > source_pages or selected_last > source_pages:
                raise ValueError("PDF layout page range exceeds source page count")
            pages: list[dict[str, object]] = []
            region_count = 0
            for page_number in range(first_page, selected_last + 1):
                page = pdf.pages[page_number - 1]
                image = page.to_image(
                    width=int(analyzer.preprocessing["render_width"]),
                    antialias=True,
                ).original
                regions = analyzer.analyze(
                    image,
                    page_width=float(page.width),
                    page_height=float(page.height),
                    threshold=threshold,
                )
                region_count += len(regions)
                pages.append(
                    {
                        "number": page_number,
                        "width": round(float(page.width), 4),
                        "height": round(float(page.height), 4),
                        "regions": [
                            {
                                "label": region.label,
                                "score": round(region.score, 6),
                                "order": region.order,
                                "box": [round(value, 4) for value in region.box],
                            }
                            for region in regions
                        ],
                    }
                )
        manifest = analyzer.manifest
        payload: dict[str, object] = {
            "schema_version": LAYOUT_EXPORT_SCHEMA_VERSION,
            "extractor": {
                "name": "pf2e-codex-pdf-layout",
                "profile_version": LAYOUT_EXTRACTOR_VERSION,
                "renderer": "pypdfium2",
                "renderer_version": getattr(pypdfium2, "__version__", "unknown"),
                "model_manifest_sha256": _canonical_digest(manifest),
                "provider": analyzer.provider,
                "threshold": threshold,
                "ocr": False,
            },
            "source": {"sha256": source_hash, "page_count": source_pages},
            "selection": {"first_page": first_page, "last_page": selected_last},
            "pages": pages,
        }
        validate_layout_export_payload(
            payload,
            expected_source_sha256=source_hash,
            expected_page_count=source_pages,
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        temporary = None
        return PdfLayoutSummary(
            output_path=output,
            source_pages=source_pages,
            exported_pages=selected_last - first_page + 1,
            regions=region_count,
            provider=analyzer.provider,
        )
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = [
    "BoundLayoutRegion",
    "BoundNativeLayout",
    "DEFAULT_LAYOUT_MODEL_DIR",
    "LAYOUT_EXPORT_SCHEMA_VERSION",
    "LayoutAnalyzer",
    "LayoutRegion",
    "PdfLayoutDependencyError",
    "PdfLayoutSummary",
    "bind_layout_to_native_export",
    "export_pdf_layout",
    "validate_layout_export_payload",
    "validate_layout_model",
]
