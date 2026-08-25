from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pf2e_codex.pdf_export import (
    _TRUSTED_PDF_ORIGIN,
    VerifiedNativeExport,
    trusted_payload_digest,
)
from pf2e_codex.pdf_layout import (
    _postprocess_layout,
    bind_layout_to_native_export,
    validate_layout_export_payload,
    validate_layout_model,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_layout_model_validation_is_onnx_only_and_hash_bound(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"graph")
    (tmp_path / "model.onnx.data").write_bytes(b"weights")
    manifest = {
        "schema_version": 1,
        "export": {
            "inputs": {"page_pixels": [1, 3, 1040, 800]},
            "outputs": ["logits", "pred_boxes", "order_logits"],
        },
        "preprocessing": {
            "canonical_page": {"height": 1040, "width": 800},
            "render_width": 800,
            "model_size": {"height": 800, "width": 800},
            "resize": "bicubic-align-corners-false-antialias-false",
            "rescale_factor": 1 / 255,
        },
        "labels": {"0": "text"},
        "files": {
            name: {"size": (tmp_path / name).stat().st_size, "sha256": _digest(tmp_path / name)}
            for name in ("model.onnx", "model.onnx.data")
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert validate_layout_model(tmp_path)["labels"] == {"0": "text"}

    (tmp_path / "model.onnx.data").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        validate_layout_model(tmp_path)


def test_layout_postprocessing_preserves_reading_order_and_pdf_coordinates() -> None:
    logits = np.full((1, 2, 2), -10.0, dtype=np.float32)
    logits[0, 0, 1] = 8.0
    logits[0, 1, 0] = 7.0
    boxes = np.array([[[0.25, 0.5, 0.2, 0.4], [0.75, 0.5, 0.2, 0.4]]], dtype=np.float32)
    orders = np.array([[[0.0, 8.0], [-8.0, 0.0]]], dtype=np.float32)
    regions = _postprocess_layout(
        logits,
        boxes,
        orders,
        labels={"0": "text", "1": "paragraph_title"},
        page_width=600.0,
        page_height=800.0,
        threshold=0.5,
    )
    assert [(region.label, region.order) for region in regions] == [
        ("paragraph_title", 0),
        ("text", 1),
    ]
    assert regions[0].box == pytest.approx((90.0, 240.0, 210.0, 560.0))
    assert regions[1].box == pytest.approx((390.0, 240.0, 510.0, 560.0))


def test_layout_artifact_is_source_bound_and_contains_no_filename() -> None:
    source_hash = "a" * 64
    payload = {
        "schema_version": 1,
        "extractor": {"name": "pf2e-codex-pdf-layout", "profile_version": 1},
        "source": {"sha256": source_hash, "page_count": 1},
        "pages": [
            {
                "number": 1,
                "width": 600.0,
                "height": 800.0,
                "regions": [
                    {"label": "text", "score": 0.9, "order": 0, "box": [1.0, 2.0, 3.0, 4.0]}
                ],
            }
        ],
    }
    validate_layout_export_payload(
        payload,
        expected_source_sha256=source_hash,
        expected_page_count=1,
    )
    assert "filename" not in payload["source"]
    with pytest.raises(ValueError, match="does not match"):
        validate_layout_export_payload(payload, expected_source_sha256="b" * 64)


def test_layout_binding_uses_opaque_anchors_and_smallest_containing_region() -> None:
    native = {
        "schema_version": 1,
        "extractor": {
            "name": "pf2e-codex-native-pdf",
            "profile_version": 1,
            "backend": "pdfplumber",
            "backend_version": "test",
            "ocr": False,
        },
        "source": {
            "filename": "PZO12001E.pdf",
            "sha256": "a" * 64,
            "size": 1,
            "page_count": 1,
        },
        "selection": {"first_page": 1, "last_page": 1},
        "pages": [
            {
                "number": 1,
                "width": 600.0,
                "height": 800.0,
                "images": [],
                "words": [
                    {
                        "text": "Rules",
                        "x0": 100.0,
                        "top": 100.0,
                        "x1": 140.0,
                        "bottom": 112.0,
                        "font": "Test",
                        "size": 10.0,
                        "upright": True,
                        "direction": "ltr",
                    },
                    {
                        "text": "outside",
                        "x0": 500.0,
                        "top": 500.0,
                        "x1": 550.0,
                        "bottom": 512.0,
                        "font": "Test",
                        "size": 10.0,
                        "upright": True,
                        "direction": "ltr",
                    },
                ],
            }
        ],
    }
    artifact = VerifiedNativeExport(
        payload=native,
        product_code="PZO12001",
        source_basename="PZO12001E.pdf",
        page_count=1,
        extractor_profile_version=1,
        pdf_verified=True,
        attestation_digest="test",
        _verification_token=_TRUSTED_PDF_ORIGIN,
        _payload_digest=trusted_payload_digest(native),
    )
    layout = {
        "schema_version": 1,
        "extractor": {"name": "pf2e-codex-pdf-layout", "profile_version": 1},
        "source": {"sha256": "a" * 64, "page_count": 1},
        "pages": [
            {
                "number": 1,
                "width": 600.0,
                "height": 800.0,
                "regions": [
                    {"label": "text", "score": 0.99, "order": 0, "box": [0, 0, 300, 300]},
                    {
                        "label": "paragraph_title",
                        "score": 0.8,
                        "order": 1,
                        "box": [90, 90, 160, 130],
                    },
                ],
            }
        ],
    }
    bound = bind_layout_to_native_export(artifact, layout)
    assert bound.bound_anchor_count == 1
    assert len(bound.unbound_native_anchors) == 1
    assert bound.regions[0].native_word_anchors == ()
    assert len(bound.regions[1].native_word_anchors) == 1
    assert "Rules" not in repr(bound)
