# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "huggingface-hub>=1.28,<2",
#   "numpy>=2.4,<3",
#   "onnx>=1.21,<2",
#   "onnxscript>=0.7,<1",
#   "safetensors>=0.8,<1",
#   "torch>=2.13,<3",
#   "torchvision>=0.28,<1",
#   "transformers>=5.15,<6",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchvision = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Export the pinned PP-DocLayoutV3 checkpoint for ONNX-only corpus use.

This script intentionally has its own PEP 723 environment. PP-DocLayoutV3 is
available only in Transformers 5, while pf2e-codex's embedding exporter still
uses Optimum-ONNX, whose current release requires Transformers 4.57 or older.
Torch is used only for this one-time graph conversion; product inference uses
the hardware-selected ONNX Runtime provider.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch
from huggingface_hub import snapshot_download
from onnx import numpy_helper
from transformers import AutoConfig, AutoModelForObjectDetection

MODEL_REPOSITORY = "PaddlePaddle/PP-DocLayoutV3_safetensors"
MODEL_REVISION = "97d101e6db2642e162a1d05392d1b0231c91033e"
MODEL_LICENSE = "Apache-2.0"
LAYOUT_MODEL_SCHEMA_VERSION = 1
ONNX_OPSET = 18


class _LayoutCore(torch.nn.Module):
    """Embed exact preprocessing and retain only native-word layout outputs."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, page_pixels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pixel_values = torch.nn.functional.interpolate(
            page_pixels.float(),
            size=(800, 800),
            mode="bicubic",
            align_corners=False,
            antialias=False,
        ) / 255.0
        pixel_mask = torch.ones(
            (page_pixels.shape[0], 800, 800),
            dtype=torch.int64,
            device=page_pixels.device,
        )
        output = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return output.logits, output.pred_boxes, output.order_logits


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fold_constant_cos(source: Path, destination: Path) -> int:
    """Fold exporter-left constant Cos nodes unsupported by MIGraphX.

    Torch 2.13 leaves two rotary-position Cos operations over fixed initializer
    tensors. They are compile-time constants, but the MIGraphX ONNX Runtime
    build cannot assign them to a provider during initialization. Only this
    exact, evidence-backed constant case is rewritten.
    """
    model = onnx.load(source, load_external_data=True)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    removed: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.op_type != "Cos":
            continue
        if len(node.input) != 1 or len(node.output) != 1 or node.input[0] not in initializers:
            raise RuntimeError("layout ONNX export contains a nonconstant Cos operation")
        value = np.cos(numpy_helper.to_array(initializers[node.input[0]]))
        model.graph.initializer.append(numpy_helper.from_array(value, name=node.output[0]))
        removed.append(node)
    for node in removed:
        model.graph.node.remove(node)
    if any(node.op_type == "Cos" for node in model.graph.node):
        raise RuntimeError("layout ONNX export still contains unsupported Cos operations")
    onnx.save_model(
        model,
        destination,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=destination.name + ".data",
        size_threshold=1024,
    )
    onnx.checker.check_model(destination)
    return len(removed)


def _version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def _export(staging: Path, model_path: Path) -> dict[str, object]:
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model = _LayoutCore(
        AutoModelForObjectDetection.from_pretrained(model_path, local_files_only=True).eval()
    ).eval()
    page_pixels = torch.full((1, 3, 1040, 800), 255, dtype=torch.uint8)
    raw_path = staging / "raw.onnx"
    torch.onnx.export(
        model,
        (page_pixels,),
        raw_path,
        input_names=["page_pixels"],
        output_names=["logits", "pred_boxes", "order_logits"],
        opset_version=ONNX_OPSET,
        dynamo=True,
    )
    model_path_out = staging / "model.onnx"
    folded = _fold_constant_cos(raw_path, model_path_out)
    raw_path.unlink(missing_ok=True)
    (staging / "raw.onnx.data").unlink(missing_ok=True)

    preprocessor = json.loads((model_path / "preprocessor_config.json").read_text())
    checkpoint = model_path / "model.safetensors"
    config_path = model_path / "config.json"
    data_path = staging / "model.onnx.data"
    manifest: dict[str, object] = {
        "schema_version": LAYOUT_MODEL_SCHEMA_VERSION,
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "license": MODEL_LICENSE,
            "checkpoint_sha256": _sha256(checkpoint),
            "config_sha256": _sha256(config_path),
        },
        "export": {
            "opset": ONNX_OPSET,
            "torch": _version("torch"),
            "transformers": _version("transformers"),
            "onnx": _version("onnx"),
            "constant_cos_nodes_folded": folded,
            "inputs": {"page_pixels": [1, 3, 1040, 800]},
            "outputs": ["logits", "pred_boxes", "order_logits"],
        },
        "preprocessing": {
            "canonical_page": {"height": 1040, "width": 800},
            "render_width": 800,
            "model_size": preprocessor["size"],
            "resize": "bicubic-align-corners-false-antialias-false",
            "rescale_factor": preprocessor["rescale_factor"],
            "padding": "white-bottom-right",
        },
        "labels": {str(key): value for key, value in config.id2label.items()},
        "files": {
            "model.onnx": {"sha256": _sha256(model_path_out), "size": model_path_out.stat().st_size},
            "model.onnx.data": {"sha256": _sha256(data_path), "size": data_path.stat().st_size},
        },
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return manifest


def export_layout_model(output: Path, *, force: bool = False) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        raise FileExistsError(f"layout model artifact already exists: {output}")
    snapshot = Path(
        snapshot_download(
            MODEL_REPOSITORY,
            revision=MODEL_REVISION,
            allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
        )
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    backup: Path | None = None
    try:
        _export(staging, snapshot)
        if output.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{output.name}.backup-", dir=output.parent))
            backup.rmdir()
            os.replace(output, backup)
        os.replace(staging, output)
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    except Exception:
        if backup is not None and not output.exists():
            os.replace(backup, output)
            backup = None
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


def main() -> None:
    default = Path.home() / ".cache" / "pf2e-codex" / "layout" / "pp-doclayout-v3"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    export_layout_model(arguments.output, force=arguments.force)
    print(f"Layout ONNX artifact ready: {arguments.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
