import tomllib
from pathlib import Path

from pf2e_codex.config import Settings
from pf2e_codex.embeddings import _select_onnx_provider

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_query_provider_defaults_to_gpu_first_auto():
    assert Settings.model_fields["query_provider"].default == "auto"


def test_provider_selection_prefers_gpu_over_cpu():
    assert (
        _select_onnx_provider(
            ["CPUExecutionProvider", "CUDAExecutionProvider"],
            detected_gpu_vendors=("NVIDIA",),
        )
        == "CUDAExecutionProvider"
    )


def test_auto_provider_rejects_cpu_only_runtime_on_supported_gpu():
    try:
        _select_onnx_provider(
            ["CPUExecutionProvider"],
            detected_gpu_vendors=("AMD",),
        )
    except RuntimeError as exc:
        assert "only CPUExecutionProvider" in str(exc)
        assert "explicitly" in str(exc)
    else:
        raise AssertionError("CPU-only runtime was silently accepted on GPU hardware")


def test_cpu_remains_fallback_when_no_supported_gpu_exists():
    assert (
        _select_onnx_provider(
            ["CPUExecutionProvider"],
            detected_gpu_vendors=(),
        )
        == "CPUExecutionProvider"
    )


def test_uv_project_pins_export_only_torch_to_cpu_index():
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())

    assert any(dependency.startswith("torch") for dependency in config["project"]["dependencies"])
    assert config["tool"]["uv"]["sources"]["torch"] == {"index": "pytorch-cpu"}
    indexes = {entry["name"]: entry for entry in config["tool"]["uv"]["index"]}
    assert indexes["pytorch-cpu"] == {
        "name": "pytorch-cpu",
        "url": "https://download.pytorch.org/whl/cpu",
        "explicit": True,
    }


def test_dev_setup_detects_gpu_before_cpu_fallback():
    makefile = (REPO_ROOT / "Makefile").read_text()

    assert "0x10de" in makefile
    assert "0x1002" in makefile
    assert "setup-dev-amd" in makefile
    assert "setup-dev-nvidia" in makefile
    assert "setup-dev-cpu" in makefile
    assert "--inexact" in makefile
