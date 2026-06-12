# SCALE Experiment

Goal: use `CUDAExecutionProvider` (via onnxruntime-gpu + SCALE) instead of
`MIGraphXExecutionProvider` on AMD GPUs. CUDA EP handles dynamic shapes
natively — no per-shape compilation.

## Why

MIGraphX compiles per input shape. With random-length queries (different
token counts before padding), it recompiles on every unique shape. CUDA EP
doesn't have this issue.

## Setup

```bash
# Download and extract (all architectures bundled)
wget https://pkgs.scale-lang.com/tar/scale-latest-amd64.tar.xz
tar xf scale-latest-amd64.tar.xz

# Activate for your GPU (7900 XTX = gfx1100)
source ./scale/bin/scaleenv gfx1100

# Install CUDA onnxruntime
uv pip install onnxruntime-gpu
```

## Test

```python
import onnxruntime as ort
print(ort.get_available_providers())
# Should show: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

If CUDA EP is present, the rest of the pipeline should work transparently —
our `_detect_onnx_provider()` prefers the first available provider in
priority order. We'd just add `CUDAExecutionProvider` before
`MIGraphXExecutionProvider` in that list.

## Priority

Current: MIGraphX → ROCm → CUDA → CPU
After:   CUDA (via SCALE) → MIGraphX → CPU

CPU is the universal fallback. CUDA (via SCALE) is preferred because it
handles dynamic shapes without per-shape recompilation.
