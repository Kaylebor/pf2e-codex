# Maintainer: Your Name <your@email.com>
pkgname=pf2e-codex
pkgver=0.1.0
pkgrel=1
pkgdesc="PF2E rules knowledge base with MCP, CLI, and SDK interfaces"
arch=('any')
url="https://github.com/Kaylebor/pf2e-codex"
license=('MIT')
depends=(
    'python'
    'python-pydantic'
    'python-pydantic-settings'
    'python-yaml'
    'python-rich'
    'python-typer'
    'python-sqlite-vec'
    'python-mcp'
    'uvicorn'
    'python-starlette'
    'python-sse-starlette'
)
optdepends=(
    'migraphx: MIGraphX library for AMD GPU ONNX acceleration'
    'rocm-hip-runtime: HIP runtime for AMD GPU (needed by MIGraphX)'
    'cuda: CUDA runtime for NVIDIA GPU ONNX acceleration'
)
makedepends=('python-pip')
# Build from local repo (for AUR: use GitHub tarball URL)
source=()
sha256sums=()
install=pf2e-codex.install

package() {
    # Force system Python (Mise may override PATH)
    export PYTHON=/usr/bin/python3
    export PIP_ROOT_USER_ACTION=ignore

    local lib="$pkgdir/usr/share/pf2e-codex/lib"
    mkdir -p "$lib"

    # --no-deps everywhere: pip never scans system packages = zero noise.
    # System provides: pydantic, rich, typer, pyyaml, sqlite-vec, mcp, etc.
    /usr/bin/pip3 install --no-deps --no-cache-dir --target "$lib" "$startdir"
    /usr/bin/pip3 install --no-deps --no-cache-dir --target "$lib" \
        'transformers>=4.40,<6.0' 'tokenizers>=0.19,<0.23'

    # ── GPU autodetection ──
    # onnxruntime-migraphx bundles its own .so, no system onnxruntime needed.
    if [ -e /opt/rocm/lib/libamdhip64.so ] || [ -e /opt/rocm/lib/libamdhip64.so.7 ]; then
        echo "==> AMD GPU/ROCm detected — installing onnxruntime-migraphx"
        /usr/bin/pip3 install --no-deps --no-cache-dir --target "$lib" \
            'onnxruntime-migraphx>=1.25'
    elif [ -e /opt/cuda/lib64/libcudart.so ]; then
        echo "==> NVIDIA GPU detected — installing onnxruntime-gpu"
        /usr/bin/pip3 install --no-deps --no-cache-dir --target "$lib" \
            'onnxruntime-gpu'
    else
        echo "==> No GPU detected — installing onnxruntime (CPU)"
        /usr/bin/pip3 install --no-deps --no-cache-dir --target "$lib" \
            'onnxruntime>=1.20'
    fi

    # Wrapper: PYTHONPATH points to private lib, uses system python3
    mkdir -p "$pkgdir/usr/bin"
    cat > "$pkgdir/usr/bin/pf2e-codex" << 'WRAPPER'
#!/bin/sh
export PYTHONPATH="/usr/share/pf2e-codex/lib${PYTHONPATH:+:$PYTHONPATH}"
exec /usr/bin/python3 -m pf2e_codex.cli "$@"
WRAPPER
    chmod 755 "$pkgdir/usr/bin/pf2e-codex"
}
