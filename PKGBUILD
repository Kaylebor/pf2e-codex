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
)
optdepends=(
    'migraphx: MIGraphX library for AMD GPU ONNX acceleration'
    'rocm-hip-runtime: HIP runtime for AMD GPU (needed by MIGraphX)'
    'cuda: CUDA runtime for NVIDIA GPU ONNX acceleration'
)
makedepends=('python-pip' 'patchelf')
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

    # Pin torch to CPU-only (onnxruntime-migraphx handles GPU inference)
    cat > /tmp/pf2e-torch-constraint.txt << 'EOF'
torch==2.12.0+cpu
EOF

    # Install pf2e-codex + deps (optimum pulls torch CPU, transformers, etc.
    # onnxruntime is handled below by GPU detection + fallback).
    /usr/bin/pip3 install --no-cache-dir --target "$lib" \
        --constraint /tmp/pf2e-torch-constraint.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "$startdir"

    rm -f /tmp/pf2e-torch-constraint.txt

    # ── GPU autodetection ──
    # --force-reinstall overwrites the CPU onnxruntime from optimum[onnxruntime]
    # with the GPU variant (includes libonnxruntime_providers_migraphx.so).
    if [ -e /opt/rocm/lib/libamdhip64.so ] || [ -e /opt/rocm/lib/libamdhip64.so.7 ]; then
        echo "==> AMD GPU/ROCm detected — installing onnxruntime-migraphx"
        /usr/bin/pip3 install --force-reinstall --no-deps --no-cache-dir --target "$lib" \
            'onnxruntime-migraphx>=1.25'
        # Fix: PyPI wheel has GNU_STACK RWE (blocked on hardened kernels)
        find "$lib/onnxruntime" -name '*.so' -exec patchelf --clear-execstack {} \; 2>/dev/null || true
    elif [ -e /opt/cuda/lib64/libcudart.so ]; then
        echo "==> NVIDIA GPU detected — installing onnxruntime-gpu"
        /usr/bin/pip3 install --force-reinstall --no-deps --no-cache-dir --target "$lib" \
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
export PYTHONPATH="/usr/share/pf2e-codex/lib"
exec /usr/bin/python3 -S -m pf2e_codex.cli "$@"
WRAPPER
    chmod 755 "$pkgdir/usr/bin/pf2e-codex"
}
