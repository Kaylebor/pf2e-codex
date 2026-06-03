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
makedepends=('python-pip')
# Build from local repo (for AUR: use GitHub tarball URL)
source=()
sha256sums=()
install=pf2e-codex.install

package() {
    # Force system Python (Mise may override PATH)
    export PYTHON=/usr/bin/python3

    local lib="$pkgdir/usr/share/pf2e-codex/lib"
    mkdir -p "$lib"

    # Install pf2e-codex + runtime deps (onnxruntime, transformers, tokenizers, etc.)
    # NO torch/optimum — ONNX export is a separate one-time step.
    /usr/bin/pip3 install --no-cache-dir --target "$lib" "$startdir"

    # ── GPU autodetection ──
    # Upgrade onnxruntime to GPU variant. Needs system libs: migraphx + rocm-hip-runtime.
    if command -v rocminfo &>/dev/null && rocminfo &>/dev/null 2>&1; then
        echo "==> AMD GPU detected — installing onnxruntime-migraphx"
        /usr/bin/pip3 install --no-cache-dir --target "$lib" 'onnxruntime-migraphx>=1.25'
    elif command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
        echo "==> NVIDIA GPU detected — installing onnxruntime-gpu"
        /usr/bin/pip3 install --no-cache-dir --target "$lib" 'onnxruntime-gpu'
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
