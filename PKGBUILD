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

    local venv="$pkgdir/usr/share/pf2e-codex/.venv"

    # Create isolated venv — all Python deps managed internally
    /usr/bin/python3 -m venv --system-site-packages "$venv"

    # Install pf2e-codex (pulls in optimum[onnxruntime] → CPU onnxruntime, + transformers etc.)
    "$venv/bin/pip" install --no-cache-dir "$startdir"

    # ── GPU autodetection ──
    # Upgrade onnxruntime to GPU variant if hardware + system libs are present.
    # onnxruntime-migraphx wheel bundles the provider .so but needs system
    # libmigraphx + libamdhip64 → provided by migraphx + rocm-hip-runtime.
    if command -v rocminfo &>/dev/null && rocminfo &>/dev/null 2>&1; then
        echo "==> AMD GPU detected — installing onnxruntime-migraphx"
        "$venv/bin/pip" install --no-cache-dir 'onnxruntime-migraphx>=1.25'
    elif command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
        echo "==> NVIDIA GPU detected — installing onnxruntime-gpu"
        "$venv/bin/pip" install --no-cache-dir 'onnxruntime-gpu'
    else
        echo "==> No GPU detected — using CPU onnxruntime"
    fi

    # Create wrapper script (uses python3 -m to avoid baked-in build-time shebang)
    mkdir -p "$pkgdir/usr/bin"
    cat > "$pkgdir/usr/bin/pf2e-codex" << 'WRAPPER'
#!/bin/sh
exec /usr/share/pf2e-codex/.venv/bin/python3 -m pf2e_codex.cli "$@"
WRAPPER
    chmod 755 "$pkgdir/usr/bin/pf2e-codex"
}
