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
    'python-onnxruntime-cpu: CPU ONNX inference (required, install one ONNX variant)'
    'python-onnxruntime-opt-rocm: AMD GPU ONNX acceleration'
    'python-onnxruntime-cuda: NVIDIA GPU ONNX acceleration'
    'migraphx: AMD graph optimization for faster inference'
    'rocm-opencl-runtime: AMD GPU compute runtime'
    'cuda: NVIDIA GPU compute runtime'
)
makedepends=('python-pip')
# Build from local repo (for AUR: use GitHub tarball URL)
source=()
sha256sums=()
install=pf2e-codex.install

package() {
    # Force system Python (Mise may override PATH)
    export PYTHON=/usr/bin/python3

    # Create isolated venv at install location
    /usr/bin/python3 -m venv --system-site-packages "$pkgdir/usr/share/pf2e-codex/.venv"

    # Install from PyPI (for AUR) or from local source directory (for dev)
    # pip will pull in all deps: transformers, tokenizers, optimum, rich, etc.
    "$pkgdir/usr/share/pf2e-codex/.venv/bin/pip" install --no-cache-dir "$startdir"

    # Create wrapper script to launch from venv
    mkdir -p "$pkgdir/usr/bin"
    cat > "$pkgdir/usr/bin/pf2e-codex" << 'WRAPPER'
#!/bin/sh
exec /usr/share/pf2e-codex/.venv/bin/pf2e-codex "$@"
WRAPPER
    chmod 755 "$pkgdir/usr/bin/pf2e-codex"
}
