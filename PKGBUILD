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
    # Copy source from PKGBUILD's directory into the build sandbox
    cd "$startdir"

    # Force system Python (Mise may override PATH)
    export PYTHON=/usr/bin/python3

    # Collect source files to package
    cp -r pyproject.toml src "$srcdir/"

    cd "$srcdir"

    # Create isolated venv — avoids all system site-package conflicts
    /usr/bin/python3 -m venv --system-site-packages "$pkgdir/usr/share/pf2e-codex/.venv"

    # Install pf2e-codex and all its Python deps into the venv
    "$pkgdir/usr/share/pf2e-codex/.venv/bin/pip" install --no-cache-dir .

    # Create wrapper script
    mkdir -p "$pkgdir/usr/bin"
    cat > "$pkgdir/usr/bin/pf2e-codex" << 'WRAPPER'
#!/bin/sh
exec /usr/share/pf2e-codex/.venv/bin/pf2e-codex "$@"
WRAPPER
    chmod 755 "$pkgdir/usr/bin/pf2e-codex"
}
