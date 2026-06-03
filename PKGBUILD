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
    'python-optimum'
    'python-pydantic'
    'python-pydantic-settings'
    'python-pyyaml'
    'python-typer'
    'python-rich'
    'python-sqlite-vec'
    'python-mcp'
)
optdepends=(
    'python-onnxruntime-opt-rocm: AMD GPU ONNX acceleration (ROCm EP, recommended)'
    'python-onnxruntime-cuda: NVIDIA GPU ONNX acceleration'
    'python-onnxruntime-cpu: CPU-only ONNX'
    'migraphx: AMD graph optimization for faster inference'
)
makedepends=('python-build' 'python-installer' 'python-hatchling')
source=("git+https://github.com/Kaylebor/pf2e-codex.git#tag=v${pkgver}")
sha256sums=('SKIP')
install=pf2e-codex.install

package() {
    cd "$srcdir/pf2e-codex"

    # Force system Python (Mise may override PATH)
    export PYTHON=/usr/bin/python3

    # Build wheel
    /usr/bin/python3 -m build --wheel --outdir dist

    # Install into system Python
    /usr/bin/python3 -m installer --destdir "$pkgdir" --prefix /usr dist/*.whl
}
