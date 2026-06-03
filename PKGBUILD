# Maintainer: Your Name <your@email.com>
pkgname=pf2e-codex
pkgver=0.1.0
pkgrel=1
pkgdesc="PF2E rules knowledge base with MCP, CLI, and SDK interfaces"
arch=('any')
url="https://github.com/Kaylebor/pf2e-codex"
license=('MIT')
depends=('python' 'python-uv' 'git')
optdepends=(
    'python-onnxruntime-opt-rocm: AMD GPU ONNX acceleration (ROCm EP, recommended)'
    'python-onnxruntime-cuda: NVIDIA GPU ONNX acceleration'
    'python-onnxruntime-cpu: CPU-only ONNX'
    'migraphx: AMD graph optimization for faster inference (any provider)'
)
makedepends=('git')
source=("git+https://github.com/Kaylebor/pf2e-codex.git#tag=v${pkgver}")
sha256sums=('SKIP')
install=pf2e-codex.install

package() {
    cd "$srcdir/pf2e-codex"

    # Install to /usr/share
    install -dm755 "$pkgdir/usr/share/pf2e-codex"
    cp -r . "$pkgdir/usr/share/pf2e-codex/"

    # Create virtual environment
    uv venv "$pkgdir/usr/share/pf2e-codex/.venv"

    # Determine ONNX extra based on installed GPU packages
    local extras=""
    if pacman -Q python-onnxruntime-opt-rocm &>/dev/null || pacman -Q python-onnxruntime-rocm &>/dev/null; then
        extras="[rocm]"
    elif pacman -Q python-onnxruntime-cuda &>/dev/null; then
        extras="[cuda]"
    else
        extras="[onnx]"
    fi

    # Install package with extras
    cd "$pkgdir/usr/share/pf2e-codex"
    if uv pip install -e ".${extras}"; then
        echo "Installed with ONNX support"
    else
        echo "ONNX extra unavailable, falling back to CPU-only"
        uv pip install -e "."
    fi

    # Create wrapper binary
    install -dm755 "$pkgdir/usr/bin"
    cat > "$pkgdir/usr/bin/pf2e-codex" <<EOF
#!/usr/bin/env bash
exec /usr/share/pf2e-codex/.venv/bin/pf2e-codex "\$@"
EOF
    chmod +x "$pkgdir/usr/bin/pf2e-codex"

    # Install license
    install -Dm644 LICENSE "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
}
