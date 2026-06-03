# Maintainer: Your Name <your@email.com>
pkgname=pf2e-codex
pkgver=0.1.0
pkgrel=1
pkgdesc="PF2E rules knowledge base with MCP, CLI, and SDK interfaces"
arch=('any')
url="https://github.com/Kaylebor/pf2e-codex"
license=('MIT')
depends=('python' 'git' 'python-pytorch')
makedepends=('python-build' 'python-installer' 'python-hatchling')
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
    uv venv --system-site-packages "$pkgdir/usr/share/pf2e-codex/.venv"

    # Determine ONNX extra based on installed GPU packages
    local extras=""
    if pacman -Q python-onnxruntime-opt-rocm &>/dev/null || pacman -Q python-onnxruntime-rocm &>/dev/null; then
        extras="[rocm]"
    elif pacman -Q python-onnxruntime-cuda &>/dev/null; then
        extras="[cuda]"
    else
        extras="[onnx]"
    fi

    # Install package
    cd "$pkgdir/usr/share/pf2e-codex"
    UV_LINK_MODE=copy uv pip install -e "."

    # Remove CUDA bloat from PyPI — system python-pytorch is used instead
    .venv/bin/pip3 uninstall -y torch nvidia-cublas-cu13 nvidia-cuda-cupti-cu13 \n        nvidia-cuda-nvrtc-cu13 nvidia-cudnn-cu13 nvidia-cufft-cu13 nvidia-cufile-cu13 \n        nvidia-curand-cu13 nvidia-cusolver-cu13 nvidia-cusparse-cu13 \n        nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink-cu13 \n        nvidia-nvshmem-cu13 cuda-bindings cuda-pathfinder cuda-toolkit triton \n        2>/dev/null || true

    # Ensure editable install works (hatchling pth file workaround)
    echo "$pkgdir/usr/share/pf2e-codex/src" > "$pkgdir/usr/share/pf2e-codex/.venv/lib/python3.13/site-packages/pf2e_codex.pth"

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
