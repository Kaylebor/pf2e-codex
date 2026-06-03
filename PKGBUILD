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

    # Detect GPU type (once)
    local gpu_type="cpu"
    if pacman -Q python-onnxruntime-opt-rocm &>/dev/null || pacman -Q python-onnxruntime-rocm &>/dev/null; then
        gpu_type="rocm"
    elif pacman -Q python-onnxruntime-cuda &>/dev/null; then
        gpu_type="cuda"
    fi

    # Install package
    cd "$pkgdir/usr/share/pf2e-codex"
    UV_LINK_MODE=copy uv pip install -e "."

    # Strip PyPI GPU packages not matching system — system python-pytorch is used instead
    if [[ "$gpu_type" == "rocm" ]]; then
        uv pip uninstall torch cuda-bindings cuda-pathfinder cuda-toolkit \n            nvidia-cublas-cu13 nvidia-cuda-cupti-cu13 nvidia-cuda-nvrtc-cu13 \n            nvidia-cudnn-cu13 nvidia-cufft-cu13 nvidia-cufile-cu13 nvidia-curand-cu13 \n            nvidia-cusolver-cu13 nvidia-cusparse-cu13 nvidia-cusparselt-cu13 \n            nvidia-nccl-cu13 nvidia-nvjitlink-cu13 nvidia-nvshmem-cu13 \n            triton 2>/dev/null || true
    elif [[ "$gpu_type" == "cuda" ]]; then
        uv pip uninstall torch-rocm triton 2>/dev/null || true
    else
        uv pip uninstall torch cuda-bindings cuda-pathfinder cuda-toolkit \n            nvidia-cublas-cu13 nvidia-cuda-cupti-cu13 nvidia-cuda-nvrtc-cu13 \n            nvidia-cudnn-cu13 nvidia-cufft-cu13 nvidia-cufile-cu13 nvidia-curand-cu13 \n            nvidia-cusolver-cu13 nvidia-cusparse-cu13 nvidia-cusparselt-cu13 \n            nvidia-nccl-cu13 nvidia-nvjitlink-cu13 nvidia-nvshmem-cu13 \n            triton torch-rocm 2>/dev/null || true
    fi

    # Ensure editable install works (hatchling pth file workaround)
    echo "$pkgdir/usr/share/pf2e-codex/src" > "$pkgdir/usr/share/pf2e-codex/.venv/lib/python3.13/site-packages/pf2e_codex.pth"

    # Create wrapper binary (uses system uv, not Mise-managed one)
    install -dm755 "$pkgdir/usr/bin"
    cat > "$pkgdir/usr/bin/pf2e-codex" <<EOF
#!/usr/bin/env bash
exec /usr/bin/uv run --project /usr/share/pf2e-codex python -m pf2e_codex.cli "\$@"
EOF
    chmod +x "$pkgdir/usr/bin/pf2e-codex"

    # Install license
    install -Dm644 LICENSE "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
}
