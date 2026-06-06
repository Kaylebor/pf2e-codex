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
    'patchelf'
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

    # Pin torch to CPU-only (ONNX export is CPU-only, GPU handles inference).
    cat > /tmp/pf2e-torch-constraint.txt << 'EOF'
torch==2.12.0+cpu
EOF

    # Install pf2e-codex + deps (no onnxruntime — pulled separately below).
    /usr/bin/pip3 install --no-cache-dir --target "$lib" \
        --constraint /tmp/pf2e-torch-constraint.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "$startdir"

    rm -f /tmp/pf2e-torch-constraint.txt

    # ── GPU detection ──
    # Install ONLY the onnxruntime variant for this hardware. No fallbacks.
    if [ -e /opt/rocm/lib/libamdhip64.so ] || [ -e /opt/rocm/lib/libamdhip64.so.7 ]; then
        echo "==> AMD GPU detected — onnxruntime-migraphx from AMD repo"
        /usr/bin/pip3 install --no-cache-dir --target "$lib" \
            'onnxruntime-migraphx>=1.25' \
            -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/
        # Fix: AMD wheel may have GNU_STACK RWE (blocked on hardened kernels)
        find "$lib/onnxruntime" -name '*.so' -exec patchelf --clear-execstack {} \; 2>/dev/null || true
    elif [ -e /opt/cuda/lib64/libcudart.so ]; then
        echo "==> NVIDIA GPU detected — onnxruntime-gpu"
        /usr/bin/pip3 install --no-cache-dir --target "$lib" 'onnxruntime-gpu'
    else
        echo "==> No GPU detected — onnxruntime (CPU)"
        /usr/bin/pip3 install --no-cache-dir --target "$lib" 'onnxruntime>=1.20'
    fi

    # PEP 420 namespace packages (optimum, etc.) collide with system packages.
    # Only target specific packages, not everything (would break .so modules).
    for pkg in optimum; do
        find "$lib/$pkg" -type d -not -path '*/__pycache__*' | while IFS= read -r dir; do
            if [ ! -f "$dir/__init__.py" ]; then
                touch "$dir/__init__.py" 2>/dev/null || true
            fi
        done
    done

    # Wrapper: PYTHONPATH with only our lib, -S excludes system site-packages.
    # onnxruntime is copied into lib from system opt-rocm during build.
    mkdir -p "$pkgdir/usr/bin"
    cat > "$pkgdir/usr/bin/pf2e-codex" << 'WRAPPER'
#!/bin/sh
export PYTHONPATH="/usr/share/pf2e-codex/lib"
exec /usr/bin/python3 -S -m pf2e_codex.cli "$@"
WRAPPER
    chmod 755 "$pkgdir/usr/bin/pf2e-codex"
}
