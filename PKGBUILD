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
    'python-onnxruntime-opt-rocm'
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

    # Pin torch to CPU-only (GPU inference handled by system opt-rocm).
    # onnxruntime installed by pip into lib will be deleted below.
    cat > /tmp/pf2e-torch-constraint.txt << 'EOF'
torch==2.12.0+cpu
EOF

    # Install pf2e-codex + deps into private lib.
    /usr/bin/pip3 install --no-cache-dir --target "$lib" \
        --constraint /tmp/pf2e-torch-constraint.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "$startdir"

    rm -f /tmp/pf2e-torch-constraint.txt

    # System python-onnxruntime-opt-rocm provides onnxruntime with MIGraphX
    # and .mxr caching. Remove pip's version and copy the system one.
    rm -rf "$lib/onnxruntime" "$lib/onnxruntime-"*.dist-info 2>/dev/null || true
    /usr/bin/python3 -c "
import onnxruntime as ort, os, shutil, sys
src = os.path.dirname(ort.__file__)
dst = sys.argv[1]
shutil.copytree(src, dst, symlinks=True)
" "$lib/onnxruntime" || echo 'Warning: onnxruntime copy failed — is opt-rocm installed?'

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
