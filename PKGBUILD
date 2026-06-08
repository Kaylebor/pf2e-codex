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
makedepends=('python-pip' 'curl')
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

    # Install pf2e-codex WITHOUT transitive deps (we pick the right onnxruntime variant below).
    /usr/bin/pip3 install --no-cache-dir --no-deps --target "$lib" "$startdir"

    # Install non-onnxruntime deps from pyproject.toml (optimum, einops, sqlite-vec, etc.)
    python3 -c "
import tomllib, json
with open('$startdir/pyproject.toml', 'rb') as f:
    deps = tomllib.load(f)['project']['dependencies']
filtered = [d for d in deps if 'onnxruntime' not in d]
print(json.dumps(filtered))
" > /tmp/pf2e-non-ort-deps.json
    /usr/bin/pip3 install --no-cache-dir --target "$lib" \
        --constraint /tmp/pf2e-torch-constraint.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        $(python3 -c "import json; print(' '.join(json.load(open('/tmp/pf2e-non-ort-deps.json'))))")
    rm -f /tmp/pf2e-non-ort-deps.json /tmp/pf2e-torch-constraint.txt

    # ── GPU detection ──
    # Install ONLY the onnxruntime variant for this hardware. No fallbacks, no CPU variant ever.
    if [ -e /opt/rocm/lib/libamdhip64.so ] || [ -e /opt/rocm/lib/libamdhip64.so.7 ]; then
        echo "==> AMD GPU detected — onnxruntime-migraphx from AMD repo"
        /usr/bin/pip3 install --no-cache-dir --target "$lib" \
            'onnxruntime-migraphx>=1.25' \
            -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/
        # ── Bundle protobuf 34 (transitive dep of MIGraphX on protobuf 35+ systems) ──
        echo "==> Bundling protobuf 34 (transitive dep of MIGraphX)"
        local pb_url="https://archive.archlinux.org/packages/p/protobuf/protobuf-34.1-1-x86_64.pkg.tar.zst"
        local pb_dir=$(mktemp -d)
        curl -sL -o "$pb_dir/protobuf-34.pkg.tar.zst" "$pb_url"
        tar -xaf "$pb_dir/protobuf-34.pkg.tar.zst" -C "$pb_dir"
        local capi_dir=$(find "$lib" -type d -name capi -path '*/onnxruntime/capi' | head -1)
        cp "$pb_dir/usr/lib/libprotobuf.so.34.1.0" "$capi_dir/"
        cp "$pb_dir/usr/lib/libutf8_validity.so.34.1.0" "$capi_dir/"
        cp "$pb_dir/usr/lib/libutf8_range.so.34.1.0" "$capi_dir/"
        (cd "$capi_dir" && for f in *.so.34.1.0; do
            ln -sf "$f" "${f%.1.0}"; ln -sf "$f" "${f%%.so.34.1.0}.so"
        done)
        rm -rf "$pb_dir"
        # Add protobuf 34 to the provider's RPATH via LD_LIBRARY_PATH in wrapper
        # (RPATH doesn't propagate through libmigraphx_c.so's own RPATH)
        # Clean up stale CPU wheel files that collide
        # (CPU 1.26.0 ships onnxruntime_pybind11_state.cpython-314-*.so which
        #  Python prefers over migraphx's onnxruntime_pybind11_state.so)
        find "$lib/onnxruntime/capi" -name '*.cpython-*-x86_64-linux-gnu.so' -delete
        find "$lib/onnxruntime/capi" -name '*.cpython-*-aarch64-linux-gnu.so' -delete 2>/dev/null || true
        find "$lib/onnxruntime" -path '*/capi/libonnxruntime.so.*' ! -name '*.1.25.0' -delete 2>/dev/null || true
        # Fix GNU_STACK RWE (hardened kernels)
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
    # Bundled protobuf 34 is pre-loaded at runtime in _preload_onnx.py.
    mkdir -p "$pkgdir/usr/bin"
    cat > "$pkgdir/usr/bin/pf2e-codex" << 'WRAPPER'
#!/bin/sh
export PYTHONPATH="/usr/share/pf2e-codex/lib"
exec /usr/bin/python3 -S -m pf2e_codex.cli "$@"
WRAPPER
    chmod 755 "$pkgdir/usr/bin/pf2e-codex"
}
