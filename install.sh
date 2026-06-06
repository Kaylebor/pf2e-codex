#!/usr/bin/env bash
# PF2E Codex — one-shot installer
# Usage: curl -sSL https://raw.githubusercontent.com/Kaylebor/pf2e-codex/main/install.sh | bash
#        ./install.sh [--onnx-rocm|--onnx-cuda|--onnx-cpu|--no-onnx] [--prefix DIR]

set -euo pipefail

REPO="https://github.com/Kaylebor/pf2e-codex.git"
PREFIX="${PREFIX:-$HOME/.local/share/pf2e-codex}"
ONNX_TARGET="auto"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --onnx-rocm    Force ONNX with ROCm provider (AMD)
  --onnx-cuda    Force ONNX with CUDA provider (NVIDIA)
  --onnx-cpu     Force ONNX with CPU provider
  --no-onnx      Skip ONNX, use sentence-transformers only
  --prefix DIR   Install prefix (default: ~/.local/share/pf2e-codex)
  -h, --help     Show this message

Default behaviour:
  1. Detect ROCm (rocminfo) → install with onnxruntime-rocm
  2. Detect CUDA (nvidia-smi) → install with onnxruntime-gpu
  3. Otherwise → install with onnxruntime (CPU)
  4. If any ONNX install fails → fall back to sentence-transformers
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --onnx-rocm) ONNX_TARGET="rocm"; shift ;;
        --onnx-cuda) ONNX_TARGET="cuda"; shift ;;
        --onnx-cpu)  ONNX_TARGET="cpu"; shift ;;
        --no-onnx)   ONNX_TARGET="none"; shift ;;
        --prefix)    PREFIX="$2"; shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

detect_gpu() {
    if [[ "$ONNX_TARGET" != "auto" ]]; then
        echo "$ONNX_TARGET"
        return
    fi

    # 1. ROCm — native AMD, preferred over ZLUDA-emulated CUDA
    if command -v rocminfo &>/dev/null || [[ -d /opt/rocm ]]; then
        echo "rocm"
        return
    fi

    # 2. CUDA — real NVIDIA
    if command -v nvidia-smi &>/dev/null; then
        echo "cuda"
        return
    fi

    # 3. CPU fallback
    echo "cpu"
}

require() {
    if ! command -v "$1" &>/dev/null; then
        echo "Error: '$1' is required but not installed." >&2
        exit 1
    fi
}

main() {
    echo "=== PF2E Codex Installer ==="
    echo "Prefix: $PREFIX"

    require git
    require python3
    require uv

    GPU=$(detect_gpu)
    echo "GPU detection: $GPU"

    # Determine ONNX extras
    EXTRAS=""
    if [[ "$ONNX_TARGET" != "none" ]]; then
        if [[ "$GPU" == "rocm" ]]; then
            EXTRAS="[rocm]"
        elif [[ "$GPU" == "cuda" ]]; then
            EXTRAS="[cuda]"
        else
            EXTRAS="[onnx]"
        fi
    fi

    # On Arch: pip install the GPU-specific onnxruntime variant
    if command -v paru &>/dev/null && [[ -n "$EXTRAS" ]]; then
        echo "Detected Arch. Installing onnxruntime variant..."
        case "$GPU" in
            rocm)  pip install 'onnxruntime-migraphx>=1.25' -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/ ;;
            cuda)  pip install onnxruntime-gpu ;;
            *)     pip install onnxruntime ;;
        esac
    fi

    if [[ -d "$PREFIX/.git" ]]; then
        echo "Updating existing installation..."
        git -C "$PREFIX" pull --ff-only
    else
        echo "Cloning repository..."
        git clone --depth 1 "$REPO" "$PREFIX"
    fi

    cd "$PREFIX"

    # Create venv
    echo "Creating virtual environment..."
    uv venv

    # Install with extras
    echo "Installing pf2e-codex${EXTRAS}..."
    if uv pip install -e ".${EXTRAS}"; then
        echo "Installed with ONNX support ($GPU)"
    else
        echo "ONNX install failed, falling back to CPU-only..."
        uv pip install -e "."
    fi

    # Create wrapper script
    WRAPPER_DIR="${PREFIX/%\/share*/}/bin"
    mkdir -p "$WRAPPER_DIR"
    cat > "$WRAPPER_DIR/pf2e-codex" <<EOF
#!/usr/bin/env bash
exec "$PREFIX/.venv/bin/pf2e-codex" "\$@"
EOF
    chmod +x "$WRAPPER_DIR/pf2e-codex"

    echo ""
    echo "=== Installation complete ==="
    echo "Binary: $WRAPPER_DIR/pf2e-codex"
    echo "Data:   $PREFIX"
    echo ""
    echo "Next steps:"
    echo "  pf2e-codex index   # build the rules database (first run)"
    echo "  pf2e-codex serve   # start the MCP server"
    echo ""
    echo "Add to your shell if pf2e-codex is not in PATH:"
    echo "  export PATH=\"$WRAPPER_DIR:\$PATH\""
}

main "$@"
