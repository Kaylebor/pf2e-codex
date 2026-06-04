#!/bin/bash
# embed-all.sh — Generate sqlite-vec DBs for all officially-supported models.
#
# Usage:
#   ./scripts/embed-all.sh [--data-dir DIR] [--concurrency N] [MODEL...]
#
# Assumes pf2e-codex is on PATH (installed via PKGBUILD or pip install).
# Runs models in parallel with configurable concurrency (default: 2, to avoid
# OOM with large 768d models on 24GB VRAM).
#
# Examples:
#   ./scripts/embed-all.sh                                    # all models
#   ./scripts/embed-all.sh snowflake-arctic-embed-s           # specific models
#   ./scripts/embed-all.sh --concurrency 3                    # more parallelism

set -euo pipefail

DATA_DIR="${PF2E_DATA_DIR:-$HOME/.local/share/pf2e-codex}"
CONCURRENCY=2
MODELS=()

# All officially-supported embedding models
ALL_MODELS=(
    "Snowflake/snowflake-arctic-embed-xs"
    "Snowflake/snowflake-arctic-embed-s"
    "Snowflake/snowflake-arctic-embed-m"
    "all-MiniLM-L6-v2"
    "intfloat/e5-small-v2"
    "nomic-ai/nomic-embed-text-v1.5"
    "BAAI/bge-m3"
)

usage() {
    echo "Usage: $0 [--data-dir DIR] [--concurrency N] [MODEL...]"
    echo ""
    echo "Generate sqlite-vec DBs for supported embedding models."
    echo "Skips models that already have a DB at the target path."
    echo ""
    echo "Supported models:"
    for m in "${ALL_MODELS[@]}"; do
        echo "  $m"
    done
    exit 1
}

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)
            DATA_DIR="$2"; shift 2 ;;
        --concurrency)
            CONCURRENCY="$2"; shift 2 ;;
        --help|-h)
            usage ;;
        -*)
            echo "Unknown flag: $1"; usage ;;
        *)
            MODELS+=("$1"); shift ;;
    esac
done

# Default to all models if none specified
if [[ ${#MODELS[@]} -eq 0 ]]; then
    MODELS=("${ALL_MODELS[@]}")
fi

mkdir -p "$DATA_DIR"

# Filter to models that don't already have a DB
PENDING=()
for model in "${MODELS[@]}"; do
    model_safe="${model//\//--}"
    db_path="$DATA_DIR/pf2e_${model_safe}.db"
    if [[ -f "$db_path" ]]; then
        echo "[skip] $model — DB exists at $db_path"
    else
        PENDING+=("$model")
    fi
done

if [[ ${#PENDING[@]} -eq 0 ]]; then
    echo "All models already indexed. Nothing to do."
    exit 0
fi

echo "Embedding ${#PENDING[@]} model(s) with concurrency=$CONCURRENCY"
echo "Data dir: $DATA_DIR"
echo ""

RESULTS_DIR=$(mktemp -d)
ACTIVE=0
FAILED=0

for model in "${PENDING[@]}"; do
    # Wait if we've hit concurrency limit
    while [[ $ACTIVE -ge $CONCURRENCY ]]; do
        wait -n 2>/dev/null || true
        ACTIVE=$((ACTIVE - 1))
    done

    (
        log="$RESULTS_DIR/${model//\//_}.log"
        echo "[start] $model"
        if pf2e-codex index -m "$model" --data-dir "$DATA_DIR" &>"$log"; then
            echo "[done]  $model"
        else
            echo "[FAIL]  $model — see $log"
            echo "FAILED" > "$RESULTS_DIR/${model//\//_}.status"
        fi
    ) &

    ACTIVE=$((ACTIVE + 1))
done

# Wait for remaining jobs
wait

# Report
echo ""
echo "=== Results ==="
for model in "${PENDING[@]}"; do
    status_file="$RESULTS_DIR/${model//\//_}.status"
    if [[ -f "$status_file" ]]; then
        echo "  FAIL: $model"
        FAILED=$((FAILED + 1))
    else
        echo "  OK:   $model"
    fi
done

rm -rf "$RESULTS_DIR"

if [[ $FAILED -gt 0 ]]; then
    echo ""
    echo "$FAILED model(s) failed."
    exit 1
fi

echo ""
echo "All models embedded successfully."
