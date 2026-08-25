#!/bin/bash
# Build all embedding databases and upload them as a GitHub Release.
# Usage: ./scripts/release-dbs.sh [version] [pf2e-release]
#   version: git tag (e.g. "v0.1.0"), defaults to current date
#   pf2e-release: exact Foundry PF2E release, defaults to pf2e-8.4.1

set -euo pipefail

REPO="Kaylebor/pf2e-codex"
VERSION="${1:-v$(date +%Y.%m.%d)}"
PF2E_RELEASE="${2:-pf2e-8.4.1}"
RELEASE_DIR="${PF2E_RELEASE_DB_DIR:-$PWD/.release-dbs/$VERSION}"
PF2E_CODEX_BIN="${PF2E_CODEX_BIN:-pf2e-codex}"
MODELS=(
    "Snowflake/snowflake-arctic-embed-xs"
    "Snowflake/snowflake-arctic-embed-s"
    "Snowflake/snowflake-arctic-embed-m"
    "all-MiniLM-L6-v2"
    "intfloat/e5-small-v2"
    "BAAI/bge-m3"
)

echo "=== Building DBs for release $VERSION ==="
echo "Build directory: $RELEASE_DIR"
echo ""
mkdir -p "$RELEASE_DIR"

# Build each DB
for model in "${MODELS[@]}"; do
    safe=$(echo "$model" | tr '/' '--')
    db="pf2e_${safe}.db"
    echo "[$model]"

    "$PF2E_CODEX_BIN" embed --models "$model" --release "$PF2E_RELEASE" --rebuild \
        --corpus-scope redistributable --data-dir "$RELEASE_DIR" 2>&1 | tail -3
    echo ""
done

echo "=== Creating release $VERSION ==="

# Collect all .db files
assets=()
for model in "${MODELS[@]}"; do
    safe=$(echo "$model" | tr '/' '--')
    db="pf2e_${safe}.db"
    db_path="$RELEASE_DIR/$db"
    if [ -f "$db_path" ]; then
        "$PF2E_CODEX_BIN" audit-db "$db_path" --strict \
            --expected-release "$PF2E_RELEASE" --expected-model "$model"
        assets+=("$db_path")
    fi
done

if [ ${#assets[@]} -eq 0 ]; then
    echo "No DBs found to upload."
    exit 1
fi

echo "Assets: ${assets[*]}"
echo ""

# Create release + upload
gh release create "$VERSION" \
    --repo "$REPO" \
    --title "Pre-built embedding DBs ($VERSION)" \
    --notes "Pre-computed sqlite-vec embedding databases for all supported models.

| Model | Size |
|---|---|
$(for f in "${assets[@]}"; do name=$(basename "$f"); printf "| %s | %s |\n" "$(echo "$name" | sed 's/pf2e_//;s/\.db$//;s/--/\//g')" "$(du -h "$f" | cut -f1)"; done)

Download with \`pf2e-codex pull --all\` or pick specific models." \
    "${assets[@]}"

echo ""
echo "Release created: https://github.com/$REPO/releases/tag/$VERSION"
