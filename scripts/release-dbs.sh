#!/bin/bash
# Build all embedding databases and upload them as a GitHub Release.
# Usage: ./scripts/release-dbs.sh [version]
#   version: git tag (e.g. "v0.1.0"), defaults to current date

set -euo pipefail

REPO="Kaylebor/pf2e-codex"
VERSION="${1:-v$(date +%Y.%m.%d)}"
MODELS=(
    "Snowflake/snowflake-arctic-embed-xs"
    "Snowflake/snowflake-arctic-embed-s"
    "Snowflake/snowflake-arctic-embed-m"
    "all-MiniLM-L6-v2"
    "intfloat/e5-small-v2"
    "BAAI/bge-m3"
)

echo "=== Building DBs for release $VERSION ==="
echo ""

# Build each DB
for model in "${MODELS[@]}"; do
    safe=$(echo "$model" | tr '/' '--')
    db="pf2e_${safe}.db"
    echo "[$model]"

    # Skip if already built
    if [ -f "$db" ]; then
        echo "  Already built, skipping"
        continue
    fi

    pf2e-codex embed --model "$model" --latest 2>&1 | tail -3
    # Move to working dir
    mv ~/.local/share/pf2e-codex/"$db" . 2>/dev/null || true
    echo ""
done

echo "=== Creating release $VERSION ==="

# Collect all .db files
assets=()
for model in "${MODELS[@]}"; do
    safe=$(echo "$model" | tr '/' '--')
    db="pf2e_${safe}.db"
    if [ -f "$db" ]; then
        assets+=("$db")
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
$(for f in "${assets[@]}"; do printf "| %s | %s |\n" "$(echo $f | sed 's/pf2e_//;s/\.db$//;s/--/\//g')" "$(du -h "$f" | cut -f1)"; done)

Download with \`pf2e-codex pull --all\` or pick specific models." \
    "${assets[@]}"

echo ""
echo "Release created: https://github.com/$REPO/releases/tag/$VERSION"
