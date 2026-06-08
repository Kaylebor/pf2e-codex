.PHONY: test test-chunker test-integration dev-build setup-dev clean lint typecheck build install \
        validate benchmark embed-all train-data mcp

UV := uv
PYTHON := .venv/bin/python3

# ── Developer setup ──

# Full dev environment: sync deps + bundle protobuf 34 + cleanup stale CPU files
setup-dev: dev-build
	@echo "==> Bundling protobuf 34 for MIGraphX (via onnxruntime-migraphx)..."
	@set -e; \
	CAPI=$$(find .venv -type d -name capi -path '*/onnxruntime/capi' 2>/dev/null | head -1); \
	if [ -z "$$CAPI" ]; then \
		echo "No onnxruntime capi dir found. Installing onnxruntime-migraphx..."; \
		$(UV) pip install onnxruntime-migraphx>=1.25; \
		CAPI=$$(find .venv -type d -name capi -path '*/onnxruntime/capi' 2>/dev/null | head -1); \
	fi; \
	if [ -z "$$CAPI" ]; then \
		echo "ERROR: Could not find onnxruntime capi dir"; exit 1; \
	fi; \
	if [ -f "$$CAPI/libprotobuf.so.34.1.0" ]; then \
		echo "  protobuf 34 already bundled — skipping"; \
	else \
		echo "  Downloading protobuf 34 from Arch archive..."; \
		PB_DIR=$$(mktemp -d); \
		curl -sL -o "$$PB_DIR/pb.pkg.tar.zst" \
			"https://archive.archlinux.org/packages/p/protobuf/protobuf-34.1-1-x86_64.pkg.tar.zst"; \
		tar -xaf "$$PB_DIR/pb.pkg.tar.zst" -C "$$PB_DIR"; \
		cp "$$PB_DIR/usr/lib/libprotobuf.so.34.1.0" "$$CAPI/"; \
		cp "$$PB_DIR/usr/lib/libutf8_validity.so.34.1.0" "$$CAPI/"; \
		cp "$$PB_DIR/usr/lib/libutf8_range.so.34.1.0" "$$CAPI/"; \
		(cd "$$CAPI" && for f in *.so.34.1.0; do \
			ln -sf "$$f" "$${f%.1.0}"; \
			ln -sf "$$f" "$${f%%.so.34.1.0}.so"; \
		done); \
		rm -rf "$$PB_DIR"; \
		echo "  protobuf 34 bundled -> $$CAPI"; \
	fi; \
	echo "==> Cleaning stale CPU onnxruntime files..."; \
	find "$$CAPI" -name '*.cpython-*-x86_64-linux-gnu.so' -delete 2>/dev/null || true; \
	find "$$CAPI" -name 'libonnxruntime.so.*' ! -name '*.1.25.0' -delete 2>/dev/null || true; \
	find ".venv/lib" -path '*/site-packages/onnxruntime-1.25.1.dist-info' -type d -exec rm -rf {} + 2>/dev/null || true; \
	echo "==> Setup complete. MIGraphX ready."

# Install dev dependencies
dev-build:
	$(UV) sync --group dev

# ── Lint / typecheck ──

lint:
	$(UV) run ruff check pf2e_codex/ scripts/ tests/
	$(UV) run ruff format --check pf2e_codex/ scripts/ tests/

typecheck:
	$(UV) run mypy pf2e_codex/

# ── Testing ──

test:
	$(UV) run pytest tests/ -v

test-chunker:
	$(UV) run pytest tests/test_chunker.py -v

test-integration:
	$(UV) run pytest tests/test_integration.py -v

# ── Build / install ──

build:
	$(UV) build

install:
	$(UV) pip install -e .

# ── Clean ──

clean:
	rm -rf .venv/ dist/ *.egg-info/ __pycache__/ .pytest_cache/ .ruff_cache/ .mypy_cache/
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
	rm -rf training_data/dataset.jsonl training_data/raw/ training_data/*.errors training_data/*.flipped.jsonl
	@echo "Cleaned."

# ── pf2e-codex commands ──

validate:
	$(UV) run pf2e-codex validate

benchmark:
	$(UV) run pf2e-codex benchmark

embed-all:
	$(UV) run pf2e-codex embed --all-models

train-data:
	$(UV) run python3 scripts/merge-training-data.py

mcp:
	$(UV) run pf2e-codex mcp

mcp-http:
	$(UV) run pf2e-codex mcp -t streamable-http --host 0.0.0.0 --port 8080

# ── Help ──

help:
	@echo "pf2e-codex Makefile targets:"
	@echo ""
	@echo "  setup-dev        Full dev setup (deps + protobuf 34 bundle + cleanup)"
	@echo "  dev-build        Install dev dependencies"
	@echo "  lint             Run ruff check + format check"
	@echo "  typecheck        Run mypy"
	@echo "  test             Run all tests"
	@echo "  test-chunker     Run chunker tests"
	@echo "  test-integration Run integration tests"
	@echo "  build            Build pip package"
	@echo "  install          Install project in editable mode"
	@echo "  clean            Remove build artifacts, caches, venv"
	@echo "  validate         Run pf2e-codex validation suite"
	@echo "  benchmark        Run pf2e-codex benchmark"
	@echo "  embed-all        Embed all models"
	@echo "  train-data       Merge training data"
	@echo "  mcp              Start MCP server (stdio)"
	@echo "  mcp-http         Start MCP server (streamable-http on :8080)"
