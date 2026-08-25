.PHONY: test test-chunker test-integration dev-build setup-dev setup-dev-amd \
        setup-dev-nvidia setup-dev-cpu clean lint typecheck build install \
        validate benchmark embed-all train-data mcp

UV := uv
PYTHON := .venv/bin/python3

# ── Developer setup ──

# Detect accelerator hardware unless PF2E_DEV_ACCELERATOR explicitly selects
# amd, nvidia, or cpu. CPU is only selected when no supported GPU is detected.
setup-dev: dev-build
	@accelerator="$${PF2E_DEV_ACCELERATOR:-auto}"; \
	if [ "$$accelerator" = auto ]; then \
		has_nvidia=0; has_amd=0; \
		for vendor_path in /sys/class/drm/card[0-9]*/device/vendor; do \
			[ -r "$$vendor_path" ] || continue; \
			read -r vendor < "$$vendor_path"; \
			case "$$vendor" in \
				0x10de) has_nvidia=1 ;; \
				0x1002) has_amd=1 ;; \
			esac; \
		done; \
		if [ "$$has_nvidia" = 1 ]; then \
			accelerator=nvidia; \
		elif [ "$$has_amd" = 1 ]; then \
			accelerator=amd; \
		else \
			accelerator=cpu; \
		fi; \
	fi; \
	case "$$accelerator" in \
		amd) $(MAKE) --no-print-directory setup-dev-amd ;; \
		nvidia) $(MAKE) --no-print-directory setup-dev-nvidia ;; \
		cpu) $(MAKE) --no-print-directory setup-dev-cpu ;; \
		*) echo "ERROR: PF2E_DEV_ACCELERATOR must be auto, amd, nvidia, or cpu"; exit 2 ;; \
	esac

# AMD dev environment: install only MIGraphX and bundle protobuf 34.
setup-dev-amd:
	@echo "==> Replacing any ONNX Runtime variant with AMD MIGraphX..."
	@$(UV) pip uninstall onnxruntime onnxruntime-rocm onnxruntime-gpu \
		onnxruntime-migraphx >/dev/null 2>&1 || true
	@$(UV) pip install --no-deps 'onnxruntime-migraphx>=1.25' \
		-f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/
	@echo "==> Bundling protobuf 34 for MIGraphX..."
	@set -e; \
	CAPI=$$(find .venv -type d -name capi -path '*/onnxruntime/capi' 2>/dev/null | head -1); \
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
	$(PYTHON) -c 'import onnxruntime as ort; providers=ort.get_available_providers(); assert "MIGraphXExecutionProvider" in providers, providers; print("==> MIGraphX ready:", providers)'

setup-dev-nvidia:
	@echo "==> Replacing any ONNX Runtime variant with NVIDIA CUDA..."
	@$(UV) pip uninstall onnxruntime onnxruntime-rocm onnxruntime-gpu \
		onnxruntime-migraphx >/dev/null 2>&1 || true
	@$(UV) pip install onnxruntime-gpu
	@$(PYTHON) -c 'import onnxruntime as ort; providers=ort.get_available_providers(); assert "CUDAExecutionProvider" in providers, providers; print("==> CUDA ready:", providers)'

setup-dev-cpu:
	@echo "==> No supported GPU detected; installing explicit CPU fallback..."
	@$(UV) pip uninstall onnxruntime onnxruntime-rocm onnxruntime-gpu \
		onnxruntime-migraphx >/dev/null 2>&1 || true
	@$(UV) pip install 'onnxruntime>=1.20'
	@$(PYTHON) -c 'import onnxruntime as ort; providers=ort.get_available_providers(); assert providers == ["CPUExecutionProvider"], providers; print("==> CPU fallback ready:", providers)'

# Install dev dependencies
dev-build:
	$(UV) sync --group dev --extra corpus --extra dev --inexact

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
	PF2E_RUN_INTEGRATION=1 $(UV) run pytest tests/test_integration.py -v

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
	@echo "  setup-dev        Auto-detect AMD/NVIDIA; CPU only when no GPU is usable"
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
