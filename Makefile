.PHONY: test test-chunker test-integration dev-build

# Run all tests via uv (isolated, no system package conflicts)
test:
	uv run --group dev pytest tests/ -v

test-chunker:
	uv run --group dev pytest tests/test_chunker.py -v

test-integration:
	uv run --group dev pytest tests/test_integration.py -v

# Install dev deps and run tests
dev-build:
	uv sync --dev
