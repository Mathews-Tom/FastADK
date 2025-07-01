.PHONY: help install install-dev test test-cov lint format type-check security clean build docs serve-docs pre-commit setup-dev

# Default target
help:
	@echo "FastADK Development Commands"
	@echo "============================"
	@echo "Setup Commands:"
	@echo "  setup-dev     - Complete development environment setup"
	@echo "  install       - Install package in production mode"
	@echo "  install-dev   - Install package with development dependencies"
	@echo ""
	@echo "Development Commands:"
	@echo "  test          - Run test suite"
	@echo "  test-cov      - Run tests with coverage report"
	@echo "  lint          - Run linting (ruff)"
	@echo "  format        - Format code (black + ruff)"
	@echo "  type-check    - Run type checking (mypy)"
	@echo "  security      - Run security scans (bandit + safety)"
	@echo "  pre-commit    - Run all pre-commit hooks"
	@echo ""
	@echo "Build & Release:"
	@echo "  clean         - Clean build artifacts"
	@echo "  build         - Build package"
	@echo ""
	@echo "Documentation:"
	@echo "  docs          - Build documentation"
	@echo "  serve-docs    - Serve documentation locally"

# Setup Commands
setup-dev: install-dev pre-commit
	@echo "Development environment setup complete!"

install:
	uv sync --no-dev

install-dev:
	uv sync --dev

# Testing
test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov=fastadk --cov-report=html --cov-report=term-missing --cov-report=xml

# Code Quality
lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run black .

type-check:
	uv run mypy src/

security:
	uv run bandit -r src/ -f json -o bandit-report.json || true
	uv run safety check --json --output safety-report.json || true
	@echo "Security reports generated: bandit-report.json, safety-report.json"

pre-commit:
	uv run pre-commit run --all-files

# Build & Clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	uv build

# Documentation (placeholder for future implementation)
docs:
	@echo "Documentation building not yet implemented"

serve-docs:
	@echo "Documentation serving not yet implemented"

# Development utilities
check-all: lint type-check test security
	@echo "All checks passed!"

# Install pre-commit hooks
install-hooks:
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg