# Development Setup

This guide will help you set up your development environment for contributing to FastADK.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver
- Git

## Clone the Repository

```bash
git clone https://github.com/aetherforge/fastadk.git
cd fastadk
```

## Setting Up a Virtual Environment

FastADK uses `uv` for dependency management. To set up your development environment:

```bash
# Create and activate a virtual environment (optional if using uv)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies including development dependencies
uv sync --dev
```

## Environment Variables

Create a `.env` file in the root directory for local development:

```env
# .env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
```

For running tests, these environment variables are not required as tests use mock providers.

## Pre-commit Hooks

FastADK uses pre-commit hooks to ensure code quality. Install them with:

```bash
uv add pre-commit
pre-commit install
```

## IDE Setup

### VSCode

FastADK includes VSCode settings in the `.vscode` directory. These settings configure:

- Black as the formatter
- Ruff for linting
- MyPy for type checking

Install the recommended extensions when prompted by VSCode.

### PyCharm

If you're using PyCharm:

1. Set the project interpreter to your virtual environment
2. Enable Black as the formatter
3. Configure Ruff for linting
4. Enable type checking with MyPy

## Running Tests

Run the test suite with:

```bash
uv run pytest
```

Run with coverage:

```bash
uv run pytest --cov=fastadk
```

## Building Documentation

To build and serve the documentation locally:

```bash
uv run -m mkdocs serve
```

Then visit `http://127.0.0.1:8000/` to view the docs.

## Development Workflow

1. Create a branch for your changes:

```bash
git checkout -b feature/your-feature-name
```

2. Make your changes
3. Run quality checks:

```bash
# Format code
uv run ruff format .
uv run black .

# Lint code
uv run ruff check .

# Type check
uv run mypy src tests

# Run tests
uv run pytest
```

4. Commit your changes following the [Conventional Commits](https://www.conventionalcommits.org/) specification
5. Push your branch and create a pull request

## Troubleshooting

### Common Issues

#### Package Not Found

If you're getting errors about packages not being found, try:

```bash
uv sync --dev
```

#### Test Failures

If tests are failing:

1. Make sure your environment variables are set correctly
2. Ensure you have the latest dependencies
3. Check for conflicting package versions

#### Documentation Build Issues

If `mkdocs build` fails:

1. Ensure you have all dependencies installed
2. Check for invalid links or references in markdown files
3. Verify image paths are correct

For further help, please open an issue on GitHub or reach out on Discord.
