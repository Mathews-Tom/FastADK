# Contribution Guidelines

Thank you for considering contributing to FastADK! This document outlines the process for contributing to the project and the standards we follow.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](code-of-conduct.md).

## How to Contribute

### Reporting Bugs

1. **Check Existing Issues**: Before creating a new issue, check if it already exists.
2. **Use the Bug Report Template**: When creating a new issue, use the bug report template.
3. **Be Specific**: Include detailed information about your environment, steps to reproduce, and error messages.
4. **Minimal Example**: If possible, provide a minimal code example that reproduces the issue.

### Suggesting Enhancements

1. **Use the Feature Request Template**: When suggesting a feature, use the feature request template.
2. **Be Clear**: Clearly describe the problem your feature would solve and how it would work.
3. **Consider Scope**: Ensure your suggestion fits with the project's scope and goals.

### Making Changes

1. **Fork the Repository**: Create your own fork of the repository.
2. **Create a Branch**: Create a branch for your changes from the `main` branch.
3. **Make Your Changes**: Make your changes following our coding standards.
4. **Write Tests**: Ensure your changes include appropriate tests.
5. **Run the Test Suite**: Make sure all tests pass.
6. **Update Documentation**: Update the documentation to reflect your changes.
7. **Submit a Pull Request**: Submit a pull request with your changes.

## Development Workflow

1. **Setup Development Environment**: Follow the [Development Setup](development-setup.md) guide.
2. **Install Dependencies**: Install development dependencies with `uv sync --dev`.
3. **Make Changes**: Make your changes in small, focused commits.
4. **Write Tests**: Write tests for any new functionality.
5. **Run Quality Checks**: Run all quality checks before submitting your changes:

```bash
# Format code
uv run ruff format .
uv run black .

# Lint code
uv run ruff check .

# Type check
uv run mypy src tests

# Run tests
uv run pytest tests/ -v --cov=fastadk
```

## Coding Standards

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines.
- Use [Black](https://black.readthedocs.io/) for code formatting.
- Follow [PEP 257](https://www.python.org/dev/peps/pep-0257/) for docstring conventions.
- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

### Type Hints

- Use type hints for all function and method signatures.
- Follow [PEP 484](https://www.python.org/dev/peps/pep-0484/) for type hints.
- Run `mypy` to check type hints.

### Imports

- Organize imports in the following order:
  1. Standard library imports
  2. Related third-party imports
  3. Local application/library-specific imports
- Sort imports alphabetically within each group.
- Use absolute imports for external packages and relative imports for internal modules.

### Documentation

- Document all public classes, methods, and functions.
- Keep documentation up-to-date with code changes.
- Use examples where appropriate.

## Git Workflow

### Branching Strategy

- `main`: The primary branch, always stable and deployable.
- `feature/*`: Feature branches for new features or enhancements.
- `bugfix/*`: Bugfix branches for bug fixes.
- `docs/*`: Documentation branches for documentation changes.
- `refactor/*`: Refactoring branches for code refactoring.

### Commit Messages

- Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification.
- Use one of the following types:
  - `feat`: A new feature
  - `fix`: A bug fix
  - `docs`: Documentation changes
  - `style`: Changes that do not affect the meaning of the code
  - `refactor`: Code changes that neither fix a bug nor add a feature
  - `perf`: Performance improvements
  - `test`: Adding or modifying tests
  - `chore`: Changes to the build process or auxiliary tools

Example:

```
feat: add semantic memory to agent context
```

### Pull Requests

- Use the pull request template.
- Link related issues.
- Describe your changes clearly.
- Ensure all checks pass.
- Request a review from maintainers.

## Release Process

1. Maintainers will periodically release new versions.
2. Releases follow [Semantic Versioning](https://semver.org/).
3. Changelogs are generated automatically from commit messages.
4. Releases are published to PyPI automatically via GitHub Actions.

## Recognition

Contributors will be recognized in the following ways:

- Listed in the project's `CONTRIBUTORS.md` file.
- Mentioned in release notes for significant contributions.
- Given credit in the documentation for features they implemented.
