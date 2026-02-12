# Contributing

## Getting Started

1. Setup: `make install-dev` (or use Docker: `docker compose up dev`)
2. Branch: `git checkout -b feature/your-feature-name`
3. Develop: Make changes following the guidelines below
4. Quality: `make format && make lint && make type-check`
5. Test: `make test` (coverage: `make test-cov`)
6. PR: Ensure all checks pass, submit with clear description

## Development Guidelines

**Code**: Run scripts from project root, use conventional commits (`feat:`, `fix:`), follow PEP 8, add type hints

**Branches**: `feature/<ticket-id>-description`, PRs require approval + passing CI

**Testing**: >70% coverage for `src/data/`, `src/models/`, `src/api/`; run `make test` before PRs

**Dependencies**: Managed in `pyproject.toml` via UV (`make install-dev`); Python version in `.python-version`

**Containerization**: All workflows should run in Docker containers for reproducibility
