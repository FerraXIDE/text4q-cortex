# Contributing to text4q Cortex

Thank you for your interest in contributing! This document explains how to get started.

## Development setup

```bash
git clone https://github.com/YOUR_USERNAME/text4q-cortex
cd text4q-cortex
pip install -e ".[dev]"
```

## Running tests

```bash
# All tests
pytest tests/

# Unit tests only (no server needed)
pytest tests/unit/

# Integration tests (starts local FastAPI server)
pytest tests/integration/
```

All 103 tests must pass before submitting a PR.

## Project structure

```
cortex/
├── nlp/          NLP engine: natural language → OpenQASM
├── connectors/   QPU backend connectors (IBM, Aer, Google)
├── cloud/        REST API, job queue, web dashboard
├── scheduler/    QAOA-based quantum job scheduler
├── core.py       Main Cortex class (public API)
└── cli.py        Command-line interface
```

## Submitting changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes with tests
4. Run `pytest tests/` — all must pass
5. Open a Pull Request with a clear description

## Areas where help is welcome

- New circuit patterns for `cortex/nlp/engine.py`
- Additional QPU connectors (Google, IonQ, Quantinuum)
- QAOA parameter optimization (classical outer loop)
- Documentation and examples
- Performance benchmarks

## Code style

- Python 3.10+, type hints everywhere
- `ruff` for linting: `ruff check cortex/`
- Docstrings on all public functions and classes

## Questions

Open a GitHub Discussion or issue — we respond within 48 hours.
