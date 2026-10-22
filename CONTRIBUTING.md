# Contributing

Use Python 3.11 or 3.12 and Node.js 22 or newer. Create a virtual environment and install `python -m pip install -e '.[dev]'`. Run commands from the repository root.

```sh
ruff check src tests scripts
ruff format --check src tests scripts
mypy src
pytest --cov=src/conversaflow
conversaflow eval --split test
python -m pip install build
python -m build
cd sdk/typescript
npm ci
npm run build
npm test
```

Keep business logic in domain/runtime modules and transport concerns in the API. Add regression tests for changed behavior, including authorization and tenant boundaries where applicable. Prefer deterministic fixtures; never commit secrets, generated model files, local databases, or environment directories.

Open an issue describing the expected behavior and reproduction steps, or send a focused pull request. Include the problem, resulting behavior, and validation evidence. Use conventional commit subjects such as `fix: enforce viewer permissions for streaming` or `docs: explain deployment boundaries`.

## Versions and tags

Python package and TypeScript SDK versions start at `1.0.0`. Update package metadata and `CHANGELOG.md` together for releases. Use annotated tags in the form `vMAJOR.MINOR.PATCH`, with breaking changes incrementing the major version. A tag identifies reviewed source; publishing packages and deploying infrastructure are separate operations.

Documentation should describe behavior verified from code and commands. Report test pass rates, source coverage, and model accuracy separately. Do not add a badge claiming measurements that CI does not produce.
