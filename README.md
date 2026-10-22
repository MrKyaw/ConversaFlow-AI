# ConversaFlow AI

[![CI](https://github.com/MrKyaw/ConversaFlow-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/MrKyaw/ConversaFlow-AI/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![MIT license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A self-contained conversational AI runtime for multi-turn support workflows. Combine a locally trained intent classifier, validated dialogue graphs, extractive knowledge retrieval, and typed tools behind a tenant-aware FastAPI service.

**No external model API is required.** The included tools are deterministic demonstration backends. Connect your own business systems before using the workflows with customers.

## What it does

- Classifies intents with TF-IDF and softmax regression, confidence thresholds, ambiguity handling, entity extraction, and sentiment signals.
- Runs eight declarative flows for orders, refunds, address changes, cancellations, appointments, billing, and technical support.
- Retrieves tenant-scoped knowledge passages with BM25 and returns citations.
- Exposes REST and WebSocket interfaces, API-key roles, session persistence, idempotent turn replay, human handoff signals, and SSML responses.
- Includes a TypeScript client, evaluation corpora, golden conversations, structured logging, Prometheus metrics, and automated checks.

## Quick start

Requires Python 3.11 or 3.12. Run commands from the repository root so the bundled `data/` directory is available.

```sh
git clone https://github.com/MrKyaw/ConversaFlow-AI.git
cd ConversaFlow-AI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
conversaflow train
conversaflow simulate "hello" "where is my order ORD-1234"
conversaflow serve --host 127.0.0.1
```

Visit [interactive API documentation](http://127.0.0.1:8000/docs) or [ReDoc](http://127.0.0.1:8000/redoc) while the server runs.

```sh
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/v1/turns \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: dev-key-123' \
  -d '{"text":"where is my order ORD-1234"}'
```

The sample API keys are for local development. Set `CONVERSAFLOW_API_KEYS` to your own secrets before exposing the service. Configuration comes from process environment variables; a `.env` file is **not loaded automatically**.

## Documentation

| Guide | Contents |
| --- | --- |
| [Architecture](docs/ARCHITECTURE.md) | Components, request lifecycle, extension points, persistence boundaries |
| [API reference](docs/API.md) | Authentication, roles, endpoints, errors, streaming frames |
| [Configuration](docs/CONFIGURATION.md) | Every environment setting and its default |
| [Deployment](docs/DEPLOYMENT.md) | Container setup, operations, backups, release checklist |
| [Development](CONTRIBUTING.md) | Setup, checks, contribution and tagging conventions |
| [Evaluation](docs/EVALUATION.md) | Dataset scope, reproduction commands, interpretation |
| [Test results](docs/TEST_RESULTS.md) | Measured validation evidence |
| [TypeScript SDK](sdk/typescript/README.md) | Build, use, and test the client |
| [Security policy](SECURITY.md) | Reporting vulnerabilities and deployment boundaries |
| [Changelog](CHANGELOG.md) | Versioned changes |

## Development

```sh
ruff check src tests scripts
ruff format --check src tests scripts
mypy src
pytest --cov=src/conversaflow
conversaflow eval --split test --markdown docs/EVALUATION_RESULTS.md
cd sdk/typescript
npm ci
npm run build
npm test
```

CI runs Python checks on 3.11 and 3.12, builds/tests the TypeScript SDK, and smoke-tests the container. Combined line/branch coverage has an explicit 80% floor. Test pass rate and source coverage are different measurements; see the generated results for the observed values. Evaluation measures the bundled synthetic corpus, not performance on customer traffic.

## Deployment scope

Use a single application worker: rate limiting, knowledge mutations, demo tool state, metrics, and dialogue coordination are process-local. SQLite persists sessions only. WhatsApp and Slack are channel labels; provider connectors, live speech transcription/audio synthesis, and external LLM integrations are not implemented. The voice interface produces SSML and supports interruption signals.

See the [deployment guide](docs/DEPLOYMENT.md) before running outside local development. This repository provides a deployable runtime foundation; production suitability depends on your integrations, traffic, data controls, and operational testing.

## License

[MIT](LICENSE). Copyright notices from the original repository are preserved.
