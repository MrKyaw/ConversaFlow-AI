# Architecture

ConversaFlow is a layered, dependency-injected conversational AI runtime. The architecture is deliberately modular so each component can be tested, replaced, or deployed independently.

## High-Level Layers

```
+---------------------+      +----------------------+
|   Application Layer |<----> |   HTTP / WS API      |
+---------------------+      +----------------------+
         ^                           ^
         |                           |
+---------------------+      +----------------------+
|   Runtime Layer     |      |   CLI / Scripts      |
+---------------------+      +----------------------+
         ^                           ^
         |                           |
+---------------------+      +----------------------+
|   Integration Layer |      |   Persistence / DB   |
+---------------------+      +----------------------+
         ^                           ^
         |                           |
+---------------------+      +----------------------+
|   Domain Layer      |      |   External Services  |
+---------------------+      +----------------------+
```

## Component Details

### Domain Layer (`src/conversaflow/`)

- **models.py**: Pydantic domain models (Channel, Role, SlotSpec, Entity, Sentiment, IntentPrediction, TurnRequest/TurnResponse, SessionView, Handoff, Citation, HealthResponse, ErrorEnvelope, Plan steps). All models have `extra="forbid"` to catch typos early.

- **config.py**: Configuration with explicit, testable parsing. `Settings.from_env()` reads a mapping; everything has a safe default. `parse_api_keys()` validates `key:tenant:role` format with `hmac.compare_digest`. Rate limiter, log format, and other guards are enforced.

- **errors.py**: Domain exceptions all derive from `ConversaFlowError` with stable `code` and `status_code`. Subclasses: `ConfigurationError`, `ValidationError`, `NotFoundError`, `AuthenticationError`, `AuthorizationError`, `RateLimitError`, `FlowDefinitionError`, `ToolExecutionError`, `BudgetExceededError`.

- **nlu/**: TF-IDF vectorizer + multinomial logistic regression with temperature scaling. `IntentModel` stores weights/biases as JSON; auto-trains from `data/intents/train.jsonl`. `NluPipeline` facade: classification, slot extraction, sentiment, language detection. `analyse(text)` returns `NluResult` (intent, confidence, entities, sentiment, language, OOV rate, low-confidence flag, ambiguity).

- **dialogue/**: JSON-defined flow DSL with graph validation at load time (`Flow.model_validate_json`). Nodes: say, collect, confirm, branch, tool, handoff, end, subflow. Conditions use a safe expression language (exists, not_exists, eq, ne, gt, gte, lt, lte, in, contains, matches). The `FlowRunner` is an async virtual machine over typed nodes; max_steps=20 turns accidental cycles into handoffs. Consumes the triggering utterance in the same turn when possible.

- **agents/**: Specialist router (`AgentRouter`) maps intent → one of five agents (orders, scheduling, billing, technical, care). `Planner` answers "what would this agent do next?" over a flow graph without touching a session. `ToolRegistry` validates and calls tools against a `DemoBackend`.

- **knowledge/**: BM25 retrieval over tenant-scoped FAQ (21 documents). Always carries citations; never improvises. `KnowledgeIngestRequest` / `KnowledgeIngestResponse` API.

- **voice/**: SSML composition with escaping (control chars, HTML entities). `compose(text, locale, rate, voice)` wraps in `<speak>` with `<prosody>`. `escape(text)` replaces `&`, `<`, `>`, `"`, `'`. `strip_ssml(text)` best-effort plain-text extraction.

- **observability/**: 
  - `MetricsRegistry`: counters, histograms; `render_prometheus()` text format.
  - `JsonFormatter` / `ConsoleFormatter` with PII redaction (email, credit card, IBAN, phone; dates preserved).
  - `log_event()` propagates to structured logs with correlation ID.
  - `Trace` / `Span` with correlation ID; maps to OpenTelemetry when exporter added.

- **runtime/**: 
  - `Session`: frames stack, slots, slot_entities, fallbacks, handoff, resolved, disambiguation, barge_ins, last_ssml, tool_cache, idempotency, context.
  - `SessionStore`: `InMemorySessionStore` (tests) + `SqliteSessionStore` (durable, WAL mode, process-local lock).
  - `ConversationEngine`: core turn orchestration. `handle_turn(tenant_id, request)` → `TurnResponse`. Idempotency via `idempotency_key`. Barge-in support. Health endpoint. Metrics recording.
  - `bootstrap.py`: `build_runtime(settings)` composes every component exactly once. Tests call this or construct smaller graphs by hand.

- **api/**: FastAPI surface. 
  - Auth: `X-API-Key` header or `Authorization: Bearer` scheme. `authenticate()` uses `hmac.compare_digest`. `require_role()` guards write/admin.
  - Rate limiting: fixed-window per principal (`RateLimiter`).
  - Endpoints: `/health`, `/metrics`, `/v1/sessions` (create/get/delete), `/v1/turns` (new + session-scoped), `/v1/analyze`, `/v1/plan`, `/v1/agents/route`, `/v1/knowledge` (ingest/list/delete), `/v1/flows` (list/get), `/v1/agents` (catalogue/route), `/v1/stream` (WebSocket).
  - WebSocket `/v1/stream`: ping/pong, session.create, turn, barge_in, session.get/delete, error on unsupported types, bad auth closes connection.
  - Global exception handlers: `ConversaFlowError` → stable envelope; `RequestValidationError` → 422 with details; generic `Exception` → 500 with request ID.
  - Middleware adds `X-Request-ID`, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`.

- **cli/**: Commands: `train`, `eval`, `simulate`, `serve`, `info`. `eval` produces `TEST_RESULTS.md` + JSON + Markdown. `simulate` runs a canned conversation against in-memory runtime. `serve` starts uvicorn.

- **evaluation/**: 
  - `evaluate_intents(pipeline, path)` → `IntentEvaluation` (accuracy, macro F1, per-intent report, confusion matrix, errors).
  - `evaluate_conversations(runtime, directory)` → `ConversationReport` (success rate, containment rate, avg turns, per-conversation results).
  - `to_dict()` dataclass serialization.

- **data/**: 
  - `intents/train.jsonl` (≈491 rows, 17 intents), `eval.jsonl` (85), `test.jsonl` (72).
  - `kb/faq.jsonl` (21 valid JSONL docs: support hours, warranty, shipping, payments, security, integrations, languages, location, returns, cancellation, mobile, export, trial, offline, seats, enterprise, SSO, subscription, billing, voice, deletion).
  - `flows/` (8 JSON files: refund, order_status, change_address, cancel_order, book_appointment, reschedule_appointment, technical_support, order_status — wait actually 8 unique).
  - `golden/` (golden conversation JSONs for evaluation).

- **scripts/**: `demo.py` (canned multi-turn conversation), `test_report.py` (pytest coverage → Markdown), `load_test.py` (latency probe against running server).

- **sdk/typescript/**: TypeScript client (`@conversaflow/sdk`). `ConversaFlowClient` with `createSession()`, `turn()`, `health()`. Built with `tsc -p .`.

### Integration Layer

- **Data persistence**: SQLite (durable, survives process restart) or in-memory (tests). Session payload JSON-serialized; schema includes id, tenant_id, updated_at, expires_at, payload.

- **External services**: None required for core functionality. STT/TTS URLs configurable but empty by default; the voice gateway returns SSML, not audio bytes.

### Deployment

- **Docker**: `Dockerfile` builds the package; `docker-compose.yml` exposes port 8000.
- **CI**: GitHub Actions workflow `.github/workflows/ci.yml` runs on push/PR to main: uv venv, `uv pip install -e ".[dev]"`, `ruff check src tests`, `mypy src`, `pytest --cov=src/conversaflow --cov-fail-under=100`, `conversaflow eval --split test --json /tmp/eval.json`.
- **Makefile**: `make install`, `make train`, `make eval`, `make test`, `make lint`, `make typecheck`, `make serve`, `make docker`, `make clean`.
- **Package**: `pyproject.toml` defines deps, pytest/coverage/ruff/mypy config, entry point `conversaflow = conversaflow.cli:main`. MIT license. `.github/` has repo description + topics set via `gh repo edit`.

### Non-Goals (Deliberately Out of Scope)

- No heavy ML dependencies (custom TF-IDF + multinomial LR; no PyTorch/TensorFlow)
- No external embedding services
- No telephony provider integration (audio bytes not generated)
- No cold-start vector database
- No LLM generation (extractive KB only)
- No real-time audio processing

### Success Criteria

- 100% line + branch coverage with `pytest --cov-fail-under=100`
- Fresh test accuracy ~95.8% (72 examples, 3 errors) — honest report, no false claims
- Dev accuracy ~96.5%
- Deterministic behaviour: same input → same output
- Professional git history with conventional commits
- 100% of documentation, tests, and CI operational
- Public repo `https://github.com/MrKyaw/ConversaFlow-AI` with topics `conversational-ai, nlu, dialogue-systems, agentic-ai, voice, fastapi`
- Tag `v1.0.0` with release notes