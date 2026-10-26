# Skills

This directory documents the executable skills within the ConversaFlow codebase. Each skill is a self-contained unit of behaviour that can be invoked independently, tested in isolation, and composed into larger conversations.

## Available Skills

| Skill | Purpose | Key Files |
|---|---|---|
| **NLU Classification** | Intent detection, entity extraction, sentiment, language | `src/conversaflow/nlu/pipeline.py`, `src/conversaflow/nlu/classifier.py`, `src/conversaflow/nlu/entities.py` |
| **Dialogue Flow Interpreter** | Async state machine over JSON flow graphs; slots, confirmations, tools, branches, subflows, retries, loop guard | `src/conversaflow/dialogue/runner.py`, `src/conversaflow/dialogue/flow.py` |
| **Knowledge Retrieval** | BM25 over tenant-scoped FAQ; always carries citations; never improvises | `src/conversaflow/knowledge/bm25.py`, `src/conversaflow/dialogue/policy.py` |
| **Tool Execution** | Validated tool calls with timeouts; deterministic demo backend; exception handling | `src/conversaflow/agents/tools.py`, `src/conversaflow/agents/demo_backend.py` |
| **Agent Routing** | Supervisor maps intent → specialist; sentiment-based escalation; catalogue of agents | `src/conversaflow/agents/registry.py`, `src/conversaflow/agents/planner.py` |
| **Policy Decision** | Smalltalk (greeting, goodbye, thanks, affirm, deny, feedback, handoff); knowledge answers; fallback ladder | `src/conversaflow/dialogue/policy.py` |
| **SSML Composition** | Escaping, pauses, prosody; same contract for browser TTS, telephony, offline synths | `src/conversaflow/voice/ssml.py` |
| **Metrics & Observability** | In-process counters/histograms; Prometheus text export; JSON logs with PII redaction; trace IDs | `src/conversaflow/observability/metrics.py`, `src/conversaflow/observability/logging.py`, `src/conversaflow/observability/tracing.py` |
| **Session Management** | In-memory + SQLite stores; serialization; idempotency; expiry; concurrency locks | `src/conversaflow/runtime/session.py`, `src/conversaflow/runtime/stores.py` |
| **Conversation Engine** | Core turn orchestration: NLU → policy → flow → tools → response; health; barge-in; idempotency; metrics | `src/conversaflow/runtime/engine.py`, `src/conversaflow/runtime/bootstrap.py` |
| **API Surface** | FastAPI app: auth, rate limiting, endpoints (/v1/sessions, turns, KB ingest, flows, agents, plan, health, WS), error envelope | `src/conversaflow/api/app.py`, `src/conversaflow/api/auth.py`, `src/conversaflow/api/ratelimit.py` |
| **CLI** | Commands: train, eval, simulate, serve, info; eval harness with golden conversations + metrics | `src/conversaflow/cli.py`, `src/conversaflow/evaluation/harness.py` |
| **Data Loading** | Flow validation; JSONL intents; knowledge ingestion; error messages with file/line | `src/conversaflow/data_loader.py`, `src/conversaflow/dialogue/flow.py` |
| **Models & Errors** | Pydantic domain models; error envelope with stable codes; configuration validation | `src/conversaflow/models.py`, `src/conversaflow/errors.py` |

## Skill Contracts

Each skill communicates through explicit interfaces (Pydantic models, JSON schemas, or protocol buffers). There are no hidden global state dependencies; all state lives in the session or is passed as arguments. Skills are deliberately small and testable in isolation with `pytest`.

## Adding a New Skill

1. Define the input/output schema in the relevant subpackage (`models.py`, `schemas.py`, or `api/`).
2. Implement the logic in `src/conversaflow/<subpackage>/<skill>.py`.
3. Add unit tests under `tests/`.
4. Register the skill in the composition root (`bootstrap.py`) if it's part of the runtime.
5. Update CI to run the new test suite.