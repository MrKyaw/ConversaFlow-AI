# Product Requirements Document (PRD)

## Overview

ConversaFlow is a multi-turn conversational AI runtime with hybrid NLU, dialogue flows, agentic orchestration, voice/SSML, REST/WebSocket APIs, observability, evaluations, and multi-tenant auth. It is designed to be 100% test coverage with professional CI/CD, Docker, and a TypeScript SDK.

## Core Themes

- **Hybrid NLU**: TF-IDF vectorizer + multinomial logistic regression + temperature calibration
- **Dialogue Flows**: JSON-defined DSL with graph validation; 8 flows: refund, order status, change address, cancel order, book/reschedule/cancel appointment, technical support
- **Agentic Orchestration**: Specialist routing with bounded planning; deterministic in-memory demo backend plus SQLite/in-memory session stores
- **Voice/SSML**: Escaping, pauses, prosody; same response contract works for browser TTS, telephony providers and offline synthesizers
- **REST/WebSocket APIs**: FastAPI app with API-key auth (X-API-Key/Bearer), roles admin/agent/viewer, rate limiting, request-ID middleware, error envelope
- **Observability**: In-process metrics (Prometheus text exposition), structured JSON logging with PII redaction, minimal request tracing
- **Evaluations**: Intent accuracy, slot F1, task success, containment, p95 latency; JSON/Markdown reports
- **Multi-tenant**: SQLite or in-memory session stores; tenant-scoped knowledge base (BM25 retrieval)

## Non-Goals

- No heavy ML dependencies (custom vectorizer, no PyTorch/TensorFlow)
- No external embedding services
- No telephony provider integration (audio bytes not generated)

## Success Metrics

- 100% line + branch coverage with `pytest --cov-fail-under=100`
- Fresh test accuracy ~95.8% (72 examples, 3 errors) on intentionally honest test split
- Dev accuracy ~96.5%
- Deterministic behaviour: same input → same output
- Professional git history with conventional commits
- 100% of documentation, tests, and CI operational

## Target Audience

- Recruiters and hiring managers evaluating conversational AI competence
- Academics studying hybrid NLU / dialogue systems
- Top GitHub users valuing testability, docs, and professional tooling