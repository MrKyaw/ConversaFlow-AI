# Changelog

Changes follow semantic versioning. Dates use ISO 8601.

## [1.0.0] - 2026-10-22

### Added

- Local intent classification, entity extraction, sentiment signals, and evaluation harness.
- Eight validated dialogue workflows, slot collection, confirmation, tool dispatch, and handoff signals.
- Tenant-aware REST and WebSocket interfaces, memory/SQLite session stores, and idempotent response replay.
- BM25 knowledge retrieval with citations, SSML composition, metrics, and structured logging.
- TypeScript SDK, automated checks, deployment guidance, and contributor documentation.

### Hardened

- Enforced writer permissions for WebSocket mutations.
- Rejected blank credential components, non-finite settings, invalid probability thresholds, and nonpositive timeouts.
- Escaped SSML locale attributes and strengthened behavioral assertions in tests.

[1.0.0]: https://github.com/MrKyaw/ConversaFlow-AI/releases/tag/v1.0.0
