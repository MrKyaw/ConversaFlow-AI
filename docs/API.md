# API reference

The running service exposes its authoritative schemas at `/openapi.json`, `/docs`, and `/redoc`.

## Authentication and roles

Send `X-API-Key: <key>` or `Authorization: Bearer <key>`. The configured key determines tenant identity. Session and knowledge access is tenant-scoped. Viewers can read; agents can read and operate conversations; administrators can additionally ingest and delete knowledge.

| Method | Route | Access |
| --- | --- | --- |
| GET | `/health` | Public |
| GET | `/v1/health`, `/metrics` | Authenticated |
| POST | `/v1/sessions` | Writer |
| GET | `/v1/sessions/{session_id}` | Reader, same tenant |
| DELETE | `/v1/sessions/{session_id}` | Writer, same tenant |
| POST | `/v1/turns`, `/v1/sessions/{session_id}/turns` | Writer |
| POST | `/v1/sessions/{session_id}/barge-in` | Writer |
| POST | `/v1/knowledge` | Administrator |
| GET | `/v1/knowledge` | Reader |
| DELETE | `/v1/knowledge/{document_id}` | Administrator |
| GET | `/v1/flows`, `/v1/flows/{flow_id}`, `/v1/agents` | Reader |
| POST | `/v1/agents/route`, `/v1/analyze`, `/v1/plan` | Reader |

Send JSON bodies for POST requests. A turn accepts `text`, optional `session_id`, `channel`, `locale`, and `idempotency_key`. Reuse the response session ID for subsequent turns. An idempotency key replays the cached response within that session; it does not guarantee exactly-once effects across external systems. Responses contain text, intent/confidence, dialogue state, citations, handoff information, and trace identifiers.

## Errors

```json
{"error":{"code":"unauthenticated","message":"invalid API key","request_id":"example"}}
```

Domain and request-validation errors use this envelope; validation errors may include `details`. HTTP statuses include 401 (authentication), 403 (role), 404 (missing resource), 422 (invalid input), 429 (HTTP rate limit), and 500 (unexpected failure). The framework may return its own envelope for unmatched routes or methods. `X-Request-ID` is returned for correlation; valid incoming IDs contain 1–64 letters, digits, dots, underscores, or hyphens.

## WebSocket

Connect to `/v1/stream` with an API-key header or bearer header. Browser clients may use `?api_key=...`; avoid logging query strings and use short-lived credentials in a gateway. An invalid key closes the connection with code 1008. A successful connection receives a `ready` frame.

```json
{"type":"session.create","channel":"web","locale":"en-US"}
{"type":"turn","session_id":"<returned session_id>","text":"hello"}
{"type":"ping"}
```

Supported types: `session.create`, `turn`, `barge_in`, `session.get`, `session.delete`, and `ping`. Turn replies wrap the REST-shaped response in `response`; session reads use `session`. Failures return `{"type":"error","error":{"code":"...","message":"..."}}`. Mutation frames require a writer role. HTTP rate limiting does not provide a per-frame WebSocket budget; configure connection/frame limits at the gateway.
