# Configuration

Settings are read from the process environment at startup. `.env.example` documents local values but is not automatically sourced. Relative paths resolve from the working directory.

| Variable (`CONVERSAFLOW_` prefix) | Default | Meaning |
| --- | --- | --- |
| `API_KEYS` | Three local demo keys | Comma-separated `key:tenant:role`; roles `admin`, `agent`, `viewer` |
| `STORE` | `sqlite` | `memory` or `sqlite` |
| `DB_PATH` | `var/conversaflow.db` | SQLite session database |
| `SESSION_TTL_SECONDS` | `3600` | Positive session TTL |
| `MAX_TURNS` | `60` | Positive dialogue turn budget |
| `MODEL_PATH` | `models/nlu.json` | Load model here; train if absent |
| `DATA_DIR` | `data` | Flow, intent, and knowledge root |
| `INTENT_THRESHOLD` | `0.30` | Finite probability in [0, 1] |
| `MARGIN_THRESHOLD` | `0.06` | Finite probability margin in [0, 1] |
| `KB_MIN_SCORE` | `1.5` | Finite nonnegative retrieval threshold |
| `RATE_LIMIT_PER_MINUTE` | `240` | Positive per-key HTTP request limit, per process |
| `REDACT_LOGS` | `true` | Reserved setting; built-in formatters always redact recognized patterns |
| `TOOL_TIMEOUT_SECONDS` | `3.0` | Positive finite async tool timeout |
| `STT_URL` | Empty | Reserved configuration; no external adapter is wired |
| `TTS_URL` | Empty | Reserved configuration; no external adapter is wired |
| `VOICE_TIMEOUT_SECONDS` | `5.0` | Reserved positive finite adapter timeout |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `LOG_FORMAT` | `json` | `json` or `console` |

The default keys are `dev-key-123:demo:admin`, `demo-agent-key:demo:agent`, and `readonly-key:demo:viewer`. Never use them for an exposed deployment. Empty key specifications disable authenticated access; duplicate keys, blank credential components, unknown roles, and invalid numerical settings fail at startup.

```sh
export CONVERSAFLOW_API_KEYS="${YOUR_API_KEY}:your-tenant:admin"
export CONVERSAFLOW_STORE=sqlite
export CONVERSAFLOW_DB_PATH=/srv/conversaflow/sessions.db
conversaflow serve --host 127.0.0.1
```

Use a secret manager or orchestrator environment injection in production. The API key format cannot contain commas or colons inside components. Changes require restarting the process.
