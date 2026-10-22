# Deployment and operations

## Container

```sh
export CONVERSAFLOW_API_KEYS='replace-with-a-secret:your-tenant:admin'
docker compose up --build -d
curl http://127.0.0.1:8000/health
```

Compose requires your API-key environment value, binds the host port to loopback, and persists sessions in a named volume. Put a TLS reverse proxy in front of the service. Keep one worker; in-process coordination and knowledge mutations are not shared across workers. The image trains the model at build time and runs as an unprivileged user.

For local development with the sample key, use `conversaflow serve --host 127.0.0.1` directly.

## Before customer traffic

1. Replace all demo credentials and install TLS, request size limits, connection limits, and WebSocket frame limits at the gateway.
2. Replace demonstration tools with tenant-authorized business integrations. Add timeouts, retry and idempotency policies appropriate to each external system.
3. Review training examples, flow text, and knowledge documents for your domain; evaluate held-out real traffic with privacy controls.
4. Pretrain and version the model and corpus together. Pin dependencies using your deployment resolver and scan the resulting image.
5. Define retention, deletion, backup, recovery, and audit policies for session payloads. SQLite storage is not encrypted by this runtime.
6. Validate resource limits and latency against expected traffic. Establish alerts and a tested rollback procedure.

## Health and monitoring

`GET /health` reports runtime health; authenticated `GET /metrics` exports Prometheus text. Scrape with a scoped key, collect stdout logs, and correlate failures using request IDs. Health is a runtime status signal, not an end-to-end check of your future business integrations. Knowledge changes, rate counters, metrics, and sample tool state reset on restart.

## Backup and restore

Use SQLite's online backup API or stop the service before copying the database. Do not copy a live database while ignoring its WAL file. Back up corpus/flow sources and the model alongside deployment configuration, excluding secrets from source control. Restore into an isolated environment and verify session access, tenant isolation, and tool configuration before routing traffic.

## Release and rollback

Run the checks in `CONTRIBUTING.md`, review the changelog and validation evidence, then create an annotated semantic version tag. Deploy a versioned image and matching data/model artifacts. Roll back code and artifacts together. No automatic database migration framework or external business-system rollback is provided.
