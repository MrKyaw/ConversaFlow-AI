# Security policy

## Reporting

Report suspected vulnerabilities privately through the repository's GitHub security advisory workflow if enabled. Otherwise contact the maintainer through the contact options on [MrKyaw's GitHub profile](https://github.com/MrKyaw). Do not publish credentials, customer transcripts, or exploit details in public issues. Include affected version, reproduction steps, impact, and a minimal redacted example. No guaranteed response SLA is offered.

## Scope

Version 1.0 is the initial maintained line. Use the latest reviewed commit and check dependency advisories in your deployment pipeline.

API keys select a tenant and role. Default credentials are public examples. The runtime has no user identity provider, secret rotation service, encrypted database layer, or distributed rate limiter. Session isolation and knowledge scoping do not make the sample business backend suitable for real multi-tenant use.

The public health and OpenAPI/documentation endpoints are exposed by default. Restrict them at your ingress if your deployment requires it. Protect WebSocket credentials and apply gateway limits. See [deployment guidance](docs/DEPLOYMENT.md) for required integration and operational review.
