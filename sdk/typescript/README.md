# ConversaFlow TypeScript SDK

Requires Node.js 22+ for the documented build/test workflow. The client uses the standard `fetch` API and allows injection of `fetchImpl` for testing or adapters. This repository does not publish the SDK to npm automatically.

```sh
npm ci
npm run build
npm test
```

```ts
import { ConversaFlowClient } from './dist/index.js';

const client = new ConversaFlowClient({
  baseUrl: 'http://127.0.0.1:8000',
  apiKey: process.env.CONVERSAFLOW_API_KEY!,
});
const session = await client.createSession();
const response = await client.turn({
  sessionId: session.sessionId,
  text: 'where is my order ORD-1234',
  idempotencyKey: 'request-001',
});
console.log(response.text);
```

Available methods: `createSession`, `turn`, and `health`. The SDK maps response session, turn, and trace identifiers to camelCase. HTTP failures reject with an error containing status and response text. It does not implement retries or streaming. Never embed an administrator key in a public browser bundle; proxy requests through an authenticated backend.
