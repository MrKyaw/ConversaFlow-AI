import test from 'node:test';
import assert from 'node:assert/strict';
import sdk from '../dist/index.js';
const { ConversaFlowClient } = sdk;

function clientWith(handler) {
  return new ConversaFlowClient({ baseUrl: 'https://example.test/', apiKey: 'test-key', fetchImpl: handler });
}

test('creates sessions with authenticated JSON requests', async () => {
  const client = clientWith(async (url, init) => {
    assert.equal(url, 'https://example.test/v1/sessions');
    assert.equal(init.headers['X-API-Key'], 'test-key');
    assert.deepEqual(JSON.parse(init.body), { channel: 'web', locale: 'en-US' });
    return Response.json({ session_id: 's1', expires_at: 'later' });
  });
  assert.deepEqual(await client.createSession(), { sessionId: 's1', expiresAt: 'later' });
});

test('maps turn responses and forwards idempotency keys', async () => {
  const client = clientWith(async (url, init) => {
    assert.equal(url, 'https://example.test/v1/sessions/s1/turns');
    assert.equal(JSON.parse(init.body).idempotency_key, 'unique');
    return Response.json({ session_id: 's1', turn_id: 't1', text: 'Hi', confidence: 0.9,
      intent: 'greet', resolved: true, handoff: { required: false }, trace_id: 'trace' });
  });
  const result = await client.turn({ text: 'hello', sessionId: 's1', idempotencyKey: 'unique' });
  assert.equal(result.sessionId, 's1');
  assert.equal(result.turnId, 't1');
  assert.equal(result.traceId, 'trace');
  assert.equal(result.text, 'Hi');
});

test('creates unscoped turns and reads health', async () => {
  const client = clientWith(async (url) => {
    assert.ok(url.endsWith('/v1/turns') || url.endsWith('/health'));
    return Response.json({ status: 'ok', version: '1.0.0' });
  });
  await client.turn({ text: 'hello' });
  assert.equal((await client.health()).status, 'ok');
});

test('exposes API failure status and body', async () => {
  const client = clientWith(async () => new Response('denied', { status: 403 }));
  await assert.rejects(client.health(), /403.*denied/);
});

test('encodes session identifiers as a single path component', async () => {
  const client = clientWith(async (url) => {
    assert.equal(url, 'https://example.test/v1/sessions/a%2Fb%3Fc/turns');
    return Response.json({});
  });
  await client.turn({ text: 'hello', sessionId: 'a/b?c' });
});
