export interface ConversaFlowOptions {
  baseUrl: string;
  apiKey: string;
  fetchImpl?: typeof fetch;
}

export interface TurnRequest {
  text: string;
  sessionId?: string;
  channel?: "web" | "voice" | "whatsapp" | "slack" | "api";
  locale?: string;
  idempotencyKey?: string;
}

export interface TurnResponse {
  sessionId: string;
  turnId: string;
  text: string;
  intent?: string;
  confidence: number;
  resolved: boolean;
  handoff: { required: boolean; department?: string | null; reason?: string | null };
  traceId: string;
}

export class ConversaFlowClient {
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly fetchImpl: typeof fetch;

  constructor(options: ConversaFlowOptions) {
    this.baseUrl = options.baseUrl.replace(/\/$/, "");
    this.apiKey = options.apiKey;
    this.fetchImpl = options.fetchImpl ?? fetch;
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": this.apiKey,
        ...(init.headers ?? {}),
      },
    });
    if (!response.ok) {
      const body = await response.text();
      throw new Error(`ConversaFlow request failed (${response.status}): ${body}`);
    }
    return (await response.json()) as T;
  }

  async createSession(
    channel = "web",
    locale = "en-US",
  ): Promise<{ sessionId: string; expiresAt: string }> {
    const payload = await this.request<{ session_id: string; expires_at: string }>("/v1/sessions", {
      method: "POST",
      body: JSON.stringify({ channel, locale }),
    });
    return { sessionId: payload.session_id, expiresAt: payload.expires_at };
  }

  async turn(request: TurnRequest): Promise<TurnResponse> {
    const path = request.sessionId
      ? `/v1/sessions/${encodeURIComponent(request.sessionId)}/turns`
      : "/v1/turns";
    const payload = await this.request<{
      session_id: string;
      turn_id: string;
      text: string;
      intent?: string;
      confidence: number;
      resolved: boolean;
      handoff: TurnResponse["handoff"];
      trace_id: string;
    }>(path, {
      method: "POST",
      body: JSON.stringify({
        text: request.text,
        channel: request.channel ?? "web",
        locale: request.locale ?? "en-US",
        idempotency_key: request.idempotencyKey,
      }),
    });
    return {
      sessionId: payload.session_id,
      turnId: payload.turn_id,
      text: payload.text,
      intent: payload.intent,
      confidence: payload.confidence,
      resolved: payload.resolved,
      handoff: payload.handoff,
      traceId: payload.trace_id,
    };
  }

  async health(): Promise<{ status: string; version: string }> {
    return this.request("/health");
  }
}
