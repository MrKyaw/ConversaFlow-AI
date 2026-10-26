# Agents

ConversaFlow uses a deterministic specialist routing system rather than a free-for-all multi-agent setup. A supervisor maps each intent to exactly one specialist, keeping tool permissions narrow and behaviour auditable.

## Specialist Agents

The platform defines six specialist agents, each with a focused intent scope and tool set:

| Agent | Department | Intents | Tools |
|---|---|---|---|
| **orders** | order_support | check_order_status, change_address, cancel_order, request_refund | order.lookup, order.update_address, order.cancel, refund.create |
| **scheduling** | scheduling | book_appointment, reschedule_appointment, cancel_appointment | appointment.book, appointment.reschedule, appointment.cancel |
| **billing** | billing | billing_question | (none dedicated; uses knowledge base) |
| **technical** | technical_support | technical_support | ticket.create |
| **care** | customer_care | provide_feedback, speak_to_human | (escalation-only) |

## Routing Logic

The `AgentRouter` maps an intent to its owning agent. If sentiment is strongly negative (score <= -0.5 for the technical agent, -0.25 for care), the request is escalated. Routing decisions include the reason, which surfaces in traces and the `/v1/agents/route` endpoint.

## Planner

The `Planner` answers "what would this agent do next?" over a flow graph without touching a customer session. It is useful for debugging, human review of new flows, and as the capability surface an orchestrator can inspect before handing work to a specialist. The plan is bounded: an 8-step max turns an accidental cycle into a human handoff rather than a hung request.

## Tool Registry

Each specialist has a narrow tool set. The `ToolRegistry` validates tool calls against registered specs and a deterministic demo backend (`DemoBackend`). Tool execution failures trigger appropriate handoffs or error messages.