"""Typed, validated tools with timeouts and a deterministic demo backend.

Tools are the only place where the dialogue can cause side effects. Every call
is validated against a JSON-schema-like contract *before* the handler runs, and
the result is a value object rather than an exception. The dialogue layer can
therefore recover from a failed tool without unwinding the conversation.

Handlers are in-memory for the demo; the seam is ``ToolRegistry.register``.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..errors import ValidationError

Handler = Callable[[dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any]]]
_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, dict[str, Any]]
    side_effect: bool = False
    timeout_seconds: float | None = None
    allow_extra: bool = False


@dataclass
class ToolResult:
    tool: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0.0
    attempts: int = 1


class ToolRegistry:
    def __init__(self, default_timeout: float = 3.0) -> None:
        self.default_timeout = default_timeout
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, Handler] = {}

    def register(self, spec: ToolSpec, handler: Handler) -> None:
        if spec.name in self._specs:
            raise ValidationError(f"tool {spec.name!r} is already registered")
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def specs(self) -> list[ToolSpec]:
        return [self._specs[name] for name in sorted(self._specs)]

    def has(self, name: str) -> bool:
        return name in self._specs

    async def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        spec = self._specs.get(name)
        if spec is None:
            return ToolResult(tool=name, ok=False, error=f"unknown tool {name!r}")
        try:
            validate_arguments(spec, args)
        except ValidationError as exc:
            return ToolResult(
                tool=name,
                ok=False,
                error=exc.message,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
            )
        timeout = spec.timeout_seconds or self.default_timeout
        try:
            outcome = self._handlers[name](args)
            if inspect.isawaitable(outcome):
                outcome = await asyncio.wait_for(outcome, timeout=timeout)
        except TimeoutError:
            return ToolResult(
                tool=name,
                ok=False,
                error=f"tool timed out after {timeout:g}s",
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
            )
        except Exception as exc:
            return ToolResult(
                tool=name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
            )
        return ToolResult(
            tool=name,
            ok=True,
            data=dict(outcome),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )


def validate_arguments(spec: ToolSpec, args: dict[str, Any]) -> None:
    schema = spec.input_schema
    unknown = set(args) - set(schema)
    if unknown and not spec.allow_extra:
        raise ValidationError(f"{spec.name}: unexpected argument(s) {', '.join(sorted(unknown))}")
    for field_name, rules in schema.items():
        required = bool(rules.get("required", False))
        if field_name not in args or args[field_name] is None:
            if required:
                raise ValidationError(f"{spec.name}: missing required argument {field_name!r}")
            continue
        value = args[field_name]
        expected = rules.get("type", "string")
        if expected == "integer" and isinstance(value, bool):
            raise ValidationError(f"{spec.name}.{field_name} must be an integer")
        if not isinstance(value, _TYPE_CHECKS[expected]):
            raise ValidationError(
                f"{spec.name}.{field_name} must be {expected}, got {type(value).__name__}"
            )
        if "enum" in rules and value not in rules["enum"]:
            raise ValidationError(
                f"{spec.name}.{field_name} must be one of {rules['enum']}, got {value!r}"
            )
        if "minimum" in rules and value < rules["minimum"]:
            raise ValidationError(f"{spec.name}.{field_name} must be >= {rules['minimum']}")
        if "maximum" in rules and value > rules["maximum"]:
            raise ValidationError(f"{spec.name}.{field_name} must be <= {rules['maximum']}")
        if "minLength" in rules:
            if not isinstance(value, str):
                raise ValidationError(f"{spec.name}.{field_name} must be a string")
            if len(value) < rules["minLength"]:
                raise ValidationError(
                    f"{spec.name}.{field_name} must be at least {rules['minLength']} characters"
                )
        if "pattern" in rules and not re.fullmatch(rules["pattern"], str(value)):
            raise ValidationError(
                f"{spec.name}.{field_name} must match {rules['pattern']!r}, got {value!r}"
            )


class DemoBackend:
    """Deterministic domain service used by the reference flows and tests."""

    def __init__(self) -> None:
        self.orders: dict[str, dict[str, Any]] = {
            "ORD-1234": {
                "order_id": "ORD-1234",
                "status": "shipped",
                "carrier": "UPS",
                "eta": "2026-10-26",
                "city": "Toronto",
                "total": 89.5,
            },
            "ORD-4821": {
                "order_id": "ORD-4821",
                "status": "processing",
                "carrier": None,
                "eta": "2026-10-29",
                "city": "Vancouver",
                "total": 42.0,
            },
            "ORD-7788": {
                "order_id": "ORD-7788",
                "status": "delivered",
                "carrier": "FedEx",
                "eta": "2026-10-20",
                "city": "Austin",
                "total": 120.0,
            },
            "ORD-3311": {
                "order_id": "ORD-3311",
                "status": "out_for_delivery",
                "carrier": "DHL",
                "eta": "2026-10-22",
                "city": "Montreal",
                "total": 31.25,
            },
            "ORD-5678": {
                "order_id": "ORD-5678",
                "status": "cancelled",
                "carrier": None,
                "eta": None,
                "city": "Seattle",
                "total": 18.0,
            },
        }
        self.appointments: dict[str, dict[str, Any]] = {
            "APT-2231": {
                "appointment_id": "APT-2231",
                "service": "repair",
                "date": "2026-11-03",
                "time": "10:00",
                "status": "booked",
            },
            "APT-8890": {
                "appointment_id": "APT-8890",
                "service": "consultation",
                "date": "2026-11-05",
                "time": "14:30",
                "status": "booked",
            },
        }
        self.refunds: list[dict[str, Any]] = []
        self.tickets: list[dict[str, Any]] = []
        self._sequence = 0

    def _next_id(self, prefix: str) -> str:
        self._sequence += 1
        return f"{prefix}-{1000 + self._sequence}"

    def _order(self, order_id: str) -> dict[str, Any]:
        order = self.orders.get(order_id)
        if order is None:
            raise ValueError(f"order {order_id} was not found")
        return order

    def order_lookup(self, args: dict[str, Any]) -> dict[str, Any]:
        return dict(self._order(args["order_id"]))

    def refund_create(self, args: dict[str, Any]) -> dict[str, Any]:
        order = self._order(args["order_id"])
        amount = float(args.get("amount") or order["total"])
        if amount > 500:
            raise ValueError("refunds above 500 require manual review")
        refund = {
            "refund_id": self._next_id("REF"),
            "order_id": order["order_id"],
            "amount": amount,
            "amount_display": f"${amount:,.2f}",
            "status": "approved",
            "eta": "3-5 business days",
            "reason": args.get("reason", "customer request"),
        }
        self.refunds.append(refund)
        return dict(refund)

    def address_update(self, args: dict[str, Any]) -> dict[str, Any]:
        order = self._order(args["order_id"])
        if order["status"] in {"shipped", "out_for_delivery", "delivered"}:
            raise ValueError(f"order {order['order_id']} has already shipped")
        order["city"] = args["city"]
        order["address"] = args["address"]
        return {
            "order_id": order["order_id"],
            "status": "updated",
            "address": order["address"],
            "city": order["city"],
        }

    def order_cancel(self, args: dict[str, Any]) -> dict[str, Any]:
        order = self._order(args["order_id"])
        if order["status"] in {"shipped", "out_for_delivery", "delivered"}:
            raise ValueError(f"order {order['order_id']} has already shipped")
        order["status"] = "cancelled"
        return {"order_id": order["order_id"], "status": "cancelled"}

    def appointment_book(self, args: dict[str, Any]) -> dict[str, Any]:
        appointment = {
            "appointment_id": self._next_id("APT"),
            "service": args["service"],
            "date": args["date"],
            "time": args["time"],
            "status": "booked",
        }
        self.appointments[appointment["appointment_id"]] = appointment
        return dict(appointment)

    def appointment_reschedule(self, args: dict[str, Any]) -> dict[str, Any]:
        appointment = self.appointments.get(args["appointment_id"])
        if appointment is None:
            raise ValueError(f"appointment {args['appointment_id']} was not found")
        if appointment["status"] != "booked":
            raise ValueError(
                f"appointment {appointment['appointment_id']} is {appointment['status']}"
            )
        appointment["date"] = args["date"]
        appointment["time"] = args["time"]
        return dict(appointment)

    def appointment_cancel(self, args: dict[str, Any]) -> dict[str, Any]:
        appointment = self.appointments.get(args["appointment_id"])
        if appointment is None:
            raise ValueError(f"appointment {args['appointment_id']} was not found")
        appointment["status"] = "cancelled"
        return {"appointment_id": appointment["appointment_id"], "status": "cancelled"}

    def ticket_create(self, args: dict[str, Any]) -> dict[str, Any]:
        ticket = {
            "ticket_id": self._next_id("TCK"),
            "subject": args["subject"],
            "priority": args.get("priority", "normal"),
            "status": "open",
        }
        self.tickets.append(ticket)
        return dict(ticket)


def build_tool_registry(backend: DemoBackend | None = None, timeout: float = 3.0) -> ToolRegistry:
    domain = backend or DemoBackend()
    registry = ToolRegistry(default_timeout=timeout)
    registry.register(
        ToolSpec(
            name="order.lookup",
            description="Look up the fulfilment status of an order.",
            input_schema={
                "order_id": {
                    "type": "string",
                    "required": True,
                    "pattern": r"ORD-\d{4,10}",
                }
            },
        ),
        domain.order_lookup,
    )
    registry.register(
        ToolSpec(
            name="refund.create",
            description="Create a refund for a delivered or faulty order.",
            input_schema={
                "order_id": {
                    "type": "string",
                    "required": True,
                    "pattern": r"ORD-\d{4,10}",
                },
                "reason": {"type": "string", "required": False},
                "amount": {"type": "number", "required": False, "minimum": 0.01},
            },
            side_effect=True,
        ),
        domain.refund_create,
    )
    registry.register(
        ToolSpec(
            name="order.update_address",
            description="Change the delivery address of an order that has not shipped.",
            input_schema={
                "order_id": {
                    "type": "string",
                    "required": True,
                    "pattern": r"ORD-\d{4,10}",
                },
                "address": {"type": "string", "required": True, "minLength": 5},
                "city": {"type": "string", "required": True},
            },
            side_effect=True,
        ),
        domain.address_update,
    )
    registry.register(
        ToolSpec(
            name="order.cancel",
            description="Cancel an order that has not shipped yet.",
            input_schema={
                "order_id": {
                    "type": "string",
                    "required": True,
                    "pattern": r"ORD-\d{4,10}",
                }
            },
            side_effect=True,
        ),
        domain.order_cancel,
    )
    registry.register(
        ToolSpec(
            name="appointment.book",
            description="Book a service appointment.",
            input_schema={
                "service": {
                    "type": "string",
                    "required": True,
                    "enum": ["repair", "consultation", "installation", "demo"],
                },
                "date": {"type": "string", "required": True, "pattern": r"\d{4}-\d{2}-\d{2}"},
                "time": {"type": "string", "required": True, "pattern": r"\d{2}:\d{2}"},
            },
            side_effect=True,
        ),
        domain.appointment_book,
    )
    registry.register(
        ToolSpec(
            name="appointment.reschedule",
            description="Move an existing appointment to a new date and time.",
            input_schema={
                "appointment_id": {
                    "type": "string",
                    "required": True,
                    "pattern": r"APT-\d{3,10}",
                },
                "date": {"type": "string", "required": True, "pattern": r"\d{4}-\d{2}-\d{2}"},
                "time": {"type": "string", "required": True, "pattern": r"\d{2}:\d{2}"},
            },
            side_effect=True,
        ),
        domain.appointment_reschedule,
    )
    registry.register(
        ToolSpec(
            name="appointment.cancel",
            description="Cancel an existing appointment.",
            input_schema={
                "appointment_id": {
                    "type": "string",
                    "required": True,
                    "pattern": r"APT-\d{3,10}",
                }
            },
            side_effect=True,
        ),
        domain.appointment_cancel,
    )
    registry.register(
        ToolSpec(
            name="ticket.create",
            description="Open a technical support ticket.",
            input_schema={
                "subject": {"type": "string", "required": True, "minLength": 3},
                "description": {"type": "string", "required": False},
                "priority": {
                    "type": "string",
                    "required": False,
                    "enum": ["low", "normal", "high", "urgent"],
                },
            },
            side_effect=True,
        ),
        domain.ticket_create,
    )
    return registry
