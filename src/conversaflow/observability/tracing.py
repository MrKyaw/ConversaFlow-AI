"""Minimal request tracing.

A trace is a correlation identifier propagated to logs, responses and tool
calls. That is enough to debug a multi-turn conversation without a collector,
and the `trace_id` field maps directly onto OpenTelemetry trace IDs when an
exporter is added.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from .logging import new_trace_id


@dataclass
class Span:
    name: str
    trace_id: str
    attributes: dict[str, object] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)

    def add_event(self, name: str) -> None:
        self.events.append(name)


@dataclass
class Trace:
    trace_id: str
    spans: list[Span] = field(default_factory=list)

    @contextmanager
    def span(self, name: str, **attributes: object) -> Iterator[Span]:
        span = Span(name=name, trace_id=self.trace_id, attributes=dict(attributes))
        self.spans.append(span)
        try:
            yield span
        finally:
            pass

    @property
    def duration_attributes(self) -> dict[str, object]:
        return {f"span.{span.name}.events": len(span.events) for span in self.spans}


def new_trace() -> Trace:
    return Trace(trace_id=new_trace_id())
