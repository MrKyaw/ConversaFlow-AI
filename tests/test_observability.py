"""Tests for observability modules."""

import logging

from conversaflow.observability.logging import (
    ConsoleFormatter,
    JsonFormatter,
    configure_logging,
    log_event,
    redact,
)
from conversaflow.observability.metrics import MetricsRegistry, percentile
from conversaflow.observability.tracing import new_trace


def test_redact():
    assert "[redacted-email]" in redact("contact me at a@b.com")
    assert "[redacted-phone]" in redact("call +1 234 567 8901")
    assert redact("date 2026-10-26") == "date 2026-10-26"


def test_configure_logging_json():
    configure_logging("INFO", "json")
    assert logging.getLogger().handlers


def test_configure_logging_console():
    configure_logging("DEBUG", "console")
    assert logging.getLogger().handlers


def test_log_event():
    logger = logging.getLogger("test")
    configure_logging("INFO", "json")
    log_event(logger, logging.INFO, "msg", foo="bar")


def test_formatters():
    record = logging.LogRecord("x", logging.INFO, "", 0, "hello %s", ("world",), None)
    assert "hello world" in JsonFormatter().format(record)
    assert "hello world" in ConsoleFormatter().format(record)


def test_metrics_basic():
    m = MetricsRegistry()
    m.increment("c", labels={"a": "b"})
    assert m.counter_value("c", labels={"a": "b"}) == 1.0
    m.observe("h", 5.0)
    assert m.histogram_values("h") == [5.0]
    assert m.summary("h")["count"] == 1
    assert m.summary("missing")["count"] == 0
    text = m.render_prometheus()
    assert "c{" in text


def test_percentile_empty():
    assert percentile([], 0.5) == 0.0


def test_tracing():
    trace = new_trace()
    with trace.span("work", x=1) as span:
        span.add_event("done")
    assert trace.spans[0].events == ["done"]
    assert "span.work.events" in trace.duration_attributes
