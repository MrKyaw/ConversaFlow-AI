"""Structured JSON logging with PII redaction.

Logs are operational data, not a transcript store. Emails, phone numbers and
payment-card-like digit runs are redacted before they reach any sink, which is
the simplest defensible default for a customer-facing service.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Any

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"(?<!\w)(\+?\d[\d ().-]{7,}\d)(?!\w)")
_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _redact_phone(match: re.Match[str]) -> str:
    """Keep ISO dates and other calendar-like values out of the phone mask."""

    return match.group(0) if _ISO_DATE_RE.match(match.group(0)) else "[redacted-phone]"


REDACTIONS = (
    (_EMAIL_RE, "[redacted-email]"),
    (_CARD_RE, "[redacted-card]"),
    (_IBAN_RE, "[redacted-iban]"),
    (_PHONE_RE, _redact_phone),
)


def redact(text: str) -> str:
    redacted = text
    for pattern, replacement in REDACTIONS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
        }
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = redact(value) if isinstance(value, str) else value
        if record.exc_info:
            payload["exception"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=UTC).strftime("%H:%M:%S")
        return f"{timestamp} {record.levelname:<7} {record.name}: {redact(record.getMessage())}"


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else ConsoleFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    **fields: Any,
) -> None:
    logger.log(level, message, extra={"extra_fields": fields})


def new_trace_id() -> str:
    import secrets

    return secrets.token_hex(16)
