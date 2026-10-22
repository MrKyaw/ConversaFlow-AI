"""Deterministic entity extraction for slot filling.

Pattern-based extraction is used where precision matters (order IDs, emails,
dates, amounts) and a small gazetteer handles cities. Every extractor returns a
canonical value plus the original span, which lets the dialogue layer quote the
user's own words in confirmations and lets evaluation score span accuracy.

Natural-language dates resolve against an injected clock. That single design
decision makes the entire dialogue testable without freezing system time.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from typing import Any

from ..models import Entity, SlotSpec
from .text import normalize

_ORDER_ID_RE = re.compile(r"\b(?:ord|order)[\s_#-]*(?:number|no\.?)?\s*[-#:]?\s*(\d{4,10})\b", re.I)
_APPOINTMENT_ID_RE = re.compile(
    r"\b(?:apt|appointment)[\s_#-]*(?:number|no\.?)?\s*[-#:]?\s*(\d{3,10})\b", re.I
)
_TRACKING_RE = re.compile(r"\b(1Z[0-9A-Z]{16}|TRK[0-9A-Z]{8,20})\b", re.I)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"(?<![\w.])(\+?\d[\d\s().-]{6,}\d)(?![\w.])")
_MONEY_RE = re.compile(
    r"(?:\$\s*(\d[\d,]*(?:\.\d{1,2})?))|(?:\b(\d[\d,]*(?:\.\d{1,2})?)\s*(?:usd|dollars?|bucks)\b)",
    re.I,
)
_NUMBER_RE = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)(?![\w.])")
_TIME_RE = re.compile(
    r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b|\b([01]?\d|2[0-3]):([0-5]\d)\b",
    re.I,
)
_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}  # fmt: skip
_MONTH_PATTERN = "|".join(sorted(_MONTHS, key=len, reverse=True))
_MONTH_DAY_RE = re.compile(
    rf"\b({_MONTH_PATTERN})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s*(\d{{4}}))?\b", re.I
)
_DAY_MONTH_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_PATTERN})\.?(?:,?\s*(\d{{4}}))?\b", re.I
)
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_SLASH_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")
_RELATIVE_DATE_RE = re.compile(
    r"\b(today|tomorrow|yesterday|next\s+(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*|"
    r"this\s+(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*|in\s+(\d{1,2})\s+(day|days|week|weeks))\b",
    re.I,
)
_WEEKDAYS = {
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
}  # fmt: skip
_CITIES = (
    "Amsterdam", "Austin", "Bangalore", "Berlin", "Boston", "Chicago", "Dubai",
    "Dublin", "London", "Los Angeles", "Madrid", "Melbourne", "Mexico City",
    "Montreal", "New York", "Paris", "San Francisco", "Seattle", "Singapore",
    "Sydney", "Tokyo", "Toronto", "Vancouver", "Yangon",
)  # fmt: skip
_CITY_RE = re.compile(r"\b(" + "|".join(re.escape(city) for city in _CITIES) + r")\b", re.I)
_PERSON_RE = re.compile(
    r"\b(?:my name is|this is|i am|i'm|call me)\s+"
    r"([A-Z][a-z]+(?:[ '-][A-Z][a-z]+){0,2})\b",
)
_NAME_TOKEN_RE = re.compile(r"\b[A-Z][a-z]+(?:[ '-][A-Z][a-z]+){1,2}\b")
_NAME_STOPWORDS = frozenset(
    """Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March
    April May June July August September October November December I The My Your Our
    Their This That These Those Please Thanks Thank Hello Hi Dear Order Invoice Refund""".split()
)


def extract_entities(
    text: str,
    specs: Mapping[str, SlotSpec],
    now: datetime | None = None,
) -> dict[str, Entity]:
    """Extract every slot in ``specs`` that has a confident match in ``text``."""

    reference = now or datetime.now()
    found: dict[str, Entity] = {}
    for name, spec in specs.items():
        extractor = _EXTRACTORS.get(spec.type, _extract_text)
        entity = extractor(text, spec, reference)
        if entity is not None and _valid(entity, spec):
            found[name] = entity
    return found


def _valid(entity: Entity, spec: SlotSpec) -> bool:
    value = entity.value
    if spec.enum:
        return str(value).casefold() in {option.casefold() for option in spec.enum}
    if spec.pattern and not re.fullmatch(spec.pattern, entity.raw):
        return False
    if spec.min_length is not None and len(str(value)) < spec.min_length:
        return False
    if spec.max_length is not None and len(str(value)) > spec.max_length:
        return False
    if spec.min_value is not None and float(value) < spec.min_value:
        return False
    return not (spec.max_value is not None and float(value) > spec.max_value)


def _extract_order_id(text: str, spec: SlotSpec, now: datetime) -> Entity | None:
    match = _ORDER_ID_RE.search(text)
    if match is None:
        return None
    digits = match.group(1)
    return Entity(
        value=f"ORD-{digits}",
        raw=match.group(0),
        type=spec.type,
        start=match.start(),
        end=match.end(),
        confidence=0.98,
    )


def _extract_appointment_id(text: str, spec: SlotSpec, now: datetime) -> Entity | None:
    match = _APPOINTMENT_ID_RE.search(text)
    if match is None:
        return None
    return Entity(
        value=f"APT-{match.group(1)}",
        raw=match.group(0),
        type=spec.type,
        start=match.start(),
        end=match.end(),
        confidence=0.98,
    )


def _extract_tracking(text: str, spec: SlotSpec, now: datetime) -> Entity | None:
    match = _TRACKING_RE.search(text)
    return _span(match, spec.type, 0.98) if match else None


def _extract_email(text: str, spec: SlotSpec, now: datetime) -> Entity | None:
    match = _EMAIL_RE.search(text)
    return _span(match, spec.type, 0.99) if match else None


def _extract_phone(text: str, spec: SlotSpec, now: datetime) -> Entity | None:
    match = _PHONE_RE.search(text)
    if match is None:
        return None
    digits = re.sub(r"\D", "", match.group(1))
    if not 7 <= len(digits) <= 15:
        return None
    return Entity(
        value=match.group(1).strip(),
        raw=match.group(1).strip(),
        type=spec.type,
        start=match.start(1),
        end=match.end(1),
        confidence=0.9,
    )


def _extract_money(text: str, spec: SlotSpec, now: datetime) -> Entity | None:
    match = _MONEY_RE.search(text)
    if match is None:
        return None
    raw = match.group(1) or match.group(2)
    value = float(raw.replace(",", ""))
    return Entity(
        value=value, raw=match.group(0).strip(), type=spec.type,
        start=match.start(), end=match.end(), confidence=0.95,
    )  # fmt: skip


def _extract_number(text: str, spec: SlotSpec, now: datetime) -> Entity | None:
    match = _NUMBER_RE.search(text)
    if match is None:
        return None
    raw = match.group(1)
    value: int | float = float(raw) if "." in raw else int(raw)
    return Entity(
        value=value, raw=raw, type=spec.type,
        start=match.start(1), end=match.end(1), confidence=0.9,
    )  # fmt: skip


def _extract_time(text: str, spec: SlotSpec, now: datetime) -> Entity | None:
    match = _TIME_RE.search(text)
    if match is None:
        return None
    if match.group(1):
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        meridiem = match.group(3).lower()
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if meridiem == "pm" else 0)
    else:
        hour = int(match.group(4))
        minute = int(match.group(5))
    return Entity(
        value=f"{hour:02d}:{minute:02d}", raw=match.group(0).strip(), type=spec.type,
        start=match.start(), end=match.end(), confidence=0.92,
    )  # fmt: skip


def _extract_date(text: str, spec: SlotSpec, now: datetime) -> Entity | None:
    today = now.date()
    for pattern, handler in (
        (_ISO_DATE_RE, _from_iso),
        (_SLASH_DATE_RE, _from_slash),
        (_RELATIVE_DATE_RE, _from_relative),
        (_MONTH_DAY_RE, _from_month_day),
        (_DAY_MONTH_RE, _from_day_month),
    ):
        match = pattern.search(text)
        if match is None:
            continue
        resolved = handler(match, today)
        if resolved is None:
            continue
        return Entity(
            value=resolved.isoformat(), raw=match.group(0).strip(), type=spec.type,
            start=match.start(), end=match.end(), confidence=0.9,
        )  # fmt: skip
    return None


def _from_iso(match: re.Match[str], today: date) -> date | None:
    return _safe_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _from_slash(match: re.Match[str], today: date) -> date | None:
    month, day, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
    if year < 100:
        year += 2000
    return _safe_date(year, month, day)


def _from_month_day(match: re.Match[str], today: date) -> date | None:
    month = _MONTHS[match.group(1).lower()]
    day = int(match.group(2))
    year = int(match.group(3)) if match.group(3) else today.year
    resolved = _safe_date(year, month, day)
    if resolved is not None and resolved < today and match.group(3) is None:
        resolved = _safe_date(year + 1, month, day)
    return resolved


def _from_day_month(match: re.Match[str], today: date) -> date | None:
    day = int(match.group(1))
    month = _MONTHS[match.group(2).lower()]
    year = int(match.group(3)) if match.group(3) else today.year
    resolved = _safe_date(year, month, day)
    if resolved is not None and resolved < today and match.group(3) is None:
        resolved = _safe_date(year + 1, month, day)
    return resolved


def _from_relative(match: re.Match[str], today: date) -> date | None:
    phrase = normalize(match.group(1))
    if phrase == "today":
        return today
    if phrase == "tomorrow":
        return today + timedelta(days=1)
    if phrase == "yesterday":
        return today - timedelta(days=1)
    if phrase.startswith("in "):
        amount = int(match.group(2))
        unit = match.group(3).lower()
        return today + timedelta(days=amount * (7 if unit.startswith("week") else 1))
    weekday = _WEEKDAYS[phrase.split()[1][:3]]
    ahead = (weekday - today.weekday()) % 7
    if phrase.startswith("next") and ahead == 0:
        ahead = 7
    return today + timedelta(days=ahead)


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _extract_city(text: str, spec: SlotSpec, now: datetime) -> Entity | None:
    match = _CITY_RE.search(text)
    if match is None:
        return None
    canonical = next(city for city in _CITIES if city.casefold() == match.group(1).casefold())
    return Entity(
        value=canonical, raw=match.group(1), type=spec.type,
        start=match.start(), end=match.end(), confidence=0.9,
    )  # fmt: skip


def _extract_person_name(text: str, spec: SlotSpec, now: datetime) -> Entity | None:
    match = _PERSON_RE.search(text)
    if match is None:
        match = _NAME_TOKEN_RE.search(text)
    if match is None:
        return None
    candidate = match.group(1) if match.re is _PERSON_RE else match.group(0)
    first_word = re.split(r"[ '-]", candidate)[0]
    if first_word in _NAME_STOPWORDS:
        return None
    return Entity(
        value=candidate.strip(), raw=candidate.strip(), type=spec.type,
        start=match.start(), end=match.end(1) if match.re is _PERSON_RE else match.end(),
        confidence=0.75,
    )  # fmt: skip


def _extract_enum(text: str, spec: SlotSpec, now: datetime) -> Entity | None:
    normalized = normalize(text)
    for option in spec.enum:
        candidate = normalize(option)
        match = re.search(rf"\b{re.escape(candidate)}\b", normalized)
        if match is not None:
            return Entity(
                value=option,
                raw=text[match.start() : match.end()],
                type=spec.type,
                start=match.start(),
                end=match.end(),
                confidence=0.95,
            )
    for index, option in enumerate(spec.enum, start=1):
        if normalized in {str(index), f"option {index}", f"number {index}"}:
            return Entity(value=option, raw=text, type=spec.type, confidence=0.7)
    return None


def _extract_text(text: str, spec: SlotSpec, now: datetime) -> Entity | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    if spec.name == "reason":
        marker = re.search(r"\b(?:because|since)\b\s+(.+)", cleaned, re.I)
        if marker is not None:
            cleaned = marker.group(1).strip()
        cleaned = _ORDER_ID_RE.sub("", cleaned)
        cleaned = re.sub(
            r"^(?:i(?:'d| would)? (?:like|want|need) (?:a|my) refund(?: for| on)?|"
            r"refund(?: for| on)?|please refund)\s*",
            "",
            cleaned,
            flags=re.I,
        )
        cleaned = cleaned.strip(" .,-")
    elif spec.name == "address":
        if not re.search(r"\d+\s+[A-Za-z]", cleaned) and not re.search(
            r"\b(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|boulevard|blvd|way|court|ct|circle|cir|apartment|apt|suite|unit)\b",
            cleaned,
            re.I,
        ):
            return None
    if not cleaned:
        return None
    return Entity(
        value=cleaned, raw=cleaned, type=spec.type,
        start=0, end=len(text), confidence=0.5,
    )  # fmt: skip


def _span(match: re.Match[str], slot_type: str, confidence: float) -> Entity:
    raw = match.group(0).strip()
    return Entity(
        value=raw, raw=raw, type=slot_type,
        start=match.start(), end=match.end(), confidence=confidence,
    )  # fmt: skip


_EXTRACTORS = {
    "order_id": _extract_order_id,
    "appointment_id": _extract_appointment_id,
    "tracking_number": _extract_tracking,
    "email": _extract_email,
    "phone": _extract_phone,
    "money": _extract_money,
    "number": _extract_number,
    "time": _extract_time,
    "date": _extract_date,
    "city": _extract_city,
    "person_name": _extract_person_name,
    "enum": _extract_enum,
    "text": _extract_text,
}


def canonicalize(value: Any, spec: SlotSpec) -> Any:
    """Normalise a value supplied through an API or a tool payload."""

    if spec.type == "money":
        return float(value)
    if spec.type == "number":
        return float(value)
    if spec.enum:
        for option in spec.enum:
            if str(value).casefold() == option.casefold():
                return option
    return value
