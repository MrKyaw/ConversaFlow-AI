"""Variable interpolation for prompts and tool arguments.

Only ``${dotted.path}`` is supported. Unknown paths render as an empty string so
a half-collected slot never leaks a literal ``${slots.foo}`` into a customer
reply. Values are stringified with a small amount of human-friendly formatting.
"""

from __future__ import annotations

import re
from typing import Any

from .flow import MISSING, lookup

_VARIABLE_RE = re.compile(r"\$\{([a-z_][a-z0-9_.]*)\}")


def render(template: str, context: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        value = lookup(context, match.group(1))
        return "" if value is MISSING else format_value(value)

    return _VARIABLE_RE.sub(replace, template).strip()


def render_value(value: Any, context: dict[str, Any]) -> Any:
    """Render nested tool arguments without changing their JSON types."""

    if isinstance(value, str):
        return render(value, context)
    if isinstance(value, list):
        return [render_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: render_value(item, context) for key, item in value.items()}
    return value


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def contains_variable(template: str) -> bool:
    return bool(_VARIABLE_RE.search(template))
