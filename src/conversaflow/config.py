"""Process configuration with explicit, testable parsing.

Configuration is read once at start-up. Everything is optional and has a safe
default so `git clone && make serve` works with no `.env` file.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ConfigurationError

VALID_ROLES = frozenset({"admin", "agent", "viewer"})
DEFAULT_API_KEYS = "dev-key-123:demo:admin,demo-agent-key:demo:agent,readonly-key:demo:viewer"


@dataclass(frozen=True)
class Principal:
    """An authenticated caller and the tenant it is allowed to touch."""

    key: str
    tenant_id: str
    role: str

    @property
    def can_write(self) -> bool:
        return self.role in {"admin", "agent"}

    @property
    def can_administer(self) -> bool:
        return self.role == "admin"


@dataclass(frozen=True)
class Settings:
    api_keys: Mapping[str, Principal]
    store: str = "sqlite"
    db_path: Path = Path("var/conversaflow.db")
    session_ttl_seconds: int = 3600
    max_turns: int = 60
    model_path: Path = Path("models/nlu.json")
    intent_threshold: float = 0.30
    margin_threshold: float = 0.06
    kb_min_score: float = 1.5
    rate_limit_per_minute: int = 240
    redact_logs: bool = True
    tool_timeout_seconds: float = 3.0
    stt_url: str = ""
    tts_url: str = ""
    voice_timeout_seconds: float = 5.0
    log_level: str = "INFO"
    log_format: str = "json"
    data_dir: Path = Path("data")
    extra: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        values = os.environ if env is None else env
        principals = parse_api_keys(values.get("CONVERSAFLOW_API_KEYS", DEFAULT_API_KEYS))
        store = values.get("CONVERSAFLOW_STORE", "sqlite").lower()
        if store not in {"memory", "sqlite"}:
            raise ConfigurationError(f"unsupported CONVERSAFLOW_STORE: {store!r}")
        log_format = values.get("CONVERSAFLOW_LOG_FORMAT", "json").lower()
        if log_format not in {"json", "console"}:
            raise ConfigurationError(f"unsupported CONVERSAFLOW_LOG_FORMAT: {log_format!r}")
        ttl = _positive_int(values, "CONVERSAFLOW_SESSION_TTL_SECONDS", 3600)
        max_turns = _positive_int(values, "CONVERSAFLOW_MAX_TURNS", 60)
        rate = _positive_int(values, "CONVERSAFLOW_RATE_LIMIT_PER_MINUTE", 240)
        return cls(
            api_keys=principals,
            store=store,
            db_path=Path(values.get("CONVERSAFLOW_DB_PATH", "var/conversaflow.db")),
            session_ttl_seconds=ttl,
            max_turns=max_turns,
            model_path=Path(values.get("CONVERSAFLOW_MODEL_PATH", "models/nlu.json")),
            intent_threshold=_float(values, "CONVERSAFLOW_INTENT_THRESHOLD", 0.30),
            margin_threshold=_float(values, "CONVERSAFLOW_MARGIN_THRESHOLD", 0.06),
            kb_min_score=_float(values, "CONVERSAFLOW_KB_MIN_SCORE", 1.5),
            rate_limit_per_minute=rate,
            redact_logs=_bool(values, "CONVERSAFLOW_REDACT_LOGS", True),
            tool_timeout_seconds=_float(values, "CONVERSAFLOW_TOOL_TIMEOUT_SECONDS", 3.0),
            stt_url=values.get("CONVERSAFLOW_STT_URL", ""),
            tts_url=values.get("CONVERSAFLOW_TTS_URL", ""),
            voice_timeout_seconds=_float(values, "CONVERSAFLOW_VOICE_TIMEOUT_SECONDS", 5.0),
            log_level=values.get("CONVERSAFLOW_LOG_LEVEL", "INFO").upper(),
            log_format=log_format,
            data_dir=Path(values.get("CONVERSAFLOW_DATA_DIR", "data")),
        )


def parse_api_keys(spec: str) -> dict[str, Principal]:
    """Parse ``key:tenant:role,key:tenant:role`` into principals.

    Empty entries are ignored so a trailing comma is harmless. Duplicate keys
    are rejected: silently letting the last one win is a security footgun.
    """
    principals: dict[str, Principal] = {}
    for raw in spec.split(","):
        entry = raw.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) != 3 or not all(part.strip() for part in parts):
            raise ConfigurationError(
                f"CONVERSAFLOW_API_KEYS entries must be key:tenant:role, got {entry!r}"
            )
        key, tenant, role = (part.strip() for part in parts)
        if role not in VALID_ROLES:
            raise ConfigurationError(
                f"invalid role {role!r}; expected one of {sorted(VALID_ROLES)}"
            )
        if key in principals:
            raise ConfigurationError(f"duplicate API key configured for tenant {tenant!r}")
        principals[key] = Principal(key=key, tenant_id=tenant, role=role)
    return principals


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive, got {value}")
    return value


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("non-finite value")
        if name.endswith("THRESHOLD") and not 0 <= value <= 1:
            raise ValueError("threshold outside [0, 1]")
        if name.endswith("SECONDS") and value <= 0:
            raise ValueError("timeout must be positive")
        if name.endswith("MIN_SCORE") and value < 0:
            raise ValueError("score must be non-negative")
        return value
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}") from exc


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be a boolean, got {raw!r}")
