"""Tests for configuration parsing."""

import pytest

from conversaflow.config import ConfigurationError, Settings, parse_api_keys


def test_default_settings():
    s = Settings.from_env({})
    assert s.store == "sqlite"
    assert "dev-key-123" in s.api_keys


def test_memory_store():
    s = Settings.from_env({"CONVERSAFLOW_STORE": "memory"})
    assert s.store == "memory"


def test_invalid_store():
    with pytest.raises(ConfigurationError):
        Settings.from_env({"CONVERSAFLOW_STORE": "redis"})


def test_invalid_log_format():
    with pytest.raises(ConfigurationError):
        Settings.from_env({"CONVERSAFLOW_LOG_FORMAT": "xml"})


def test_positive_int_invalid():
    with pytest.raises(ConfigurationError):
        Settings.from_env({"CONVERSAFLOW_SESSION_TTL_SECONDS": "0"})


def test_float_invalid():
    with pytest.raises(ConfigurationError):
        Settings.from_env({"CONVERSAFLOW_INTENT_THRESHOLD": "abc"})


def test_bool_invalid():
    with pytest.raises(ConfigurationError):
        Settings.from_env({"CONVERSAFLOW_REDACT_LOGS": "maybe"})


def test_parse_api_keys_duplicate():
    with pytest.raises(ConfigurationError):
        parse_api_keys("a:t:admin,a:t2:viewer")


def test_parse_api_keys_malformed():
    with pytest.raises(ConfigurationError):
        parse_api_keys("onlytwo:parts")


def test_parse_api_keys_empty_entries():
    result = parse_api_keys("a:t:admin,, b:t2:viewer")
    assert len(result) == 2


def test_parse_api_keys_invalid_role():
    with pytest.raises(ConfigurationError):
        parse_api_keys("a:t:superuser")


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-0.1", "1.1"])
def test_threshold_must_be_finite_probability(value):
    with pytest.raises(ConfigurationError):
        Settings.from_env({"CONVERSAFLOW_INTENT_THRESHOLD": value})


@pytest.mark.parametrize(
    "name,value",
    [
        ("CONVERSAFLOW_TOOL_TIMEOUT_SECONDS", "0"),
        ("CONVERSAFLOW_VOICE_TIMEOUT_SECONDS", "-1"),
        ("CONVERSAFLOW_KB_MIN_SCORE", "-1"),
    ],
)
def test_invalid_runtime_limits(name, value):
    with pytest.raises(ConfigurationError):
        Settings.from_env({name: value})


@pytest.mark.parametrize("spec", [" :tenant:admin", "key: :admin"])
def test_whitespace_credentials_rejected(spec):
    with pytest.raises(ConfigurationError):
        parse_api_keys(spec)
