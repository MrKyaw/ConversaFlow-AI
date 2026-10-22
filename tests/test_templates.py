"""Tests for templates."""

from conversaflow.dialogue.templates import contains_variable, format_value, render, render_value


def test_render_basic():
    assert render("hi ${slots.name}", {"slots": {"name": "Ada"}}) == "hi Ada"


def test_render_missing():
    assert render("hi ${slots.missing}", {"slots": {}}) == "hi"


def test_render_format_float():
    assert format_value(5.0) == "5"
    assert format_value(True) == "yes"


def test_render_value_nested():
    ctx = {"slots": {"x": 5}}
    assert render_value({"a": "${slots.x}", "b": [1, "${slots.x}"]}, ctx) == {
        "a": "5",
        "b": [1, "5"],
    }


def test_render_value_passthrough():
    assert render_value(42, {}) == 42


def test_contains_variable():
    assert contains_variable("x ${y}") is True
    assert contains_variable("x") is False
