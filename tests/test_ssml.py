"""Tests for SSML composition."""

from conversaflow.voice.ssml import compose, escape, strip_ssml


def test_escape():
    assert escape("<b>x&y</b>") == "&lt;b&gt;x&amp;y&lt;/b&gt;"


def test_compose_basic():
    out = compose("Hello world. How are you?")
    assert out.startswith('<speak version="1.0"')
    assert "<s>Hello world.</s>" in out


def test_compose_empty():
    out = compose("   ")
    assert "<s>" not in out


def test_compose_rate_clamp():
    out = compose("hi", rate=10)
    assert 'rate="50%"' in out
    out2 = compose("hi", rate=500)
    assert 'rate="200%"' in out2


def test_compose_voice():
    out = compose("hi", voice="Joanna")
    assert '<voice name="Joanna">' in out


def test_strip_ssml():
    assert strip_ssml("<speak>hi</speak>") == "hi"


def test_locale_attribute_is_escaped():
    import xml.etree.ElementTree as ET

    from conversaflow.voice.ssml import compose

    locale = 'en-US" injected="yes'
    for text in ("hello", ""):
        root = ET.fromstring(compose(text, locale=locale))
        assert root.attrib == {
            "version": "1.0",
            "{http://www.w3.org/XML/1998/namespace}lang": locale,
        }
