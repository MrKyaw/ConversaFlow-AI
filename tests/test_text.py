"""Tests for text normalisation."""

from conversaflow.nlu.text import (
    content_tokens,
    ngrams,
    normalize,
    tokenize,
    truncate,
)


def test_normalize_basic():
    assert normalize("  Hello, World!  ") == "hello world"


def test_normalize_apostrophe():
    assert normalize("Don't") == "don't"


def test_normalize_unicode():
    assert normalize("Café") == "café"


def test_tokenize():
    assert tokenize("Hello, world!") == ["hello", "world"]


def test_content_tokens_removes_stopwords():
    assert "the" not in content_tokens("the quick brown fox")


def test_ngrams():
    tokens = ["a", "b", "c"]
    result = ngrams(tokens)
    assert "a" in result
    assert any(item.startswith("b:") for item in result)


def test_truncate_short():
    assert truncate("hello", 80) == "hello"


def test_truncate_long():
    text = "word " * 30
    result = truncate(text, 20)
    assert len(result) <= 20
    assert result.endswith("…")
