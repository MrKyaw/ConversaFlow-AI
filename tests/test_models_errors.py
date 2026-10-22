"""Tests for domain errors and models."""

from conversaflow.errors import ConversaFlowError, ValidationError
from conversaflow.models import Entity, Sentiment, TurnRequest, utcnow


def test_error_base():
    err = ConversaFlowError("boom")
    assert err.message == "boom"
    assert err.code == "internal_error"
    assert err.status_code == 500


def test_validation_error():
    err = ValidationError("bad")
    assert err.status_code == 422


def test_turn_request_blank():
    import pytest
    from pydantic import ValidationError as PydanticValidationError

    with pytest.raises(PydanticValidationError):
        TurnRequest(text="   ")


def test_turn_request_extra():
    import pytest
    from pydantic import ValidationError as PydanticValidationError

    with pytest.raises(PydanticValidationError):
        TurnRequest(text="hi", foo="bar")


def test_utcnow():
    assert utcnow().tzinfo is not None


def test_entity_and_sentiment():
    e = Entity(value=1, raw="1", type="number")
    assert e.value == 1
    s = Sentiment(score=0.0, label="neutral", intensity=0.0)
    assert s.label == "neutral"
