"""The NLU facade: classification, slot extraction, sentiment, language."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from ..models import Entity, IntentPrediction, Sentiment, SlotSpec
from .classifier import IntentModel
from .entities import extract_entities
from .language import detect_language
from .sentiment import analyse_sentiment
from .text import content_tokens, normalize


@dataclass(frozen=True)
class NluConfig:
    intent_threshold: float = 0.30
    margin_threshold: float = 0.06
    oov_suspicion: float = 0.70


@dataclass
class NluResult:
    text: str
    normalized: str
    intent: str
    confidence: float
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    entities: dict[str, Entity] = field(default_factory=dict)
    sentiment: Sentiment = field(
        default_factory=lambda: Sentiment(score=0.0, label="neutral", intensity=0.0)
    )
    language: str = "en"
    out_of_vocabulary: float = 0.0
    low_confidence: bool = False
    ambiguous: bool = False

    @property
    def runner_up(self) -> str | None:
        return self.alternatives[1][0] if len(self.alternatives) > 1 else None

    @property
    def margin(self) -> float:
        if len(self.alternatives) < 2:
            return self.alternatives[0][1] if self.alternatives else 0.0
        return self.alternatives[0][1] - self.alternatives[1][1]

    def prediction(self) -> IntentPrediction:
        return IntentPrediction(
            name=self.intent,
            confidence=round(self.confidence, 4),
            runner_up=self.runner_up,
            margin=round(self.margin, 4),
            alternatives=[(name, round(score, 4)) for name, score in self.alternatives[:3]],
        )


class NluPipeline:
    """Stateless analysis; all state lives in the dialogue layer."""

    def __init__(self, model: IntentModel, config: NluConfig | None = None) -> None:
        self.model = model
        self.config = config or NluConfig()

    def analyse(
        self,
        text: str,
        specs: Mapping[str, SlotSpec] | None = None,
        now: datetime | None = None,
        locale: str = "en-US",
    ) -> NluResult:
        ranked = self.model.ranked(text)
        intent, confidence = ranked[0]
        alternatives = [(name, round(score, 4)) for name, score in ranked[:5]]
        margin = confidence - ranked[1][1] if len(ranked) > 1 else confidence
        tokens = content_tokens(text)
        known = sum(1 for token in tokens if token in self.model.vectorizer.vocabulary)
        oov = 1.0 - (known / len(tokens)) if tokens else 1.0
        low_confidence = (
            confidence < self.config.intent_threshold or oov >= self.config.oov_suspicion
        )
        ambiguous = (
            margin < self.config.margin_threshold
            and confidence >= self.config.intent_threshold
            and len(ranked) > 1
        )
        return NluResult(
            text=text,
            normalized=normalize(text),
            intent=intent,
            confidence=round(confidence, 4),
            alternatives=alternatives,
            entities=extract_entities(text, specs or {}, now),
            sentiment=analyse_sentiment(text),
            language=detect_language(text, default=locale.split("-")[0]),
            out_of_vocabulary=round(oov, 4),
            low_confidence=low_confidence,
            ambiguous=ambiguous,
        )
