"""SSML composition for voice channels.

The voice gateway returns SSML rather than audio bytes so the same response
contract works for browser TTS, telephony providers and offline synthesizers.
Escaping is not optional: user text is untrusted and must never be able to
inject SSML tags.
"""

from __future__ import annotations

import re

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_ESCAPES = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;"}


def escape(text: str) -> str:
    cleaned = _CONTROL_RE.sub("", text)
    return "".join(_ESCAPES.get(character, character) for character in cleaned)


def compose(
    text: str,
    *,
    locale: str = "en-US",
    rate: int = 100,
    voice: str | None = None,
) -> str:
    """Wrap text in a minimal, widely supported SSML document."""

    locale = escape(locale)
    safe = escape(text.strip())
    if not safe:
        return f'<speak version="1.0" xml:lang="{locale}"></speak>'
    sentences = [sentence.strip() for sentence in _SENTENCE_RE.split(safe) if sentence.strip()]
    body = '<break time="250ms"/>'.join(f"<s>{sentence}</s>" for sentence in sentences)
    rate_value = min(max(rate, 50), 200)
    voice_tag = f'<voice name="{escape(voice)}">' if voice else ""
    close_tag = "</voice>" if voice else ""
    return (
        f'<speak version="1.0" xml:lang="{locale}">'
        f'<prosody rate="{rate_value}%">{voice_tag}{body}{close_tag}</prosody></speak>'
    )


def strip_ssml(text: str) -> str:
    """Best-effort plain text for logs and transcripts."""

    return re.sub(r"<[^>]+>", " ", text).strip()
