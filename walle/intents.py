"""Turning a raw transcript into an action.

Vosk returns lower-case words with no punctuation, so matching is done on
normalised token sequences rather than on exact strings.

Routing is deliberately kept as pure functions with no I/O: this is the part of
the robot most likely to need tuning once you hear how it mishears you, and it
is the part that can be tested without a microphone attached.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .cities import normalise


class Mode(str, Enum):
    CITY = "CITY"
    TRANSLATE = "TRANSLATE"


class IntentKind(str, Enum):
    MODE_SWITCH = "mode_switch"
    SET_LANGUAGE = "set_language"
    MOTION = "motion"
    CITY_QUERY = "city_query"
    TRANSLATE_QUERY = "translate_query"
    STATUS = "status"
    HELP = "help"
    SHUTDOWN = "shutdown"
    EMPTY = "empty"


@dataclass(frozen=True)
class Intent:
    kind: IntentKind
    text: str = ""
    mode: Mode | None = None
    gesture: str | None = None
    language: str | None = None


# Argos language packs most people install for travel use. Spoken language
# names are mapped to ISO codes; the reverse map gives Piper a voice key.
LANGUAGE_CODES: dict[str, str] = {
    "arabic": "ar",
    "chinese": "zh",
    "mandarin": "zh",
    "dutch": "nl",
    "english": "en",
    "french": "fr",
    "german": "de",
    "hindi": "hi",
    "indonesian": "id",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "polish": "pl",
    "portuguese": "pt",
    "russian": "ru",
    "spanish": "es",
    "turkish": "tr",
    "ukrainian": "uk",
    "vietnamese": "vi",
}

LANGUAGE_NAMES = {code: name for name, code in LANGUAGE_CODES.items()}

MODE_PHRASES: tuple[tuple[str, Mode], ...] = (
    ("switch to translator", Mode.TRANSLATE),
    ("switch to translate", Mode.TRANSLATE),
    ("translator mode", Mode.TRANSLATE),
    ("translation mode", Mode.TRANSLATE),
    ("start translating", Mode.TRANSLATE),
    ("switch to city guide", Mode.CITY),
    ("switch to city", Mode.CITY),
    ("city guide mode", Mode.CITY),
    ("city mode", Mode.CITY),
    ("travel guide mode", Mode.CITY),
)

GESTURE_PHRASES: tuple[tuple[str, str], ...] = (
    ("wave your hand", "wave"),
    ("wave hello", "wave"),
    ("say hello", "wave"),
    ("wave", "wave"),
    ("nod your head", "nod"),
    ("nod", "nod"),
    ("look at me", "center"),
    ("eyes front", "center"),
)

SHUTDOWN_PHRASES = (
    "shut down",
    "shutdown",
    "power off",
    "go to sleep",
    "goodbye wall e",
)

STATUS_PHRASES = (
    "are you online",
    "status report",
    "system status",
    "how are you",
    "battery status",
)

HELP_PHRASES = (
    "what can you do",
    "help me",
    "list commands",
    "what are your commands",
)

_TRANSLATE_TO = re.compile(
    r"^(?:please\s+)?(?:translate|say)\s+(?P<body>.+?)\s+"
    r"(?:in|into|to)\s+(?P<lang>[a-z]+)$"
)
_SET_LANGUAGE = re.compile(
    r"^(?:switch|set|change)\s+(?:the\s+)?language\s+(?:to\s+)?(?P<lang>[a-z]+)$"
)
# The body is optional: a clipped "translate" on its own must ask what to
# translate, not fall through and get looked up as a city name.
_TRANSLATE_PREFIX = re.compile(
    r"^(?:please\s+)?(?:translate|say)(?:\s+(?P<body>.+))?$"
)


def _matches_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def parse(transcript: str, mode: Mode) -> Intent:
    """Classify one utterance in the context of the current mode."""
    text = normalise(transcript)
    if not text:
        return Intent(IntentKind.EMPTY)

    # Explicit commands win over mode routing, so "wave" never gets looked up
    # as a city and "shut down" is never translated into Spanish.
    if _matches_any(text, SHUTDOWN_PHRASES):
        return Intent(IntentKind.SHUTDOWN, text=text)

    for phrase, target in MODE_PHRASES:
        if phrase in text:
            return Intent(IntentKind.MODE_SWITCH, text=text, mode=target)

    if match := _SET_LANGUAGE.match(text):
        code = LANGUAGE_CODES.get(match.group("lang"))
        if code:
            return Intent(IntentKind.SET_LANGUAGE, text=text, language=code)

    for phrase, gesture in GESTURE_PHRASES:
        if phrase in text:
            return Intent(IntentKind.MOTION, text=text, gesture=gesture)

    if _matches_any(text, STATUS_PHRASES):
        return Intent(IntentKind.STATUS, text=text)

    if _matches_any(text, HELP_PHRASES):
        return Intent(IntentKind.HELP, text=text)

    # "translate the museum is closed into french" carries its own target.
    if match := _TRANSLATE_TO.match(text):
        code = LANGUAGE_CODES.get(match.group("lang"))
        if code:
            return Intent(
                IntentKind.TRANSLATE_QUERY,
                text=match.group("body"),
                language=code,
            )

    if match := _TRANSLATE_PREFIX.match(text):
        return Intent(IntentKind.TRANSLATE_QUERY, text=match.group("body") or "")

    if mode is Mode.TRANSLATE:
        return Intent(IntentKind.TRANSLATE_QUERY, text=text)
    return Intent(IntentKind.CITY_QUERY, text=text)
