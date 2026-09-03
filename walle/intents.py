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
    CHAT = "CHAT"


class IntentKind(str, Enum):
    MODE_SWITCH = "mode_switch"
    SET_LANGUAGE = "set_language"
    SET_LANGUAGE_PAIR = "set_language_pair"
    MOTION = "motion"
    CITY_QUERY = "city_query"
    TRANSLATE_QUERY = "translate_query"
    CHAT_QUERY = "chat_query"
    MAP_QUERY = "map_query"
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
    """Target language for translation."""

    source_language: str | None = None
    """Source language, when the speaker named a whole pair."""


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
    ("switch to chat", Mode.CHAT),
    ("chat mode", Mode.CHAT),
    ("lets talk", Mode.CHAT),
    ("let us talk", Mode.CHAT),
    ("conversation mode", Mode.CHAT),
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
    "battery status",
)

# Phrases that mean "report your state" when the robot is working, and mean
# small talk when you have explicitly asked it to chat. Routing them as status
# in chat mode makes the robot answer a friendly question with a diagnostic.
CONVERSATIONAL_STATUS = ("how are you", "how do you feel", "are you okay")

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
# "translate from english to spanish" names a language PAIR and configures the
# robot. "translate the museum is closed into spanish" names a PHRASE and is a
# one-off. Both start with the same word, so the two are told apart by whether
# the middle is itself a language name - see parse().
_LANGUAGE_PAIR = re.compile(
    r"\b(?:translate|translating|translation|convert)\s+(?:from\s+)?"
    r"(?P<src>[a-z]+)\s+(?:in)?to\s+(?P<dst>[a-z]+)\b"
)

# "speak spanish", and the tail of "switch to translator mode and speak spanish".
_SPEAK_LANGUAGE = re.compile(
    r"\b(?:speak|speaking|talk in|say it in|reply in|answer in)\s+(?P<lang>[a-z]+)\b"
)

_TRANSLATE_PREFIX = re.compile(
    r"^(?:please\s+)?(?:translate|say)(?:\s+(?P<body>.+))?$"
)

# "show me a map of kyoto" / "show kyoto on the map" / "map of kyoto".
_MAP_OF = re.compile(
    r"^(?:(?:can you\s+)?(?:show|display|pull up|bring up)\s+(?:me\s+)?)?"
    r"(?:a\s+|the\s+)?map\s+(?:of\s+|for\s+)?(?P<place>.+)$"
)
_MAP_ON = re.compile(
    r"^(?:show|display|find|put)\s+(?:me\s+)?(?P<place>.+?)\s+on\s+"
    r"(?:a|the)\s+map$"
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

    # A named language pair configures the robot and switches it into
    # translator mode in one breath, which is how people actually ask.
    if match := _LANGUAGE_PAIR.search(text):
        source = LANGUAGE_CODES.get(match.group("src"))
        target = LANGUAGE_CODES.get(match.group("dst"))
        if source and target and source != target:
            return Intent(
                IntentKind.SET_LANGUAGE_PAIR,
                text=text,
                mode=Mode.TRANSLATE,
                language=target,
                source_language=source,
            )

    for phrase, target in MODE_PHRASES:
        if phrase in text:
            # "switch to translator mode and speak spanish" carries both the
            # mode and the language; dropping the second half silently leaves
            # the robot translating into whatever it used last.
            language = None
            if spoken := _SPEAK_LANGUAGE.search(text):
                language = LANGUAGE_CODES.get(spoken.group("lang"))
            return Intent(
                IntentKind.MODE_SWITCH, text=text, mode=target, language=language
            )

    if match := _SET_LANGUAGE.match(text):
        code = LANGUAGE_CODES.get(match.group("lang"))
        if code:
            return Intent(IntentKind.SET_LANGUAGE, text=text, language=code)

    # A bare "speak spanish" sets the target language without changing mode.
    if match := _SPEAK_LANGUAGE.search(text):
        if code := LANGUAGE_CODES.get(match.group("lang")):
            return Intent(IntentKind.SET_LANGUAGE, text=text, language=code)

    for phrase, gesture in GESTURE_PHRASES:
        if phrase in text:
            return Intent(IntentKind.MOTION, text=text, gesture=gesture)

    if _matches_any(text, STATUS_PHRASES):
        return Intent(IntentKind.STATUS, text=text)

    if _matches_any(text, CONVERSATIONAL_STATUS):
        if mode is Mode.CHAT:
            return Intent(IntentKind.CHAT_QUERY, text=text)
        return Intent(IntentKind.STATUS, text=text)

    if _matches_any(text, HELP_PHRASES):
        return Intent(IntentKind.HELP, text=text)

    # Maps are checked before translation so "show me a map of berlin" is not
    # read as a phrase to translate while in translator mode.
    for pattern in (_MAP_ON, _MAP_OF):
        if match := pattern.match(text):
            place = match.group("place").strip()
            if place:
                return Intent(IntentKind.MAP_QUERY, text=place)

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
    if mode is Mode.CHAT:
        # Chat mode does not try to guess from phrasing whether something is a
        # place question - "what is the weather like" starts with the same
        # words as "what is Prague". The assistant resolves it against the
        # database instead, which is a fact rather than a heuristic.
        return Intent(IntentKind.CHAT_QUERY, text=text)
    return Intent(IntentKind.CITY_QUERY, text=text)
