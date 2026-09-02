"""The main controller: hear, decide, move, speak.

The most serious behavioural fix relative to the draft lives here. That version
branched on connectivity like this::

    if is_connected():
        print("Online Mode Active")     # ... and nothing else
    else:
        # the entire question-answering pipeline

Every answer was in the offline branch, so the robot fell silent the moment it
had Wi-Fi - the one condition the user would assume made it work better.

Here the offline pipeline is the guaranteed path. Connectivity is treated as an
optional accelerator: if an online backend is configured and reachable it gets
first refusal, and anything it cannot answer falls through to the local models.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from .cities import CityDatabase, CityNotFound
from .config import Config
from .intents import Intent, IntentKind, LANGUAGE_NAMES, Mode, parse
from .net import ConnectivityMonitor
from .translation import ArgosTranslator, TranslationUnavailable

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Reply:
    """What the robot should do in response to one utterance."""

    text: str
    lang: str = "en"
    gesture: str | None = None
    stop: bool = False


class Speaker(Protocol):
    @property
    def is_speaking(self) -> bool: ...
    def speak(self, text: str, lang: str = "en") -> bool: ...
    def close(self) -> None: ...


class Recogniser(Protocol):
    def listen(self) -> Iterator[str]: ...
    def close(self) -> None: ...


class Motion(Protocol):
    def wave_hand(self, times: int = 2, servo: str = "right_arm") -> None: ...
    def nod(self, times: int = 2, servo: str = "neck_tilt") -> None: ...
    def rest(self) -> None: ...
    def close(self) -> None: ...


class OnlineBackend(Protocol):
    """Optional cloud helper. Return None to defer to the offline models."""

    def answer(self, intent: Intent) -> Reply | None: ...


HELP_TEXT = (
    "I can do four things. Ask me about a city and I will tell you where it is "
    "and how big it is. Say translate, followed by a phrase, and I will say it "
    "in your target language. Say switch to translator or switch to city guide "
    "to change mode. Say wave or nod and I will move."
)


class Assistant:
    """Wires the speech, knowledge, motion and audio components together."""

    def __init__(
        self,
        config: Config,
        speaker: Speaker,
        recogniser: Recogniser,
        motion: Motion | None = None,
        cities: CityDatabase | None = None,
        translator: ArgosTranslator | None = None,
        connectivity: ConnectivityMonitor | None = None,
        online: OnlineBackend | None = None,
    ) -> None:
        self.config = config
        self.speaker = speaker
        self.recogniser = recogniser
        self.motion = motion
        self.cities = cities
        self.translator = translator or ArgosTranslator(
            config.translate.source_lang, config.translate.target_lang
        )
        self.connectivity = connectivity or ConnectivityMonitor(config.network)
        self.online = online

        try:
            self.mode = Mode(config.default_mode)
        except ValueError:
            log.warning("unknown default_mode %r; using CITY", config.default_mode)
            self.mode = Mode.CITY

        self.target_lang = config.translate.target_lang
        self._running = False

    # -- decision layer (no I/O, directly unit-testable) --------------------

    def respond(self, transcript: str) -> Reply | None:
        """Map one transcript to a reply. Returns None when nothing to say."""
        intent = parse(transcript, self.mode)

        if intent.kind is IntentKind.EMPTY:
            return None

        if self.online is not None and self.connectivity.is_online():
            try:
                if (reply := self.online.answer(intent)) is not None:
                    return reply
            except Exception as exc:  # noqa: BLE001 - never let the cloud win
                log.warning("online backend failed, using local models: %s", exc)

        return self._respond_offline(intent)

    def _respond_offline(self, intent: Intent) -> Reply | None:
        match intent.kind:
            case IntentKind.SHUTDOWN:
                return Reply("Powering down. Goodbye.", stop=True)

            case IntentKind.MODE_SWITCH:
                assert intent.mode is not None
                self.mode = intent.mode
                label = (
                    "translator" if intent.mode is Mode.TRANSLATE else "city guide"
                )
                return Reply(f"Switched to {label} mode.", gesture="wave")

            case IntentKind.SET_LANGUAGE:
                assert intent.language is not None
                self.target_lang = intent.language
                name = LANGUAGE_NAMES.get(intent.language, intent.language)
                return Reply(f"I will translate into {name}.", gesture="nod")

            case IntentKind.MOTION:
                return Reply("", gesture=intent.gesture)

            case IntentKind.STATUS:
                where = "online" if self.connectivity.is_online() else "offline"
                mode = "city guide" if self.mode is Mode.CITY else "translator"
                name = LANGUAGE_NAMES.get(self.target_lang, self.target_lang)
                return Reply(
                    f"I am {where}, in {mode} mode, translating into {name}."
                )

            case IntentKind.HELP:
                return Reply(HELP_TEXT)

            case IntentKind.CITY_QUERY:
                return self._answer_city(intent)

            case IntentKind.TRANSLATE_QUERY:
                return self._answer_translation(intent)

        return None

    def _answer_city(self, intent: Intent) -> Reply:
        if self.cities is None:
            return Reply(
                "My city database is not loaded, so I cannot look that up."
            )
        try:
            city = self.cities.find_in_utterance(
                intent.text, max_words=self.config.city.max_name_words
            )
        except CityNotFound:
            return Reply(
                "Sorry, I could not find that city in my offline database."
            )
        return Reply(city.summary(), gesture="nod")

    def _answer_translation(self, intent: Intent) -> Reply:
        target = intent.language or self.target_lang
        body = intent.text.strip()
        if not body:
            return Reply("Tell me what you would like me to translate.")

        try:
            translated = self.translator.translate(
                body, from_code=self.config.translate.source_lang, to_code=target
            )
        except TranslationUnavailable as exc:
            log.error("%s", exc)
            name = LANGUAGE_NAMES.get(target, target)
            # Answer in English: speaking untranslated English through a foreign
            # voice is what the draft did, and it sounds like a working answer.
            return Reply(f"I do not have the {name} language pack installed.")

        return Reply(translated, lang=target)

    # -- I/O layer ----------------------------------------------------------

    def deliver(self, reply: Reply) -> None:
        """Perform the gesture and speak the text."""
        if reply.gesture and self.motion is not None:
            try:
                match reply.gesture:
                    case "wave":
                        self.motion.wave_hand()
                    case "nod":
                        self.motion.nod()
                    case "center":
                        self.motion.rest()
            except Exception as exc:  # noqa: BLE001 - a stuck servo must not
                log.error("gesture %r failed: %s", reply.gesture, exc)

        if reply.text:
            spoken = self.speaker.speak(reply.text, lang=reply.lang)
            if not spoken and reply.lang != "en":
                # The translation worked but there is no voice to say it in.
                # This is easy to hit now that the online backend can translate
                # into any language while Piper only has the voices you
                # installed. Standing there silent looks like a crash.
                name = LANGUAGE_NAMES.get(reply.lang, reply.lang)
                self.speaker.speak(
                    f"I translated that, but I have no {name} voice installed."
                )

    def greet(self) -> None:
        self.speaker.speak("Wall E system ready.")

    def run(self) -> None:
        """Listen and respond until shutdown is requested or the stream ends."""
        self._running = True
        self.greet()
        for transcript in self.recogniser.listen():
            if not self._running:
                break
            log.info("heard: %s", transcript)
            try:
                reply = self.respond(transcript)
            except Exception as exc:  # noqa: BLE001 - one bad utterance must
                # never take the robot down; say so and keep listening.
                log.exception("failed to handle %r", transcript)
                reply = Reply("Sorry, something went wrong handling that.")
                del exc

            if reply is None:
                continue
            self.deliver(reply)
            if reply.stop:
                self._running = False
                break

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        """Release hardware in the reverse order it was acquired."""
        for name, component in (
            ("recogniser", self.recogniser),
            ("speaker", self.speaker),
            ("motion", self.motion),
            ("cities", self.cities),
        ):
            if component is None:
                continue
            try:
                component.close()
            except Exception as exc:  # noqa: BLE001 - keep tearing down
                log.warning("error closing %s: %s", name, exc)
