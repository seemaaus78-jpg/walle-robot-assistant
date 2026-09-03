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
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

from .chat import ChatHistory, OfflineChat
from .cities import City, CityDatabase, CityNotFound
from .config import Config
from .display import Card, Display
from .emotion import EmotionEngine, Event
from .face import Emotion
from .intents import Intent, IntentKind, LANGUAGE_NAMES, Mode, parse
from .maps import MapRenderer, MapRequest, zoom_for_population
from .net import ConnectivityMonitor
from .translation import ArgosTranslator, TranslationUnavailable

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Visual:
    """Something to put on the panel alongside the spoken answer.

    ``city`` asks for a map centred on that place; ``card`` is a block of text.
    A reply with neither leaves the animated face up.
    """

    card: Card | None = None
    city: City | None = None


@dataclass(frozen=True)
class Reply:
    """What the robot should do in response to one utterance."""

    text: str
    lang: str = "en"
    gesture: str | None = None
    stop: bool = False
    visual: Visual | None = None
    emotion: Emotion | None = None
    """Overrides the emotion the event would otherwise produce."""


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

    def answer(
        self, intent: Intent, history: ChatHistory | None = None
    ) -> Reply | None: ...


# A two-way conditional cannot name a third mode, and silently mislabelling the
# mode you just switched to is worse than not announcing it at all.
MODE_LABELS: dict[Mode, str] = {
    Mode.CITY: "city guide",
    Mode.TRANSLATE: "translator",
    Mode.CHAT: "chat",
}

HELP_TEXT = (
    "Ask me about a city and I will tell you where it is and put it on the "
    "map. Say translate, followed by a phrase, and I will say it in your "
    "target language. Say lets talk and we can just chat. Say switch to "
    "translator or switch to city guide to change mode. Say wave or nod and I "
    "will move."
)


def city_card(city: City) -> Card:
    """A city's facts, laid out for a small panel.

    Short lines: at 240 pixels wide there is room for a label and a number, and
    a wrapped sentence is unreadable from desk distance.
    """
    lines = [city.country if not city.admin else f"{city.admin}, {city.country}"]
    if city.population:
        lines.append(f"Population {city.population:,}")
    if city.has_location:
        lines.append(f"{city.latitude:.2f}, {city.longitude:.2f}")
    return Card(title=city.name, lines=tuple(lines))


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
        display: Display | None = None,
        emotions: EmotionEngine | None = None,
        chat: OfflineChat | None = None,
        maps: MapRenderer | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.speaker = speaker
        self.recogniser = recogniser
        self.motion = motion
        self.cities = cities
        self.display = display
        self.emotions = emotions or EmotionEngine(
            clock, sleep_after_s=config.display.sleep_after_s
        )
        self.chat = chat or OfflineChat(cities)
        self.maps = maps
        self.history = ChatHistory(config.chat.history_turns)
        self._clock = clock
        self._visual_until = 0.0
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

        self.source_lang = config.translate.source_lang
        self.target_lang = config.translate.target_lang
        self._running = False

    # -- decision layer (no I/O, directly unit-testable) --------------------

    def respond(self, transcript: str) -> Reply | None:
        """Map one transcript to a reply. Returns None when nothing to say."""
        intent = parse(transcript, self.mode)

        if intent.kind is IntentKind.EMPTY:
            return None

        self._feel(Event.HEARD)
        self.history.add_user(intent.text or transcript)

        if self.online is not None and self.connectivity.is_online():
            # The face shows it is working before the network round trip, not
            # after: several seconds of a blank stare reads as a crash.
            self._feel(Event.THINKING)
            try:
                if (reply := self.online.answer(intent, self.history)) is not None:
                    return reply
            except Exception as exc:  # noqa: BLE001 - never let the cloud win
                log.warning("online backend failed, using local models: %s", exc)

        return self._respond_offline(intent)

    def _feel(self, event: Event) -> None:
        """Report an event to the face."""
        self.emotions.on_event(event)
        self._refresh_face()

    def _refresh_face(self) -> None:
        if self.display is not None:
            self.display.set_emotion(self.emotions.current(), self.emotions.gaze())

    def _respond_offline(self, intent: Intent) -> Reply | None:
        match intent.kind:
            case IntentKind.SHUTDOWN:
                return Reply("Powering down. Goodbye.", stop=True)

            case IntentKind.MODE_SWITCH:
                assert intent.mode is not None
                self.mode = intent.mode
                # "switch to translator mode and speak spanish" sets both.
                if intent.language:
                    self.target_lang = intent.language
                    name = LANGUAGE_NAMES.get(intent.language, intent.language)
                    return Reply(
                        f"Translator mode, speaking {name}.", gesture="wave"
                    )
                return Reply(
                    f"Switched to {MODE_LABELS[intent.mode]} mode.", gesture="wave"
                )

            case IntentKind.SET_LANGUAGE_PAIR:
                assert intent.language is not None
                assert intent.source_language is not None
                self.source_lang = intent.source_language
                self.target_lang = intent.language
                self.mode = intent.mode or Mode.TRANSLATE
                source = LANGUAGE_NAMES.get(
                    intent.source_language, intent.source_language
                )
                target = LANGUAGE_NAMES.get(intent.language, intent.language)
                # Naming both ends back confirms it heard the pair correctly,
                # which matters when the recogniser is the weak link.
                return Reply(
                    f"Translating {source} to {target}. Go ahead.", gesture="nod"
                )

            case IntentKind.SET_LANGUAGE:
                assert intent.language is not None
                self.target_lang = intent.language
                name = LANGUAGE_NAMES.get(intent.language, intent.language)
                return Reply(f"I will translate into {name}.", gesture="nod")

            case IntentKind.MOTION:
                return Reply("", gesture=intent.gesture)

            case IntentKind.STATUS:
                where = "online" if self.connectivity.is_online() else "offline"
                mode = MODE_LABELS[self.mode]
                source = LANGUAGE_NAMES.get(self.source_lang, self.source_lang)
                target = LANGUAGE_NAMES.get(self.target_lang, self.target_lang)
                return Reply(
                    f"I am {where}, in {mode} mode, "
                    f"translating {source} to {target}."
                )

            case IntentKind.HELP:
                return Reply(HELP_TEXT)

            case IntentKind.CITY_QUERY:
                return self._answer_city(intent)

            case IntentKind.TRANSLATE_QUERY:
                return self._answer_translation(intent)

            case IntentKind.CHAT_QUERY:
                return self._answer_chat(intent)

            case IntentKind.MAP_QUERY:
                return self._answer_map(intent)

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
                "Sorry, I could not find that city in my offline database.",
                emotion=Emotion.CONFUSED,
            )
        return Reply(city.summary(), gesture="nod", visual=Visual(card=city_card(city)))

    def _answer_translation(self, intent: Intent) -> Reply:
        target = intent.language or self.target_lang
        if target == self.source_lang:
            name = LANGUAGE_NAMES.get(target, target)
            return Reply(f"That is already {name}. Tell me which language to use.")
        body = intent.text.strip()
        if not body:
            return Reply("Tell me what you would like me to translate.")

        try:
            translated = self.translator.translate(
                body, from_code=self.source_lang, to_code=target
            )
        except TranslationUnavailable as exc:
            log.error("%s", exc)
            name = LANGUAGE_NAMES.get(target, target)
            # Answer in English: speaking untranslated English through a foreign
            # voice is what the draft did, and it sounds like a working answer.
            return Reply(f"I do not have the {name} language pack installed.")

        return Reply(translated, lang=target)

    def _answer_chat(self, intent: Intent) -> Reply:
        """Small talk, but reach for real knowledge first.

        Chat mode does not guess from phrasing whether something is a place
        question. It asks the database, which either knows the place or does
        not - a fact rather than a heuristic. Only if that misses does the
        conversational responder take over.
        """
        body = intent.text.strip()
        if not body:
            return Reply("I did not catch that.", emotion=Emotion.CONFUSED)

        if self.cities is not None:
            try:
                city = self.cities.find_in_utterance(
                    body, max_words=self.config.city.max_name_words
                )
            except CityNotFound:
                pass
            else:
                return Reply(
                    city.summary(), gesture="nod", visual=Visual(card=city_card(city))
                )

        return Reply(self.chat.reply(body, self.history))

    def _answer_map(self, intent: Intent) -> Reply:
        place = intent.text.strip()
        if not place:
            return Reply("Which place should I show you?", emotion=Emotion.CONFUSED)

        if self.cities is None:
            return Reply("My city database is not loaded, so I have no map.")

        try:
            city = self.cities.find_in_utterance(
                place, max_words=self.config.city.max_name_words
            )
        except CityNotFound:
            return Reply(
                f"I could not find {place} to put on the map.",
                emotion=Emotion.CONFUSED,
            )

        if not city.has_location:
            return Reply(
                f"I know {city.name}, but I do not have its coordinates.",
                visual=Visual(card=city_card(city)),
            )

        # The map is fetched during delivery, not here, so a slow tile server
        # cannot stall the decision layer.
        return Reply(
            f"Here is {city.name}.", gesture="nod", visual=Visual(city=city)
        )

    # -- I/O layer ----------------------------------------------------------

    def deliver(self, reply: Reply) -> None:
        """Show, gesture and speak, in that order.

        The panel updates first because it is instantaneous and the speech is
        not: the face should already be reacting while Piper is still starting
        up.
        """
        self._show(reply)

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
            self.history.add_robot(reply.text)
            self._feel(Event.SPEAKING_STARTED)
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
            self._feel(Event.SPEAKING_FINISHED)

        # The emotion an answer leaves behind: a found city is a small success,
        # a miss is a shrug. Explicit overrides win.
        if reply.emotion is not None:
            self.emotions.on_event(
                Event.NOT_UNDERSTOOD
                if reply.emotion is Emotion.CONFUSED
                else Event.FAILED
                if reply.emotion is Emotion.SAD
                else Event.ANSWERED
            )
        elif reply.text:
            self._feel(Event.ANSWERED)
        self._refresh_face()

    def _show(self, reply: Reply) -> None:
        """Put a card or a map on the panel, if the reply asked for one."""
        if self.display is None or reply.visual is None:
            return

        visual = reply.visual
        try:
            if visual.city is not None and self.maps is not None:
                if self._show_map(visual.city):
                    return
                # A map that could not be built still deserves the facts rather
                # than a blank screen.
                self.display.show_card(city_card(visual.city))
                self._hold_visual()
                return
            if visual.card is not None:
                self.display.show_card(visual.card)
                self._hold_visual()
        except Exception as exc:  # noqa: BLE001 - a dead panel must not stop
            log.error("display update failed: %s", exc)

    def _show_map(self, city: City) -> bool:
        request = MapRequest(
            latitude=city.latitude,
            longitude=city.longitude,
            zoom=zoom_for_population(city.population),
            grid=self.config.maps.grid,
        )
        image = self.maps.render(request, self.display.size)
        if image is None:
            return False
        self.display.show_image(image)
        self._hold_visual()
        return True

    def _hold_visual(self) -> None:
        self._visual_until = self._clock() + self.config.display.card_seconds

    def _tick_display(self) -> None:
        """Return to the face once a card or map has had its time."""
        if self.display is None:
            return
        if self._visual_until and self._clock() >= self._visual_until:
            self._visual_until = 0.0
            self.display.show_face()
        self._refresh_face()

    def greet(self) -> None:
        self._feel(Event.WAKE)
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
                reply = Reply(
                    "Sorry, something went wrong handling that.",
                    emotion=Emotion.SAD,
                )
                del exc

            if reply is None:
                self._tick_display()
                continue
            self.deliver(reply)
            self._tick_display()
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
            ("display", self.display),
            ("cities", self.cities),
        ):
            if component is None:
                continue
            try:
                component.close()
            except Exception as exc:  # noqa: BLE001 - keep tearing down
                log.warning("error closing %s: %s", name, exc)
