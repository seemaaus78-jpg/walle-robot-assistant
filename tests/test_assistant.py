"""End-to-end behaviour of the controller, with every device faked.

The first test in this file is the important one: it pins the regression that
made the draft robot fall silent whenever it had Wi-Fi.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from tests.test_cities import make_db
from walle.assistant import Assistant, Reply
from walle.cities import CityDatabase
from walle.config import Config
from walle.face import Emotion
from walle.intents import Intent, IntentKind, Mode
from walle.stt import ScriptedRecogniser
from walle.translation import TranslationUnavailable
from walle.tts import NullSpeaker


class FakeConnectivity:
    def __init__(self, online: bool = False) -> None:
        self.online = online

    def is_online(self, *, force: bool = False) -> bool:
        return self.online

    def invalidate(self) -> None:
        return None


class FakeTranslator:
    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.calls: list[tuple[str, str, str]] = []

    def translate(self, text, from_code=None, to_code=None):
        self.calls.append((text, from_code or "en", to_code or "es"))
        if not self.available:
            raise TranslationUnavailable("no package installed")
        return f"[{to_code}] {text}"


class FakeMotion:
    def __init__(self) -> None:
        self.gestures: list[str] = []

    def wave_hand(self, times: int = 2, servo: str = "right_arm") -> None:
        self.gestures.append("wave")

    def nod(self, times: int = 2, servo: str = "neck_tilt") -> None:
        self.gestures.append("nod")

    def rest(self) -> None:
        self.gestures.append("rest")

    def close(self) -> None:
        return None


class ExplodingMotion(FakeMotion):
    def wave_hand(self, times: int = 2, servo: str = "right_arm") -> None:
        raise RuntimeError("servo jammed")


class AssistantHarness(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        db_path = Path(self._tmp.name) / "cities.db"
        make_db(db_path)
        self.cities = CityDatabase(db_path)
        self.speaker = NullSpeaker()
        self.motion = FakeMotion()
        self.translator = FakeTranslator()
        self.connectivity = FakeConnectivity(online=False)

    def tearDown(self):
        self.cities.close()
        self._tmp.cleanup()

    def make(self, transcripts=(), online=False, backend=None, **kwargs):
        self.connectivity.online = online
        return Assistant(
            config=Config(),
            speaker=self.speaker,
            recogniser=ScriptedRecogniser(list(transcripts)),
            motion=self.motion,
            cities=self.cities,
            translator=self.translator,
            connectivity=self.connectivity,
            online=backend,
            **kwargs,
        )


class OnlineRegressionTests(AssistantHarness):
    def test_online_robot_still_answers(self):
        """The draft printed 'Online Mode Active' and answered nothing."""
        assistant = self.make(online=True)
        reply = assistant.respond("tell me about tokyo")
        self.assertIsNotNone(reply)
        self.assertIn("Tokyo is in Japan", reply.text)

    def test_answers_are_identical_online_and_offline(self):
        offline = self.make(online=False).respond("tell me about tokyo")
        online = self.make(online=True).respond("tell me about tokyo")
        self.assertEqual(offline.text, online.text)

    def test_every_mode_answers_while_online(self):
        assistant = self.make(online=True)
        for utterance in (
            "tell me about tokyo",
            "switch to translator",
            "translate good morning",
            "what can you do",
            "are you online",
        ):
            reply = assistant.respond(utterance)
            self.assertIsNotNone(reply, utterance)
            self.assertTrue(reply.text or reply.gesture, utterance)


class OnlineBackendTests(AssistantHarness):
    def test_backend_answer_is_preferred_when_online(self):
        class Backend:
            def answer(self, intent: Intent, history=None):
                return Reply("Tokyo is the capital of Japan, live from the cloud.")

        assistant = self.make(online=True, backend=Backend())
        self.assertIn("live from the cloud", assistant.respond("tell me about tokyo").text)

    def test_backend_is_skipped_when_offline(self):
        class Backend:
            def answer(self, intent: Intent, history=None):
                raise AssertionError("must not be called while offline")

        assistant = self.make(online=False, backend=Backend())
        self.assertIn("Tokyo is in Japan", assistant.respond("tell me about tokyo").text)

    def test_backend_declining_falls_through_to_local_models(self):
        class Backend:
            def answer(self, intent: Intent, history=None):
                return None

        assistant = self.make(online=True, backend=Backend())
        self.assertIn("Tokyo is in Japan", assistant.respond("tell me about tokyo").text)

    def test_backend_raising_falls_through_to_local_models(self):
        class Backend:
            def answer(self, intent: Intent, history=None):
                raise ConnectionError("504 from the API")

        assistant = self.make(online=True, backend=Backend())
        self.assertIn("Tokyo is in Japan", assistant.respond("tell me about tokyo").text)


class RoutingTests(AssistantHarness):
    def test_mode_switch_persists_across_utterances(self):
        assistant = self.make()
        assistant.respond("switch to translator")
        self.assertIs(assistant.mode, Mode.TRANSLATE)
        reply = assistant.respond("the museum is closed")
        self.assertEqual(reply.lang, "es")
        self.assertEqual(reply.text, "[es] the museum is closed")

    def test_set_language_changes_the_default_target(self):
        assistant = self.make()
        assistant.respond("set language to german")
        assistant.respond("switch to translator")
        self.assertEqual(assistant.respond("good evening").lang, "de")

    def test_inline_target_does_not_change_the_default(self):
        assistant = self.make()
        assistant.respond("switch to translator")
        self.assertEqual(assistant.respond("translate hello into french").lang, "fr")
        self.assertEqual(assistant.respond("goodbye").lang, "es")

    def test_missing_city_gives_a_spoken_apology_not_an_exception(self):
        reply = self.make().respond("tell me about atlantis")
        self.assertIn("could not find", reply.text)

    def test_city_lookup_without_a_database_degrades_gracefully(self):
        assistant = self.make()
        assistant.cities = None
        self.assertIn("not loaded", assistant.respond("tell me about tokyo").text)

    def test_empty_transcript_produces_no_reply(self):
        self.assertIsNone(self.make().respond("   "))

    def test_gesture_only_reply_has_no_speech(self):
        reply = self.make().respond("wave your hand")
        self.assertEqual(reply.text, "")
        self.assertEqual(reply.gesture, "wave")


class TranslationFailureTests(AssistantHarness):
    def test_missing_language_pack_is_reported_in_english(self):
        """The draft returned the untranslated English and spoke it with a
        Spanish voice, which sounds like a working answer."""
        self.translator.available = False
        assistant = self.make()
        assistant.respond("switch to translator")
        reply = assistant.respond("where is the station")
        self.assertEqual(reply.lang, "en")
        self.assertIn("language pack", reply.text)
        self.assertNotIn("where is the station", reply.text)

    def test_empty_translation_body_is_prompted_for(self):
        assistant = self.make()
        self.assertIn("what you would like", assistant.respond("translate").text)


class RunLoopTests(AssistantHarness):
    def test_full_scripted_session(self):
        assistant = self.make(
            [
                "tell me about tokyo",
                "switch to translator",
                "translate the museum is closed",
                "wave your hand",
            ]
        )
        assistant.run()
        spoken = [text for _, text in self.speaker.spoken]
        self.assertIn("Wall E system ready.", spoken)
        self.assertTrue(any("Tokyo is in Japan" in line for line in spoken))
        self.assertIn("[es] the museum is closed", spoken)
        self.assertIn("wave", self.motion.gestures)

    def test_shutdown_stops_the_loop_early(self):
        assistant = self.make(["shut down", "tell me about tokyo"])
        assistant.run()
        spoken = [text for _, text in self.speaker.spoken]
        self.assertIn("Powering down. Goodbye.", spoken)
        self.assertFalse(any("Tokyo" in line for line in spoken))

    def test_a_failing_utterance_does_not_kill_the_loop(self):
        class Boom(FakeTranslator):
            def translate(self, text, from_code=None, to_code=None):
                raise RuntimeError("unexpected")

        self.translator = Boom()
        assistant = self.make(["translate hello", "tell me about tokyo"])
        assistant.run()
        spoken = [text for _, text in self.speaker.spoken]
        self.assertTrue(any("something went wrong" in line for line in spoken))
        self.assertTrue(any("Tokyo is in Japan" in line for line in spoken))

    def test_a_jammed_servo_does_not_stop_the_answer(self):
        self.motion = ExplodingMotion()
        assistant = self.make(["switch to translator"])
        assistant.run()
        spoken = [text for _, text in self.speaker.spoken]
        self.assertIn("Switched to translator mode.", spoken)

    def test_close_tolerates_components_that_raise(self):
        class BadSpeaker(NullSpeaker):
            def close(self):
                raise RuntimeError("device busy")

        self.speaker = BadSpeaker()
        assistant = self.make()
        assistant.close()  # must not raise

    def test_unknown_default_mode_falls_back_to_city(self):
        assistant = Assistant(
            config=Config(default_mode="NONSENSE"),
            speaker=self.speaker,
            recogniser=ScriptedRecogniser([]),
            cities=self.cities,
            translator=self.translator,
            connectivity=self.connectivity,
        )
        self.assertIs(assistant.mode, Mode.CITY)


if __name__ == "__main__":
    unittest.main()


class HybridHandoffTests(AssistantHarness):
    """The whole point of the design: one robot, two answering paths."""

    def backend(self, *responses):
        from tests.test_online import FakeClock, FakeTransport
        from walle.config import OnlineConfig
        from walle.online import GeminiBackend

        self.transport = FakeTransport(*responses)
        return GeminiBackend(
            OnlineConfig(), "key", transport=self.transport, clock=FakeClock()
        )

    def gemini(self, text):
        from tests.test_online import gemini_response

        return gemini_response(text)

    def test_online_answer_is_used_when_reachable(self):
        backend = self.backend(self.gemini("Tokyo is enormous and worth a week."))
        assistant = self.make(online=True, backend=backend)
        self.assertIn("worth a week", assistant.respond("tell me about tokyo").text)

    def test_same_question_answered_locally_when_offline(self):
        backend = self.backend()  # any request would raise
        assistant = self.make(online=False, backend=backend)
        self.assertIn("Tokyo is in Japan", assistant.respond("tell me about tokyo").text)
        self.assertEqual(self.transport.calls, [])

    def test_rate_limited_backend_hands_back_to_sqlite(self):
        import urllib.error

        backend = self.backend(
            urllib.error.HTTPError("http://x", 429, "rate limited", {}, None)
        )
        assistant = self.make(online=True, backend=backend)
        # First question burns the quota, second must not even try.
        self.assertIn("Tokyo is in Japan", assistant.respond("tell me about tokyo").text)
        self.assertIn("Tokyo is in Japan", assistant.respond("tell me about tokyo").text)
        self.assertEqual(len(self.transport.calls), 1)

    def test_gestures_work_with_a_dead_api(self):
        backend = self.backend()  # any request raises
        assistant = self.make(["wave your hand", "shut down"], online=True, backend=backend)
        assistant.run()
        self.assertIn("wave", self.motion.gestures)
        self.assertEqual(self.transport.calls, [])

    def test_online_translation_reaches_the_speaker_in_the_target_language(self):
        backend = self.backend(self.gemini("Le musée est fermé."))
        assistant = self.make(online=True, backend=backend)
        reply = assistant.respond("translate the museum is closed into french")
        self.assertEqual(reply.lang, "fr")
        self.assertEqual(reply.text, "Le musée est fermé.")


class MissingVoiceTests(AssistantHarness):
    def test_a_translation_with_no_voice_is_explained_not_silent(self):
        """Gemini can translate into any language; Piper only has the voices
        that are installed. The gap must be spoken, not swallowed."""

        class VoicelessSpeaker(NullSpeaker):
            def speak(self, text, lang="en"):
                if lang != "en":
                    return False
                return super().speak(text, lang)

        self.speaker = VoicelessSpeaker()
        assistant = self.make()
        assistant.deliver(Reply("Le musée est fermé.", lang="fr"))
        spoken = [text for _, text in self.speaker.spoken]
        self.assertTrue(any("no french voice" in line.lower() for line in spoken), spoken)


class FakeDisplay:
    """Records what would have gone on the panel."""

    def __init__(self, size=(240, 240)):
        self.size = size
        self.cards = []
        self.images = []
        self.emotions = []
        self.face_shown = 0

    def set_emotion(self, emotion, gaze=(0.0, 0.0)):
        self.emotions.append(emotion)

    def show_card(self, card):
        self.cards.append(card)

    def show_image(self, image):
        self.images.append(image)

    def show_face(self):
        self.face_shown += 1

    def close(self):
        return None


class FakeMaps:
    def __init__(self, image="a-map"):
        self.image = image
        self.requests = []

    def render(self, request, size):
        self.requests.append((request, size))
        return self.image


class ChatModeTests(AssistantHarness):
    def test_chat_mode_still_answers_place_questions_from_the_database(self):
        """Offline, a real city answer beats an apology for having no signal."""
        assistant = self.make()
        assistant.respond("lets talk")
        reply = assistant.respond("tell me about tokyo")
        self.assertIn("Tokyo is in Japan", reply.text)

    def test_chat_mode_falls_back_to_small_talk(self):
        assistant = self.make()
        assistant.respond("lets talk")
        self.assertIn("Wall E", assistant.respond("who are you").text)

    def test_offline_chat_admits_its_limits(self):
        assistant = self.make()
        assistant.respond("lets talk")
        reply = assistant.respond("what do you think about monetary policy")
        self.assertIn("connection", reply.text)

    def test_mode_label_is_announced_correctly(self):
        # A two-way conditional cannot name a third mode.
        assistant = self.make()
        self.assertIn("chat", assistant.respond("lets talk").text)
        self.assertIn("translator", assistant.respond("switch to translator").text)
        self.assertIn("city guide", assistant.respond("switch to city guide").text)

    def test_history_records_both_sides(self):
        assistant = self.make()
        assistant.deliver(assistant.respond("tell me about tokyo"))
        roles = [turn.role for turn in assistant.history.turns()]
        self.assertIn("user", roles)
        self.assertIn("model", roles)


class MapTests(AssistantHarness):
    def setUp(self):
        super().setUp()
        self.display = FakeDisplay()
        self.maps = FakeMaps()

    def make_with_display(self, transcripts=(), maps=None):
        from walle.emotion import EmotionEngine

        clock = lambda: 1000.0  # noqa: E731 - frozen so cards never expire mid-test
        return Assistant(
            config=Config(),
            speaker=self.speaker,
            recogniser=ScriptedRecogniser(list(transcripts)),
            motion=self.motion,
            cities=self.cities,
            translator=self.translator,
            connectivity=self.connectivity,
            display=self.display,
            maps=self.maps if maps is None else maps,
            emotions=EmotionEngine(clock),
            clock=clock,
        )

    def test_map_request_is_centred_on_the_city(self):
        assistant = self.make_with_display()
        assistant.deliver(assistant.respond("show me a map of tokyo"))
        self.assertEqual(len(self.maps.requests), 1)
        request, size = self.maps.requests[0]
        self.assertAlmostEqual(request.latitude, 35.6895, places=2)
        self.assertAlmostEqual(request.longitude, 139.6917, places=2)
        self.assertEqual(size, (240, 240))
        self.assertEqual(self.display.images, ["a-map"])

    def test_zoom_follows_city_size(self):
        assistant = self.make_with_display()
        assistant.deliver(assistant.respond("map of tokyo"))
        assistant.deliver(assistant.respond("map of nuuk"))
        big, small = (r.zoom for r, _ in self.maps.requests)
        self.assertLess(big, small)

    def test_unrenderable_map_falls_back_to_the_facts(self):
        # A grey square looks like a rendering bug; the facts do not.
        assistant = self.make_with_display(maps=FakeMaps(image=None))
        assistant.deliver(assistant.respond("show me a map of tokyo"))
        self.assertEqual(self.display.images, [])
        self.assertEqual(self.display.cards[-1].title, "Tokyo")

    def test_city_without_coordinates_is_reported(self):
        # Known place, no coordinates: say so and show the facts rather than
        # rendering a map of the middle of the ocean.
        assistant = self.make_with_display()
        reply = assistant.respond("show me a map of innsbruck")
        self.assertIn("do not have its coordinates", reply.text)
        self.assertEqual(self.maps.requests, [])

    def test_unknown_place_is_not_a_crash(self):
        assistant = self.make_with_display()
        reply = assistant.respond("show me a map of atlantis")
        self.assertIn("could not find", reply.text.lower())
        self.assertEqual(self.maps.requests, [])

    def test_city_answers_put_a_card_on_the_panel(self):
        assistant = self.make_with_display()
        assistant.deliver(assistant.respond("tell me about tokyo"))
        card = self.display.cards[-1]
        self.assertEqual(card.title, "Tokyo")
        self.assertTrue(any("8,336,599" in line for line in card.lines))

    def test_a_dead_panel_does_not_stop_the_answer(self):
        class BrokenDisplay(FakeDisplay):
            def show_card(self, card):
                raise RuntimeError("SPI error")

        self.display = BrokenDisplay()
        assistant = self.make_with_display()
        assistant.deliver(assistant.respond("tell me about tokyo"))
        spoken = [text for _, text in self.speaker.spoken]
        self.assertTrue(any("Tokyo is in Japan" in line for line in spoken))


class EmotionIntegrationTests(AssistantHarness):
    def setUp(self):
        super().setUp()
        self.display = FakeDisplay()

    def make_expressive(self, transcripts=()):
        from walle.emotion import EmotionEngine

        self.time = [1000.0]
        clock = lambda: self.time[0]  # noqa: E731
        return Assistant(
            config=Config(),
            speaker=self.speaker,
            recogniser=ScriptedRecogniser(list(transcripts)),
            cities=self.cities,
            translator=self.translator,
            connectivity=self.connectivity,
            display=self.display,
            emotions=EmotionEngine(clock),
            clock=clock,
        )

    def test_hearing_something_shows_on_the_face(self):
        assistant = self.make_expressive()
        assistant.respond("tell me about tokyo")
        self.assertIn(Emotion.LISTENING, self.display.emotions)

    def test_a_miss_looks_confused(self):
        assistant = self.make_expressive()
        assistant.deliver(assistant.respond("tell me about atlantis"))
        self.assertIs(assistant.emotions.current(), Emotion.CONFUSED)

    def test_a_hit_looks_pleased(self):
        assistant = self.make_expressive()
        assistant.deliver(assistant.respond("tell me about tokyo"))
        self.assertIs(assistant.emotions.current(), Emotion.HAPPY)

    def test_an_error_looks_sad(self):
        class Boom(FakeTranslator):
            def translate(self, text, from_code=None, to_code=None):
                raise RuntimeError("unexpected")

        self.translator = Boom()
        assistant = self.make_expressive(["translate hello"])
        assistant.run()
        self.assertIs(assistant.emotions.current(), Emotion.SAD)

    def test_boot_greeting_is_happy(self):
        assistant = self.make_expressive()
        assistant.greet()
        self.assertIs(assistant.emotions.current(), Emotion.HAPPY)

    def test_it_falls_asleep_when_left_alone(self):
        assistant = self.make_expressive()
        assistant.greet()
        self.time[0] += Config().display.sleep_after_s + 1
        self.assertTrue(assistant.emotions.is_asleep)


class LanguagePairFlowTests(AssistantHarness):
    """The specification's flow: ask for a pair in your own language, then just
    talk and have everything translated."""

    def test_pair_switches_mode_and_both_languages(self):
        assistant = self.make()
        reply = assistant.respond("translate from english to spanish")
        self.assertIs(assistant.mode, Mode.TRANSLATE)
        self.assertEqual(assistant.source_lang, "en")
        self.assertEqual(assistant.target_lang, "es")
        self.assertIn("english to spanish", reply.text.lower())

    def test_everything_after_the_pair_is_translated(self):
        assistant = self.make()
        assistant.respond("translate from english to spanish")
        reply = assistant.respond("where is the train station")
        self.assertEqual(reply.lang, "es")
        self.assertEqual(reply.text, "[es] where is the train station")

    def test_source_language_reaches_the_translator(self):
        # source_lang was previously frozen at the config value forever.
        assistant = self.make()
        assistant.respond("translate from french to german")
        assistant.respond("bonjour")
        self.assertEqual(self.translator.calls[-1][1], "fr")
        self.assertEqual(self.translator.calls[-1][2], "de")

    def test_mode_switch_with_language_sets_both(self):
        assistant = self.make()
        reply = assistant.respond("switch to translator mode and speak french")
        self.assertIs(assistant.mode, Mode.TRANSLATE)
        self.assertEqual(assistant.target_lang, "fr")
        self.assertIn("french", reply.text.lower())

    def test_translating_into_the_source_language_is_refused_clearly(self):
        assistant = self.make()
        assistant.respond("switch to translator")
        reply = assistant.respond("translate hello into english")
        self.assertIn("already english", reply.text.lower())

    def test_status_reports_the_whole_pair(self):
        assistant = self.make()
        assistant.respond("translate from english to german")
        assistant.mode = Mode.CITY
        self.assertIn("english to german", assistant.respond("system status").text)


class GuideCacheTests(AssistantHarness):
    """Ask online, keep the answer, use it offline, delete it when done."""

    GUIDE = {
        "summary": "Tokyo is enormous and endlessly walkable.",
        "food": ["Tsukiji outer market", "Omoide Yokocho for grilled skewers"],
        "emergency": ["Emergency number is 119"],
    }

    def setUp(self):
        super().setUp()
        import tempfile
        from dataclasses import replace as dc_replace
        from pathlib import Path

        from walle.config import Config as Cfg
        from walle.guides import GuideStore

        self._gtmp = tempfile.TemporaryDirectory()
        base = Cfg()
        self.config = dc_replace(
            base,
            guides=dc_replace(
                base.guides, database=Path(self._gtmp.name) / "guides.db"
            ),
        )
        self.store = GuideStore(self.config.guides.database)

    def tearDown(self):
        self.store.close()
        self._gtmp.cleanup()
        super().tearDown()

    def backend(self, sections=None, fail=False):
        outer = self

        class Backend:
            def __init__(self):
                self.fetches = 0

            def answer(self, intent, history=None):
                return None

            def fetch_guide(self, city_name, country=None):
                self.fetches += 1
                if fail:
                    raise ConnectionError("no route")
                return outer.GUIDE if sections is None else sections

        self.online_backend = Backend()
        return self.online_backend

    def make_guided(self, online=True, backend=None, auto_save=None):
        from dataclasses import replace as dc_replace

        config = self.config
        if auto_save is not None:
            config = dc_replace(
                config, guides=dc_replace(config.guides, auto_save=auto_save)
            )
        self.connectivity.online = online
        return Assistant(
            config=config,
            speaker=self.speaker,
            recogniser=ScriptedRecogniser([]),
            cities=self.cities,
            translator=self.translator,
            connectivity=self.connectivity,
            guides=self.store,
            online=backend if backend is not None else self.backend(),
        )

    def test_online_fetch_is_saved_to_the_card(self):
        assistant = self.make_guided(online=True)
        reply = assistant.respond("travel guide for tokyo")
        self.assertIn("endlessly walkable", reply.text)
        self.assertIn("saved it", reply.text)
        self.assertEqual(self.store.count(), 1)

    def test_offline_serves_the_saved_guide_without_calling_out(self):
        backend = self.backend()
        self.make_guided(online=True, backend=backend).respond("travel guide for tokyo")
        self.assertEqual(backend.fetches, 1)

        offline = self.make_guided(online=False, backend=backend)
        reply = offline.respond("travel guide for tokyo")
        self.assertIn("endlessly walkable", reply.text)
        self.assertEqual(backend.fetches, 1, "must not call out while offline")

    def test_offline_section_question_answers_from_the_cache(self):
        self.make_guided(online=True).respond("travel guide for tokyo")
        offline = self.make_guided(online=False)
        reply = offline.respond("what are the restaurants in tokyo")
        self.assertIn("Tsukiji", reply.text)

    def test_every_answer_says_how_old_it_is(self):
        """A restaurant list is a snapshot, not a fact."""
        self.make_guided(online=True).respond("travel guide for tokyo")
        reply = self.make_guided(online=False).respond("travel guide for tokyo")
        self.assertIn("saved today", reply.text)

    def test_emergency_answers_carry_the_caution(self):
        from walle.guides import EMERGENCY_CAUTION

        self.make_guided(online=True).respond("travel guide for tokyo")
        reply = self.make_guided(online=False).respond(
            "where is the police station in tokyo"
        )
        self.assertIn(EMERGENCY_CAUTION, reply.text)

    def test_offline_with_nothing_saved_says_so(self):
        reply = self.make_guided(online=False).respond("travel guide for tokyo")
        self.assertIn("no saved guide", reply.text)

    def test_a_failed_fetch_does_not_break_anything(self):
        assistant = self.make_guided(online=True, backend=self.backend(fail=True))
        reply = assistant.respond("travel guide for tokyo")
        self.assertIsNotNone(reply)
        self.assertEqual(self.store.count(), 0)

    def test_an_empty_response_is_not_saved(self):
        assistant = self.make_guided(online=True, backend=self.backend(sections={}))
        assistant.respond("travel guide for tokyo")
        self.assertEqual(self.store.count(), 0)

    def test_unknown_city_is_not_fetched(self):
        backend = self.backend()
        assistant = self.make_guided(online=True, backend=backend)
        reply = assistant.respond("travel guide for atlantis")
        self.assertIn("do not know", reply.text)
        self.assertEqual(backend.fetches, 0)

    def test_city_questions_pick_up_the_saved_guide(self):
        self.make_guided(online=True).respond("travel guide for tokyo")
        reply = self.make_guided(online=False).respond("tell me about tokyo")
        self.assertIn("Tokyo is in Japan", reply.text)
        self.assertIn("endlessly walkable", reply.text)

    def test_auto_save_can_be_turned_off(self):
        backend = self.backend()
        assistant = self.make_guided(online=True, backend=backend, auto_save=False)
        assistant.respond("tell me about tokyo")
        self.assertEqual(backend.fetches, 0)
        self.assertEqual(self.store.count(), 0)

    def test_delete_one_confirms_with_the_stored_name(self):
        self.make_guided(online=True).respond("travel guide for tokyo")
        reply = self.make_guided(online=False).respond("forget tokyo")
        self.assertIn("Deleted my guide for Tokyo", reply.text)
        self.assertEqual(self.store.count(), 0)

    def test_delete_something_not_saved(self):
        reply = self.make_guided(online=False).respond("forget tokyo")
        self.assertIn("nothing saved", reply.text)

    def test_delete_everything(self):
        self.make_guided(online=True).respond("travel guide for tokyo")
        reply = self.make_guided(online=False).respond("delete all guides")
        self.assertIn("Deleted all 1", reply.text)
        self.assertEqual(self.store.count(), 0)

    def test_listing_what_is_saved(self):
        self.make_guided(online=True).respond("travel guide for tokyo")
        reply = self.make_guided(online=False).respond("what have you saved")
        self.assertIn("Tokyo", reply.text)

    def test_listing_when_empty(self):
        reply = self.make_guided(online=False).respond("what have you saved")
        self.assertIn("not saved any", reply.text)


class VisionTests(AssistantHarness):
    """Looking at things, and being honest that it needs a connection to."""

    IMAGE = b"\xff\xd8\xff\xe0fake-jpeg"

    class FakeVision:
        def __init__(self, answer="A cat asleep on a keyboard.", fail=False):
            self.answer = answer
            self.fail = fail
            self.calls = []

        def answer_(self):
            return None

        def answer(self, intent, history=None):
            return None

        def look(self, image, task="describe", question=None, target_language=None):
            self.calls.append((image, task, target_language))
            if self.fail:
                raise ConnectionError("no route")
            return self.answer

    def make_seeing(self, online=True, camera=None, vision=None):
        from walle.camera import NullCamera

        self.vision = vision or self.FakeVision()
        self.camera = camera if camera is not None else NullCamera(self.IMAGE)
        self.connectivity.online = online
        return Assistant(
            config=Config(),
            speaker=self.speaker,
            recogniser=ScriptedRecogniser([]),
            cities=self.cities,
            translator=self.translator,
            connectivity=self.connectivity,
            camera=self.camera,
            online=self.vision,
        )

    def test_it_describes_what_it_sees(self):
        reply = self.make_seeing().respond("what do you see")
        self.assertIn("cat", reply.text.lower())
        self.assertEqual(self.vision.calls[0][1], "describe")

    def test_it_announces_itself_before_taking_a_picture(self):
        """A camera that fires silently is a worse thing to have on a desk."""
        self.make_seeing().respond("what do you see")
        spoken = [text for _, text in self.speaker.spoken]
        self.assertIn("Let me look.", spoken)

    def test_reading_text_uses_the_read_task(self):
        self.make_seeing().respond("read this")
        self.assertEqual(self.vision.calls[0][1], "read")

    def test_translating_a_sign_carries_the_language(self):
        assistant = self.make_seeing()
        reply = assistant.respond("what does this sign say in french")
        self.assertEqual(self.vision.calls[0][1], "translate")
        self.assertEqual(self.vision.calls[0][2], "fr")
        # A translated sign is spoken in the language it was translated into.
        self.assertEqual(reply.lang, "fr")

    def test_a_description_is_spoken_in_english(self):
        reply = self.make_seeing().respond("what do you see")
        self.assertEqual(reply.lang, "en")

    def test_offline_says_so_rather_than_guessing(self):
        """Vision is the one thing that genuinely cannot work offline."""
        assistant = self.make_seeing(online=False)
        reply = assistant.respond("what do you see")
        self.assertIn("internet", reply.text.lower())
        self.assertEqual(self.vision.calls, [])

    def test_offline_does_not_even_take_the_picture(self):
        assistant = self.make_seeing(online=False)
        assistant.respond("what do you see")
        self.assertEqual(self.camera.captures, 0)

    def test_no_camera_is_reported_clearly(self):
        assistant = self.make_seeing()
        assistant.camera = None
        self.assertIn("camera", assistant.respond("what do you see").text.lower())

    def test_a_camera_that_returns_nothing(self):
        from walle.camera import NullCamera

        assistant = self.make_seeing(camera=NullCamera(None))
        self.assertIn("did not give me a picture", assistant.respond("look at this").text)

    def test_a_failed_vision_request_does_not_crash(self):
        assistant = self.make_seeing(vision=self.FakeVision(fail=True))
        reply = assistant.respond("what do you see")
        self.assertIn("could not make sense", reply.text)

    def test_an_empty_answer_is_reported(self):
        assistant = self.make_seeing(vision=self.FakeVision(answer=""))
        self.assertIn("could not make sense", assistant.respond("what do you see").text)

    def test_looking_never_reaches_the_city_database(self):
        # "read this" must not be looked up as a place called "this".
        reply = self.make_seeing().respond("read this")
        self.assertNotIn("could not find", reply.text.lower())


class DriveTests(AssistantHarness):
    class FakeDrive:
        def __init__(self):
            self.moves = []
            self.stops = 0

        def move(self, direction, seconds=None, speed=None):
            self.moves.append((str(getattr(direction, "value", direction)), seconds))

        def stop(self):
            self.stops += 1

        @property
        def is_moving(self):
            return False

        def close(self):
            return None

    def make_driving(self, online=False, backend=None):
        self.drive = self.FakeDrive()
        self.connectivity.online = online
        return Assistant(
            config=Config(),
            speaker=self.speaker,
            recogniser=ScriptedRecogniser([]),
            cities=self.cities,
            translator=self.translator,
            connectivity=self.connectivity,
            drive=self.drive,
            online=backend,
        )

    def test_directions_reach_the_motors(self):
        assistant = self.make_driving()
        for phrase, expected in [
            ("go forward", "forward"),
            ("move backward", "backward"),
            ("turn left", "left"),
            ("turn right", "right"),
        ]:
            assistant.respond(phrase)
            self.assertEqual(self.drive.moves[-1][0], expected, phrase)

    def test_stop_stops(self):
        assistant = self.make_driving()
        assistant.respond("stop")
        self.assertEqual(self.drive.stops, 1)

    def test_stop_never_waits_on_the_network(self):
        """A robot rolling toward the edge must not pause for an API call."""

        class Backend:
            def answer(self, intent, history=None):
                raise AssertionError("stop must not reach the cloud")

        assistant = self.make_driving(online=True, backend=Backend())
        assistant.respond("stop")
        self.assertEqual(self.drive.stops, 1)

    def test_driving_never_reaches_the_cloud_either(self):
        class Backend:
            def answer(self, intent, history=None):
                raise AssertionError("driving must not reach the cloud")

        assistant = self.make_driving(online=True, backend=Backend())
        assistant.respond("go forward")
        self.assertEqual(self.drive.moves[-1][0], "forward")

    def test_movement_is_not_narrated(self):
        # Talking over your own motors is noise.
        assistant = self.make_driving()
        self.assertEqual(assistant.respond("go forward").text, "")

    def test_turn_around_uses_a_longer_run(self):
        assistant = self.make_driving()
        assistant.respond("turn around")
        self.assertEqual(self.drive.moves[-1][1], Config().drive.max_seconds / 2)

    def test_no_motors_is_reported_clearly(self):
        assistant = self.make_driving()
        assistant.drive = None
        self.assertIn("motors", assistant.respond("go forward").text.lower())
