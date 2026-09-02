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
            def answer(self, intent: Intent):
                return Reply("Tokyo is the capital of Japan, live from the cloud.")

        assistant = self.make(online=True, backend=Backend())
        self.assertIn("live from the cloud", assistant.respond("tell me about tokyo").text)

    def test_backend_is_skipped_when_offline(self):
        class Backend:
            def answer(self, intent: Intent):
                raise AssertionError("must not be called while offline")

        assistant = self.make(online=False, backend=Backend())
        self.assertIn("Tokyo is in Japan", assistant.respond("tell me about tokyo").text)

    def test_backend_declining_falls_through_to_local_models(self):
        class Backend:
            def answer(self, intent: Intent):
                return None

        assistant = self.make(online=True, backend=Backend())
        self.assertIn("Tokyo is in Japan", assistant.respond("tell me about tokyo").text)

    def test_backend_raising_falls_through_to_local_models(self):
        class Backend:
            def answer(self, intent: Intent):
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
