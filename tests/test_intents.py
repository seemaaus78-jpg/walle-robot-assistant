"""Intent routing: the layer most likely to need tuning once you hear the
recogniser mishear you, and the one that needs no hardware to exercise."""

import unittest

from walle.intents import Intent, IntentKind, Mode, parse


class ParseTests(unittest.TestCase):
    def assert_kind(self, transcript: str, kind: IntentKind, mode=Mode.CITY) -> Intent:
        intent = parse(transcript, mode)
        self.assertIs(intent.kind, kind, f"{transcript!r} -> {intent}")
        return intent

    def test_empty_transcript(self):
        self.assert_kind("", IntentKind.EMPTY)
        self.assert_kind("   ", IntentKind.EMPTY)

    def test_mode_switches(self):
        self.assertIs(
            parse("switch to translator", Mode.CITY).mode, Mode.TRANSLATE
        )
        self.assertIs(
            parse("switch to city guide", Mode.TRANSLATE).mode, Mode.CITY
        )
        self.assertIs(parse("translator mode please", Mode.CITY).mode, Mode.TRANSLATE)

    def test_gestures(self):
        self.assertEqual(parse("wave your hand", Mode.CITY).gesture, "wave")
        self.assertEqual(parse("please nod", Mode.CITY).gesture, "nod")

    def test_gesture_beats_mode_routing(self):
        # In translator mode "wave" must move the arm, not be translated.
        intent = self.assert_kind("wave", IntentKind.MOTION, mode=Mode.TRANSLATE)
        self.assertEqual(intent.gesture, "wave")

    def test_shutdown_is_never_translated(self):
        self.assert_kind("shut down", IntentKind.SHUTDOWN, mode=Mode.TRANSLATE)

    def test_translate_with_explicit_target(self):
        intent = self.assert_kind(
            "translate the museum is closed into french", IntentKind.TRANSLATE_QUERY
        )
        self.assertEqual(intent.text, "the museum is closed")
        self.assertEqual(intent.language, "fr")

    def test_translate_without_target_uses_default(self):
        intent = self.assert_kind("translate where is the station", IntentKind.TRANSLATE_QUERY)
        self.assertEqual(intent.text, "where is the station")
        self.assertIsNone(intent.language)

    def test_unknown_language_is_not_treated_as_target(self):
        # "klingon" is not an Argos package, so this must stay a plain phrase
        # rather than silently translating into the default language.
        intent = self.assert_kind(
            "translate good morning into klingon", IntentKind.TRANSLATE_QUERY
        )
        self.assertIsNone(intent.language)
        self.assertEqual(intent.text, "good morning into klingon")

    def test_set_language(self):
        intent = self.assert_kind("set language to german", IntentKind.SET_LANGUAGE)
        self.assertEqual(intent.language, "de")

    def test_mode_decides_the_fallthrough(self):
        self.assert_kind("tell me about lisbon", IntentKind.CITY_QUERY, mode=Mode.CITY)
        self.assert_kind(
            "tell me about lisbon", IntentKind.TRANSLATE_QUERY, mode=Mode.TRANSLATE
        )

    def test_punctuation_and_case_are_normalised(self):
        # Vosk emits bare lower-case words, but a scripted --text run may not.
        self.assertIs(parse("Switch to Translator!", Mode.CITY).mode, Mode.TRANSLATE)

    def test_status_and_help(self):
        self.assert_kind("are you online", IntentKind.STATUS)
        self.assert_kind("what can you do", IntentKind.HELP)


if __name__ == "__main__":
    unittest.main()
