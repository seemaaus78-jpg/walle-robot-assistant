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


class LanguagePairTests(unittest.TestCase):
    """Naming a pair configures the robot; naming a phrase translates it.

    Both start with the word "translate", so this is the boundary that decides
    whether the robot reconfigures itself or answers.
    """

    def test_from_x_to_y_sets_the_pair_and_switches_mode(self):
        intent = parse("translate from english to spanish", Mode.CITY)
        self.assertIs(intent.kind, IntentKind.SET_LANGUAGE_PAIR)
        self.assertEqual(intent.source_language, "en")
        self.assertEqual(intent.language, "es")
        self.assertIs(intent.mode, Mode.TRANSLATE)

    def test_the_word_from_is_optional(self):
        intent = parse("translate english to french", Mode.CITY)
        self.assertIs(intent.kind, IntentKind.SET_LANGUAGE_PAIR)
        self.assertEqual((intent.source_language, intent.language), ("en", "fr"))

    def test_a_phrase_is_still_translated_not_treated_as_a_pair(self):
        intent = parse("translate the museum is closed into french", Mode.CITY)
        self.assertIs(intent.kind, IntentKind.TRANSLATE_QUERY)
        self.assertEqual(intent.text, "the museum is closed")

    def test_a_phrase_that_merely_mentions_a_language(self):
        # "spanish lessons" is a thing to translate, not a language pair.
        intent = parse("translate spanish lessons are expensive into german", Mode.CITY)
        self.assertIs(intent.kind, IntentKind.TRANSLATE_QUERY)
        self.assertEqual(intent.language, "de")

    def test_unknown_language_names_are_not_a_pair(self):
        intent = parse("translate klingon to elvish", Mode.CITY)
        self.assertIsNot(intent.kind, IntentKind.SET_LANGUAGE_PAIR)

    def test_identical_languages_are_not_a_pair(self):
        intent = parse("translate english to english", Mode.CITY)
        self.assertIsNot(intent.kind, IntentKind.SET_LANGUAGE_PAIR)

    def test_mode_switch_carries_a_spoken_language(self):
        # The example phrase from the specification.
        intent = parse("switch to translator mode and speak spanish", Mode.CITY)
        self.assertIs(intent.kind, IntentKind.MODE_SWITCH)
        self.assertIs(intent.mode, Mode.TRANSLATE)
        self.assertEqual(intent.language, "es")

    def test_mode_switch_without_a_language_still_works(self):
        intent = parse("switch to translator", Mode.CITY)
        self.assertIs(intent.kind, IntentKind.MODE_SWITCH)
        self.assertIsNone(intent.language)

    def test_bare_speak_language_sets_the_target(self):
        intent = parse("speak spanish", Mode.CITY)
        self.assertIs(intent.kind, IntentKind.SET_LANGUAGE)
        self.assertEqual(intent.language, "es")

    def test_speak_variants(self):
        for phrase in ("reply in german", "answer in italian", "say it in japanese"):
            intent = parse(phrase, Mode.CITY)
            self.assertIs(intent.kind, IntentKind.SET_LANGUAGE, phrase)
