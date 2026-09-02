"""Offline conversation and its honest floor."""

import unittest

from walle.chat import NO_CONNECTION, ChatHistory, OfflineChat
from walle.cities import City


class FakeCities:
    def __init__(self, city=None):
        self._city = city

    def random_city(self, min_population=250_000):
        return self._city


class HistoryTests(unittest.TestCase):
    def test_history_is_bounded(self):
        # An assistant left running for days must not grow an endless prompt.
        history = ChatHistory(max_turns=4)
        for i in range(10):
            history.add_user(f"turn {i}")
        self.assertEqual(len(history), 4)
        self.assertEqual(history.turns()[0].text, "turn 6")

    def test_roles_are_recorded(self):
        history = ChatHistory()
        history.add_user("hello")
        history.add_robot("hi")
        self.assertEqual([t.role for t in history.turns()], ["user", "model"])

    def test_clear(self):
        history = ChatHistory()
        history.add_user("hello")
        history.clear()
        self.assertEqual(len(history), 0)


class OfflineChatTests(unittest.TestCase):
    def setUp(self):
        self.chat = OfflineChat()

    def test_greetings_are_answered(self):
        for greeting in ("hello", "hi there", "good morning", "hey"):
            self.assertNotEqual(self.chat.reply(greeting), NO_CONNECTION, greeting)

    def test_identity_questions(self):
        self.assertIn("Wall E", self.chat.reply("who are you"))

    def test_thanks_and_farewells(self):
        self.assertNotEqual(self.chat.reply("thank you"), NO_CONNECTION)
        self.assertNotEqual(self.chat.reply("goodbye"), NO_CONNECTION)

    def test_replies_rotate_rather_than_repeat(self):
        first = self.chat.reply("hello")
        second = self.chat.reply("hello")
        self.assertNotEqual(first, second)

    def test_rotation_is_deterministic(self):
        a, b = OfflineChat(), OfflineChat()
        self.assertEqual(
            [a.reply("hello") for _ in range(4)],
            [b.reply("hello") for _ in range(4)],
        )

    def test_unknown_input_says_so_plainly(self):
        """The floor of offline chat. It must not bluff."""
        reply = self.chat.reply("what do you think about quantitative easing")
        self.assertEqual(reply, NO_CONNECTION)
        self.assertIn("connection", reply)

    def test_empty_input(self):
        self.assertIn("catch that", self.chat.reply("   "))

    def test_punctuation_and_case_do_not_matter(self):
        self.assertNotEqual(self.chat.reply("Hello!"), NO_CONNECTION)

    def test_small_talk_uses_real_data_when_available(self):
        # Offline the robot has one real body of knowledge; it should reach for
        # that rather than inventing filler.
        city = City("Kyoto", "Japan", 1_463_723, admin="Kyoto")
        chat = OfflineChat(FakeCities(city))
        reply = chat.reply("tell me something interesting")
        self.assertIn("Kyoto", reply)

    def test_small_talk_without_a_database_falls_back(self):
        self.assertEqual(
            OfflineChat(FakeCities(None)).reply("surprise me"), NO_CONNECTION
        )

    def test_no_database_configured(self):
        self.assertEqual(OfflineChat(None).reply("surprise me"), NO_CONNECTION)


if __name__ == "__main__":
    unittest.main()
