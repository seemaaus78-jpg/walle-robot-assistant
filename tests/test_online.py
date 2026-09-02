"""The Gemini backend, driven through a fake transport.

No test here touches the network. The point of the module is what it does when
the API misbehaves, and that is far easier to provoke with a fake.
"""

import unittest
import urllib.error

from walle.config import OnlineConfig
from walle.intents import Intent, IntentKind, Mode, parse
from walle.online import GeminiBackend, build_online_backend, clean_for_speech


def gemini_response(text: str, finish: str = "STOP") -> dict:
    return {
        "candidates": [
            {"content": {"parts": [{"text": text}]}, "finishReason": finish}
        ]
    }


class FakeTransport:
    """Records requests and replays queued responses or raises queued errors."""

    def __init__(self, *responses):
        self.queue = list(responses)
        self.calls = []

    def post(self, url, payload, headers, timeout):
        self.calls.append(
            {
                "url": url,
                "payload": payload,
                "headers": headers,
                "timeout": timeout,
            }
        )
        if not self.queue:
            raise AssertionError("unexpected extra request")
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_backend(transport, config=None, clock=None):
    return GeminiBackend(
        config or OnlineConfig(),
        api_key="test-key",
        transport=transport,
        clock=clock or FakeClock(),
    )


class AnswerTests(unittest.TestCase):
    def test_city_question_is_answered(self):
        transport = FakeTransport(
            gemini_response("Kyoto is in Japan. Go in November for the maples.")
        )
        reply = make_backend(transport).answer(parse("tell me about kyoto", Mode.CITY))
        self.assertIsNotNone(reply)
        self.assertIn("Kyoto is in Japan", reply.text)
        self.assertEqual(reply.lang, "en")
        self.assertEqual(reply.gesture, "nod")

    def test_translation_carries_the_target_language(self):
        transport = FakeTransport(gemini_response("Le musée est fermé."))
        intent = parse("translate the museum is closed into french", Mode.CITY)
        reply = make_backend(transport).answer(intent)
        self.assertEqual(reply.text, "Le musée est fermé.")
        self.assertEqual(reply.lang, "fr")

    def test_translation_falls_back_to_the_configured_language(self):
        transport = FakeTransport(gemini_response("El museo está cerrado."))
        intent = parse("the museum is closed", Mode.TRANSLATE)
        reply = make_backend(transport).answer(intent)
        self.assertEqual(reply.lang, "es")


class LocalOnlyIntentTests(unittest.TestCase):
    """Control of the robot itself must never depend on the network."""

    def setUp(self):
        self.transport = FakeTransport()  # any call raises
        self.backend = make_backend(self.transport)

    def test_gestures_are_never_sent_upstream(self):
        self.assertIsNone(self.backend.answer(parse("wave your hand", Mode.CITY)))
        self.assertEqual(self.transport.calls, [])

    def test_shutdown_is_never_sent_upstream(self):
        self.assertIsNone(self.backend.answer(parse("shut down", Mode.CITY)))
        self.assertEqual(self.transport.calls, [])

    def test_mode_switch_is_never_sent_upstream(self):
        self.assertIsNone(
            self.backend.answer(parse("switch to translator", Mode.CITY))
        )
        self.assertEqual(self.transport.calls, [])

    def test_status_and_help_stay_local(self):
        self.assertIsNone(self.backend.answer(parse("are you online", Mode.CITY)))
        self.assertIsNone(self.backend.answer(parse("what can you do", Mode.CITY)))
        self.assertEqual(self.transport.calls, [])

    def test_empty_body_is_not_sent(self):
        self.assertIsNone(
            self.backend.answer(Intent(IntentKind.CITY_QUERY, text="  "))
        )
        self.assertEqual(self.transport.calls, [])


class RequestShapeTests(unittest.TestCase):
    def setUp(self):
        self.transport = FakeTransport(gemini_response("ok"))
        self.backend = make_backend(self.transport)
        self.backend.answer(parse("tell me about lisbon", Mode.CITY))
        self.call = self.transport.calls[0]

    def test_key_travels_in_a_header_not_the_url(self):
        # A key in the query string ends up in proxy logs and crash reports.
        self.assertEqual(self.call["headers"]["x-goog-api-key"], "test-key")
        self.assertNotIn("test-key", self.call["url"])

    def test_model_is_in_the_endpoint(self):
        self.assertIn("gemini-2.5-flash:generateContent", self.call["url"])

    def test_timeout_is_passed_through(self):
        self.assertEqual(self.call["timeout"], OnlineConfig().timeout_s)

    def test_output_is_capped_so_the_robot_does_not_monologue(self):
        limits = self.call["payload"]["generationConfig"]
        self.assertEqual(limits["maxOutputTokens"], OnlineConfig().max_output_tokens)

    def test_system_instruction_forbids_markdown(self):
        system = self.call["payload"]["systemInstruction"]["parts"][0]["text"]
        self.assertIn("markdown", system.lower())


class FailureTests(unittest.TestCase):
    def http_error(self, code):
        return urllib.error.HTTPError("http://x", code, "err", {}, None)

    def test_network_error_returns_none_and_does_not_raise(self):
        transport = FakeTransport(urllib.error.URLError("no route to host"))
        backend = make_backend(transport)
        self.assertIsNone(backend.answer(parse("tell me about lisbon", Mode.CITY)))

    def test_timeout_returns_none(self):
        transport = FakeTransport(TimeoutError("timed out"))
        backend = make_backend(transport)
        self.assertIsNone(backend.answer(parse("tell me about lisbon", Mode.CITY)))

    def test_rate_limit_starts_a_cooldown(self):
        # The free tier limits requests per minute. Retrying into the limit just
        # burns the next minute too.
        clock = FakeClock()
        transport = FakeTransport(self.http_error(429))
        backend = make_backend(transport, clock=clock)

        self.assertIsNone(backend.answer(parse("tell me about lisbon", Mode.CITY)))
        self.assertEqual(len(transport.calls), 1)

        # Further questions are answered locally without another request.
        for _ in range(5):
            self.assertIsNone(backend.answer(parse("tell me about porto", Mode.CITY)))
        self.assertEqual(len(transport.calls), 1)

    def test_cooldown_expires(self):
        clock = FakeClock()
        transport = FakeTransport(self.http_error(429), gemini_response("Porto is in Portugal."))
        backend = make_backend(transport, clock=clock)

        backend.answer(parse("tell me about lisbon", Mode.CITY))
        clock.advance(OnlineConfig().cooldown_s + 1)
        reply = backend.answer(parse("tell me about porto", Mode.CITY))
        self.assertEqual(reply.text, "Porto is in Portugal.")

    def test_bad_key_starts_a_cooldown_rather_than_hammering(self):
        clock = FakeClock()
        transport = FakeTransport(self.http_error(403))
        backend = make_backend(transport, clock=clock)
        backend.answer(parse("tell me about lisbon", Mode.CITY))
        backend.answer(parse("tell me about porto", Mode.CITY))
        self.assertEqual(len(transport.calls), 1)

    def test_server_error_starts_a_cooldown(self):
        clock = FakeClock()
        transport = FakeTransport(self.http_error(503))
        backend = make_backend(transport, clock=clock)
        backend.answer(parse("tell me about lisbon", Mode.CITY))
        backend.answer(parse("tell me about porto", Mode.CITY))
        self.assertEqual(len(transport.calls), 1)

    def test_client_error_does_not_cool_down(self):
        # A malformed request is our bug, not a reason to stop using the API.
        clock = FakeClock()
        transport = FakeTransport(self.http_error(400), gemini_response("Porto is in Portugal."))
        backend = make_backend(transport, clock=clock)
        backend.answer(parse("tell me about lisbon", Mode.CITY))
        self.assertIsNotNone(backend.answer(parse("tell me about porto", Mode.CITY)))


class ResponseParsingTests(unittest.TestCase):
    def parse_response(self, body):
        transport = FakeTransport(body)
        return make_backend(transport).answer(parse("tell me about lisbon", Mode.CITY))

    def test_empty_candidates_yield_no_answer(self):
        self.assertIsNone(self.parse_response({"candidates": []}))

    def test_safety_block_yields_no_answer(self):
        # A blocked candidate has no parts; the local models answer instead.
        self.assertIsNone(
            self.parse_response(
                {"candidates": [{"finishReason": "SAFETY", "content": {}}]}
            )
        )

    def test_unexpected_shape_yields_no_answer(self):
        self.assertIsNone(self.parse_response({"error": {"message": "nope"}}))
        self.assertIsNone(self.parse_response({"candidates": "not a list"}))

    def test_whitespace_only_text_yields_no_answer(self):
        self.assertIsNone(self.parse_response(gemini_response("   \n  ")))

    def test_multipart_responses_are_joined(self):
        body = {
            "candidates": [
                {"content": {"parts": [{"text": "Lisbon is "}, {"text": "in Portugal."}]}}
            ]
        }
        self.assertEqual(self.parse_response(body).text, "Lisbon is in Portugal.")


class SpeechCleaningTests(unittest.TestCase):
    def test_markdown_is_stripped(self):
        # Piper reads asterisks and backticks aloud.
        self.assertEqual(
            clean_for_speech("**Lisbon** is in `Portugal`."), "Lisbon is in Portugal."
        )

    def test_headings_and_newlines_collapse(self):
        self.assertEqual(clean_for_speech("# Lisbon\n\nIn Portugal."), "Lisbon In Portugal.")

    def test_wrapping_quotes_are_removed(self):
        self.assertEqual(clean_for_speech('"El museo está cerrado."'), "El museo está cerrado.")

    def test_accents_survive(self):
        self.assertEqual(clean_for_speech("Le musée est fermé."), "Le musée est fermé.")


class BuilderTests(unittest.TestCase):
    def test_no_key_means_offline_only(self):
        self.assertIsNone(build_online_backend(OnlineConfig(), None))
        self.assertIsNone(build_online_backend(OnlineConfig(), ""))

    def test_disabled_in_config(self):
        self.assertIsNone(build_online_backend(OnlineConfig(enabled=False), "key"))

    def test_unknown_provider_is_refused_not_guessed(self):
        config = OnlineConfig(provider="some-other-llm")
        self.assertIsNone(build_online_backend(config, "key"))

    def test_configured_backend_is_built(self):
        self.assertIsInstance(build_online_backend(OnlineConfig(), "key"), GeminiBackend)

    def test_backend_refuses_to_construct_without_a_key(self):
        with self.assertRaises(ValueError):
            GeminiBackend(OnlineConfig(), api_key="")


if __name__ == "__main__":
    unittest.main()
