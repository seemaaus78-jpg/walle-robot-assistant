"""Emotional state machine, on an injected clock."""

import unittest

from walle.emotion import EmotionEngine, Event
from walle.face import Emotion


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.engine = EmotionEngine(self.clock, sleep_after_s=60.0)

    def test_starts_neutral(self):
        self.assertIs(self.engine.current(), Emotion.NEUTRAL)

    def test_every_event_is_handled(self):
        for event in Event:
            self.assertIsInstance(self.engine.on_event(event), Emotion)

    def test_events_map_to_expressions(self):
        self.assertIs(self.engine.on_event(Event.HEARD), Emotion.LISTENING)
        self.assertIs(self.engine.on_event(Event.THINKING), Emotion.THINKING)
        self.assertIs(self.engine.on_event(Event.NOT_UNDERSTOOD), Emotion.CONFUSED)
        self.assertIs(self.engine.on_event(Event.FAILED), Emotion.SAD)

    def test_transient_expressions_decay_to_resting(self):
        self.engine.on_event(Event.ANSWERED)
        self.assertIs(self.engine.current(), Emotion.HAPPY)
        self.clock.advance(1.0)
        self.assertIs(self.engine.current(), Emotion.HAPPY)
        self.clock.advance(2.0)
        self.assertIs(self.engine.current(), Emotion.NEUTRAL)

    def test_open_ended_states_hold_until_the_next_event(self):
        # Listening ends when something else happens, not on a timer.
        self.engine.on_event(Event.HEARD)
        self.clock.advance(30.0)
        self.assertIs(self.engine.current(), Emotion.LISTENING)

    def test_falls_asleep_after_silence(self):
        self.clock.advance(61.0)
        self.assertIs(self.engine.current(), Emotion.SLEEPING)
        self.assertTrue(self.engine.is_asleep)

    def test_activity_wakes_it_up(self):
        self.clock.advance(61.0)
        self.assertTrue(self.engine.is_asleep)
        self.engine.on_event(Event.HEARD)
        self.assertIs(self.engine.current(), Emotion.LISTENING)
        self.assertFalse(self.engine.is_asleep)

    def test_the_sleep_event_does_not_count_as_activity(self):
        self.engine.on_event(Event.SLEEP)
        self.clock.advance(61.0)
        self.assertTrue(self.engine.is_asleep)

    def test_unknown_events_are_ignored(self):
        self.engine.on_event(Event.HEARD)
        self.assertIs(self.engine.on_event("not an event"), Emotion.LISTENING)


class GazeTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.engine = EmotionEngine(self.clock, sleep_after_s=600.0)

    def test_engaged_states_hold_the_commanded_gaze(self):
        self.engine.on_event(Event.HEARD)
        self.engine.look_at(-0.8, 0.2)
        self.assertEqual(self.engine.gaze(), (-0.8, 0.2))

    def test_gaze_is_clamped(self):
        self.engine.on_event(Event.HEARD)
        self.engine.look_at(-9.0, 9.0)
        self.assertEqual(self.engine.gaze(), (-1.0, 1.0))

    def test_idle_gaze_wanders(self):
        samples = set()
        for _ in range(8):
            self.clock.advance(1.7)
            samples.add(tuple(round(v, 3) for v in self.engine.gaze()))
        self.assertGreater(len(samples), 4, samples)

    def test_idle_gaze_stays_in_range(self):
        for _ in range(50):
            self.clock.advance(0.9)
            dx, dy = self.engine.gaze()
            self.assertGreaterEqual(dx, -1.0)
            self.assertLessEqual(dx, 1.0)
            self.assertGreaterEqual(dy, -1.0)
            self.assertLessEqual(dy, 1.0)

    def test_sleeping_looks_down(self):
        engine = EmotionEngine(self.clock, sleep_after_s=1.0)
        self.clock.advance(2.0)
        self.assertGreater(engine.gaze()[1], 0.0)


if __name__ == "__main__":
    unittest.main()
