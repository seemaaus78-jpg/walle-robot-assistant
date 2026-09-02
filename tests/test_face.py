"""Eye geometry: the whole expression system, with no display attached."""

import unittest

from walle.face import (
    BLINK_DURATION_S,
    BlinkClock,
    Emotion,
    STYLES,
    build_face,
)

SIZE = (240, 240)


def face(emotion=Emotion.NEUTRAL, t=0.0, gaze=(0.0, 0.0), blink=None):
    return build_face(emotion, t, SIZE, gaze, blink)


class StyleCoverageTests(unittest.TestCase):
    def test_every_emotion_has_a_style(self):
        for emotion in Emotion:
            self.assertIn(emotion, STYLES, emotion)

    def test_emotions_are_visually_distinct(self):
        """On a face with two features, emotions must differ in *shape*.

        Colour alone is not enough: it is the first thing lost to a cheap
        panel, a colour-blind viewer, or the swap_bytes setting being wrong.
        """
        shapes = {
            (round(f.left.width, 2), round(f.left.height, 2), round(f.left.tilt, 1))
            for f in (face(e) for e in Emotion)
        }
        self.assertGreaterEqual(len(shapes), len(Emotion) - 1, shapes)


class GeometryTests(unittest.TestCase):
    def test_eyes_are_level_and_symmetric_by_default(self):
        f = face()
        self.assertEqual(f.left.cy, f.right.cy)
        self.assertLess(f.left.cx, f.right.cx)
        centre = SIZE[0] / 2
        self.assertAlmostEqual(centre - f.left.cx, f.right.cx - centre, places=5)

    def test_tilt_is_mirrored_between_the_eyes(self):
        # Both eyes leaning the same way reads as a broken render, not a mood.
        f = face(Emotion.SAD)
        self.assertNotEqual(f.left.tilt, 0.0)
        self.assertAlmostEqual(f.left.tilt, -f.right.tilt)

    def test_geometry_scales_with_the_panel(self):
        small = build_face(Emotion.NEUTRAL, 0.0, (240, 240))
        large = build_face(Emotion.NEUTRAL, 0.0, (480, 480))
        self.assertAlmostEqual(large.left.width, small.left.width * 2, places=5)

    def test_eyes_stay_inside_the_panel(self):
        for emotion in Emotion:
            for gaze in ((0, 0), (-1, -1), (1, 1)):
                f = build_face(emotion, 3.0, SIZE, gaze)
                for eye in (f.left, f.right):
                    self.assertGreaterEqual(eye.cx - eye.width / 2, -1, emotion)
                    self.assertLessEqual(eye.cx + eye.width / 2, SIZE[0] + 1, emotion)
                    self.assertGreaterEqual(eye.cy - eye.height / 2, -1, emotion)
                    self.assertLessEqual(eye.cy + eye.height / 2, SIZE[1] + 1, emotion)

    def test_rendering_is_deterministic(self):
        # Two processes must not disagree about the face at the same moment.
        self.assertEqual(face(Emotion.HAPPY, t=12.5), face(Emotion.HAPPY, t=12.5))

    def test_unknown_emotion_falls_back_to_neutral(self):
        self.assertEqual(build_face("nonsense", 0.0, SIZE).left, face().left)


class GazeTests(unittest.TestCase):
    def test_gaze_moves_both_eyes_together(self):
        centre = face()
        left_look = face(gaze=(-1.0, 0.0))
        self.assertLess(left_look.left.cx, centre.left.cx)
        self.assertLess(left_look.right.cx, centre.right.cx)

    def test_gaze_is_clamped(self):
        extreme = face(gaze=(-40.0, 40.0))
        limit = face(gaze=(-1.0, 1.0))
        self.assertAlmostEqual(extreme.left.cx, limit.left.cx)
        self.assertAlmostEqual(extreme.left.cy, limit.left.cy)


class BlinkTests(unittest.TestCase):
    def setUp(self):
        self.clock = BlinkClock()

    def test_eyes_are_open_between_blinks(self):
        self.assertEqual(self.clock.openness(0.0), 1.0)

    def test_a_blink_closes_and_reopens(self):
        # Find the first scheduled blink and sample through it.
        self.clock._extend_to(30.0)
        start = self.clock._times[0]
        mid = self.clock.openness(start + BLINK_DURATION_S / 2)
        after = self.clock.openness(start + BLINK_DURATION_S + 0.01)
        self.assertLess(mid, 0.1)
        self.assertEqual(after, 1.0)

    def test_blinking_squashes_eye_height_not_width(self):
        self.clock._extend_to(30.0)
        start = self.clock._times[0]
        open_face = build_face(Emotion.NEUTRAL, 0.0, SIZE, blink=self.clock)
        shut_face = build_face(
            Emotion.NEUTRAL, start + BLINK_DURATION_S / 2, SIZE, blink=self.clock
        )
        self.assertLess(shut_face.left.height, open_face.left.height * 0.2)
        self.assertAlmostEqual(shut_face.left.width, open_face.left.width)

    def test_a_closed_eye_is_still_a_visible_line(self):
        self.clock._extend_to(30.0)
        start = self.clock._times[0]
        f = build_face(Emotion.NEUTRAL, start + BLINK_DURATION_S / 2, SIZE, blink=self.clock)
        self.assertGreater(f.left.height, 0.0)

    def test_sleeping_does_not_blink(self):
        # Sleeping eyes still drift slightly - they breathe - but they must not
        # dip through a blink, which would read as waking up.
        self.clock._extend_to(30.0)
        start = self.clock._times[0]
        a = build_face(Emotion.SLEEPING, start, SIZE, blink=self.clock)
        b = build_face(
            Emotion.SLEEPING, start + BLINK_DURATION_S / 2, SIZE, blink=self.clock
        )
        self.assertAlmostEqual(a.left.height, b.left.height, delta=a.left.height * 0.15)

    def test_blink_curve_is_continuous_at_both_edges(self):
        # The value must meet the 1.0 returned outside the window, or every
        # blink shows a visible step.
        self.clock._extend_to(30.0)
        start = self.clock._times[0]
        self.assertAlmostEqual(self.clock.openness(start), 1.0, places=6)
        self.assertAlmostEqual(
            self.clock.openness(start + BLINK_DURATION_S * 0.999), 1.0, places=3
        )
        self.assertAlmostEqual(
            self.clock.openness(start + BLINK_DURATION_S / 2), 0.0, places=6
        )

    def test_sleeping_eyes_are_nearly_shut(self):
        awake = face(Emotion.NEUTRAL)
        asleep = face(Emotion.SLEEPING)
        self.assertLess(asleep.left.height, awake.left.height * 0.3)

    def test_schedule_is_deterministic_and_ordered(self):
        first, second = BlinkClock(), BlinkClock()
        first._extend_to(120.0)
        second._extend_to(120.0)
        self.assertEqual(first._times, second._times)
        self.assertEqual(first._times, sorted(first._times))

    def test_negative_time_is_safe(self):
        self.assertEqual(self.clock.openness(-5.0), 1.0)


if __name__ == "__main__":
    unittest.main()
