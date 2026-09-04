"""The tracks. Driven through the null GPIO backend, no motors attached."""

import time
import unittest

from walle.config import DriveConfig
from walle.drive import MOVEMENTS, Direction, DriveBase, Wheels
from walle.motion import NullBackend

CFG = DriveConfig(
    left_forward_line=20, left_backward_line=21,
    right_forward_line=22, right_backward_line=23,
    default_seconds=0.3, max_seconds=1.0, pwm_hz=200.0,
)


class MovementTableTests(unittest.TestCase):
    def test_forward_runs_both_tracks_the_same_way(self):
        self.assertEqual(MOVEMENTS[Direction.FORWARD], Wheels(1, 1))

    def test_turning_runs_the_tracks_opposite_ways(self):
        # A tracked chassis has nothing to steer with; it turns on the spot.
        self.assertEqual(MOVEMENTS[Direction.LEFT], Wheels(-1, 1))
        self.assertEqual(MOVEMENTS[Direction.RIGHT], Wheels(1, -1))

    def test_stop_is_both_tracks_still(self):
        self.assertEqual(MOVEMENTS[Direction.STOP], Wheels(0, 0))


class ConfigValidationTests(unittest.TestCase):
    def test_duplicate_lines_are_rejected(self):
        bad = DriveConfig(left_forward_line=5, left_backward_line=5)
        with self.assertRaises(ValueError):
            DriveBase(bad, backend=NullBackend())


class BridgeSafetyTests(unittest.TestCase):
    """Both pins of a pair high is a short through the H-bridge."""

    def setUp(self):
        self.backend = NullBackend()
        self.drive = DriveBase(CFG, backend=self.backend)

    def tearDown(self):
        self.drive.close()

    def test_a_pair_is_never_driven_high_together(self):
        for direction in Direction:
            self.drive.move(direction, seconds=0.2)
            time.sleep(0.08)
            for write in self.backend.writes:
                self.assertFalse(
                    write.get(20) == 1 and write.get(21) == 1,
                    f"left bridge shorted during {direction}",
                )
                self.assertFalse(
                    write.get(22) == 1 and write.get(23) == 1,
                    f"right bridge shorted during {direction}",
                )

    def test_forward_drives_only_the_forward_pins(self):
        self.backend.writes.clear()
        self.drive.move(Direction.FORWARD, seconds=0.3)
        time.sleep(0.08)
        highs = [w for w in self.backend.writes if w.get(20) == 1]
        self.assertGreater(len(highs), 0)
        for write in highs:
            self.assertEqual(write[21], 0)
            self.assertEqual(write[22], 1)
            self.assertEqual(write[23], 0)


class TimeLimitTests(unittest.TestCase):
    """A bounded run is what stops it driving off the desk."""

    def setUp(self):
        self.backend = NullBackend()
        self.drive = DriveBase(CFG, backend=self.backend)

    def tearDown(self):
        self.drive.close()

    def test_movement_stops_by_itself(self):
        self.drive.move(Direction.FORWARD, seconds=0.25)
        self.assertTrue(self.drive.is_moving)
        time.sleep(0.5)
        self.assertFalse(self.drive.is_moving)

    def test_the_pins_actually_go_low_afterwards(self):
        self.drive.move(Direction.FORWARD, seconds=0.2)
        time.sleep(0.45)
        self.backend.writes.clear()
        time.sleep(0.1)
        for write in self.backend.writes:
            self.assertEqual(set(write.values()), {0})

    def test_a_long_request_is_clamped(self):
        self.drive.move(Direction.FORWARD, seconds=600)
        time.sleep(CFG.max_seconds + 0.4)
        self.assertFalse(self.drive.is_moving)

    def test_stop_is_immediate(self):
        self.drive.move(Direction.FORWARD, seconds=5)
        self.drive.stop()
        self.assertFalse(self.drive.is_moving)
        self.assertEqual(set(self.backend.writes[-1].values()), {0})


class SpeedTests(unittest.TestCase):
    def setUp(self):
        self.backend = NullBackend()
        self.drive = DriveBase(CFG, backend=self.backend)

    def tearDown(self):
        self.drive.close()

    def test_speed_is_clamped_above_the_stall_floor(self):
        # Below roughly a third a small gear motor buzzes without turning.
        self.assertEqual(self.drive._clamp_speed(0.01), CFG.min_speed)
        self.assertEqual(self.drive._clamp_speed(None), CFG.default_speed)
        self.assertEqual(self.drive._clamp_speed(9.0), 1.0)


class LifecycleTests(unittest.TestCase):
    def test_close_stops_the_motors(self):
        backend = NullBackend()
        drive = DriveBase(CFG, backend=backend)
        drive.move(Direction.FORWARD, seconds=5)
        drive.close()
        self.assertEqual(set(backend.writes[-1].values()), {0})

    def test_close_is_idempotent(self):
        drive = DriveBase(CFG, backend=NullBackend())
        drive.close()
        drive.close()

    def test_context_manager_stops_on_exit(self):
        backend = NullBackend()
        with DriveBase(CFG, backend=backend) as drive:
            drive.move(Direction.FORWARD, seconds=5)
        self.assertEqual(set(backend.writes[-1].values()), {0})

    def test_an_idle_drive_does_not_pulse(self):
        backend = NullBackend()
        drive = DriveBase(CFG, backend=backend)
        try:
            backend.writes.clear()
            time.sleep(0.15)
            self.assertTrue(all(set(w.values()) <= {0} for w in backend.writes))
        finally:
            drive.close()


if __name__ == "__main__":
    unittest.main()
