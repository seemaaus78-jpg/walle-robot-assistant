"""Servo maths and the PWM thread, driven through the null GPIO backend."""

import time
import unittest

from walle.config import MotionConfig, ServoConfig
from walle.motion import NullBackend, ServoBank, angle_to_pulse_us


class PulseWidthTests(unittest.TestCase):
    def setUp(self):
        self.servo = ServoConfig("arm", line=17)  # 500-2500 us over 0-180 deg

    def test_endpoints(self):
        self.assertEqual(angle_to_pulse_us(self.servo, 0), 500)
        self.assertEqual(angle_to_pulse_us(self.servo, 180), 2500)

    def test_centre_is_the_conventional_1500us(self):
        self.assertEqual(angle_to_pulse_us(self.servo, 90), 1500)

    def test_angles_are_clamped_to_the_mechanical_range(self):
        # Commanding past the end stop makes an SG90 stall against it, which
        # heats the gearbox and browns out the shared 5 V rail.
        self.assertEqual(angle_to_pulse_us(self.servo, -45), 500)
        self.assertEqual(angle_to_pulse_us(self.servo, 900), 2500)

    def test_pulse_is_always_inside_the_servo_window(self):
        for angle in range(-90, 271, 7):
            pulse = angle_to_pulse_us(self.servo, angle)
            self.assertGreaterEqual(pulse, self.servo.min_pulse_us)
            self.assertLessEqual(pulse, self.servo.max_pulse_us)

    def test_narrowed_range_is_respected(self):
        limited = ServoConfig("neck", line=1, min_pulse_us=1000, max_pulse_us=2000)
        self.assertEqual(angle_to_pulse_us(limited, 90), 1500)
        self.assertEqual(angle_to_pulse_us(limited, 0), 1000)

    def test_degenerate_range_does_not_divide_by_zero(self):
        fixed = ServoConfig("fixed", line=2, min_angle=90.0, max_angle=90.0)
        self.assertEqual(angle_to_pulse_us(fixed, 90), 500)


class ConfigValidationTests(unittest.TestCase):
    def test_duplicate_gpio_lines_are_rejected(self):
        config = MotionConfig(
            servos=(ServoConfig("a", line=5), ServoConfig("b", line=5))
        )
        with self.assertRaises(ValueError):
            ServoBank(config, backend=NullBackend())

    def test_empty_servo_list_is_rejected(self):
        with self.assertRaises(ValueError):
            ServoBank(MotionConfig(servos=()), backend=NullBackend())


class PwmThreadTests(unittest.TestCase):
    def setUp(self):
        self.backend = NullBackend()
        self.config = MotionConfig(
            servos=(ServoConfig("arm", line=17), ServoConfig("neck", line=18)),
            hold_s=0.2,
        )
        self.bank = ServoBank(self.config, backend=self.backend)

    def tearDown(self):
        self.bank.close()

    def test_idle_bank_does_not_pulse(self):
        self.backend.writes.clear()
        time.sleep(0.15)
        # Only all-low writes while idle: no rising edges.
        self.assertTrue(all(set(w.values()) <= {0} for w in self.backend.writes))

    def test_commanding_a_servo_produces_pulses(self):
        self.backend.writes.clear()
        self.bank.set_angle("arm", 120)
        time.sleep(0.12)
        highs = [w for w in self.backend.writes if w.get(17) == 1]
        lows = [w for w in self.backend.writes if w.get(17) == 0]
        # At 50 Hz, ~100 ms is about five frames; allow slack for scheduling.
        self.assertGreaterEqual(len(highs), 2, self.backend.writes[:10])
        self.assertGreaterEqual(len(lows), 2)

    def test_servo_detaches_after_the_hold_window(self):
        # An SG90 left pulsing buzzes and keeps drawing current; the draft never
        # stopped driving the line at all.
        self.bank.set_angle("arm", 120)
        time.sleep(self.config.hold_s + 0.25)
        self.backend.writes.clear()
        time.sleep(0.12)
        self.assertTrue(all(set(w.values()) <= {0} for w in self.backend.writes))

    def test_unknown_servo_name_is_ignored_not_fatal(self):
        self.bank.set_angle("tail", 10)  # must not raise

    def test_close_is_idempotent(self):
        self.bank.close()
        self.bank.close()

    def test_detach_all_drives_every_line_low(self):
        self.bank.set_angle("arm", 120)
        self.bank.set_angle("neck", 45)
        self.backend.writes.clear()
        self.bank.detach_all()
        self.assertEqual(self.backend.writes[-1], {17: 0, 18: 0})


class GestureTests(unittest.TestCase):
    def test_wave_returns_to_rest_and_detaches(self):
        backend = NullBackend()
        config = MotionConfig(
            servos=(ServoConfig("right_arm", line=17, rest_angle=90.0),), hold_s=0.1
        )
        bank = ServoBank(config, backend=backend)
        try:
            bank.wave_hand(times=1)
        finally:
            bank.close()
        self.assertEqual(backend.writes[-1], {17: 0})

    def test_gesture_on_a_missing_servo_is_a_no_op(self):
        config = MotionConfig(servos=(ServoConfig("neck_tilt", line=3),))
        bank = ServoBank(config, backend=NullBackend())
        try:
            bank.wave_hand()  # no right_arm configured
        finally:
            bank.close()


if __name__ == "__main__":
    unittest.main()
