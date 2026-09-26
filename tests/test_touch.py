"""The XPT2046 protocol, sampling and calibration arithmetic.

All against fakes: the point is that a wrong shift or a swapped axis is caught
here rather than by poking a screen and squinting at where the cursor lands.
"""

import unittest

from walle.touch import (
    ADC_FULL_SCALE,
    MAX_CLOCK_HZ,
    TouchCalibration,
    TouchPanel,
    scale,
)

_READ_X, _READ_Y, _READ_Z1, _READ_Z2 = 0xD0, 0x90, 0xB0, 0xC0


class FakeSpi:
    """Answers each control byte with a programmed 12-bit reading."""

    def __init__(self, values: dict) -> None:
        self.values = values
        self.speeds: list = []
        self.sent: list = []

    def transfer(self, data, speed_hz=None) -> bytes:
        self.sent.append(bytes(data))
        self.speeds.append(speed_hz)
        raw = self.values.get(data[0], 0) << 3       # 12 bits, left aligned
        return bytes([0x00, (raw >> 8) & 0xFF, raw & 0xFF])


class FakeLines:
    def __init__(self) -> None:
        self.writes: list = []
        self.closed = False

    def set_values(self, values: dict) -> None:
        self.writes.append(dict(values))

    def close(self) -> None:
        self.closed = True


def build_panel(values, **kwargs):
    spi, cs = FakeSpi(values), FakeLines()
    return TouchPanel(spi, cs, 312, **kwargs), spi, cs


class ScaleTests(unittest.TestCase):
    def test_maps_the_calibrated_range_onto_pixels(self):
        self.assertEqual(scale(300, 300, 3800, 320, False), 0)
        self.assertEqual(scale(3800, 300, 3800, 320, False), 319)

    def test_clamps_outside_the_calibrated_range(self):
        # A press on the bezel reads past the calibrated edge; the nearest
        # pixel is the honest answer, not an exception.
        self.assertEqual(scale(0, 300, 3800, 320, False), 0)
        self.assertEqual(scale(9999, 300, 3800, 320, False), 319)

    def test_inverting_mirrors_the_axis(self):
        self.assertEqual(scale(300, 300, 3800, 320, True), 319)
        self.assertEqual(scale(3800, 300, 3800, 320, True), 0)

    def test_degenerate_range_does_not_divide_by_zero(self):
        self.assertEqual(scale(500, 1000, 1000, 320, False), 0)


class ProtocolTests(unittest.TestCase):
    def test_twelve_bit_reply_is_decoded_from_the_trailing_bytes(self):
        panel, _, _ = build_panel({_READ_X: 1234})
        self.assertEqual(panel._channel(_READ_X), 1234)

    def test_each_read_sends_a_control_byte_and_two_idle_bytes(self):
        panel, spi, _ = build_panel({_READ_X: 1})
        panel._channel(_READ_X)
        self.assertEqual(spi.sent[-1], bytes([_READ_X, 0x00, 0x00]))

    def test_clock_is_held_below_the_controllers_ceiling(self):
        # The panel shares this bus at 32 MHz; reading touch that fast returns
        # plausible nonsense rather than an error.
        panel, spi, _ = build_panel({_READ_X: 1}, speed_hz=32_000_000)
        panel._channel(_READ_X)
        self.assertEqual(spi.speeds[-1], MAX_CLOCK_HZ)

    def test_chip_select_brackets_a_read(self):
        panel, _, cs = build_panel({_READ_X: 100})
        cs.writes.clear()
        panel.read_raw()
        self.assertEqual(cs.writes[0], {312: 0})
        self.assertEqual(cs.writes[-1], {312: 1})

    def test_chip_select_idles_high(self):
        _, _, cs = build_panel({})
        self.assertEqual(cs.writes[0], {312: 1})


class ReadTests(unittest.TestCase):
    def values(self, x, y, z1, z2):
        return {_READ_X: x, _READ_Y: y, _READ_Z1: z1, _READ_Z2: z2}

    def test_light_contact_is_not_a_touch(self):
        panel, _, _ = build_panel(self.values(2000, 2000, 10, ADC_FULL_SCALE - 10))
        self.assertIsNone(panel.read())

    def test_a_firm_press_reports_a_point(self):
        panel, _, _ = build_panel(self.values(2000, 2000, 900, 100))
        point = panel.read()
        self.assertIsNotNone(point)
        self.assertTrue(0 <= point.x < 320 and 0 <= point.y < 240)

    def test_swap_xy_transposes_the_axes(self):
        # The panel is 240x320 in silicon and the face runs at 320x240, so the
        # touch axes transpose for the same reason the display's do.
        straight = TouchCalibration(swap_xy=False, invert_x=False, invert_y=False)
        swapped = TouchCalibration(swap_xy=True, invert_x=False, invert_y=False)
        vals = self.values(3800, 300, 900, 100)
        a, _, _ = build_panel(vals, calibration=straight)
        b, _, _ = build_panel(vals, calibration=swapped)
        self.assertNotEqual((a.read().x, a.read().y), (b.read().x, b.read().y))

    def test_pressure_rises_as_the_press_hardens(self):
        soft, _, _ = build_panel(self.values(2000, 2000, 400, 2000))
        hard, _, _ = build_panel(self.values(2000, 2000, 1200, 200))
        self.assertGreater(hard.read_raw()[2], soft.read_raw()[2])

    def test_median_rejects_a_single_outlier(self):
        panel, spi, _ = build_panel({_READ_X: 2000}, samples=3)
        readings = iter([4095, 2000, 2000])

        def flaky(data, speed_hz=None):
            raw = (next(readings) if data[0] == _READ_X else 0) << 3
            return bytes([0, (raw >> 8) & 0xFF, raw & 0xFF])

        spi.transfer = flaky
        self.assertEqual(panel._median(_READ_X), 2000)

    def test_close_parks_the_chip_select_high_and_releases_it(self):
        panel, _, cs = build_panel({})
        panel.close()
        self.assertEqual(cs.writes[-1], {312: 1})
        self.assertTrue(cs.closed)


if __name__ == "__main__":
    unittest.main()
