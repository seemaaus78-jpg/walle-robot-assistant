"""Pixel packing, layout arithmetic, and graceful degradation.

Pillow is optional and absent on plenty of machines, so these tests cover the
parts that work without it plus the behaviour when it is missing.
"""

import unittest

from walle.display import Card, Display, NullBackend, fit_box, pack_rgb565


class Rgb565Tests(unittest.TestCase):
    def test_primary_colours(self):
        self.assertEqual(pack_rgb565([(255, 0, 0)]), b"\xf8\x00")
        self.assertEqual(pack_rgb565([(0, 255, 0)]), b"\x07\xe0")
        self.assertEqual(pack_rgb565([(0, 0, 255)]), b"\x00\x1f")

    def test_black_and_white(self):
        self.assertEqual(pack_rgb565([(0, 0, 0)]), b"\x00\x00")
        self.assertEqual(pack_rgb565([(255, 255, 255)]), b"\xff\xff")

    def test_two_bytes_per_pixel(self):
        self.assertEqual(len(pack_rgb565([(1, 2, 3)] * 100)), 200)

    def test_byte_swap_reverses_each_pair(self):
        # Some ILI9341 boards are wired big-endian; getting this wrong gives a
        # recognisable but wrongly-coloured image, not a blank screen.
        self.assertEqual(pack_rgb565([(255, 0, 0)], swap_bytes=True), b"\x00\xf8")

    def test_empty_input(self):
        self.assertEqual(pack_rgb565([]), b"")

    def test_cyan_matches_the_face_colour(self):
        # The default eye colour must survive the round trip recognisably.
        packed = pack_rgb565([(62, 207, 207)])
        value = (packed[0] << 8) | packed[1]
        red = ((value >> 11) & 0x1F) << 3
        green = ((value >> 5) & 0x3F) << 2
        blue = (value & 0x1F) << 3
        self.assertLess(abs(red - 62), 10)
        self.assertLess(abs(green - 207), 10)
        self.assertLess(abs(blue - 207), 10)


class FitBoxTests(unittest.TestCase):
    def test_square_into_square(self):
        self.assertEqual(fit_box((512, 512), (240, 240)), (0, 0, 240, 240))

    def test_wide_source_is_letterboxed_vertically(self):
        x, y, w, h = fit_box((640, 480), (240, 240))
        self.assertEqual(w, 240)
        self.assertLess(h, 240)
        self.assertGreater(y, 0)
        self.assertEqual(x, 0)

    def test_tall_source_is_letterboxed_horizontally(self):
        x, y, w, h = fit_box((480, 640), (240, 240))
        self.assertEqual(h, 240)
        self.assertLess(w, 240)
        self.assertGreater(x, 0)

    def test_aspect_ratio_is_preserved(self):
        # Cropping a map to fill the panel loses exactly the edges you wanted.
        _, _, w, h = fit_box((800, 400), (320, 240))
        self.assertAlmostEqual(w / h, 2.0, places=1)

    def test_result_never_exceeds_the_panel(self):
        for source in ((10, 4000), (4000, 10), (1, 1), (333, 777)):
            x, y, w, h = fit_box(source, (320, 240))
            self.assertLessEqual(w, 320)
            self.assertLessEqual(h, 240)
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)

    def test_degenerate_source_does_not_divide_by_zero(self):
        self.assertEqual(fit_box((0, 0), (240, 240)), (0, 0, 240, 240))


class NullBackendTests(unittest.TestCase):
    def test_records_frames(self):
        backend = NullBackend((320, 240))
        self.assertEqual(backend.size, (320, 240))
        backend.show("frame")
        self.assertEqual(backend.frames, 1)
        self.assertEqual(backend.last, "frame")


class DegradationTests(unittest.TestCase):
    """A missing screen is a normal configuration, not an error."""

    def test_disabled_display_is_inert(self):
        display = Display(backend=None, enabled=False)
        display.set_emotion.__call__  # attribute exists
        display.show_card(Card("Kyoto", ("Japan",)))
        display.show_image(None)
        display.clear()
        display.close()
        self.assertFalse(display.enabled)

    def test_disabled_display_reports_zero_size(self):
        self.assertEqual(Display(backend=None, enabled=False).size, (0, 0))

    def test_show_image_bytes_reports_failure_without_pillow(self):
        display = Display(backend=None, enabled=False)
        self.assertFalse(display.show_image_bytes(b"not an image"))

    def test_context_manager(self):
        with Display(backend=None, enabled=False) as display:
            display.show_face()

    def test_close_is_idempotent(self):
        display = Display(backend=None, enabled=False)
        display.close()
        display.close()


if __name__ == "__main__":
    unittest.main()
