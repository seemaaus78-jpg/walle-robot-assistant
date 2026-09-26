"""Pixel packing, layout arithmetic, and graceful degradation.

Pillow is optional and absent on plenty of machines, so these tests cover the
parts that work without it plus the behaviour when it is missing.
"""

import unittest

from walle.config import DisplayConfig
from walle.display import (
    ROTATIONS,
    Card,
    Display,
    Ili9341Backend,
    NullBackend,
    SPI_IOC_WR_BITS_PER_WORD,
    SPI_IOC_WR_MAX_SPEED_HZ,
    SPI_IOC_WR_MODE,
    SpiWriter,
    _ioc_write,
    build_display,
    fit_box,
    pack_rgb565,
)


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


class FakeSpi:
    """Records what would go down the wire."""

    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.closed = False
        self.max_speed_hz = 0
        self.mode = None

    def writebytes(self, data) -> None:
        self.writes.append(bytes(data))

    def close(self) -> None:
        self.closed = True


class FakeLines:
    """Stands in for GpiodBackend."""

    def __init__(self) -> None:
        self.writes: list[dict] = []
        self.closed = False

    def set_values(self, values: dict) -> None:
        self.writes.append(dict(values))

    def close(self) -> None:
        self.closed = True


class FakeImage:
    """Just enough of a Pillow image for the packing path."""

    def __init__(self, pixels) -> None:
        self._pixels = list(pixels)

    def convert(self, mode: str) -> "FakeImage":
        return self

    def getdata(self):
        return self._pixels


def build_panel(**kwargs):
    """An Ili9341Backend wired entirely to fakes."""
    spi, dc, reset = FakeSpi(), FakeLines(), FakeLines()
    kwargs.setdefault("rotation", 90)
    panel = Ili9341Backend(
        "/dev/spidev1.0", spi=spi, dc_lines=dc, reset_lines=reset, **kwargs
    )
    return panel, spi, dc, reset


def commands_sent(spi: FakeSpi) -> list[int]:
    """Single-byte writes are command codes; payloads follow them."""
    return [write[0] for write in spi.writes if len(write) == 1]


class Ili9341GeometryTests(unittest.TestCase):
    def test_rotation_picks_the_panel_orientation(self):
        for rotation, expected in ((0, (240, 320)), (90, (320, 240)),
                                   (180, (240, 320)), (270, (320, 240))):
            panel, _, _, _ = build_panel(rotation=rotation)
            self.assertEqual(panel.size, expected, rotation)

    def test_unknown_rotation_is_rejected_at_construction(self):
        # Better than silently rendering a sheared image on the bench.
        with self.assertRaises(ValueError):
            build_panel(rotation=45)

    def test_madctl_carries_the_rotation_bits(self):
        panel, spi, _, _ = build_panel(rotation=90)
        index = spi.writes.index(bytes([0x36]))
        self.assertEqual(spi.writes[index + 1], bytes([ROTATIONS[90][0]]))


class Ili9341InitTests(unittest.TestCase):
    def test_software_reset_precedes_sleep_out_and_display_on(self):
        _, spi, _, _ = build_panel()
        sent = commands_sent(spi)
        self.assertLess(sent.index(0x01), sent.index(0x11))
        self.assertLess(sent.index(0x11), sent.index(0x29))

    def test_pixel_format_is_sixteen_bit(self):
        _, spi, _, _ = build_panel()
        index = spi.writes.index(bytes([0x3A]))
        self.assertEqual(spi.writes[index + 1], b"\x55")

    def test_reset_line_is_pulsed_low_then_released(self):
        _, _, _, reset = build_panel()
        self.assertEqual([w[313] for w in reset.writes], [1, 0, 1])

    def test_no_reset_line_is_allowed(self):
        # Some modules tie RESET high; the panel must still come up.
        spi, dc = FakeSpi(), FakeLines()
        panel = Ili9341Backend(
            "/dev/spidev1.0", spi=spi, dc_lines=dc, reset=None, reset_lines=None
        )
        self.assertIn(0x29, commands_sent(spi))
        panel.close()

    def test_command_drops_dc_and_payload_raises_it(self):
        _, _, dc, _ = build_panel()
        levels = [w[5] for w in dc.writes]
        self.assertEqual(levels[0], 0)
        self.assertIn(1, levels)


class Ili9341FrameTests(unittest.TestCase):
    def test_show_writes_two_bytes_for_every_pixel(self):
        panel, spi, _, _ = build_panel(rotation=90)
        width, height = panel.size
        before = len(spi.writes)
        panel.show(FakeImage([(255, 0, 0)] * (width * height)))
        payload = b"".join(spi.writes[before:])
        # The window commands precede the frame, hence >= rather than ==.
        self.assertGreaterEqual(len(payload), width * height * 2)

    def test_show_sets_a_full_screen_window(self):
        panel, spi, _, _ = build_panel(rotation=90)
        width, height = panel.size
        before = len(spi.writes)
        panel.show(FakeImage([(0, 0, 0)] * (width * height)))
        tail = spi.writes[before:]
        caset = tail.index(bytes([0x2A]))
        self.assertEqual(tail[caset + 1], bytes([0, 0, (width - 1) >> 8, (width - 1) & 0xFF]))
        paset = tail.index(bytes([0x2B]))
        self.assertEqual(tail[paset + 1], bytes([0, 0, (height - 1) >> 8, (height - 1) & 0xFF]))

    def test_frame_is_split_to_the_spidev_transfer_limit(self):
        # spidev rejects anything over its bufsiz, which defaults to 4096.
        panel, spi, _, _ = build_panel(rotation=90)
        width, height = panel.size
        panel.show(FakeImage([(0, 255, 0)] * (width * height)))
        self.assertTrue(all(len(write) <= 4096 for write in spi.writes))

    def test_pixels_are_packed_big_endian_by_default(self):
        panel, _, _, _ = build_panel(rotation=90)
        self.assertEqual(panel._encode(FakeImage([(255, 0, 0)])), b"\xf8\x00")

    def test_swap_bytes_reverses_each_pair(self):
        panel, _, _, _ = build_panel(rotation=90, swap_bytes=True)
        self.assertEqual(panel._encode(FakeImage([(255, 0, 0)])), b"\x00\xf8")

    def test_byteswap_is_pairwise(self):
        self.assertEqual(Ili9341Backend._byteswap(b"\x12\x34\x56\x78"), b"\x34\x12\x78\x56")

    def test_close_releases_the_bus_and_both_gpio_lines(self):
        panel, spi, dc, reset = build_panel()
        panel.close()
        self.assertTrue(spi.closed and dc.closed and reset.closed)


class BuildDisplayTests(unittest.TestCase):
    def test_backend_none_gives_a_silent_display(self):
        display = build_display(DisplayConfig(backend="none"))
        self.assertFalse(display.enabled)

    def test_command_line_veto_beats_the_config(self):
        display = build_display(DisplayConfig(backend="spi"), enabled=False)
        self.assertFalse(display.enabled)

    def test_missing_hardware_degrades_instead_of_raising(self):
        # No /dev/fb1 and no /dev/spidev1.0 on a dev machine: the robot still
        # talks, it just has no face.
        display = build_display(DisplayConfig(backend="auto"))
        self.assertFalse(display.enabled)


class SpiWriterTests(unittest.TestCase):
    """The ioctl numbers are the whole point of the spidev C extension; if they
    are wrong the panel stays dark with no error, so pin them down here."""

    def test_ioc_write_matches_the_linux_macro(self):
        # _IOW(type, nr, size) = (1 << 30) | (size << 16) | (type << 8) | nr
        self.assertEqual(_ioc_write(1, 1), 0x40016B01)
        self.assertEqual(_ioc_write(3, 1), 0x40016B03)
        self.assertEqual(_ioc_write(4, 4), 0x40046B04)

    def test_request_numbers_are_the_documented_ones(self):
        self.assertEqual(SPI_IOC_WR_MODE, 0x40016B01)
        self.assertEqual(SPI_IOC_WR_BITS_PER_WORD, 0x40016B03)
        self.assertEqual(SPI_IOC_WR_MAX_SPEED_HZ, 0x40046B04)

    def test_missing_device_raises_rather_than_half_opening(self):
        with self.assertRaises(OSError):
            SpiWriter("/dev/spidev-does-not-exist", 32_000_000)
