"""The touch layer on the front of the panel.

The 2.8" modules carry an XPT2046 resistive controller behind the glass,
sharing the display's SPI bus. Two things make it awkward, and both are handled
here rather than by the caller:

*   It tops out near 2 MHz while the panel runs at 32, so every read overrides
    the bus clock for the duration of that one transfer.
*   SPI1 brings out a single hardware chip select and there are two devices, so
    this one takes a GPIO chip select of its own. See walle/display.py.

Resistive panels are noisy. A single sample is worthless; the reads here are
medianed and gated on measured pressure, which is also how a touch is detected
at all - the T_IRQ line is left unconnected, so nothing has to poll a GPIO.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# Control bytes: start bit, channel, 12-bit differential mode, power down
# between conversions. Differential rejects the supply noise that single-ended
# mode picks up straight off a robot's motor rail.
_READ_X = 0xD0
_READ_Y = 0x90
_READ_Z1 = 0xB0
_READ_Z2 = 0xC0

MAX_CLOCK_HZ = 2_000_000
"""The controller's ceiling. Reads above this return plausible-looking
nonsense rather than failing, which is the hard kind of bug to find."""

ADC_FULL_SCALE = 4095


@dataclass(frozen=True)
class TouchCalibration:
    """Maps the controller's raw ADC range onto pixels.

    The defaults are typical for these modules but are not a substitute for
    measuring your own: the resistive film's usable range varies panel to
    panel, and an uncalibrated screen reads every press a centimetre or so out.
    Run scripts/calibrate_touch.py and paste the numbers into config.toml.
    """

    x_min: int = 300
    x_max: int = 3800
    y_min: int = 300
    y_max: int = 3800
    swap_xy: bool = True
    """The panel is 240x320 in silicon and the face runs at 320x240, so the
    touch axes are transposed for the same reason the display's are."""

    invert_x: bool = False
    invert_y: bool = True
    pressure_threshold: int = 200
    """Below this, treat it as no contact. Too low and the panel reports
    phantom presses from its own noise floor."""


@dataclass(frozen=True)
class TouchPoint:
    x: int
    y: int
    pressure: int


def scale(raw: int, low: int, high: int, span: int, invert: bool) -> int:
    """Map a raw ADC reading onto 0..span-1, clamped.

    Clamping rather than rejecting: a press right on the bezel reads outside
    the calibrated range, and the nearest edge pixel is the honest answer.
    """
    if high == low:
        return 0
    position = (raw - low) / (high - low)
    if invert:
        position = 1.0 - position
    return max(0, min(span - 1, int(position * span)))


class TouchPanel:
    """An XPT2046 sharing the panel's SPI bus.

    ``spi`` needs a ``transfer(data, speed_hz)`` returning the MISO bytes, and
    ``cs_lines`` a ``set_values``. Both are injected so the protocol is
    testable without hardware.
    """

    def __init__(
        self,
        spi: Any,
        cs_lines: Any,
        cs_line: int,
        size: tuple[int, int] = (320, 240),
        calibration: TouchCalibration | None = None,
        samples: int = 3,
        speed_hz: int = MAX_CLOCK_HZ,
    ) -> None:
        self._spi = spi
        self._cs = cs_lines
        self._cs_line = cs_line
        self._size = size
        self.calibration = calibration or TouchCalibration()
        self._samples = max(1, samples)
        self._speed_hz = min(speed_hz, MAX_CLOCK_HZ)
        self._deselect()

    def _select(self) -> None:
        self._cs.set_values({self._cs_line: 0})

    def _deselect(self) -> None:
        self._cs.set_values({self._cs_line: 1})

    def _channel(self, control: int) -> int:
        """One conversion: control byte out, 12 bits back, MSB first.

        The reply straddles the two bytes after the control byte and is left
        aligned in them, hence the shift.
        """
        reply = self._spi.transfer(bytes([control, 0x00, 0x00]), self._speed_hz)
        if len(reply) < 3:
            return 0
        return ((reply[1] << 8) | reply[2]) >> 3

    def _median(self, control: int) -> int:
        readings = sorted(self._channel(control) for _ in range(self._samples))
        return readings[len(readings) // 2]

    def read_raw(self) -> tuple[int, int, int]:
        """Raw ADC x, y and pressure, without calibration. For calibrating."""
        self._select()
        try:
            x = self._median(_READ_X)
            y = self._median(_READ_Y)
            z1 = self._channel(_READ_Z1)
            z2 = self._channel(_READ_Z2)
        finally:
            self._deselect()
        # The datasheet's pressure estimate, reduced to the part that varies:
        # resistance falls as the press hardens, so this rises with it.
        pressure = z1 + (ADC_FULL_SCALE - z2)
        return x, y, max(0, pressure)

    def read(self) -> TouchPoint | None:
        """A calibrated point in screen pixels, or None if nothing is touching."""
        raw_x, raw_y, pressure = self.read_raw()
        cal = self.calibration
        if pressure < cal.pressure_threshold:
            return None
        if not (cal.x_min <= raw_x <= cal.x_max or cal.y_min <= raw_y <= cal.y_max):
            # Both axes outside the calibrated window is the signature of a
            # read taken as the finger lifted, not a press near a corner.
            return None

        width, height = self._size
        if cal.swap_xy:
            x = scale(raw_y, cal.y_min, cal.y_max, width, cal.invert_y)
            y = scale(raw_x, cal.x_min, cal.x_max, height, cal.invert_x)
        else:
            x = scale(raw_x, cal.x_min, cal.x_max, width, cal.invert_x)
            y = scale(raw_y, cal.y_min, cal.y_max, height, cal.invert_y)
        return TouchPoint(x, y, pressure)

    def close(self) -> None:
        try:
            self._deselect()
            self._cs.close()
        except Exception as exc:  # noqa: BLE001 - releasing must never raise
            log.warning("error releasing touch chip select: %s", exc)
