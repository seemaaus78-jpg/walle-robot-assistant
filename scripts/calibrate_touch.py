#!/usr/bin/env python3
"""Measure this panel's touch range and print a [touch] block to paste.

The defaults in config.example.toml are typical for these modules, not correct
for yours: the usable range of the resistive film varies panel to panel, and
guessing puts every press about a centimetre out. Run this once, on the board,
with the screen wired:

    python3 scripts/calibrate_touch.py

Press and hold each corner when asked. Nothing is written to disk - the block
is printed for you to paste into config.toml, so a bad run costs nothing.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from walle.config import load_config  # noqa: E402
from walle.display import SpiWriter  # noqa: E402
from walle.motion import GpiodBackend  # noqa: E402
from walle.touch import TouchPanel  # noqa: E402

CORNERS = (
    ("top left", "x_min", "y_min"),
    ("top right", "x_max", "y_min"),
    ("bottom left", "x_min", "y_max"),
    ("bottom right", "x_max", "y_max"),
)

HOLD_SAMPLES = 12
"""Enough readings to settle. The first few after contact are always low as
the film compresses, so the extremes are taken over the whole hold."""


def sample_corner(panel: TouchPanel, name: str) -> tuple[int, int]:
    input("Press and hold the %s corner, then press Enter... " % name)
    readings = []
    for _ in range(HOLD_SAMPLES):
        x, y, pressure = panel.read_raw()
        if pressure >= panel.calibration.pressure_threshold:
            readings.append((x, y))
        time.sleep(0.05)
    if not readings:
        raise SystemExit(
            "No contact detected at the %s corner.\n"
            "Either nothing is touching the glass, or T_DO is not on pin 21 -\n"
            "without MISO every read comes back as zero." % name
        )
    # Median of each axis independently: a finger rolls slightly during a hold.
    xs = sorted(r[0] for r in readings)
    ys = sorted(r[1] for r in readings)
    x, y = xs[len(xs) // 2], ys[len(ys) // 2]
    print("  %s -> raw x=%d y=%d  (%d good samples)" % (name, x, y, len(readings)))
    return x, y


def main() -> int:
    config = load_config()
    if not config.touch.enabled:
        print("note: [touch] enabled is false in config; calibrating anyway\n")

    spi = SpiWriter(config.display.spi_device, config.touch.speed_hz)
    lines = GpiodBackend(
        config.touch.cs_chip, [config.touch.cs_line], consumer="walle-calibrate"
    )
    panel = TouchPanel(
        spi, lines, config.touch.cs_line, speed_hz=config.touch.speed_hz
    )

    print("Four corners. Hold each one firmly until the reading prints.\n")
    try:
        measured = {name: sample_corner(panel, name) for name, _, _ in CORNERS}
    finally:
        panel.close()
        spi.close()

    xs = [x for x, _ in measured.values()]
    ys = [y for _, y in measured.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    if x_max - x_min < 500 or y_max - y_min < 500:
        print(
            "\nThe four corners barely differ, so something is wrong with the\n"
            "wiring rather than the calibration. Check T_CS is on pin 16 and\n"
            "T_DO on pin 21 before trusting these numbers.",
            file=sys.stderr,
        )

    # Which way round the axes run is whatever the film says, not an assumption.
    left_x = measured["top left"][0]
    right_x = measured["top right"][0]
    top_y = measured["top left"][1]
    bottom_y = measured["bottom left"][1]

    print("\nPaste this into config.toml:\n")
    print("[touch]")
    print("enabled = true")
    print("x_min = %d" % x_min)
    print("x_max = %d" % x_max)
    print("y_min = %d" % y_min)
    print("y_max = %d" % y_max)
    print("swap_xy = %s" % str(config.touch.swap_xy).lower())
    print("invert_x = %s" % ("true" if left_x > right_x else "false"))
    print("invert_y = %s" % ("true" if top_y > bottom_y else "false"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
