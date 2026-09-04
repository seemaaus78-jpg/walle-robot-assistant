"""Driving the tracks: two DC gear motors through an H-bridge.

Separate from ``walle/motion.py`` on purpose. That module positions servos -
it thinks in angles, and holds a pose. This one drives wheels: it thinks in
direction and speed, and a wheel has no position to hold.

The reference build uses two 17-68 RPM gear motors and a mini L298N. That board
takes four control lines - two per motor - and decides direction from which of
the pair is high. Speed comes from pulsing those lines, since the mini L298N
usually has its enable pins tied high rather than broken out.

Why an H-bridge at all: a GPIO pin can signal but not push. It supplies a few
milliamps, and a small gear motor wants hundreds. Wiring a motor straight to a
pin destroys the pin, and often the board. The driver takes its power from the
battery and only takes instructions from the pins.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum

from .config import DriveConfig
from .motion import GpiodBackend, LineBackend, NullBackend

log = logging.getLogger(__name__)


class Direction(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    STOP = "stop"


@dataclass(frozen=True)
class Wheels:
    """Which way each track turns, as -1, 0 or +1."""

    left: int
    right: int


# Turning is done on the spot, tracks running opposite ways. A tracked chassis
# cannot steer any other way - there is nothing to point.
MOVEMENTS: dict[Direction, Wheels] = {
    Direction.FORWARD: Wheels(1, 1),
    Direction.BACKWARD: Wheels(-1, -1),
    Direction.LEFT: Wheels(-1, 1),
    Direction.RIGHT: Wheels(1, -1),
    Direction.STOP: Wheels(0, 0),
}


class DriveBase:
    """Two tracks, four control lines, one software PWM thread for speed.

    Every movement is time-limited. A robot told to go forward and then left
    alone drives off the desk; there is no floor sensor here to stop it, so the
    limit is the safety mechanism.
    """

    def __init__(self, config: DriveConfig, backend: LineBackend | None = None) -> None:
        self._config = config
        self._lines = [
            config.left_forward_line,
            config.left_backward_line,
            config.right_forward_line,
            config.right_backward_line,
        ]
        if len(set(self._lines)) != len(self._lines):
            raise ValueError(f"duplicate GPIO line in drive config: {self._lines}")

        self._backend = backend if backend is not None else self._make_backend()
        self._lock = threading.Lock()
        self._wheels = Wheels(0, 0)
        self._speed = config.default_speed
        self._deadline = 0.0
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, name="drive", daemon=True)
        self._thread.start()

    def _make_backend(self) -> LineBackend:
        if not self._config.enabled:
            log.info("drive disabled in config; motors will be simulated")
            return NullBackend()
        try:
            return GpiodBackend(self._config.chip, self._lines, consumer="walle-drive")
        except Exception as exc:  # noqa: BLE001 - never refuse to boot over wheels
            log.warning(
                "motor driver unavailable (%s); the robot will not move. Check "
                "the chip name with `gpioinfo` and that the user is in the "
                "`gpio` group.",
                exc,
            )
            return NullBackend()

    # -- public API ---------------------------------------------------------

    def move(
        self,
        direction: Direction,
        seconds: float | None = None,
        speed: float | None = None,
    ) -> None:
        """Drive in one direction for a bounded time, then stop by itself."""
        wheels = MOVEMENTS.get(direction)
        if wheels is None:
            log.warning("unknown direction %r", direction)
            return

        duration = self._config.default_seconds if seconds is None else seconds
        duration = max(0.0, min(duration, self._config.max_seconds))
        with self._lock:
            self._wheels = wheels
            self._speed = self._clamp_speed(speed)
            self._deadline = 0.0 if wheels == MOVEMENTS[Direction.STOP] else (
                time.monotonic() + duration
            )
        self._wake.set()

    def _clamp_speed(self, speed: float | None) -> float:
        value = self._config.default_speed if speed is None else speed
        # Below the stall floor the motors buzz and heat without turning.
        return max(self._config.min_speed, min(1.0, value))

    def stop(self) -> None:
        with self._lock:
            self._wheels = MOVEMENTS[Direction.STOP]
            self._deadline = 0.0
        self._all_low()
        self._wake.set()

    @property
    def is_moving(self) -> bool:
        with self._lock:
            return self._wheels != MOVEMENTS[Direction.STOP]

    def close(self) -> None:
        if self._stop.is_set():
            return
        try:
            self.stop()
        finally:
            self._stop.set()
            self._wake.set()
            self._thread.join(timeout=2.0)
            self._backend.close()

    def __enter__(self) -> "DriveBase":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- driving ------------------------------------------------------------

    def _all_low(self) -> None:
        self._backend.set_values({line: 0 for line in self._lines})

    def _levels(self, wheels: Wheels, on: bool) -> dict[int, int]:
        """Pin levels for this instant of the PWM cycle.

        Both pins of a pair low is coast. Both high would be a short through
        the bridge, so it is never produced.
        """
        cfg = self._config
        drive = 1 if on else 0
        return {
            cfg.left_forward_line: drive if wheels.left > 0 else 0,
            cfg.left_backward_line: drive if wheels.left < 0 else 0,
            cfg.right_forward_line: drive if wheels.right > 0 else 0,
            cfg.right_backward_line: drive if wheels.right < 0 else 0,
        }

    def _run(self) -> None:
        period = 1.0 / self._config.pwm_hz
        while not self._stop.is_set():
            now = time.monotonic()
            with self._lock:
                wheels, speed, deadline = self._wheels, self._speed, self._deadline
                # Time is up: stop before anything else happens.
                if deadline and now >= deadline:
                    self._wheels = wheels = MOVEMENTS[Direction.STOP]
                    self._deadline = 0.0

            if wheels == MOVEMENTS[Direction.STOP]:
                self._all_low()
                self._wake.wait(timeout=0.4)
                self._wake.clear()
                continue

            start = time.perf_counter()
            try:
                self._backend.set_values(self._levels(wheels, True))
                self._sleep_until(start + period * speed)
                self._backend.set_values(self._levels(wheels, False))
            except Exception as exc:  # noqa: BLE001 - stop rather than run away
                log.error("motor write failed, stopping: %s", exc)
                self._stop.set()
                break
            self._sleep_until(start + period)

    @staticmethod
    def _sleep_until(deadline: float) -> None:
        remaining = deadline - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
