"""Servo motion for the neck and arms.

The draft ``wave_hand()`` ran ``gpioset gpiochip1 15=1``, which drives the line
high and leaves it there. An SG90 is not an on/off actuator: it reads a 50 Hz
pulse train and positions its horn according to the pulse *width* (roughly
500 us to 2500 us for 0 to 180 degrees). A constantly-high line is an invalid
signal, so the servo either sits still or drifts against its end stop while
drawing stall current from the shared 5 V rail. The original also spawned a
process per movement and never returned the horn to rest.

This module generates a real pulse train in a single background thread and
detaches (stops pulsing) once a pose has settled, which stops the servos
buzzing and keeps them off the battery budget between gestures.

Timing caveat, stated plainly: this is *software* PWM from CPython. Kernel
scheduling jitter of a few hundred microseconds is normal, and that is visible
as a slight tremble in the horn. It is fine for gestures. If you want steady
holds, drive the servos from a PCA9685 over I2C, or from a hardware PWM channel,
and keep this module only as the fallback - see docs/hardware.md.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Protocol

from .config import MotionConfig, ServoConfig

log = logging.getLogger(__name__)


class LineBackend(Protocol):
    """Minimal GPIO surface used by :class:`ServoBank`."""

    def set_values(self, values: dict[int, int]) -> None: ...

    def close(self) -> None: ...


class NullBackend:
    """Records writes instead of touching hardware (dev machines, tests)."""

    def __init__(self) -> None:
        self.writes: list[dict[int, int]] = []

    def set_values(self, values: dict[int, int]) -> None:
        self.writes.append(dict(values))

    def close(self) -> None:
        return None


class GpiodBackend:
    """libgpiod backed lines, supporting both the v1 and v2 Python bindings."""

    def __init__(self, chip: str, offsets: list[int], consumer: str = "walle") -> None:
        import gpiod  # noqa: PLC0415 - optional, hardware-only dependency

        self._gpiod = gpiod
        self._offsets = offsets
        # The v1 binding writes the whole line set at once, so the current level
        # of every line has to be tracked here; otherwise dropping one servo's
        # pulse would also drop every other servo still mid-pulse.
        self._state = {offset: 0 for offset in offsets}
        self._v2 = hasattr(gpiod, "request_lines")

        if self._v2:
            from gpiod.line import Direction, Value  # noqa: PLC0415

            self._Value = Value
            settings = gpiod.LineSettings(
                direction=Direction.OUTPUT, output_value=Value.INACTIVE
            )
            self._request = gpiod.request_lines(
                self._chip_path(chip),
                consumer=consumer,
                config={offset: settings for offset in offsets},
            )
        else:
            self._chip = gpiod.Chip(chip)
            self._lines = self._chip.get_lines(offsets)
            self._lines.request(
                consumer=consumer,
                type=gpiod.LINE_REQ_DIR_OUT,
                default_vals=[0] * len(offsets),
            )

    @staticmethod
    def _chip_path(chip: str) -> str:
        return chip if chip.startswith("/dev/") else f"/dev/{chip}"

    def set_values(self, values: dict[int, int]) -> None:
        if self._v2:
            self._request.set_values(
                {
                    offset: (self._Value.ACTIVE if level else self._Value.INACTIVE)
                    for offset, level in values.items()
                }
            )
        else:
            self._state.update(values)
            self._lines.set_values([self._state[offset] for offset in self._offsets])

    def close(self) -> None:
        try:
            if self._v2:
                self._request.release()
            else:
                self._lines.release()
                self._chip.close()
        except Exception as exc:  # noqa: BLE001 - releasing must never raise
            log.warning("error releasing GPIO lines: %s", exc)


def angle_to_pulse_us(servo: ServoConfig, angle: float) -> int:
    """Map an angle onto a pulse width, clamped to the servo's safe range.

    Clamping matters: commanding an SG90 past its mechanical stop makes it push
    against the stop indefinitely, which overheats the gearbox and browns out the
    5 V rail the board shares.
    """
    lo, hi = servo.min_angle, servo.max_angle
    angle = max(lo, min(hi, angle))
    span = hi - lo
    fraction = 0.0 if span == 0 else (angle - lo) / span
    pulse = servo.min_pulse_us + fraction * (servo.max_pulse_us - servo.min_pulse_us)
    return int(round(pulse))


class ServoBank:
    """Drives every configured servo from one 50 Hz software PWM thread."""

    def __init__(self, config: MotionConfig, backend: LineBackend | None = None) -> None:
        self._config = config
        self._servos = {servo.name: servo for servo in config.servos}
        if not self._servos:
            raise ValueError("no servos configured")

        offsets = [servo.line for servo in config.servos]
        if len(set(offsets)) != len(offsets):
            raise ValueError(f"duplicate GPIO line in servo config: {offsets}")

        self._backend = backend if backend is not None else self._make_backend(offsets)
        self._frame_s = 1.0 / config.frame_hz
        self._targets: dict[int, int] = {}
        self._deadlines: dict[int, float] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="servo-pwm", daemon=True
        )
        self._thread.start()

    def _make_backend(self, offsets: list[int]) -> LineBackend:
        if not self._config.enabled:
            log.info("motion disabled by config; using null GPIO backend")
            return NullBackend()
        try:
            return GpiodBackend(self._config.chip, offsets)
        except Exception as exc:  # noqa: BLE001 - any import/permission failure
            log.warning(
                "GPIO unavailable (%s); motion will be simulated. On the board, "
                "check the chip name with `gpioinfo` and that the user is in the "
                "`gpio` group.",
                exc,
            )
            return NullBackend()

    # -- public API ---------------------------------------------------------

    def set_angle(self, name: str, angle: float) -> None:
        """Command one servo and keep it powered for ``hold_s``."""
        servo = self._servos.get(name)
        if servo is None:
            log.warning("unknown servo %r; known: %s", name, sorted(self._servos))
            return
        pulse = angle_to_pulse_us(servo, angle)
        with self._lock:
            self._targets[servo.line] = pulse
            self._deadlines[servo.line] = time.monotonic() + self._config.hold_s
        self._wake.set()

    def detach_all(self) -> None:
        """Stop pulsing every servo immediately."""
        with self._lock:
            self._targets.clear()
            self._deadlines.clear()
        self._backend.set_values({servo.line: 0 for servo in self._config.servos})

    def rest(self) -> None:
        """Return every servo to its neutral pose, then let it go slack."""
        for servo in self._config.servos:
            self.set_angle(servo.name, servo.rest_angle)
        time.sleep(self._config.hold_s)
        self.detach_all()

    def wave_hand(self, times: int = 2, servo: str = "right_arm") -> None:
        """The acknowledgement gesture used when a mode changes."""
        target = self._servos.get(servo)
        if target is None:
            return
        low = target.rest_angle - 40
        high = target.rest_angle + 40
        for _ in range(times):
            self.set_angle(servo, high)
            time.sleep(0.28)
            self.set_angle(servo, low)
            time.sleep(0.28)
        self.set_angle(servo, target.rest_angle)
        time.sleep(0.25)
        self.detach_all()

    def nod(self, times: int = 2, servo: str = "neck_tilt") -> None:
        """Acknowledge that a command was understood."""
        target = self._servos.get(servo)
        if target is None:
            return
        for _ in range(times):
            self.set_angle(servo, target.rest_angle - 25)
            time.sleep(0.22)
            self.set_angle(servo, target.rest_angle + 15)
            time.sleep(0.22)
        self.set_angle(servo, target.rest_angle)
        time.sleep(0.2)
        self.detach_all()

    def close(self) -> None:
        """Rest the servos, stop the PWM thread and release the GPIO lines."""
        if self._stop.is_set():
            return
        try:
            self.detach_all()
        finally:
            self._stop.set()
            self._wake.set()
            self._thread.join(timeout=2.0)
            self._backend.close()

    def __enter__(self) -> "ServoBank":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- PWM thread ---------------------------------------------------------

    def _active_pulses(self) -> dict[int, int]:
        """Targets whose hold window has not expired yet."""
        now = time.monotonic()
        with self._lock:
            expired = [line for line, due in self._deadlines.items() if due <= now]
            for line in expired:
                self._targets.pop(line, None)
                self._deadlines.pop(line, None)
            return dict(self._targets)

    def _run(self) -> None:
        while not self._stop.is_set():
            pulses = self._active_pulses()
            if not pulses:
                # Nothing to drive: idle until a command arrives instead of
                # spinning at 50 Hz for no reason.
                self._backend.set_values(
                    {servo.line: 0 for servo in self._config.servos}
                )
                self._wake.wait(timeout=0.5)
                self._wake.clear()
                continue

            frame_start = time.perf_counter()
            try:
                self._backend.set_values({line: 1 for line in pulses})
                # Drop each line at its own pulse width, shortest first.
                for line, pulse_us in sorted(pulses.items(), key=lambda kv: kv[1]):
                    self._sleep_until(frame_start + pulse_us / 1_000_000)
                    self._backend.set_values({line: 0})
            except Exception as exc:  # noqa: BLE001 - keep the robot alive
                log.error("servo write failed: %s", exc)
                self._stop.set()
                break

            self._sleep_until(frame_start + self._frame_s)

    @staticmethod
    def _sleep_until(deadline: float) -> None:
        remaining = deadline - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
