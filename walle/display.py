"""The screen: animated face, information cards, maps and photos.

The reference build puts a large panel across the whole front of the robot and
uses it for two quite different jobs - being a face, and showing things. Both
live here.

Pillow is imported lazily and is optional. Without it the display degrades to a
logging stub and the rest of the robot is unaffected, which keeps the assistant
runnable on a machine that has no imaging stack at all. The pixel packing and
the layout arithmetic are kept free of Pillow so they stay testable either way.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from .config import DisplayConfig
from .face import BlinkClock, Emotion, FaceGeometry, build_face

log = logging.getLogger(__name__)

FACE_FPS = 20.0
"""Fast enough that a 160 ms blink reads as smooth; slow enough that redrawing
costs a rounding error of CPU on a board this size."""


def pack_rgb565(pixels: Sequence[tuple[int, int, int]], swap_bytes: bool = False) -> bytes:
    """Pack 8-bit RGB triples into RGB565, the format most SPI panels want.

    Kept separate from any imaging library so the packing - the part that
    silently produces a psychedelic screen when it is wrong - can be tested
    directly.
    """
    out = bytearray(len(pixels) * 2)
    for index, (r, g, b) in enumerate(pixels):
        value = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
        if swap_bytes:
            value = ((value & 0xFF) << 8) | (value >> 8)
        out[index * 2] = value >> 8
        out[index * 2 + 1] = value & 0xFF
    return bytes(out)


def fit_box(
    source: tuple[int, int], target: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Letterbox ``source`` inside ``target``: returns (x, y, width, height).

    Used for maps and photos, which never match the panel's aspect ratio.
    Cropping a map to fill the screen loses exactly the edges you wanted.
    """
    src_w, src_h = source
    dst_w, dst_h = target
    if src_w <= 0 or src_h <= 0:
        return (0, 0, dst_w, dst_h)

    scale = min(dst_w / src_w, dst_h / src_h)
    width = max(1, int(round(src_w * scale)))
    height = max(1, int(round(src_h * scale)))
    return ((dst_w - width) // 2, (dst_h - height) // 2, width, height)


class DisplayBackend(Protocol):
    """Somewhere to put a finished frame."""

    @property
    def size(self) -> tuple[int, int]: ...

    def show(self, image: Any) -> None: ...

    def close(self) -> None: ...


class NullBackend:
    """Keeps the last frame instead of displaying it. Dev machines and tests."""

    def __init__(self, size: tuple[int, int] = (240, 240)) -> None:
        self._size = size
        self.frames = 0
        self.last: Any = None

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    def show(self, image: Any) -> None:
        self.frames += 1
        self.last = image

    def close(self) -> None:
        return None


class FramebufferBackend:
    """Writes frames to a Linux framebuffer.

    This covers both an SPI panel bound through ``fbtft`` (typically
    ``/dev/fb1``) and an HDMI output, without a per-panel Python driver. The
    geometry is read from sysfs rather than configured, because getting it
    wrong produces a diagonally sheared image that is easy to misdiagnose as a
    wiring fault.
    """

    def __init__(self, device: str = "/dev/fb1", swap_bytes: bool = False) -> None:
        self.device = device
        self._swap = swap_bytes
        self._size, self._bpp = self._probe(device)
        if self._bpp != 16:
            raise RuntimeError(
                f"{device} is {self._bpp} bits per pixel; this backend writes "
                "RGB565. Set the panel to 16bpp or use a different backend."
            )
        self._handle = open(device, "wb", buffering=0)

    @staticmethod
    def _probe(device: str) -> tuple[tuple[int, int], int]:
        name = Path(device).name
        sysfs = Path("/sys/class/graphics") / name
        try:
            width, height = (
                int(part)
                for part in (sysfs / "virtual_size").read_text().strip().split(",")
            )
            bpp = int((sysfs / "bits_per_pixel").read_text().strip())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"cannot read framebuffer geometry for {device}: {exc}")
        return (width, height), bpp

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    def show(self, image: Any) -> None:
        pixels = list(image.convert("RGB").getdata())
        self._handle.seek(0)
        self._handle.write(pack_rgb565(pixels, swap_bytes=self._swap))

    def close(self) -> None:
        try:
            self._handle.close()
        except OSError as exc:
            log.warning("error closing %s: %s", self.device, exc)


# -- ILI9341 over spidev ------------------------------------------------------
#
# Radxa OS ships fb_ili9341 but no overlay that binds it to a bus, and writing
# an untested sunxi overlay risks a board that will not boot. The stock
# sun60iw2p1-spi1-spidev overlay exposes /dev/spidev1.0 instead, so the panel is
# driven from here: SPI carries pixels, two plain GPIOs carry DC and RESET.

_SWRESET = 0x01
_SLPOUT = 0x11
_DISPON = 0x29
_CASET = 0x2A
_PASET = 0x2B
_RAMWR = 0x2C
_MADCTL = 0x36
_PIXFMT = 0x3A

_MY, _MX, _MV, _BGR = 0x80, 0x40, 0x20, 0x08

ROTATIONS: dict[int, tuple[int, tuple[int, int]]] = {
    0: (_MX | _BGR, (240, 320)),
    90: (_MV | _BGR, (320, 240)),
    180: (_MY | _BGR, (240, 320)),
    270: (_MX | _MY | _MV | _BGR, (320, 240)),
}
"""MADCTL value and the resulting (width, height) for each rotation. The panel
is 240x320 in silicon; 90 and 270 transpose it via the MV bit rather than by
rotating every frame in software."""

# (command, payload, delay_s). Straight from the ILI9341 datasheet's power-on
# recommendations - the gamma and power blocks are opaque magic numbers there
# too, and changing them produces a dim or colour-shifted panel.
_INIT: tuple[tuple[int, bytes, float], ...] = (
    (0xEF, b"\x03\x80\x02", 0),
    (0xCF, b"\x00\xC1\x30", 0),
    (0xED, b"\x64\x03\x12\x81", 0),
    (0xE8, b"\x85\x00\x78", 0),
    (0xCB, b"\x39\x2C\x00\x34\x02", 0),
    (0xF7, b"\x20", 0),
    (0xEA, b"\x00\x00", 0),
    (0xC0, b"\x23", 0),                                   # power control 1
    (0xC1, b"\x10", 0),                                   # power control 2
    (0xC5, b"\x3E\x28", 0),                               # VCOM control 1
    (0xC7, b"\x86", 0),                                   # VCOM control 2
    (_PIXFMT, b"\x55", 0),                                # 16 bits per pixel
    (0xB1, b"\x00\x18", 0),                               # frame rate, ~79 Hz
    (0xB6, b"\x08\x82\x27", 0),                           # display function
    (0xF2, b"\x00", 0),                                   # 3-gamma off
    (0x26, b"\x01", 0),                                   # gamma curve 1
    (0xE0, b"\x0F\x31\x2B\x0C\x0E\x08\x4E\xF1"
           b"\x37\x07\x10\x03\x0E\x09\x00", 0),
    (0xE1, b"\x00\x0E\x14\x03\x11\x07\x31\xC1"
           b"\x48\x08\x0F\x0C\x31\x36\x0F", 0),
    (_SLPOUT, b"", 0.120),
    (_DISPON, b"", 0.020),
)

_CHUNK = 4096
"""spidev refuses a transfer larger than its bufsiz module parameter, which
defaults to 4096 bytes. A full frame is 150 KB, so it goes out in pieces."""


def _ioc_write(nr: int, size: int) -> int:
    """Linux _IOW(SPI_IOC_MAGIC, nr, size), which is 'k' for the SPI driver.

    Spelled out rather than imported because the request numbers are the whole
    reason the spidev PyPI package needs a C compiler, and the board cannot
    run one: Debian 11 went end-of-life in August 2026 and its security
    archive no longer serves python3-dev.
    """
    return (1 << 30) | (size << 16) | (0x6B << 8) | nr


SPI_IOC_WR_MODE = _ioc_write(1, 1)
SPI_IOC_WR_BITS_PER_WORD = _ioc_write(3, 1)
SPI_IOC_WR_MAX_SPEED_HZ = _ioc_write(4, 4)


class SpiWriter:
    """Write-only SPI over a /dev/spidev node, using only the standard library.

    The panel is never read from - no touch controller, and the display's own
    SDO line is left unconnected - so a half-duplex ``write()`` is the entire
    requirement, and the kernel's spidev driver supports exactly that. Three
    ioctls set the mode up first. Presents ``writebytes`` and ``close`` so it
    is interchangeable with the spidev package's SpiDev object.
    """

    def __init__(self, device: str, speed_hz: int, mode: int = 0) -> None:
        import fcntl  # noqa: PLC0415 - Linux only, and only on real hardware
        import struct  # noqa: PLC0415

        self.device = device
        self._fd = os.open(device, os.O_RDWR)
        try:
            fcntl.ioctl(self._fd, SPI_IOC_WR_MODE, struct.pack("B", mode))
            fcntl.ioctl(self._fd, SPI_IOC_WR_BITS_PER_WORD, struct.pack("B", 8))
            fcntl.ioctl(self._fd, SPI_IOC_WR_MAX_SPEED_HZ, struct.pack("I", speed_hz))
        except OSError:
            os.close(self._fd)
            raise

    def writebytes(self, data) -> None:
        payload = bytes(data)
        written = 0
        while written < len(payload):
            written += os.write(self._fd, payload[written:])

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1


class Ili9341Backend:
    """An ILI9341 panel on /dev/spidev, with DC and RESET on GPIO lines.

    ``spi`` and the two line backends are injectable so the whole protocol -
    init sequence, window arithmetic, pixel packing - can be tested without
    hardware. Left to itself the backend opens the real devices.
    """

    def __init__(
        self,
        device: str = "/dev/spidev1.0",
        *,
        dc: tuple[str, int] = ("gpiochip1", 5),
        reset: tuple[str, int] | None = ("gpiochip0", 313),
        rotation: int = 90,
        speed_hz: int = 32_000_000,
        swap_bytes: bool = False,
        spi: Any | None = None,
        dc_lines: Any | None = None,
        reset_lines: Any | None = None,
    ) -> None:
        if rotation not in ROTATIONS:
            raise ValueError(
                f"rotation must be one of {sorted(ROTATIONS)}, not {rotation!r}"
            )
        madctl, self._size = ROTATIONS[rotation]
        self.device = device
        self._swap = swap_bytes
        self._fast: tuple[str, bool] | None = None
        self._calibrated = False

        self._spi = spi if spi is not None else self._open_spi(device, speed_hz)
        self._dc_line = dc[1]
        self._dc = dc_lines if dc_lines is not None else self._open_lines(*dc)
        self._reset_line = reset[1] if reset else None
        if reset_lines is not None:
            self._reset = reset_lines
        elif reset is not None:
            self._reset = self._open_lines(*reset)
        else:
            self._reset = None

        self._hard_reset()
        self._command(_SWRESET)
        time.sleep(0.150)
        for cmd, payload, delay in _INIT:
            self._command(cmd, payload)
            if delay:
                time.sleep(delay)
        self._command(_MADCTL, bytes([madctl]))

    # -- device opening ----------------------------------------------------

    @staticmethod
    def _open_spi(device: str, speed_hz: int) -> Any:
        return SpiWriter(device, speed_hz)

    @staticmethod
    def _open_lines(chip: str, offset: int) -> Any:
        from .motion import GpiodBackend  # noqa: PLC0415 - avoids a cycle

        return GpiodBackend(chip, [offset], consumer="walle-display")

    # -- the wire ----------------------------------------------------------

    def _command(self, code: int, payload: bytes = b"") -> None:
        self._dc.set_values({self._dc_line: 0})
        self._write(bytes([code]))
        if payload:
            self._dc.set_values({self._dc_line: 1})
            self._write(payload)

    def _write(self, data: bytes) -> None:
        for start in range(0, len(data), _CHUNK):
            self._spi.writebytes(list(data[start : start + _CHUNK]))

    def _hard_reset(self) -> None:
        if self._reset is None or self._reset_line is None:
            return
        for level, pause in ((1, 0.005), (0, 0.020), (1, 0.150)):
            self._reset.set_values({self._reset_line: level})
            time.sleep(pause)

    def _set_window(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self._command(_CASET, bytes([x0 >> 8, x0 & 0xFF, x1 >> 8, x1 & 0xFF]))
        self._command(_PASET, bytes([y0 >> 8, y0 & 0xFF, y1 >> 8, y1 & 0xFF]))
        self._command(_RAMWR)
        self._dc.set_values({self._dc_line: 1})

    # -- pixels ------------------------------------------------------------

    def _calibrate(self) -> None:
        """Decide once whether Pillow can pack RGB565 for us, and how.

        A full frame is 76800 pixels; packing that in Python costs more than
        the SPI transfer does. Pillow's 16-bit raw modes are C-speed but their
        byte order varies between builds, and getting it wrong yields a
        psychedelic screen rather than an error. So rather than guess, encode a
        known swatch both ways and keep whichever agrees with pack_rgb565.
        """
        self._calibrated = True
        try:
            from PIL import Image  # noqa: PLC0415 - optional, probed at runtime
        except ImportError:
            return
        swatch = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
        probe = Image.new("RGB", (2, 2))
        probe.putdata(swatch)
        want = pack_rgb565(swatch)
        for mode in ("BGR;16",):
            for swapped in (False, True):
                try:
                    got = probe.tobytes("raw", mode)
                except Exception:  # noqa: BLE001 - unsupported mode, try the next
                    break
                if swapped:
                    got = self._byteswap(got)
                if got == want:
                    self._fast = (mode, swapped)
                    return
        log.info("no fast RGB565 path on this Pillow; packing in Python")

    @staticmethod
    def _byteswap(data: bytes) -> bytes:
        out = bytearray(data)
        out[0::2], out[1::2] = out[1::2], out[0::2]
        return bytes(out)

    def _encode(self, image: Any) -> bytes:
        rgb = image.convert("RGB")
        if not self._calibrated:
            self._calibrate()
        if self._fast is not None:
            mode, swapped = self._fast
            data = rgb.tobytes("raw", mode)
            if swapped:
                data = self._byteswap(data)
        else:
            data = pack_rgb565(list(rgb.getdata()))
        return self._byteswap(data) if self._swap else data

    # -- DisplayBackend ----------------------------------------------------

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    def show(self, image: Any) -> None:
        width, height = self._size
        self._set_window(0, 0, width - 1, height - 1)
        self._write(self._encode(image))

    def close(self) -> None:
        for name, handle in (("spi", self._spi), ("dc", self._dc), ("reset", self._reset)):
            if handle is None:
                continue
            try:
                handle.close()
            except Exception as exc:  # noqa: BLE001 - closing must never raise
                log.warning("error closing display %s: %s", name, exc)


@dataclass(frozen=True)
class Card:
    """A short block of text to put on screen beside an answer."""

    title: str
    lines: tuple[str, ...] = ()
    accent: tuple[int, int, int] = (62, 207, 207)


class Display:
    """High-level screen control.

    The face animates continuously in a background thread. Showing a card, a
    map or a photo suspends the animation until :meth:`show_face` resumes it,
    so a map does not flicker under a blinking pair of eyes.
    """

    def __init__(
        self,
        backend: DisplayBackend | None = None,
        enabled: bool = True,
        font_path: str | None = None,
    ) -> None:
        self._backend = backend
        self._enabled = enabled and backend is not None
        self._image_module = None
        self._draw_module = None
        self._font_path = font_path
        self._fonts: dict[int, Any] = {}

        self._blink = BlinkClock()
        self._emotion = Emotion.NEUTRAL
        self._gaze = (0.0, 0.0)
        self._face_active = True
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        if self._enabled and self._load_pillow():
            self._thread = threading.Thread(
                target=self._animate, name="face", daemon=True
            )
            self._thread.start()
        elif self._enabled:
            log.warning(
                "Pillow is not installed; the display will stay blank. "
                "Install it with `pip install Pillow` to get the face."
            )
            self._enabled = False

    # -- setup --------------------------------------------------------------

    def _load_pillow(self) -> bool:
        try:
            from PIL import Image, ImageDraw  # noqa: PLC0415 - optional dep

            self._image_module = Image
            self._draw_module = ImageDraw
            return True
        except ImportError:
            return False

    def _font(self, size: int):
        """A truetype font if one can be found, else Pillow's bitmap default.

        The default font is tiny and fixed-size, which makes a city name
        unreadable from across a desk, so a real font is worth hunting for.
        """
        if size in self._fonts:
            return self._fonts[size]

        from PIL import ImageFont  # noqa: PLC0415

        candidates = [self._font_path] if self._font_path else []
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        ]
        font = None
        for path in candidates:
            if path and Path(path).is_file():
                try:
                    font = ImageFont.truetype(path, size)
                    break
                except OSError:
                    continue
        if font is None:
            font = ImageFont.load_default()
        self._fonts[size] = font
        return font

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def size(self) -> tuple[int, int]:
        return self._backend.size if self._backend else (0, 0)

    # -- public API ---------------------------------------------------------

    def set_emotion(self, emotion: Emotion, gaze: tuple[float, float] = (0.0, 0.0)) -> None:
        """Update what the face is doing. Cheap; safe to call every frame."""
        with self._lock:
            self._emotion = emotion
            self._gaze = gaze

    def show_face(self) -> None:
        """Resume the animated face after a card, map or photo."""
        with self._lock:
            self._face_active = True

    def show_card(self, card: Card) -> None:
        """Put a block of text on screen, e.g. a city's facts."""
        if not self._enabled:
            log.info("[display] %s | %s", card.title, " / ".join(card.lines))
            return
        with self._lock:
            self._face_active = False
        self._present(self._render_card(card))

    def show_image(self, image: Any) -> None:
        """Put an already-open Pillow image on screen, letterboxed."""
        if not self._enabled:
            log.info("[display] image")
            return
        with self._lock:
            self._face_active = False
        self._present(self._render_image(image))

    def show_image_bytes(self, data: bytes) -> bool:
        """Decode and display an image downloaded as bytes (a map tile, say)."""
        if not self._enabled:
            log.info("[display] image (%d bytes)", len(data))
            return False
        import io  # noqa: PLC0415

        try:
            image = self._image_module.open(io.BytesIO(data))
            image.load()
        except Exception as exc:  # noqa: BLE001 - any decode failure
            log.warning("could not decode image for display: %s", exc)
            return False
        self.show_image(image)
        return True

    def clear(self) -> None:
        if not self._enabled:
            return
        blank = self._image_module.new("RGB", self.size, (0, 0, 0))
        self._present(blank)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        if self._backend is not None:
            try:
                if self._enabled:
                    self.clear()
            finally:
                self._backend.close()

    def __enter__(self) -> "Display":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- rendering ----------------------------------------------------------

    def _present(self, image: Any) -> None:
        try:
            self._backend.show(image)
        except Exception as exc:  # noqa: BLE001 - a dead screen must not stop
            log.error("display write failed: %s", exc)

    def _render_face(self, geometry: FaceGeometry) -> Any:
        image = self._image_module.new("RGB", self.size, geometry.background)
        for eye in (geometry.left, geometry.right):
            self._draw_eye(image, eye, geometry.colour)
        return image

    def _draw_eye(self, image: Any, eye, colour: tuple[int, int, int]) -> None:
        """Draw one rounded, possibly tilted eye.

        Tilted eyes are drawn on their own transparent layer and rotated before
        compositing, because Pillow's rounded_rectangle cannot draw at an angle
        and a slant is what separates 'sad' from 'neutral' on a face this
        simple.
        """
        width = max(2, int(round(eye.width)))
        height = max(2, int(round(eye.height)))
        radius = max(0, min(int(round(eye.radius)), min(width, height) // 2))

        # Pad so the corners survive rotation without being clipped.
        pad = int(max(width, height) * 0.5) + 2
        layer = self._image_module.new(
            "RGBA", (width + pad * 2, height + pad * 2), (0, 0, 0, 0)
        )
        pen = self._draw_module.Draw(layer)
        pen.rounded_rectangle(
            [pad, pad, pad + width - 1, pad + height - 1],
            radius=radius,
            fill=(*colour, 255),
        )
        if eye.tilt:
            layer = layer.rotate(
                eye.tilt, resample=self._image_module.BICUBIC, expand=False
            )

        image.paste(
            layer,
            (int(round(eye.cx - layer.width / 2)), int(round(eye.cy - layer.height / 2))),
            layer,
        )

    def _render_card(self, card: Card) -> Any:
        width, height = self.size
        image = self._image_module.new("RGB", (width, height), (8, 10, 14))
        pen = self._draw_module.Draw(image)

        title_size = max(14, int(height * 0.13))
        body_size = max(11, int(height * 0.085))

        pen.rectangle([0, 0, width, int(height * 0.03)], fill=card.accent)
        pen.text(
            (int(width * 0.06), int(height * 0.10)),
            card.title[:40],
            font=self._font(title_size),
            fill=card.accent,
        )

        y = int(height * 0.10) + int(title_size * 1.6)
        for line in card.lines[:6]:
            pen.text(
                (int(width * 0.06), y),
                line[:52],
                font=self._font(body_size),
                fill=(226, 232, 240),
            )
            y += int(body_size * 1.5)
        return image

    def _render_image(self, source: Any) -> Any:
        canvas = self._image_module.new("RGB", self.size, (0, 0, 0))
        x, y, width, height = fit_box(source.size, self.size)
        resized = source.convert("RGB").resize(
            (width, height), self._image_module.LANCZOS
        )
        canvas.paste(resized, (x, y))
        return canvas

    # -- animation thread ---------------------------------------------------

    def _animate(self) -> None:
        period = 1.0 / FACE_FPS
        start = time.monotonic()
        while not self._stop.is_set():
            frame_start = time.monotonic()
            with self._lock:
                active = self._face_active
                emotion = self._emotion
                gaze = self._gaze

            if active:
                geometry = build_face(
                    emotion, frame_start - start, self.size, gaze, self._blink
                )
                try:
                    self._present(self._render_face(geometry))
                except Exception as exc:  # noqa: BLE001 - never kill the thread
                    log.error("face render failed: %s", exc)

            elapsed = time.monotonic() - frame_start
            self._stop.wait(max(0.0, period - elapsed))


_BACKEND_ORDER = {
    "auto": ("framebuffer", "spi"),
    "framebuffer": ("framebuffer",),
    "spi": ("spi",),
}


def build_display(config: DisplayConfig, enabled: bool = True) -> Display:
    """Open the panel, falling back to a silent no-op display if it is absent.

    A missing screen is a normal configuration, not an error: the robot talks.
    ``enabled`` is the command line's veto over the config's own setting.
    """
    if not (enabled and config.enabled) or config.backend == "none":
        log.info("display disabled")
        return Display(backend=None, enabled=False)

    order = _BACKEND_ORDER.get(config.backend)
    if order is None:
        log.warning("unknown display backend %r; trying both", config.backend)
        order = _BACKEND_ORDER["auto"]

    failures = []
    for kind in order:
        try:
            backend: DisplayBackend = (
                FramebufferBackend(config.device, swap_bytes=config.swap_bytes)
                if kind == "framebuffer"
                else Ili9341Backend(
                    config.spi_device,
                    dc=(config.dc_chip, config.dc_line),
                    reset=(config.reset_chip, config.reset_line),
                    rotation=config.rotation,
                    speed_hz=config.spi_speed_hz,
                    swap_bytes=config.swap_bytes,
                )
            )
        except Exception as exc:  # noqa: BLE001 - try the next, then degrade
            failures.append("%s: %s" % (kind, exc))
            continue
        log.info("display: %s at %dx%d", kind, *backend.size)
        return Display(backend=backend, enabled=True, font_path=config.font_path)

    log.warning(
        "display unavailable (%s); running without a face", "; ".join(failures)
    )
    return Display(backend=None, enabled=False)
