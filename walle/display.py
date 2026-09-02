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
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

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


def build_display(
    enabled: bool,
    device: str,
    swap_bytes: bool = False,
    font_path: str | None = None,
) -> Display:
    """Open the panel, falling back to a silent no-op display if it is absent.

    A missing screen is a normal configuration, not an error: the robot talks.
    """
    if not enabled:
        log.info("display disabled in config")
        return Display(backend=None, enabled=False)
    try:
        backend: DisplayBackend = FramebufferBackend(device, swap_bytes=swap_bytes)
        log.info("display: %s at %dx%d", device, *backend.size)
    except Exception as exc:  # noqa: BLE001 - degrade rather than refuse to boot
        log.warning("display unavailable (%s); running without a face", exc)
        return Display(backend=None, enabled=False)
    return Display(backend=backend, enabled=True, font_path=font_path)
