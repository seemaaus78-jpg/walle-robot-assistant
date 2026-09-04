"""The eye: capturing a single still when asked.

Two rules shape this module, and both are deliberate.

**It captures only on request.** There is no background loop, no preview
stream, no motion trigger. A frame is grabbed when someone asks the robot to
look at something, and at no other time. A microphone that is always listening
is what a voice assistant is; a camera that is always watching is a different
kind of object to have in a room, and this is not that.

**Nothing is kept unless you ask for it.** Captured frames live in memory,
go to the model, and are dropped. ``save_captures`` writes them to the card
instead, and defaults to off.

Capture goes through an external command rather than a Python imaging library,
because the right tool differs per board - ``libcamera-still`` for a ribbon
camera, ``fswebcam`` or ``ffmpeg`` for a USB webcam - and shelling out costs
nothing on a still taken once a minute.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)

# Tried in order; the first whose program is installed wins. {path}, {width}
# and {height} are filled in.
DEFAULT_COMMANDS: tuple[tuple[str, ...], ...] = (
    # Ribbon (CSI) cameras on modern Debian-based images.
    ("libcamera-still", "-n", "--immediate", "-o", "{path}",
     "--width", "{width}", "--height", "{height}"),
    ("rpicam-still", "-n", "--immediate", "-o", "{path}",
     "--width", "{width}", "--height", "{height}"),
    # USB webcams. fswebcam is the simplest thing that works.
    ("fswebcam", "-q", "--no-banner", "-d", "{device}",
     "-r", "{width}x{height}", "{path}"),
    ("ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
     "-f", "v4l2", "-video_size", "{width}x{height}",
     "-i", "{device}", "-frames:v", "1", "{path}"),
)

MAX_IMAGE_BYTES = 4 * 1024 * 1024
"""Refuse to send anything larger. A still at these sizes is well under this;
something much bigger means the capture tool did not do what was expected."""


class Camera(Protocol):
    def capture(self) -> bytes | None: ...
    def close(self) -> None: ...


class NullCamera:
    """No camera. Returns nothing, cheerfully."""

    def __init__(self, payload: bytes | None = None) -> None:
        self.payload = payload
        self.captures = 0

    def capture(self) -> bytes | None:
        self.captures += 1
        return self.payload

    def close(self) -> None:
        return None


class CommandCamera:
    """Grabs one JPEG by running a capture program."""

    def __init__(
        self,
        device: str = "/dev/video0",
        width: int = 1024,
        height: int = 768,
        warmup_s: float = 0.0,
        timeout_s: float = 12.0,
        command: list[str] | None = None,
        save_dir: Path | None = None,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.warmup_s = warmup_s
        self.timeout_s = timeout_s
        self.save_dir = Path(save_dir) if save_dir else None
        self._template = command or self._detect()
        if self._template is None:
            raise RuntimeError(
                "no capture program found. Install one of: libcamera-still "
                "(ribbon camera), fswebcam or ffmpeg (USB webcam)."
            )
        log.info("camera: %s via %s", device, self._template[0])

    @staticmethod
    def _detect() -> list[str] | None:
        for template in DEFAULT_COMMANDS:
            if shutil.which(template[0]):
                return list(template)
        return None

    def _build(self, path: Path) -> list[str]:
        return [
            part.format(
                path=str(path),
                device=self.device,
                width=self.width,
                height=self.height,
            )
            for part in self._template
        ]

    def capture(self) -> bytes | None:
        """Take one still. Returns JPEG bytes, or None if it did not work."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.jpg"
            if self.warmup_s:
                # Some webcams hand back a black or badly exposed first frame.
                time.sleep(self.warmup_s)
            try:
                result = subprocess.run(
                    self._build(path),
                    capture_output=True,
                    timeout=self.timeout_s,
                    check=False,
                )
            except FileNotFoundError:
                log.error("capture program %r disappeared", self._template[0])
                return None
            except subprocess.TimeoutExpired:
                log.error("camera timed out after %.0f s", self.timeout_s)
                return None
            except OSError as exc:
                log.error("camera failed: %s", exc)
                return None

            if result.returncode != 0:
                log.error(
                    "camera exited %s: %s",
                    result.returncode,
                    result.stderr.decode("utf-8", "replace").strip()[:200],
                )
                return None
            if not path.is_file():
                log.error("camera produced no file")
                return None

            data = path.read_bytes()

        if not data:
            log.error("camera produced an empty image")
            return None
        if len(data) > MAX_IMAGE_BYTES:
            log.error("captured image is %d bytes; refusing to send", len(data))
            return None

        if self.save_dir is not None:
            self._save(data)
        return data

    def _save(self, data: bytes) -> None:
        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            (self.save_dir / f"{stamp}.jpg").write_bytes(data)
        except OSError as exc:
            log.warning("could not save capture: %s", exc)

    def close(self) -> None:
        return None


def build_camera(
    enabled: bool,
    device: str,
    width: int,
    height: int,
    warmup_s: float,
    save_dir: Path | None,
) -> Camera | None:
    """Open the camera, or return None. A missing camera is not an error."""
    if not enabled:
        log.info("camera disabled in config")
        return None
    try:
        return CommandCamera(
            device=device,
            width=width,
            height=height,
            warmup_s=warmup_s,
            save_dir=save_dir,
        )
    except Exception as exc:  # noqa: BLE001 - degrade rather than refuse to boot
        log.warning("camera unavailable (%s); the robot will not see", exc)
        return None
