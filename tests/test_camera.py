"""Camera capture, with the capture program faked."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from walle.camera import MAX_IMAGE_BYTES, CommandCamera, NullCamera

JPEG = b"\xff\xd8\xff\xe0" + b"fake image data" * 4


class FakeRun:
    """Stands in for subprocess.run, writing a file like a real tool would."""

    def __init__(self, payload=JPEG, returncode=0, raises=None, write=True):
        self.payload = payload
        self.returncode = returncode
        self.raises = raises
        self.write = write
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        if self.raises:
            raise self.raises
        if self.write:
            Path(command[-1] if command[-1].endswith(".jpg") else command[
                command.index("-o") + 1 if "-o" in command else -1
            ]).write_bytes(self.payload)
        return subprocess.CompletedProcess(command, self.returncode, b"", b"err")


def camera(runner, **kwargs):
    import walle.camera as mod

    mod.subprocess.run = runner
    return CommandCamera(command=["fswebcam", "-d", "{device}", "{path}"], **kwargs)


class CaptureTests(unittest.TestCase):
    def test_returns_image_bytes(self):
        cam = camera(FakeRun())
        self.assertEqual(cam.capture(), JPEG)

    def test_device_and_size_are_substituted(self):
        runner = FakeRun()
        cam = CommandCamera(
            command=["fswebcam", "-d", "{device}", "-r", "{width}x{height}", "{path}"],
            device="/dev/video2",
            width=640,
            height=480,
        )
        import walle.camera as mod

        mod.subprocess.run = runner
        cam.capture()
        self.assertIn("/dev/video2", runner.calls[0])
        self.assertIn("640x480", runner.calls[0])

    def test_a_failing_program_returns_none(self):
        self.assertIsNone(camera(FakeRun(returncode=1, write=False)).capture())

    def test_a_timeout_returns_none(self):
        raiser = FakeRun(raises=subprocess.TimeoutExpired("fswebcam", 1))
        self.assertIsNone(camera(raiser).capture())

    def test_a_missing_program_returns_none(self):
        self.assertIsNone(camera(FakeRun(raises=FileNotFoundError())).capture())

    def test_no_file_produced_returns_none(self):
        self.assertIsNone(camera(FakeRun(write=False)).capture())

    def test_an_empty_image_is_rejected(self):
        self.assertIsNone(camera(FakeRun(payload=b"")).capture())

    def test_an_absurdly_large_image_is_refused(self):
        # A capture tool misbehaving should not push megabytes at the API.
        huge = b"x" * (MAX_IMAGE_BYTES + 1)
        self.assertIsNone(camera(FakeRun(payload=huge)).capture())

    def test_nothing_is_kept_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            cam = camera(FakeRun())
            cam.capture()
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_captures_are_saved_only_when_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = Path(tmp) / "captures"
            cam = camera(FakeRun(), save_dir=saved)
            cam.capture()
            files = list(saved.glob("*.jpg"))
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].read_bytes(), JPEG)

    def test_a_save_failure_does_not_lose_the_capture(self):
        cam = camera(FakeRun(), save_dir=Path("/proc/nonexistent/nope"))
        self.assertEqual(cam.capture(), JPEG)

    def test_no_capture_program_available(self):
        import walle.camera as mod

        original = mod.shutil.which
        mod.shutil.which = lambda _name: None
        try:
            with self.assertRaises(RuntimeError):
                CommandCamera()
        finally:
            mod.shutil.which = original


class NullCameraTests(unittest.TestCase):
    def test_returns_nothing_by_default(self):
        cam = NullCamera()
        self.assertIsNone(cam.capture())
        self.assertEqual(cam.captures, 1)

    def test_can_serve_a_fixed_frame(self):
        self.assertEqual(NullCamera(JPEG).capture(), JPEG)


if __name__ == "__main__":
    unittest.main()
