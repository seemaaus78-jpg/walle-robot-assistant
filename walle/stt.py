"""Speech-to-text from the INMP441 I2S microphone via Vosk.

Two corrections to the draft capture loop:

* It opened the stream with ``frames_per_buffer=8000`` but read 4000 frames per
  call, so the ring buffer filled faster than it drained and PortAudio dropped
  audio at the front of every other sentence. The two values are now the same
  number.
* It listened continuously with no gate, so the microphone picked up the
  robot's own Piper output through the speaker, transcribed it, and answered
  itself. Audio captured while the speaker is active is now drained and thrown
  away, and the recogniser is reset afterwards so no half-heard fragment of the
  robot's own voice survives into the next utterance.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path

from .config import AudioConfig, SpeechConfig

log = logging.getLogger(__name__)


class SpeechRecogniser:
    """Yields final transcripts from the microphone.

    ``mute`` is a callable, not a flag, so the assistant can pass in
    ``speaker.is_speaking`` and have the gate follow playback automatically.
    """

    def __init__(
        self,
        speech: SpeechConfig,
        audio: AudioConfig,
        mute: Callable[[], bool] | None = None,
    ) -> None:
        self._speech = speech
        self._audio = audio
        self._mute = mute or (lambda: False)
        self._closed = False

        model_path = Path(speech.model_path)
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"Vosk model directory not found: {model_path}. Download the "
                "small English model and unpack it there - see docs/setup.md."
            )

        import vosk  # noqa: PLC0415 - heavy optional dependency
        import pyaudio  # noqa: PLC0415

        vosk.SetLogLevel(-1)
        self._vosk = vosk
        self._model = vosk.Model(str(model_path))
        self._recogniser = vosk.KaldiRecognizer(self._model, audio.sample_rate)

        self._pyaudio = pyaudio.PyAudio()
        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=audio.channels,
            rate=audio.sample_rate,
            input=True,
            frames_per_buffer=audio.block_frames,
            input_device_index=audio.input_device,
        )
        self._stream.start_stream()
        log.info("microphone open at %d Hz", audio.sample_rate)

    def reset(self) -> None:
        """Drop any partially accumulated utterance."""
        self._recogniser = self._vosk.KaldiRecognizer(
            self._model, self._audio.sample_rate
        )

    def listen(self) -> Iterator[str]:
        """Block, yielding one complete transcript at a time."""
        was_muted = False
        while not self._closed:
            try:
                data = self._stream.read(
                    self._audio.block_frames, exception_on_overflow=False
                )
            except OSError as exc:
                log.error("microphone read failed: %s", exc)
                break

            if self._mute():
                # Keep draining the stream so PortAudio's buffer does not
                # overflow, but do not let the robot hear itself.
                was_muted = True
                continue

            if was_muted:
                self.reset()
                was_muted = False

            if self._recogniser.AcceptWaveform(data):
                text = json.loads(self._recogniser.Result()).get("text", "").strip()
                if text and self._passes_wake_word(text):
                    yield self._strip_wake_word(text)

    def _passes_wake_word(self, text: str) -> bool:
        if not self._speech.wake_words:
            return True
        return any(text.startswith(word) for word in self._speech.wake_words)

    def _strip_wake_word(self, text: str) -> str:
        for word in self._speech.wake_words:
            if text.startswith(word):
                return text[len(word) :].strip() or text
        return text

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.stop_stream()
            self._stream.close()
        except Exception as exc:  # noqa: BLE001 - teardown must not raise
            log.warning("error closing microphone stream: %s", exc)
        finally:
            self._pyaudio.terminate()

    def __enter__(self) -> "SpeechRecogniser":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class ScriptedRecogniser:
    """Replays a fixed list of transcripts. Used by ``--text`` mode and tests."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def listen(self) -> Iterator[str]:
        yield from self._lines

    def reset(self) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> "ScriptedRecogniser":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None
