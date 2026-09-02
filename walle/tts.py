"""Text-to-speech via Piper, played out through the MAX98357A I2S amplifier.

Three things the draft script got wrong are fixed here:

1. ``os.system(f'echo "{text}" | piper ...')`` ran every spoken string through a
   shell. Recognised speech is untrusted input; a transcript containing a quote,
   ``$``, a backtick or a semicolon either mangled the sentence or executed as a
   command. Text is now written to Piper's stdin and no shell is involved.
2. The playback rate was hard-coded to 22050 Hz. Piper voices ship at 16000 Hz
   (``low``), 22050 Hz (``medium``) or 24000 Hz (``high``); a mismatch plays back
   at the wrong pitch and speed. The rate is read from the voice's sidecar JSON.
3. The voice file was derived as ``{lang}_US.onnx``, which produces the
   non-existent ``es_US.onnx`` for Spanish. Voices are now mapped explicitly.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import unicodedata
from pathlib import Path

from .config import TTSConfig, VoiceConfig

log = logging.getLogger(__name__)

MAX_UTTERANCE_CHARS = 1000


class VoiceNotAvailable(RuntimeError):
    """Raised when no Piper voice is configured or installed for a language."""


def sanitise(text: str) -> str:
    """Strip control characters and clamp length.

    Piper reads stdin line by line, so embedded newlines would be spoken as
    separate utterances; they are collapsed to spaces.
    """
    cleaned = "".join(
        " " if unicodedata.category(ch)[0] == "C" else ch for ch in text
    )
    return " ".join(cleaned.split())[:MAX_UTTERANCE_CHARS]


def voice_sample_rate(voice: VoiceConfig, fallback: int) -> int:
    """Read the sample rate Piper will emit for this voice."""
    try:
        with open(voice.config_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        rate = int(meta["audio"]["sample_rate"])
    except (OSError, KeyError, ValueError, TypeError) as exc:
        log.warning(
            "could not read sample rate from %s (%s); falling back to %d Hz",
            voice.config_path,
            exc,
            fallback,
        )
        return fallback
    return rate


def build_commands(
    voice: VoiceConfig,
    sample_rate: int,
    binary: str,
    playback_device: str,
) -> tuple[list[str], list[str]]:
    """Return the (piper, aplay) argument vectors.

    Split out from :class:`PiperSpeaker` so the command construction can be
    tested without a synthesiser or a sound card present.
    """
    piper_cmd = [binary, "--model", str(voice.model), "--output-raw"]
    if voice.speaker is not None:
        piper_cmd += ["--speaker", str(voice.speaker)]

    aplay_cmd = [
        "aplay",
        "-q",
        "-D",
        playback_device,
        "-r",
        str(sample_rate),
        "-f",
        "S16_LE",
        "-c",
        "1",
        "-t",
        "raw",
    ]
    return piper_cmd, aplay_cmd


class PiperSpeaker:
    """Synthesise speech and block until playback finishes.

    ``is_speaking`` lets the capture loop discard audio recorded while the robot
    is talking, so it does not transcribe and answer its own voice.
    """

    def __init__(self, config: TTSConfig, playback_device: str = "default") -> None:
        self._config = config
        self._playback_device = playback_device
        self._rates: dict[str, int] = {}
        self._speaking = threading.Event()
        self._lock = threading.Lock()

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def available_languages(self) -> list[str]:
        return sorted(self._config.voices)

    def _voice_for(self, lang: str) -> VoiceConfig:
        voice = self._config.voices.get(lang)
        if voice is None:
            raise VoiceNotAvailable(f"no Piper voice configured for {lang!r}")
        if not voice.model.is_file():
            raise VoiceNotAvailable(f"Piper voice file missing: {voice.model}")
        return voice

    def _sample_rate_for(self, lang: str, voice: VoiceConfig) -> int:
        if lang not in self._rates:
            self._rates[lang] = voice_sample_rate(
                voice, self._config.fallback_sample_rate
            )
        return self._rates[lang]

    def speak(self, text: str, lang: str = "en") -> bool:
        """Speak ``text``. Returns True when audio was played."""
        payload = sanitise(text)
        if not payload:
            return False

        try:
            voice = self._voice_for(lang)
        except VoiceNotAvailable as exc:
            log.error("%s", exc)
            return False

        rate = self._sample_rate_for(lang, voice)
        piper_cmd, aplay_cmd = build_commands(
            voice, rate, self._config.binary, self._playback_device
        )

        with self._lock:
            self._speaking.set()
            try:
                return self._run(piper_cmd, aplay_cmd, payload)
            finally:
                self._speaking.clear()

    def _run(
        self, piper_cmd: list[str], aplay_cmd: list[str], payload: str
    ) -> bool:
        piper = None
        aplay = None
        try:
            piper = subprocess.Popen(
                piper_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            aplay = subprocess.Popen(
                aplay_cmd, stdin=piper.stdout, stderr=subprocess.PIPE
            )
            # Let Piper see EPIPE if aplay dies first.
            assert piper.stdout is not None
            piper.stdout.close()

            assert piper.stdin is not None
            piper.stdin.write(payload.encode("utf-8") + b"\n")
            piper.stdin.close()

            aplay.wait(timeout=self._config.timeout_s)
            piper.wait(timeout=5)
        except FileNotFoundError as exc:
            log.error("text-to-speech binary missing: %s", exc)
            return False
        except subprocess.TimeoutExpired:
            log.error("text-to-speech timed out; killing pipeline")
            for proc in (aplay, piper):
                if proc and proc.poll() is None:
                    proc.kill()
            return False
        except OSError as exc:
            log.error("text-to-speech failed: %s", exc)
            return False

        if aplay.returncode != 0:
            stderr = (aplay.stderr.read() if aplay.stderr else b"").decode(
                "utf-8", "replace"
            )
            log.error("aplay exited %s: %s", aplay.returncode, stderr.strip())
            return False
        return True

    def close(self) -> None:
        """No persistent resources; present so callers can treat it uniformly."""
        return None


class NullSpeaker:
    """Logs instead of speaking. Used on development machines and in tests."""

    def __init__(self) -> None:
        self.spoken: list[tuple[str, str]] = []

    @property
    def is_speaking(self) -> bool:
        return False

    def available_languages(self) -> list[str]:
        return ["en"]

    def speak(self, text: str, lang: str = "en") -> bool:
        self.spoken.append((lang, text))
        log.info("[speak:%s] %s", lang, text)
        return True

    def close(self) -> None:
        return None
