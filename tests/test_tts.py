"""Speech synthesis command construction.

No audio is produced here; these tests cover the parts that decide *what* is
run, which is where the draft's shell-injection and sample-rate bugs lived.
"""

import json
import tempfile
import unittest
from pathlib import Path

from walle.config import TTSConfig, VoiceConfig
from walle.tts import (
    MAX_UTTERANCE_CHARS,
    NullSpeaker,
    PiperSpeaker,
    VoiceNotAvailable,
    build_commands,
    sanitise,
    voice_sample_rate,
)


class SanitiseTests(unittest.TestCase):
    def test_newlines_collapse(self):
        # Piper treats a newline as an utterance boundary.
        self.assertEqual(sanitise("hello\nthere\r\nfriend"), "hello there friend")

    def test_control_characters_removed(self):
        self.assertEqual(sanitise("bad\x00text\x07here"), "bad text here")

    def test_length_is_clamped(self):
        self.assertEqual(len(sanitise("a" * 5000)), MAX_UTTERANCE_CHARS)

    def test_accented_text_survives(self):
        self.assertEqual(sanitise("¿Dónde está la estación?"), "¿Dónde está la estación?")


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.voice = VoiceConfig(Path("/models/en_US-lessac-medium.onnx"))

    def test_text_never_reaches_the_argument_vector(self):
        # The draft interpolated the transcript into a shell string. Whatever a
        # transcript contains, it must not appear in either command.
        hostile = '"; rm -rf $HOME; echo "pwned'
        piper_cmd, aplay_cmd = build_commands(self.voice, 22050, "piper", "default")
        joined = " ".join(piper_cmd + aplay_cmd)
        self.assertNotIn(hostile, joined)
        self.assertNotIn("rm", piper_cmd + aplay_cmd)
        # And nothing in either vector is a shell metacharacter payload.
        for token in piper_cmd + aplay_cmd:
            self.assertNotIn(";", token)
            self.assertNotIn("|", token)

    def test_model_path_is_passed_verbatim(self):
        piper_cmd, _ = build_commands(self.voice, 22050, "piper", "default")
        self.assertIn("--model", piper_cmd)
        self.assertEqual(piper_cmd[piper_cmd.index("--model") + 1], str(self.voice.model))

    def test_sample_rate_reaches_aplay(self):
        _, aplay_cmd = build_commands(self.voice, 16000, "piper", "default")
        self.assertEqual(aplay_cmd[aplay_cmd.index("-r") + 1], "16000")

    def test_playback_device_is_honoured(self):
        _, aplay_cmd = build_commands(self.voice, 22050, "piper", "plughw:0,0")
        self.assertEqual(aplay_cmd[aplay_cmd.index("-D") + 1], "plughw:0,0")

    def test_speaker_id_only_appears_when_set(self):
        piper_cmd, _ = build_commands(self.voice, 22050, "piper", "default")
        self.assertNotIn("--speaker", piper_cmd)
        multi = VoiceConfig(Path("/models/multi.onnx"), speaker=3)
        piper_cmd, _ = build_commands(multi, 22050, "piper", "default")
        self.assertEqual(piper_cmd[piper_cmd.index("--speaker") + 1], "3")


class SampleRateTests(unittest.TestCase):
    def test_rate_is_read_from_the_sidecar_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "voice.onnx"
            model.write_bytes(b"")
            model.with_suffix(".onnx.json").write_text(
                json.dumps({"audio": {"sample_rate": 16000}})
            )
            self.assertEqual(voice_sample_rate(VoiceConfig(model), 22050), 16000)

    def test_high_quality_voice_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "voice.onnx"
            model.with_suffix(".onnx.json").write_text(
                json.dumps({"audio": {"sample_rate": 24000}})
            )
            # 24 kHz played at the draft's hard-coded 22050 Hz runs ~9% slow.
            self.assertEqual(voice_sample_rate(VoiceConfig(model), 22050), 24000)

    def test_missing_sidecar_falls_back(self):
        voice = VoiceConfig(Path("/nonexistent/voice.onnx"))
        self.assertEqual(voice_sample_rate(voice, 22050), 22050)

    def test_malformed_sidecar_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "voice.onnx"
            model.with_suffix(".onnx.json").write_text("{not json")
            self.assertEqual(voice_sample_rate(VoiceConfig(model), 22050), 22050)

    def test_config_path_keeps_the_onnx_suffix(self):
        voice = VoiceConfig(Path("/m/en_US-lessac-medium.onnx"))
        self.assertEqual(voice.config_path.name, "en_US-lessac-medium.onnx.json")


class VoiceSelectionTests(unittest.TestCase):
    def test_unknown_language_is_reported_not_guessed(self):
        # The draft built "es_US.onnx" by string formatting, a file that does
        # not exist for any Piper voice.
        speaker = PiperSpeaker(TTSConfig())
        with self.assertRaises(VoiceNotAvailable):
            speaker._voice_for("kl")

    def test_configured_but_missing_file_is_reported(self):
        config = TTSConfig(voices={"en": VoiceConfig(Path("/nope/en.onnx"))})
        speaker = PiperSpeaker(config)
        with self.assertRaises(VoiceNotAvailable):
            speaker._voice_for("en")

    def test_speak_returns_false_instead_of_raising(self):
        config = TTSConfig(voices={"en": VoiceConfig(Path("/nope/en.onnx"))})
        self.assertFalse(PiperSpeaker(config).speak("hello"))

    def test_empty_text_is_not_spoken(self):
        self.assertFalse(PiperSpeaker(TTSConfig()).speak("   "))


class NullSpeakerTests(unittest.TestCase):
    def test_records_utterances(self):
        speaker = NullSpeaker()
        speaker.speak("hola", lang="es")
        self.assertEqual(speaker.spoken, [("es", "hola")])
        self.assertFalse(speaker.is_speaking)


if __name__ == "__main__":
    unittest.main()
