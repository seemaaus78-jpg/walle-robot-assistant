"""Configuration loading and path resolution."""

import tempfile
import unittest
from pathlib import Path

from walle.config import Config, load_config


class DefaultsTests(unittest.TestCase):
    def test_defaults_load_without_a_file(self):
        config = load_config(Path("/nonexistent/config.toml"))
        self.assertEqual(config.default_mode, "CITY")
        self.assertEqual(config.audio.sample_rate, 16000)

    def test_capture_block_matches_the_stream_buffer(self):
        # The draft opened the stream with frames_per_buffer=8000 and then read
        # 4000 frames per call, so PortAudio dropped audio. One value now feeds
        # both, so they cannot disagree.
        self.assertEqual(Config().audio.block_frames, 4000)

    def test_spanish_voice_is_a_real_piper_name(self):
        # "es_US" was never a Piper voice; the locale must be a real one.
        spanish = Config().tts.voices["es"].model.name
        self.assertTrue(spanish.startswith("es_ES-"), spanish)

    def test_paths_are_resolved_against_base_dir(self):
        config = Config(base_dir=Path("/opt/walle")).resolve_paths()
        self.assertEqual(config.city.database, Path("/opt/walle/world_cities.db"))
        self.assertTrue(config.speech.model_path.is_absolute())
        self.assertTrue(config.tts.voices["en"].model.is_absolute())

    def test_absolute_paths_are_left_alone(self):
        config = Config(base_dir=Path("/opt/walle"))
        config = config.__class__(
            base_dir=config.base_dir,
            city=config.city.__class__(database=Path("/mnt/sd/cities.db")),
        ).resolve_paths()
        self.assertEqual(config.city.database, Path("/mnt/sd/cities.db"))


class TomlTests(unittest.TestCase):
    def write(self, body: str) -> Path:
        tmp = Path(tempfile.mkdtemp())
        path = tmp / "config.toml"
        path.write_text(body)
        return path

    def test_overrides_are_applied(self):
        path = self.write(
            """
            default_mode = "TRANSLATE"

            [translate]
            source_lang = "en"
            target_lang = "fr"

            [network]
            cache_ttl_s = 5.0

            [city]
            database = "data/cities.db"
            """
        )
        config = load_config(path)
        self.assertEqual(config.default_mode, "TRANSLATE")
        self.assertEqual(config.translate.target_lang, "fr")
        self.assertEqual(config.network.cache_ttl_s, 5.0)
        self.assertEqual(config.city.database.name, "cities.db")
        self.assertTrue(config.city.database.is_absolute())

    def test_servos_are_parsed_into_dataclasses(self):
        path = self.write(
            """
            [motion]
            chip = "gpiochip0"

            [[motion.servos]]
            name = "right_arm"
            line = 22
            rest_angle = 80.0
            """
        )
        config = load_config(path)
        self.assertEqual(config.motion.chip, "gpiochip0")
        self.assertEqual(len(config.motion.servos), 1)
        self.assertEqual(config.motion.servos[0].name, "right_arm")
        self.assertEqual(config.motion.servos[0].line, 22)

    def test_voices_accept_the_short_string_form(self):
        path = self.write(
            """
            [tts.voices]
            en = "voices/en_GB-alan-medium.onnx"
            """
        )
        config = load_config(path)
        self.assertEqual(config.tts.voices["en"].model.name, "en_GB-alan-medium.onnx")

    def test_voices_accept_the_table_form_with_a_speaker_id(self):
        path = self.write(
            """
            [tts.voices.es]
            model = "voices/es_ES-sharvard-medium.onnx"
            speaker = 1
            """
        )
        config = load_config(path)
        self.assertEqual(config.tts.voices["es"].speaker, 1)

    def test_unknown_default_mode_is_kept_for_the_assistant_to_reject(self):
        config = load_config(self.write('default_mode = "nonsense"'))
        self.assertEqual(config.default_mode, "NONSENSE")


if __name__ == "__main__":
    unittest.main()
