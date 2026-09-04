"""Configuration loading.

Defaults are tuned for the Radxa Cubie A7Z build described in docs/hardware.md.
Every value can be overridden from a TOML file (see config.example.toml); the
file is looked up at $WALLE_CONFIG, then ./config.toml, then
~/travel_assistant/config.toml.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_LOCATIONS = (
    Path("config.toml"),
    Path.home() / "travel_assistant" / "config.toml",
)


@dataclass(frozen=True)
class AudioConfig:
    """Capture settings for the INMP441 I2S microphone."""

    sample_rate: int = 16000
    """Vosk's small English model is trained at 16 kHz. Do not change."""

    block_frames: int = 4000
    """Frames handed to Vosk per read. 4000 frames = 250 ms at 16 kHz."""

    channels: int = 1
    input_device: int | None = None
    """PortAudio device index. None lets PortAudio pick the default."""

    playback_device: str = "default"
    """ALSA device name passed to aplay for the MAX98357A amplifier."""


@dataclass(frozen=True)
class SpeechConfig:
    """Vosk speech-to-text settings."""

    model_path: Path = Path("models_stt/vosk-model-small-en-us-0.15")
    wake_words: tuple[str, ...] = ()
    """If non-empty, an utterance is only acted on when it starts with one of
    these phrases. Empty means always-on listening."""


@dataclass(frozen=True)
class VoiceConfig:
    """A single Piper voice: the model plus its sidecar JSON."""

    model: Path
    speaker: int | None = None

    @property
    def config_path(self) -> Path:
        """Piper ships <voice>.onnx alongside <voice>.onnx.json."""
        return self.model.with_suffix(self.model.suffix + ".json")


@dataclass(frozen=True)
class TTSConfig:
    """Piper text-to-speech settings.

    Voices are mapped explicitly per language code. Piper voice names encode a
    locale and a quality tier (``es_ES-davefx-medium``), so they cannot be
    derived from a bare language code such as ``es``.
    """

    binary: str = "piper"
    voices_dir: Path = Path("models_tts")
    voices: dict[str, VoiceConfig] = field(
        default_factory=lambda: {
            "en": VoiceConfig(Path("models_tts/en_US-lessac-medium.onnx")),
            "es": VoiceConfig(Path("models_tts/es_ES-davefx-medium.onnx")),
        }
    )
    fallback_sample_rate: int = 22050
    """Used only if a voice's sidecar JSON is missing or unreadable."""

    timeout_s: float = 30.0


@dataclass(frozen=True)
class ServoConfig:
    """One SG90 servo on one GPIO line."""

    name: str
    line: int
    min_pulse_us: int = 500
    max_pulse_us: int = 2500
    min_angle: float = 0.0
    max_angle: float = 180.0
    rest_angle: float = 90.0


@dataclass(frozen=True)
class DriveConfig:
    """The tracks: two DC gear motors through a mini L298N H-bridge.

    Four control lines, two per motor. Direction comes from which of a pair is
    high; speed comes from pulsing them, because the mini L298N normally has
    its enable pins tied high rather than broken out.
    """

    enabled: bool = True
    chip: str = "gpiochip1"

    left_forward_line: int = 20
    left_backward_line: int = 21
    right_forward_line: int = 22
    right_backward_line: int = 23

    pwm_hz: float = 200.0
    """Fast enough that the motor averages it out rather than stepping."""

    default_speed: float = 0.75
    min_speed: float = 0.35
    """Below roughly a third, a small gear motor buzzes and warms without
    actually turning. Speeds are clamped up to this."""

    default_seconds: float = 1.2
    max_seconds: float = 5.0
    """Every movement is time-limited. There is no floor sensor on this robot,
    so a bounded run is what stops it driving off the edge of the desk."""


@dataclass(frozen=True)
class MotionConfig:
    """Servo bank settings.

    ``gpiochip1`` line numbers below are placeholders: run ``gpioinfo`` on the
    board and set the real offsets before wiring anything up.
    """

    chip: str = "gpiochip1"
    enabled: bool = True
    frame_hz: float = 50.0
    """SG90s expect a 20 ms frame."""

    hold_s: float = 0.45
    """How long to keep pulsing after reaching a position before detaching."""

    servos: tuple[ServoConfig, ...] = (
        ServoConfig("neck_pan", line=15),
        ServoConfig("neck_tilt", line=16),
        ServoConfig("left_arm", line=17),
        ServoConfig("right_arm", line=18),
    )


@dataclass(frozen=True)
class NetworkConfig:
    """Connectivity probing.

    A plain TCP connect is used rather than an HTTPS request: opening
    ``https://1.1.1.1`` completes a TLS handshake against an IP address, which
    is both slower and prone to certificate errors that look like an outage.
    """

    probe_host: str = "1.1.1.1"
    probe_port: int = 53
    timeout_s: float = 1.5
    cache_ttl_s: float = 30.0
    """Probe results are reused for this long so the main loop never blocks on
    a network round trip for every utterance."""


@dataclass(frozen=True)
class DisplayConfig:
    """The panel on the robot's face.

    ``device`` is a Linux framebuffer: ``/dev/fb1`` for an SPI panel bound
    through fbtft, ``/dev/fb0`` for HDMI. Geometry is read from sysfs, not
    configured, so there is nothing here to get wrong.
    """

    enabled: bool = True
    device: str = "/dev/fb1"
    swap_bytes: bool = False
    """Some ILI9341 boards are wired big-endian. If the face renders in the
    wrong colours, flip this before suspecting the wiring."""

    font_path: str | None = None
    sleep_after_s: float = 120.0
    """Silence before the eyes close. Also the point at which the robot
    visibly stops listening, which is worth being able to see."""

    card_seconds: float = 6.0
    """How long a map or an information card stays up before the face
    returns."""


@dataclass(frozen=True)
class ChatConfig:
    """Open-ended conversation.

    Offline chat is pattern-matched, not a language model - see walle/chat.py
    for why a local model does not fit alongside Vosk, Argos and Piper on a
    1 GB board. The llama.cpp fields are a seam for the 2 GB build.
    """

    enabled: bool = True
    history_turns: int = 8
    local_model_binary: str | None = None
    local_model_path: str | None = None
    local_model_timeout_s: float = 25.0


@dataclass(frozen=True)
class CameraConfig:
    """The camera. Captures only when asked - never a background stream.

    Vision is online-only. There is no room on a 1 GB board for a local vision
    model alongside speech, translation and synthesis, so if this matters to
    you offline the answer is a bigger board, not a setting.
    """

    enabled: bool = True
    device: str = "/dev/video0"
    """Ignored by ribbon-camera tools, which address the camera directly."""

    width: int = 1024
    height: int = 768
    warmup_s: float = 0.0
    """Some USB webcams hand back a black first frame. Try 0.5 if yours does."""

    save_captures: bool = False
    """Off by default. Photographs are sent and dropped, not collected."""

    capture_dir: Path = Path("captures")


@dataclass(frozen=True)
class GuideConfig:
    """Travel guides fetched online and kept on the card.

    Guides live in their own database so "delete everything I saved" is one
    file, and can never damage the cities database the robot needs to work.
    """

    enabled: bool = True
    database: Path = Path("travel_guides.db")

    auto_save: bool = True
    """Fetch and keep a full guide the first time a city is asked about while
    online. Turn off to only save when explicitly asked."""

    max_guides: int = 200
    """Oldest are dropped past this. At a few kilobytes each this is a
    housekeeping limit, not a space one."""

    refresh_after_days: int = 90
    """A cached guide older than this is refetched when online, and read back
    with a warning when offline."""

    save_map_tiles: bool = True
    """Also pull the map tiles for the city while saving, so the map works
    offline too."""


@dataclass(frozen=True)
class MapConfig:
    enabled: bool = True
    tile_cache: Path = Path("tiles")
    """Doubles as the offline tile pack: anything already here is used without
    a connection."""

    url_template: str = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    grid: int = 2
    timeout_s: float = 5.0
    user_agent: str = ""
    """Left empty to use the project default. OpenStreetMap's tile policy
    requires a real identifying User-Agent; put your own contact here if you
    use this heavily."""


@dataclass(frozen=True)
class OnlineConfig:
    """Cloud answering. Optional by design - the robot is complete without it.

    The API key is read from the environment, never from the config file, so a
    key cannot be committed to git along with the GPIO offsets.
    """

    enabled: bool = True
    provider: str = "gemini"
    api_key_env: str = "GEMINI_API_KEY"

    model: str = "gemini-2.5-flash"
    """Free-tier model names change. Check the current list at
    https://ai.google.dev/gemini-api/docs/models before assuming this one is
    still available."""

    timeout_s: float = 6.0
    """Kept short: the robot is standing there silent while this runs, and a
    local answer now beats a better answer in fifteen seconds."""

    cooldown_s: float = 60.0
    """How long to stop calling the API after a rate limit or server error."""

    temperature: float = 0.4
    max_output_tokens: int = 160

    max_vision_tokens: int = 320
    """A photo of a dense menu needs more room than a spoken sentence."""

    vision_timeout_s: float = 20.0
    """Uploading an image takes longer than sending a sentence."""

    max_guide_tokens: int = 1400
    """A whole travel guide truncated mid-object parses as nothing at all, so
    it gets a far larger budget than a single spoken sentence."""

    default_target_lang: str = "es"


@dataclass(frozen=True)
class CityConfig:
    database: Path = Path("world_cities.db")
    max_name_words: int = 4
    """Longest multi-word city name to try when scanning an utterance
    ("San Cristobal de las Casas" style names need a wide window)."""


@dataclass(frozen=True)
class TranslateConfig:
    source_lang: str = "en"
    target_lang: str = "es"


@dataclass(frozen=True)
class Config:
    base_dir: Path = Path.cwd()
    default_mode: str = "CITY"
    audio: AudioConfig = field(default_factory=AudioConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    drive: DriveConfig = field(default_factory=DriveConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    city: CityConfig = field(default_factory=CityConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    online: OnlineConfig = field(default_factory=OnlineConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    maps: MapConfig = field(default_factory=MapConfig)
    guides: GuideConfig = field(default_factory=GuideConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)

    def resolve_paths(self) -> "Config":
        """Make every model/database path absolute against ``base_dir``."""
        base = self.base_dir.expanduser().resolve()

        def under(path: Path) -> Path:
            path = path.expanduser()
            return path if path.is_absolute() else base / path

        voices = {
            code: replace(voice, model=under(voice.model))
            for code, voice in self.tts.voices.items()
        }
        return replace(
            self,
            base_dir=base,
            speech=replace(self.speech, model_path=under(self.speech.model_path)),
            tts=replace(self.tts, voices_dir=under(self.tts.voices_dir), voices=voices),
            city=replace(self.city, database=under(self.city.database)),
            maps=replace(self.maps, tile_cache=under(self.maps.tile_cache)),
            guides=replace(self.guides, database=under(self.guides.database)),
            camera=replace(self.camera, capture_dir=under(self.camera.capture_dir)),
        )


def _config_path() -> Path | None:
    env = os.environ.get("WALLE_CONFIG")
    if env:
        return Path(env)
    for candidate in DEFAULT_CONFIG_LOCATIONS:
        if candidate.is_file():
            return candidate
    return None


def _servos_from(raw: list[dict[str, Any]]) -> tuple[ServoConfig, ...]:
    return tuple(ServoConfig(**entry) for entry in raw)


def _voices_from(raw: dict[str, Any]) -> dict[str, VoiceConfig]:
    voices: dict[str, VoiceConfig] = {}
    for code, entry in raw.items():
        if isinstance(entry, str):
            voices[code] = VoiceConfig(Path(entry))
        else:
            voices[code] = VoiceConfig(
                model=Path(entry["model"]), speaker=entry.get("speaker")
            )
    return voices


def load_config(path: Path | None = None) -> Config:
    """Load configuration, falling back to defaults when no file is present."""
    path = path or _config_path()
    if path is None or not Path(path).is_file():
        return Config().resolve_paths()

    with open(path, "rb") as handle:
        raw = tomllib.load(handle)

    cfg = Config(
        base_dir=Path(raw.get("base_dir", Path(path).resolve().parent)),
        default_mode=str(raw.get("default_mode", "CITY")).upper(),
    )

    if section := raw.get("audio"):
        cfg = replace(cfg, audio=AudioConfig(**section))
    if section := raw.get("speech"):
        section = dict(section)
        if "model_path" in section:
            section["model_path"] = Path(section["model_path"])
        if "wake_words" in section:
            section["wake_words"] = tuple(section["wake_words"])
        cfg = replace(cfg, speech=SpeechConfig(**section))
    if section := raw.get("tts"):
        section = dict(section)
        if "voices_dir" in section:
            section["voices_dir"] = Path(section["voices_dir"])
        if "voices" in section:
            section["voices"] = _voices_from(section["voices"])
        cfg = replace(cfg, tts=TTSConfig(**section))
    if section := raw.get("motion"):
        section = dict(section)
        if "servos" in section:
            section["servos"] = _servos_from(section["servos"])
        cfg = replace(cfg, motion=MotionConfig(**section))
    if section := raw.get("drive"):
        cfg = replace(cfg, drive=DriveConfig(**section))
    if section := raw.get("network"):
        cfg = replace(cfg, network=NetworkConfig(**section))
    if section := raw.get("city"):
        section = dict(section)
        if "database" in section:
            section["database"] = Path(section["database"])
        cfg = replace(cfg, city=CityConfig(**section))
    if section := raw.get("translate"):
        cfg = replace(cfg, translate=TranslateConfig(**section))
    if section := raw.get("online"):
        cfg = replace(cfg, online=OnlineConfig(**section))
    if section := raw.get("display"):
        cfg = replace(cfg, display=DisplayConfig(**section))
    if section := raw.get("chat"):
        cfg = replace(cfg, chat=ChatConfig(**section))
    if section := raw.get("camera"):
        section = dict(section)
        if "capture_dir" in section:
            section["capture_dir"] = Path(section["capture_dir"])
        cfg = replace(cfg, camera=CameraConfig(**section))
    if section := raw.get("guides"):
        section = dict(section)
        if "database" in section:
            section["database"] = Path(section["database"])
        cfg = replace(cfg, guides=GuideConfig(**section))
    if section := raw.get("maps"):
        section = dict(section)
        if "tile_cache" in section:
            section["tile_cache"] = Path(section["tile_cache"])
        cfg = replace(cfg, maps=MapConfig(**section))

    return cfg.resolve_paths()
