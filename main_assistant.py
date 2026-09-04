#!/usr/bin/env python3
"""Entry point for the WALL-E Hybrid Robot Assistant.

On the robot::

    python3 main_assistant.py

On any machine, with no microphone, speaker, servos or models present::

    python3 main_assistant.py --text "tell me about tokyo" "switch to translator"

The dry-run mode exercises the full intent-routing and response path, which is
the fastest way to check a change before copying it to the SD card.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

from walle.assistant import Assistant
from walle.chat import OfflineChat
from walle.cities import CityDatabase
from walle.config import Config, load_config
from walle.display import build_display
from walle.guides import GuideStore
from walle.maps import MapRenderer, TileSource
from walle.motion import ServoBank
from walle.net import ConnectivityMonitor
from walle.online import build_online_backend
from walle.stt import ScriptedRecogniser, SpeechRecogniser
from walle.translation import ArgosTranslator
from walle.tts import NullSpeaker, PiperSpeaker

log = logging.getLogger("walle")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=None, help="path to a TOML config file"
    )
    parser.add_argument(
        "--text",
        nargs="*",
        metavar="UTTERANCE",
        help="skip the microphone and process these phrases instead; with no "
        "arguments, read one phrase per line from stdin",
    )
    parser.add_argument(
        "--speak",
        action="store_true",
        help="use real audio output in --text mode (default: print only)",
    )
    parser.add_argument(
        "--no-motion", action="store_true", help="do not drive the servos"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="never call the cloud, even with an API key and a connection",
    )
    parser.add_argument(
        "--no-display", action="store_true", help="do not drive the face panel"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="log at DEBUG level"
    )
    return parser


def open_cities(config: Config) -> CityDatabase | None:
    """The robot stays useful without the database, so this never raises."""
    try:
        return CityDatabase(config.city.database)
    except Exception as exc:  # noqa: BLE001 - degrade instead of refusing to boot
        log.warning(
            "city lookups disabled (%s). Build the database with "
            "scripts/build_city_db.py.",
            exc,
        )
        return None


def open_motion(config: Config, disabled: bool) -> ServoBank | None:
    if disabled or not config.motion.enabled:
        log.info("motion disabled")
        return None
    try:
        return ServoBank(config.motion)
    except Exception as exc:  # noqa: BLE001 - gestures are not load-bearing
        log.warning("motion disabled (%s)", exc)
        return None


def open_guides(config: Config) -> GuideStore | None:
    """The saved-guide store. Missing it costs the cache, not the robot."""
    if not config.guides.enabled:
        log.info("travel guides disabled in config")
        return None
    try:
        store = GuideStore(config.guides.database, config.guides.max_guides)
    except Exception as exc:  # noqa: BLE001 - degrade rather than refuse to boot
        log.warning("saved guides unavailable (%s)", exc)
        return None
    log.info("saved guides: %d in %s", store.count(), config.guides.database)
    return store


def open_maps(config: Config, offline: bool) -> MapRenderer:
    """Tiles come from the cache first, so a downloaded pack works offline."""
    source = TileSource(
        cache_dir=config.maps.tile_cache,
        user_agent=config.maps.user_agent or None,
        url_template=config.maps.url_template,
        timeout_s=config.maps.timeout_s,
        offline=offline or not config.maps.enabled,
    )
    if not config.maps.user_agent:
        source.user_agent = TileSource().user_agent
    return MapRenderer(source)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    log.debug("configuration: %s", config)

    dry_run = args.text is not None
    if dry_run:
        lines = args.text or [line.strip() for line in sys.stdin if line.strip()]
        recogniser = ScriptedRecogniser(lines)
        speaker = PiperSpeaker(config.tts, config.audio.playback_device) if args.speak else NullSpeaker()
    else:
        speaker = PiperSpeaker(config.tts, config.audio.playback_device)
        try:
            recogniser = SpeechRecogniser(
                config.speech, config.audio, mute=lambda: speaker.is_speaking
            )
        except Exception as exc:  # noqa: BLE001 - no microphone means no robot
            log.error("cannot start speech recognition: %s", exc)
            return 1

    cities = open_cities(config)
    display = build_display(
        enabled=config.display.enabled and not args.no_display,
        device=config.display.device,
        swap_bytes=config.display.swap_bytes,
        font_path=config.display.font_path,
    )

    assistant = Assistant(
        config=config,
        speaker=speaker,
        recogniser=recogniser,
        motion=open_motion(config, args.no_motion or dry_run),
        cities=cities,
        display=display,
        chat=OfflineChat(cities),
        maps=open_maps(config, offline=args.offline),
        guides=open_guides(config),
        translator=ArgosTranslator(
            config.translate.source_lang, config.translate.target_lang
        ),
        connectivity=ConnectivityMonitor(config.network),
        online=None
        if args.offline
        else build_online_backend(
            config.online, os.environ.get(config.online.api_key_env)
        ),
    )

    def handle_signal(signum: int, _frame: object) -> None:
        log.info("received signal %d; shutting down", signum)
        assistant.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_signal)

    try:
        assistant.run()
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        assistant.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
