"""Offline translation through Argos Translate.

The draft version swallowed every failure and returned the *input* string. The
caller then handed that untranslated English back to a Spanish Piper voice, so a
missing language pack produced confident-sounding gibberish rather than an
error. Failures now raise, and the assistant reports them in English.

The language pair is also resolved once and cached: ``get_installed_languages()``
walks the package directory on every call, which is wasted work on an SD card.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

log = logging.getLogger(__name__)


class TranslationUnavailable(RuntimeError):
    """Argos is not installed, or the requested language pair is missing."""


class ArgosTranslator:
    """Lazily-loaded wrapper around one or more Argos language pairs."""

    def __init__(self, source_lang: str = "en", target_lang: str = "es") -> None:
        self.source_lang = source_lang
        self.target_lang = target_lang
        self._pairs: dict[tuple[str, str], Any] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _load_module() -> Any:
        try:
            from argostranslate import translate  # noqa: PLC0415 - optional dep
        except ImportError as exc:  # pragma: no cover - depends on host install
            raise TranslationUnavailable(
                "argostranslate is not installed; run "
                "`pip install argostranslate` and install a language package"
            ) from exc
        return translate

    def _pair(self, from_code: str, to_code: str) -> Any:
        key = (from_code, to_code)
        with self._lock:
            if key in self._pairs:
                return self._pairs[key]

            translate = self._load_module()
            languages = translate.get_installed_languages()
            by_code = {lang.code: lang for lang in languages}

            source = by_code.get(from_code)
            target = by_code.get(to_code)
            if source is None or target is None:
                installed = ", ".join(sorted(by_code)) or "none"
                raise TranslationUnavailable(
                    f"no Argos package for {from_code}->{to_code}; "
                    f"installed languages: {installed}"
                )

            pair = source.get_translation(target)
            if pair is None:
                raise TranslationUnavailable(
                    f"Argos has both languages but no {from_code}->{to_code} path"
                )
            self._pairs[key] = pair
            return pair

    def translate(
        self, text: str, from_code: str | None = None, to_code: str | None = None
    ) -> str:
        """Translate ``text``, raising :class:`TranslationUnavailable` on failure."""
        text = text.strip()
        if not text:
            return ""

        from_code = from_code or self.source_lang
        to_code = to_code or self.target_lang
        if from_code == to_code:
            return text

        pair = self._pair(from_code, to_code)
        try:
            return pair.translate(text)
        except Exception as exc:  # noqa: BLE001 - Argos raises bare exceptions
            raise TranslationUnavailable(
                f"translation {from_code}->{to_code} failed: {exc}"
            ) from exc

    def warm_up(self) -> bool:
        """Pre-load the default pair at boot so the first request is not slow."""
        try:
            self._pair(self.source_lang, self.target_lang)
        except TranslationUnavailable as exc:
            log.warning("translator unavailable: %s", exc)
            return False
        return True
