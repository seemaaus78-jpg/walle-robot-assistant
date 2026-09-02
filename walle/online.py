"""Online answering through the Gemini API.

This is the "hybrid" half of the assistant. When the robot has Wi-Fi and an API
key, Gemini answers city questions and translations with far more range than a
200,000-row SQLite table and a single Argos language pair. When it does not -
no signal, no key, rate limited, request failed - every one of those paths
returns ``None`` and the local models answer instead.

Three deliberate constraints:

* **stdlib only.** A plain ``urllib`` POST costs nothing on a 1 GB board, where
  pulling in ``requests`` plus the Google SDK and its transitive dependencies
  costs tens of megabytes of RAM for one HTTP call.
* **Motion and shutdown never leave the robot.** Only questions are sent
  upstream. "wave", "nod" and "shut down" are handled locally whatever the
  network is doing, so the robot still obeys you when the API is down.
* **Failure is always downward.** Nothing in this module can raise into the
  main loop; the worst case is a slightly less interesting local answer.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Callable

from .assistant import Reply
from .config import OnlineConfig
from .intents import LANGUAGE_NAMES, Intent, IntentKind

log = logging.getLogger(__name__)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

# Answers are spoken aloud, so they must not contain markdown, lists, emoji or
# parentheticals that a text-to-speech engine will read out as punctuation.
SYSTEM_INSTRUCTION = (
    "You are the voice of a small desk robot travel assistant. "
    "Answer in at most two short sentences of plain spoken English. "
    "Never use markdown, bullet points, headings, emoji, or parentheses. "
    "Write numbers and units as a person would say them aloud. "
    "If you do not know something, say so in one short sentence."
)

TRANSLATE_INSTRUCTION = (
    "You are a translation engine. Reply with the translation only: "
    "no quotation marks, no transliteration, no explanation, no alternatives, "
    "and no note about what language it is."
)

# Characters that a speech synthesiser reads aloud or stumbles over.
_MARKDOWN = str.maketrans({"*": None, "_": None, "`": None, "#": None})


class Transport:
    """Seam for tests. The default implementation is a plain HTTPS POST."""

    def post(self, url: str, payload: dict, headers: dict, timeout: float) -> dict:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))


def clean_for_speech(text: str) -> str:
    """Strip formatting a synthesiser would read out or trip over."""
    cleaned = text.translate(_MARKDOWN)
    cleaned = " ".join(cleaned.split())
    return cleaned.strip().strip('"').strip()


class GeminiBackend:
    """An :class:`~walle.assistant.OnlineBackend` backed by the Gemini API.

    A free-tier key is rate limited per minute. Rather than retrying into the
    limit, a 429 puts the backend to sleep for ``cooldown_s`` and the robot
    answers locally in the meantime - which is indistinguishable to the user
    apart from a less detailed answer.
    """

    def __init__(
        self,
        config: OnlineConfig,
        api_key: str,
        transport: Transport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not api_key:
            raise ValueError("GeminiBackend requires an API key")
        self._config = config
        self._api_key = api_key
        self._transport = transport or Transport()
        self._clock = clock
        self._cooldown_until = 0.0

    # -- OnlineBackend -----------------------------------------------------

    def answer(self, intent: Intent) -> Reply | None:
        """Answer if this is a question and the API is usable, else None."""
        if self._cooling_down():
            return None

        match intent.kind:
            case IntentKind.CITY_QUERY:
                return self._answer_city(intent)
            case IntentKind.TRANSLATE_QUERY:
                return self._answer_translation(intent)
            case _:
                # Mode switches, gestures, status, help and shutdown are local
                # concerns. Sending them upstream would make the robot stop
                # obeying "shut down" whenever the network hiccuped.
                return None

    # -- intent handlers ---------------------------------------------------

    def _answer_city(self, intent: Intent) -> Reply | None:
        question = intent.text.strip()
        if not question:
            return None
        text = self._generate(
            SYSTEM_INSTRUCTION,
            f"A traveller asks about a place: {question}. "
            "Say where it is and one genuinely useful thing about visiting.",
        )
        if text is None:
            return None
        return Reply(text, gesture="nod")

    def _answer_translation(self, intent: Intent) -> Reply | None:
        body = intent.text.strip()
        if not body:
            return None
        target = intent.language or self._config.default_target_lang
        name = LANGUAGE_NAMES.get(target, target)
        text = self._generate(
            TRANSLATE_INSTRUCTION, f"Translate into {name}: {body}"
        )
        if text is None:
            return None
        return Reply(text, lang=target)

    # -- API plumbing ------------------------------------------------------

    def _cooling_down(self) -> bool:
        return self._clock() < self._cooldown_until

    def _start_cooldown(self, reason: str) -> None:
        self._cooldown_until = self._clock() + self._config.cooldown_s
        log.warning(
            "Gemini unavailable (%s); using local models for %.0f s",
            reason,
            self._config.cooldown_s,
        )

    def _generate(self, system: str, prompt: str) -> str | None:
        url = f"{API_ROOT}/{self._config.model}:generateContent"
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self._config.temperature,
                "maxOutputTokens": self._config.max_output_tokens,
            },
        }
        headers = {"x-goog-api-key": self._api_key}

        try:
            body = self._transport.post(
                url, payload, headers, self._config.timeout_s
            )
        except urllib.error.HTTPError as exc:
            # 429 is the free tier's per-minute limit; 5xx is Google's problem.
            # Both mean "stop asking for a while".
            if exc.code == 429 or exc.code >= 500:
                self._start_cooldown(f"HTTP {exc.code}")
            elif exc.code in (401, 403):
                self._start_cooldown(f"HTTP {exc.code}, check the API key")
            else:
                log.warning("Gemini request failed: HTTP %s", exc.code)
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            log.warning("Gemini request failed: %s", exc)
            return None
        except json.JSONDecodeError as exc:
            log.warning("Gemini returned invalid JSON: %s", exc)
            return None

        return self._extract(body)

    @staticmethod
    def _extract(body: dict) -> str | None:
        """Pull the text out of a generateContent response.

        A response can legitimately carry no text: the model may stop on a
        safety filter or hit the token limit mid-word. Both come back as a
        candidate with no usable parts, and both mean "let the local models
        answer" rather than "speak an empty sentence".
        """
        try:
            candidates = body["candidates"]
            if not candidates:
                log.debug("Gemini returned no candidates: %s", body)
                return None
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts)
        except (KeyError, TypeError, AttributeError, IndexError):
            log.warning("unexpected Gemini response shape: %s", body)
            return None

        cleaned = clean_for_speech(text)
        if not cleaned:
            reason = body["candidates"][0].get("finishReason", "unknown")
            log.info("Gemini produced no usable text (finishReason=%s)", reason)
            return None
        return cleaned


def build_online_backend(config: OnlineConfig, api_key: str | None):
    """Construct the configured backend, or None if it cannot be used.

    Returning None is the normal, non-exceptional outcome: no key configured is
    exactly the offline-only setup the robot is designed around.
    """
    if not config.enabled:
        log.info("online backend disabled in config")
        return None
    if not api_key:
        log.info(
            "no API key in $%s; running offline-only", config.api_key_env
        )
        return None
    if config.provider != "gemini":
        log.warning("unknown online provider %r; running offline-only", config.provider)
        return None

    log.info("online backend: Gemini (%s)", config.model)
    return GeminiBackend(config, api_key)
