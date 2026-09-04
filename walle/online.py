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

import base64
import json
import logging
import time
import urllib.error
import urllib.request
from typing import Callable

from .assistant import Reply
from .chat import ChatHistory
from .config import OnlineConfig
from .guides import SECTIONS
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

CHAT_INSTRUCTION = (
    "You are the voice of a small desk robot called Wall E, a travel "
    "assistant with a dry, warm manner. Reply in at most two short sentences "
    "of plain spoken English. Never use markdown, bullet points, emoji or "
    "parentheses. Do not offer lists of options. If you do not know something, "
    "say so plainly."
)

VISION_INSTRUCTION = (
    "You are the eyes of a small desk robot. You are shown one photograph "
    "taken by the robot's camera. Answer in at most two short sentences of "
    "plain spoken English. Never use markdown, emoji or parentheses. "
    "Describe only what is actually visible. If the photo is too dark, too "
    "blurred or too close to tell, say exactly that instead of guessing. "
    "Never guess at a person's name, age, mood, health or nationality."
)

READ_INSTRUCTION = (
    "You are shown a photograph. Read the text that appears in it and reply "
    "with that text only. Do not describe the image, do not explain, do not "
    "add quotation marks. If there is no readable text, reply exactly: "
    "I cannot see any text."
)

GUIDE_INSTRUCTION = (
    "You write short, practical travel notes for a traveller who will read "
    "them with no internet connection. Be concrete and name real places. "
    "Never use markdown, bullet characters, emoji or parentheses - every "
    "string you write will be read aloud by a speech synthesiser. "
    "If you are unsure about a fact, leave it out rather than guessing. "
    "Return JSON only, matching the requested shape exactly."
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

    def answer(
        self, intent: Intent, history: ChatHistory | None = None
    ) -> Reply | None:
        """Answer if this is a question and the API is usable, else None."""
        if self._cooling_down():
            return None

        match intent.kind:
            case IntentKind.CITY_QUERY:
                return self._answer_city(intent)
            case IntentKind.TRANSLATE_QUERY:
                return self._answer_translation(intent)
            case IntentKind.CHAT_QUERY:
                return self._answer_chat(intent, history)
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

    def _answer_chat(
        self, intent: Intent, history: ChatHistory | None
    ) -> Reply | None:
        """Open-ended conversation, with the last few turns for context."""
        body = intent.text.strip()
        if not body:
            return None
        text = self._generate(
            CHAT_INSTRUCTION, body, history=history
        )
        if text is None:
            return None
        return Reply(text)

    def look(
        self,
        image: bytes,
        task: str = "describe",
        question: str | None = None,
        target_language: str | None = None,
    ) -> str | None:
        """Send one photograph and get a spoken answer back.

        ``task`` is "describe" (what is in front of me), "read" (the words in
        this photo) or "translate" (read the words, then say them in another
        language) - which together are most of what a traveller actually points
        a camera at: a view, a menu, a sign in a script they cannot read.
        """
        if self._cooling_down() or not image:
            return None

        if task == "read":
            system, prompt = READ_INSTRUCTION, "Read the text in this photo."
        elif task == "translate":
            name = LANGUAGE_NAMES.get(target_language or "", target_language or "English")
            system = READ_INSTRUCTION
            prompt = (
                f"Read the text in this photo and give only its {name} "
                "translation. Nothing else."
            )
        else:
            system = VISION_INSTRUCTION
            prompt = question or "What is in front of you?"

        return self._generate(system, prompt, image=image)

    def fetch_guide(self, city_name: str, country: str | None = None) -> dict | None:
        """Ask for a whole travel guide, as sections, ready to store.

        Requested as JSON rather than prose so each part can be saved and read
        back on its own later - "what about restaurants in Kyoto" should not
        have to replay the entire guide.
        """
        if self._cooling_down():
            return None

        where = f"{city_name}, {country}" if country else city_name
        wanted = "\n".join(
            f'  "{section.key}": {section.prompt}' for section in SECTIONS
        )
        prompt = (
            f"Write travel notes for {where}.\n"
            f"Return a JSON object with exactly these keys:\n{wanted}\n"
            'Use a plain string for "summary", and an array of short strings '
            "for every other key. Six items maximum per array."
        )

        body = self._generate_json(GUIDE_INSTRUCTION, prompt)
        if not isinstance(body, dict):
            return None

        # Keep only the sections we asked for, and only ones with content, so a
        # partial answer stores cleanly rather than saving empty keys.
        sections: dict[str, object] = {}
        for section in SECTIONS:
            value = body.get(section.key)
            if isinstance(value, str) and value.strip():
                sections[section.key] = clean_for_speech(value)
            elif isinstance(value, list):
                items = [
                    clean_for_speech(str(item))
                    for item in value
                    if str(item).strip()
                ]
                if items:
                    sections[section.key] = items[:6]
        return sections or None

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

    def _generate(
        self,
        system: str,
        prompt: str,
        history: ChatHistory | None = None,
        json_mode: bool = False,
        image: bytes | None = None,
    ) -> str | None:
        url = f"{API_ROOT}/{self._config.model}:generateContent"

        contents: list[dict] = []
        if history is not None:
            # The current utterance is already the last user turn in history,
            # so it is dropped here and re-added as the prompt to avoid saying
            # it twice.
            turns = history.turns()
            if turns and turns[-1].role == "user":
                turns = turns[:-1]
            contents = [
                {"role": turn.role, "parts": [{"text": turn.text}]} for turn in turns
            ]
        parts: list[dict] = []
        if image is not None:
            # The picture goes before the question: the model reads the parts
            # in order, and asking about an image it has not been shown yet
            # gives noticeably vaguer answers.
            parts.append(
                {
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": base64.b64encode(image).decode("ascii"),
                    }
                }
            )
        parts.append({"text": prompt})
        contents.append({"role": "user", "parts": parts})

        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {
                "temperature": self._config.temperature,
                "maxOutputTokens": (
                    self._config.max_guide_tokens
                    if json_mode
                    else self._config.max_output_tokens
                ),
                # A photo of a dense menu needs room; a spoken sentence does not.
                **({"maxOutputTokens": self._config.max_vision_tokens}
                   if image is not None and not json_mode else {}),
            },
        }
        if json_mode:
            # A whole guide truncated mid-object parses as nothing at all, so
            # it gets a much larger budget than a spoken sentence needs.
            payload["generationConfig"]["responseMimeType"] = "application/json"
        headers = {"x-goog-api-key": self._api_key}

        try:
            timeout = (
                self._config.vision_timeout_s
                if image is not None
                else self._config.timeout_s
            )
            body = self._transport.post(url, payload, headers, timeout)
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

        return self._extract(body, raw=json_mode)

    def _generate_json(self, system: str, prompt: str) -> object | None:
        """Same call, but asking the API for JSON and parsing it."""
        raw = self._generate(system, prompt, json_mode=True)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Models occasionally wrap JSON in a code fence even when asked not
            # to. Salvage the outermost object rather than losing the whole
            # answer over punctuation.
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    pass
            log.warning("guide response was not valid JSON")
            return None

    @staticmethod
    def _extract(body: dict, raw: bool = False) -> str | None:
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

        # JSON must survive byte-for-byte: the speech cleaner strips
        # underscores and asterisks, which are legal inside JSON values.
        cleaned = text.strip() if raw else clean_for_speech(text)
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
