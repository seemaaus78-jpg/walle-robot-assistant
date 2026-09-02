"""Open-ended conversation, on both sides of the connectivity divide.

Online this is Gemini and it is genuinely open-ended. Offline it is not, and
this module does not pretend otherwise.

**What offline chat actually is.** A pattern-matched responder with a fixed
repertoire, plus the city database for small talk with real content in it. It
handles greetings, identity, thanks and a few conversational turns, and for
anything outside that it says plainly that it needs a connection for a real
conversation. It is not a language model and does not imitate one.

**Why not a local model.** A 0.5-1B parameter model through llama.cpp needs
roughly 400-700 MB resident at 4-bit quantisation. The board is already carrying
Vosk, Argos and Piper at around 620 MB of 1 GB. There is no room, and a robot
that starts swapping mid-sentence on an SD card is worse than one that says "ask
me when I have signal".

:class:`LlamaCppChat` is the seam for anyone running the 2 GB board who wants to
try anyway - it shells out to a llama.cpp binary and is off by default.
"""

from __future__ import annotations

import logging
import re
import subprocess
from collections import deque
from dataclasses import dataclass
from typing import Protocol

from .cities import normalise

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Turn:
    role: str
    text: str


class ChatHistory:
    """A bounded conversation buffer.

    Bounded because an assistant that runs for days would otherwise grow an
    unbounded prompt, and because the free tier charges by token.
    """

    def __init__(self, max_turns: int = 8) -> None:
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    def add_user(self, text: str) -> None:
        self._turns.append(Turn("user", text))

    def add_robot(self, text: str) -> None:
        self._turns.append(Turn("model", text))

    def turns(self) -> list[Turn]:
        return list(self._turns)

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)


class ChatResponder(Protocol):
    def reply(self, text: str, history: ChatHistory) -> str | None: ...


# Ordered: the first pattern that matches wins, so specific phrases come before
# general ones.
RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        r"\b(who are you|what are you|your name|whats your name)\b",
        (
            "I am Wall E, a small travel assistant. I know cities, and I can translate.",
            "Wall E. I sit on your desk and answer questions about places.",
        ),
    ),
    (
        r"\b(are you (alive|real|a robot|human))\b",
        (
            "I am a robot. I have four servos and strong opinions about cities.",
            "Not alive, but I am paying attention.",
        ),
    ),
    (
        r"\b(how are you|how do you feel|you okay|are you okay)\b",
        (
            "All systems nominal. Ask me about somewhere.",
            "Good. Warm, listening, and slightly curious.",
        ),
    ),
    (
        r"\b(hello|hi|hey|good morning|good afternoon|good evening|greetings)\b",
        (
            "Hello. Where shall we go?",
            "Hi. Name a city and I will tell you about it.",
            "Hey. I am listening.",
        ),
    ),
    (
        r"\b(thank you|thanks|cheers|appreciate it)\b",
        ("Any time.", "You are welcome.", "Happy to help."),
    ),
    (
        r"\b(good ?bye|bye|see you|good night)\b",
        ("See you.", "Goodbye. Wake me whenever."),
    ),
    (
        r"\b(tell me a joke|make me laugh|say something funny)\b",
        (
            "I know a hundred thousand cities and I have visited none of them.",
            "My sense of direction is excellent. My sense of movement is four small servos.",
        ),
    ),
    (
        r"\b(i am (bored|lonely)|im bored|entertain me)\b",
        ("Ask me about a city you have never heard of. I will find one.",),
    ),
    (
        r"\b(i (love|like) you|good robot|nice robot)\b",
        ("That is kind. I will remember it until the next power cycle.",),
    ),
    (
        r"\b(what time|what is the date|what day)\b",
        ("I do not have a clock I trust. Check your phone.",),
    ),
    (
        r"\b(sorry|my mistake)\b",
        ("No harm done.",),
    ),
)

COMPILED: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = tuple(
    (re.compile(pattern), replies) for pattern, replies in RULES
)

NO_CONNECTION = (
    "I need a connection for a real conversation. Offline I can still tell you "
    "about cities and translate for you."
)


class OfflineChat:
    """Pattern-matched small talk with an honest floor.

    Replies rotate rather than being chosen at random, so the same input in the
    same session gives a predictable sequence - easier to test, and it stops the
    robot repeating itself twice in a row.
    """

    def __init__(self, cities=None) -> None:
        self._cities = cities
        self._counters: dict[int, int] = {}

    def reply(self, text: str, history: ChatHistory | None = None) -> str:
        cleaned = normalise(text)
        if not cleaned:
            return "I did not catch that."

        for index, (pattern, replies) in enumerate(COMPILED):
            if pattern.search(cleaned):
                turn = self._counters.get(index, 0)
                self._counters[index] = turn + 1
                return replies[turn % len(replies)]

        if suggestion := self._city_smalltalk(cleaned):
            return suggestion

        return NO_CONNECTION

    def _city_smalltalk(self, text: str) -> str | None:
        """Use the city database rather than inventing filler.

        Offline the robot has one real body of knowledge. When someone asks for
        something interesting, it should reach for that instead of a canned
        platitude.
        """
        if self._cities is None:
            return None
        if not re.search(r"\b(something interesting|a fact|surprise me|random)\b", text):
            return None
        city = getattr(self._cities, "random_city", lambda: None)()
        if city is None:
            return None
        return f"Here is one. {city.summary()}"


class LlamaCppChat:
    """Optional local language model through a llama.cpp binary.

    Off by default and not recommended on the 1 GB board - see this module's
    docstring for the arithmetic. Provided so the seam exists rather than
    because it is a good idea at this memory budget.
    """

    def __init__(
        self,
        binary: str,
        model_path: str,
        timeout_s: float = 25.0,
        max_tokens: int = 80,
    ) -> None:
        self._binary = binary
        self._model = model_path
        self._timeout_s = timeout_s
        self._max_tokens = max_tokens

    def reply(self, text: str, history: ChatHistory | None = None) -> str | None:
        prompt = (
            "You are a small desk robot. Answer in one or two short spoken "
            f"sentences, no markdown.\nUser: {text}\nRobot:"
        )
        command = [
            self._binary,
            "-m", self._model,
            "-p", prompt,
            "-n", str(self._max_tokens),
            "--temp", "0.6",
            "--log-disable",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
                check=False,
            )
        except FileNotFoundError:
            log.error("llama.cpp binary not found: %s", self._binary)
            return None
        except subprocess.TimeoutExpired:
            log.warning("local model timed out after %.0f s", self._timeout_s)
            return None

        if result.returncode != 0:
            log.warning("local model exited %s: %s", result.returncode, result.stderr[:200])
            return None

        answer = result.stdout.replace(prompt, "").strip()
        return answer.split("\n")[0].strip() or None
