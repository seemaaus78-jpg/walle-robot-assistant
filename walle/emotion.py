"""Emotional state: what the robot is feeling, and for how long.

The assistant reports *events* ("I heard something", "the lookup failed"), not
emotions. This module decides what those events look like on the face, how long
each expression holds, and when the robot drifts back to resting or falls
asleep. Keeping that policy in one place means the expression can be retuned
without touching the speech or lookup code.

Like ``walle/face.py`` this is pure arithmetic over an injected clock, so the
whole behaviour is testable with no display attached.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .face import Emotion


class Event(str, Enum):
    """Things the assistant tells the face about."""

    WAKE = "wake"
    HEARD = "heard"
    THINKING = "thinking"
    ANSWERED = "answered"
    NOT_UNDERSTOOD = "not_understood"
    FAILED = "failed"
    SPEAKING_STARTED = "speaking_started"
    SPEAKING_FINISHED = "speaking_finished"
    SLEEP = "sleep"


@dataclass(frozen=True)
class Reaction:
    """How one event is expressed."""

    emotion: Emotion
    hold_s: float
    """0 means "hold until something else happens" - used for states that end
    on their own event, like listening and speaking."""


REACTIONS: dict[Event, Reaction] = {
    Event.WAKE: Reaction(Emotion.HAPPY, 2.0),
    Event.HEARD: Reaction(Emotion.LISTENING, 0.0),
    Event.THINKING: Reaction(Emotion.THINKING, 0.0),
    Event.ANSWERED: Reaction(Emotion.HAPPY, 2.5),
    Event.NOT_UNDERSTOOD: Reaction(Emotion.CONFUSED, 2.5),
    Event.FAILED: Reaction(Emotion.SAD, 3.0),
    Event.SPEAKING_STARTED: Reaction(Emotion.SPEAKING, 0.0),
    Event.SPEAKING_FINISHED: Reaction(Emotion.NEUTRAL, 0.0),
    Event.SLEEP: Reaction(Emotion.SLEEPING, 0.0),
}

RESTING = Emotion.NEUTRAL


class EmotionEngine:
    """Tracks the current expression and where the eyes are looking."""

    def __init__(
        self,
        clock: Callable[[], float],
        sleep_after_s: float = 120.0,
    ) -> None:
        self._clock = clock
        self._sleep_after_s = sleep_after_s
        self._emotion = RESTING
        self._since = clock()
        self._hold_s = 0.0
        self._last_activity = clock()
        self._gaze = (0.0, 0.0)

    # -- input --------------------------------------------------------------

    def on_event(self, event: Event) -> Emotion:
        """Apply an event and return the resulting emotion."""
        reaction = REACTIONS.get(event)
        if reaction is None:
            return self._emotion

        now = self._clock()
        self._emotion = reaction.emotion
        self._hold_s = reaction.hold_s
        self._since = now
        if event is not Event.SLEEP:
            self._last_activity = now
        return self._emotion

    def look_at(self, dx: float, dy: float = 0.0) -> None:
        """Point the eyes somewhere until the next idle wander."""
        self._gaze = (max(-1.0, min(1.0, dx)), max(-1.0, min(1.0, dy)))

    # -- output -------------------------------------------------------------

    def current(self) -> Emotion:
        """The emotion to render right now, applying decay and sleep."""
        now = self._clock()

        # A long silence puts the robot to sleep. This is the single biggest
        # thing that makes it read as alive rather than as a frozen prop, and
        # it also tells you at a glance that it stopped listening.
        if now - self._last_activity >= self._sleep_after_s:
            self._emotion = Emotion.SLEEPING
            return self._emotion

        # Transient expressions decay back to resting; hold_s == 0 means the
        # state ends on its own event instead.
        if self._hold_s > 0 and (now - self._since) >= self._hold_s:
            self._emotion = RESTING
            self._hold_s = 0.0

        return self._emotion

    def gaze(self) -> tuple[float, float]:
        """Where the eyes are pointing, as (dx, dy) in [-1, 1].

        While idle the gaze wanders slowly. Anything else looks straight at
        whoever it is dealing with, which is what makes eye contact read as
        deliberate when it happens.
        """
        emotion = self.current()
        if emotion in (Emotion.LISTENING, Emotion.SPEAKING, Emotion.THINKING):
            return self._gaze

        if emotion is Emotion.SLEEPING:
            return (0.0, 0.35)

        # Two sine waves at unrelated frequencies, so the drift never visibly
        # repeats.
        t = self._clock()
        return (0.42 * math.sin(t * 0.23), 0.22 * math.sin(t * 0.17 + 1.1))

    @property
    def is_asleep(self) -> bool:
        return self.current() is Emotion.SLEEPING
