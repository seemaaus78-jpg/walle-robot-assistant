"""The face: emotion and animation phase in, eye geometry out.

Deliberately contains no drawing code and no imaging dependency. Everything
here is arithmetic over dataclasses, which means the whole expression system -
what each emotion looks like, when it blinks, where it is looking - can be
tested on a machine with no display, no Pillow and no framebuffer.
``walle/display.py`` turns the geometry this produces into pixels.

The reference build's eyes change shape between frames: tall rounded squares at
rest, narrow horizontal slots at other moments. That is the behaviour modelled
here - eye *shape* carries the emotion, and blinking is a separate, continuous
animation layered on top.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

# Blink timing. Humans blink every 2-8 seconds; anything faster reads as
# twitchy on a face this simple.
BLINK_MIN_GAP_S = 2.4
BLINK_MAX_GAP_S = 6.5
BLINK_DURATION_S = 0.16

# How far the eyes can travel from centre when looking around, as a fraction of
# the gap between them. Beyond about a third it stops reading as a glance and
# starts reading as a rendering bug.
MAX_GAZE_FRACTION = 0.32


class Emotion(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    LISTENING = "listening"
    THINKING = "thinking"
    CONFUSED = "confused"
    SAD = "sad"
    SLEEPING = "sleeping"
    SPEAKING = "speaking"


@dataclass(frozen=True)
class EyeStyle:
    """How one emotion shapes the eyes.

    Fractions are of the display's short edge, so the same style renders
    correctly on a 240x240 panel and a 320x240 one.
    """

    width: float
    height: float
    radius: float
    """Corner radius as a fraction of the eye's shorter side. 0.5 is a
    stadium/oval, 0.0 is a hard rectangle."""

    tilt: float = 0.0
    """Degrees. Positive slants the inner corners down (angry/determined);
    negative slants them up (sad/pleading). Mirrored between the two eyes."""

    openness: float = 1.0
    """Baseline lid position before blinking is applied."""

    blinks: bool = True
    breathe: float = 0.0
    """Amplitude of a slow idle pulse, as a fraction of eye height."""


# Tuned so the emotions are distinguishable at a glance on a small panel, which
# means changing *shape*, not just colour - the two eyes are the only feature
# this face has.
STYLES: dict[Emotion, EyeStyle] = {
    #                       width  height  radius  tilt  open  blink  breathe
    Emotion.NEUTRAL:   EyeStyle(0.22, 0.30, 0.34, 0.0, 1.00, True, 0.02),
    Emotion.HAPPY:     EyeStyle(0.24, 0.22, 0.50, -6.0, 1.00, True, 0.04),
    Emotion.LISTENING: EyeStyle(0.24, 0.34, 0.34, 0.0, 1.00, True, 0.05),
    Emotion.THINKING:  EyeStyle(0.20, 0.16, 0.45, 4.0, 0.85, True, 0.00),
    Emotion.CONFUSED:  EyeStyle(0.22, 0.28, 0.36, 0.0, 0.95, True, 0.00),
    Emotion.SAD:       EyeStyle(0.21, 0.20, 0.40, -14.0, 0.70, True, 0.00),
    Emotion.SLEEPING:  EyeStyle(0.24, 0.06, 0.50, 0.0, 0.18, False, 0.03),
    Emotion.SPEAKING:  EyeStyle(0.22, 0.28, 0.36, 0.0, 1.00, True, 0.06),
}

# Cyan, like the reference build. Emotions that need to be legible from across
# a desk get their own colour; the rest share the default.
PALETTE: dict[Emotion, tuple[int, int, int]] = {
    Emotion.NEUTRAL: (62, 207, 207),
    Emotion.HAPPY: (86, 230, 200),
    Emotion.LISTENING: (86, 220, 255),
    Emotion.THINKING: (120, 170, 240),
    Emotion.CONFUSED: (240, 186, 90),
    Emotion.SAD: (90, 130, 210),
    Emotion.SLEEPING: (36, 96, 104),
    Emotion.SPEAKING: (62, 207, 207),
}

BACKGROUND = (8, 10, 14)


@dataclass(frozen=True)
class Eye:
    """One eye, in pixels, ready to draw."""

    cx: float
    cy: float
    width: float
    height: float
    radius: float
    tilt: float


@dataclass(frozen=True)
class FaceGeometry:
    left: Eye
    right: Eye
    colour: tuple[int, int, int]
    background: tuple[int, int, int]
    emotion: Emotion


def _hash01(index: int) -> float:
    """Deterministic pseudo-random value in [0, 1).

    A hash rather than ``random`` so the blink schedule is reproducible: the
    same time always yields the same frame, which is what makes the animation
    testable and what stops two processes disagreeing about the face.
    """
    value = math.sin(index * 12.9898 + 78.233) * 43758.5453
    return value - math.floor(value)


class BlinkClock:
    """Deterministic blink schedule.

    Blink times are generated lazily and cached, so asking for the openness at
    t=900 does not walk 900 seconds of history more than once.
    """

    def __init__(self) -> None:
        self._times: list[float] = []
        self._next_index = 0
        self._cursor = 0.0

    def _extend_to(self, t: float) -> None:
        while self._cursor <= t + BLINK_MAX_GAP_S:
            gap = BLINK_MIN_GAP_S + _hash01(self._next_index) * (
                BLINK_MAX_GAP_S - BLINK_MIN_GAP_S
            )
            self._cursor += gap
            self._times.append(self._cursor)
            self._next_index += 1

    def openness(self, t: float) -> float:
        """1.0 fully open, 0.0 fully closed, at time ``t`` seconds."""
        if t < 0:
            return 1.0
        self._extend_to(t)
        for start in self._times:
            if start > t:
                break
            if start <= t < start + BLINK_DURATION_S:
                # Open at both ends of the window, fully shut in the middle.
                # The cosine also makes the value continuous with the 1.0
                # returned outside the window, so there is no snap on either
                # edge - a linear ramp both reads as mechanical and shows a
                # visible step at the boundary.
                phase = (t - start) / BLINK_DURATION_S
                return (1.0 + math.cos(2.0 * math.pi * phase)) / 2.0
        return 1.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_face(
    emotion: Emotion,
    t: float,
    size: tuple[int, int],
    gaze: tuple[float, float] = (0.0, 0.0),
    blink: BlinkClock | None = None,
) -> FaceGeometry:
    """Compute both eyes for one frame.

    ``gaze`` is (dx, dy) in [-1, 1], where (0, 0) looks straight ahead and
    (-1, 0) looks fully left. The assistant uses it to turn the eyes toward
    whoever is speaking.
    """
    width, height = size
    short_edge = min(width, height)
    style = STYLES.get(emotion, STYLES[Emotion.NEUTRAL])

    eye_w = style.width * short_edge
    eye_h = style.height * short_edge

    # Blinking scales height only, so the eye squashes rather than shrinking.
    openness = style.openness
    if style.blinks and blink is not None:
        openness *= blink.openness(t)
    # A sliver is kept so a closed eye is still a visible line, not a gap.
    eye_h = max(eye_h * openness, short_edge * 0.012)

    if style.breathe:
        eye_h *= 1.0 + style.breathe * math.sin(t * 1.9)

    separation = width * 0.24
    gaze_x = clamp(gaze[0], -1.0, 1.0) * separation * MAX_GAZE_FRACTION
    gaze_y = clamp(gaze[1], -1.0, 1.0) * height * 0.09

    cy = height * 0.5 + gaze_y
    left_cx = width * 0.5 - separation * 0.5 + gaze_x
    right_cx = width * 0.5 + separation * 0.5 + gaze_x

    radius = style.radius * min(eye_w, eye_h)

    return FaceGeometry(
        left=Eye(left_cx, cy, eye_w, eye_h, radius, style.tilt),
        # Mirrored so a slant is symmetric about the centre line rather than
        # both eyes leaning the same way, which reads as a broken render.
        right=Eye(right_cx, cy, eye_w, eye_h, radius, -style.tilt),
        colour=PALETTE.get(emotion, PALETTE[Emotion.NEUTRAL]),
        background=BACKGROUND,
        emotion=emotion,
    )
