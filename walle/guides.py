"""Travel guides cached on the SD card.

When the robot has a connection it can ask Gemini for a full guide to a city -
where to eat, how to get around, which areas are worth walking, where the
hospital is - and keep the answer. Later, with no connection at all, that guide
is still there.

This is what makes the online half worth having: every question asked with
signal permanently improves what the robot knows without it.

Three design decisions worth stating:

**Guides live in their own database file.** ``travel_guides.db`` is separate
from ``world_cities.db``. The cities database is reference data that ships with
the robot and never changes; guides are the user's own collected data. Keeping
them apart means "delete everything I have saved" is one file, and can never
damage the thing the robot needs to work.

**Every guide records when it was fetched.** A restaurant list is not a fact,
it is a snapshot. The robot says how old a guide is when it reads one back, so
nobody acts on a four-month-old opening time believing it is current.

**Emergency information is treated differently.** A language model can invent a
plausible hospital address as easily as a real one, and acting on a wrong one
matters in a way that a wrong restaurant does not. That section is always read
with a caution attached - see :data:`EMERGENCY_CAUTION`.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .cities import normalise

log = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS guides (
    city_key    TEXT PRIMARY KEY,
    city_name   TEXT NOT NULL,
    country     TEXT,
    latitude    REAL,
    longitude   REAL,
    fetched_at  TEXT NOT NULL,
    source      TEXT NOT NULL,
    model       TEXT,
    sections    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_guides_fetched ON guides(fetched_at);
"""


@dataclass(frozen=True)
class Section:
    """One part of a guide, and how the robot refers to it out loud."""

    key: str
    label: str
    prompt: str
    """What to ask the model for this section."""

    aliases: tuple[str, ...] = ()
    """Spoken words that select this section."""


# The order here is the order the robot reads a full guide in: what the place is,
# then what to do, then how to move, then where to sleep, then safety.
SECTIONS: tuple[Section, ...] = (
    Section(
        "summary", "overview",
        "two sentences on what this city is like to visit",
        # Deliberately narrow. "about" would match "tell me about Tokyo" - the
        # commonest city question there is - and steal it from the plain city
        # answer, dropping the country and population entirely.
        ("overview", "summary", "in general"),
    ),
    Section(
        "areas", "areas to visit",
        "five neighbourhoods or areas worth walking around, each with a few "
        "words on what it is known for",
        ("areas", "neighbourhoods", "districts", "places to visit", "sights",
         "attractions", "what to see"),
    ),
    Section(
        "food", "food and restaurants",
        "five well-known places or streets to eat, mixing famous and everyday, "
        "each with a few words on what it serves",
        ("food", "restaurants", "restaurant", "eat", "eating", "dinner",
         "lunch", "breakfast", "cafes", "cafe", "street food"),
    ),
    Section(
        "transport", "getting around",
        "the main stations and airports, and how people usually get around "
        "the city",
        ("transport", "trains", "train", "stations", "station", "metro",
         "buses", "bus", "roads", "getting around", "get around", "airport",
         "taxi", "subway"),
    ),
    Section(
        "stay", "where to stay",
        "which areas are good to stay in and roughly what they cost",
        ("stay", "hotel", "hotels", "accommodation", "sleep", "hostel",
         "hostels", "where to stay"),
    ),
    Section(
        "emergency", "emergency and health",
        "the emergency telephone number for this country, and the names of the "
        "main public hospitals and central police stations",
        # "police station" is listed explicitly: longest-alias-wins would
        # otherwise hand it to transport, which owns the word "station".
        ("emergency", "hospital", "hospitals", "police", "police station",
         "fire station", "doctor", "pharmacy", "chemist", "ambulance",
         "safety", "emergency number"),
    ),
)

SECTION_BY_KEY = {section.key: section for section in SECTIONS}

EMERGENCY_CAUTION = (
    "Please check this against a local source before you rely on it. "
    "I got it from an AI, not from an official record."
)

STALE_AFTER_DAYS = 90
"""Past this, the robot warns that a guide is old when reading it back."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Guide:
    """One city's cached travel guide."""

    city_name: str
    country: str | None
    fetched_at: str
    source: str
    sections: dict[str, object] = field(default_factory=dict)
    model: str | None = None
    latitude: float | None = None
    longitude: float | None = None

    @property
    def age_days(self) -> int:
        try:
            fetched = datetime.fromisoformat(self.fetched_at)
        except ValueError:
            return 0
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        return max(0, (datetime.now(timezone.utc) - fetched).days)

    @property
    def is_stale(self) -> bool:
        return self.age_days >= STALE_AFTER_DAYS

    def age_phrase(self) -> str:
        """How old this guide is, in words a person would use."""
        days = self.age_days
        if days == 0:
            return "saved today"
        if days == 1:
            return "saved yesterday"
        if days < 14:
            return f"saved {days} days ago"
        if days < 60:
            return f"saved {days // 7} weeks ago"
        return f"saved {days // 30} months ago"

    def has(self, key: str) -> bool:
        return bool(self.sections.get(key))

    def section_text(self, key: str) -> str | None:
        """One section, flattened into something speakable."""
        value = self.sections.get(key)
        if not value:
            return None
        if isinstance(value, str):
            text = value.strip()
        elif isinstance(value, (list, tuple)):
            parts = [str(item).strip() for item in value if str(item).strip()]
            text = ". ".join(parts)
        else:
            text = str(value).strip()
        if not text:
            return None
        # End on punctuation so whatever follows does not run into it - a
        # synthesiser reads "for mochi This guide" as one breathless phrase.
        if text[-1] not in ".!?":
            text += "."
        if key == "emergency":
            # Never read this one without the caution attached.
            return f"{text} {EMERGENCY_CAUTION}"
        return text

    def spoken_summary(self) -> str:
        """A short answer covering the whole guide."""
        opening = self.section_text("summary") or f"I have a guide for {self.city_name}."
        available = [
            SECTION_BY_KEY[key].label
            for key in SECTION_BY_KEY
            if key != "summary" and self.has(key)
        ]
        if not available:
            return opening
        listed = ", ".join(available[:-1]) + f" and {available[-1]}" if len(available) > 1 else available[0]
        return f"{opening} I can also tell you about {listed}."


class GuideStore:
    """Reads and writes cached guides. Creates its database on first use."""

    def __init__(self, path: Path, max_guides: int = 200) -> None:
        self.path = Path(path)
        self.max_guides = max_guides
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    # -- reading ------------------------------------------------------------

    def get(self, city_name: str) -> Guide | None:
        row = self._conn.execute(
            "SELECT * FROM guides WHERE city_key = ?;", (normalise(city_name),)
        ).fetchone()
        return self._to_guide(row) if row else None

    def list_cities(self) -> list[tuple[str, str]]:
        """Saved cities, newest first, as (name, age phrase)."""
        rows = self._conn.execute(
            "SELECT city_name, fetched_at FROM guides ORDER BY fetched_at DESC;"
        ).fetchall()
        return [
            (
                row["city_name"],
                Guide(row["city_name"], None, row["fetched_at"], "").age_phrase(),
            )
            for row in rows
        ]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM guides;").fetchone()[0]

    @staticmethod
    def _to_guide(row: sqlite3.Row) -> Guide:
        try:
            sections = json.loads(row["sections"])
        except (ValueError, TypeError):
            log.warning("guide for %s has unreadable sections", row["city_name"])
            sections = {}
        return Guide(
            city_name=row["city_name"],
            country=row["country"],
            fetched_at=row["fetched_at"],
            source=row["source"],
            sections=sections if isinstance(sections, dict) else {},
            model=row["model"],
            latitude=row["latitude"],
            longitude=row["longitude"],
        )

    # -- writing ------------------------------------------------------------

    def save(
        self,
        city_name: str,
        sections: dict[str, object],
        country: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        source: str = "gemini",
        model: str | None = None,
    ) -> Guide:
        """Store a guide, replacing any earlier one for the same city."""
        guide = Guide(
            city_name=city_name,
            country=country,
            fetched_at=_now(),
            source=source,
            sections=sections,
            model=model,
            latitude=latitude,
            longitude=longitude,
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO guides "
            "(city_key, city_name, country, latitude, longitude, fetched_at, "
            " source, model, sections) VALUES (?,?,?,?,?,?,?,?,?);",
            (
                normalise(city_name),
                city_name,
                country,
                latitude,
                longitude,
                guide.fetched_at,
                source,
                model,
                json.dumps(sections, ensure_ascii=False),
            ),
        )
        self._conn.commit()
        self._evict_oldest()
        return guide

    def _evict_oldest(self) -> None:
        """Keep the store bounded so it cannot quietly fill the card."""
        if self.max_guides <= 0:
            return
        excess = self.count() - self.max_guides
        if excess <= 0:
            return
        self._conn.execute(
            "DELETE FROM guides WHERE city_key IN ("
            "  SELECT city_key FROM guides ORDER BY fetched_at ASC LIMIT ?"
            ");",
            (excess,),
        )
        self._conn.commit()
        log.info("removed %d oldest guide(s) to stay under the limit", excess)

    # -- deleting -----------------------------------------------------------

    def delete(self, city_name: str) -> bool:
        """Forget one city. Returns True if there was something to forget."""
        cursor = self._conn.execute(
            "DELETE FROM guides WHERE city_key = ?;", (normalise(city_name),)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def clear(self) -> int:
        """Forget everything. Returns how many guides were removed."""
        removed = self.count()
        self._conn.execute("DELETE FROM guides;")
        self._conn.commit()
        # Actually give the space back rather than leaving it claimed inside
        # the file, which matters on a card the user is trying to free up.
        self._conn.execute("VACUUM;")
        return removed

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "GuideStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def match_section(text: str) -> Section | None:
    """Find which section of a guide someone is asking about.

    Matching is on whole words. A plain substring test looks fine until
    "what is the weather like" selects the food section, because "weather"
    contains "eat" - and so do "theatre", "great" and "defeat".
    """
    cleaned = normalise(text)
    if not cleaned:
        return None

    best: Section | None = None
    best_len = 0
    for section in SECTIONS:
        for alias in (*section.aliases, section.key):
            if len(alias) <= best_len:
                continue
            # Longest alias wins, so "places to visit" beats a bare "visit".
            if re.search(rf"\b{re.escape(alias)}\b", cleaned):
                best, best_len = section, len(alias)
    return best
