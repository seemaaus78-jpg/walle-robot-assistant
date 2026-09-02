"""Offline city lookup backed by SQLite.

Improvements over the draft implementation:

* ``city = words[-1]`` only ever looked at the final word, so "tell me about
  new york" searched for "york" and "how big is san francisco" searched for
  "francisco". Candidates are now generated longest-first across the whole
  utterance.
* Vosk emits lower-case, unaccented, unpunctuated text. Matching against a raw
  ``name`` column therefore misses "bogota" -> "Bogotá". Lookups go through a
  normalised, indexed column.
* ``f"{pop:,}"`` raised ``TypeError`` whenever the population column was NULL,
  which turned a missing data point into a crash mid-sentence.
* A ``description`` column was assumed to exist. Its presence is now detected.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Phrases people put in front of a place name. Stripped before matching so the
# n-gram scan does not waste passes on them.
LEADING_FILLER = (
    "tell me about",
    "tell me something about",
    "what can you tell me about",
    "what do you know about",
    "do you know about",
    "how big is",
    "where is",
    "what about",
    "what is",
    "whats",
    "about",
)

# Words that are never a city on their own, so a one-word candidate matching
# them is discarded before it reaches the database.
NOISE_WORDS = frozenset(
    """
    a an and are as at be by can city could do does for from go going have how
    i in is it its me my of on or place say tell that the there these they this
    to town travel trip visit want was we what when where which who why will
    with would you your
    """.split()
)

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")

# The single source of truth for the table contract. scripts/build_city_db.py
# writes it, CityDatabase reads it, and the tests build fixtures from it, so the
# three cannot drift apart.
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cities (
    id           INTEGER PRIMARY KEY,
    name         TEXT    NOT NULL,
    name_norm    TEXT    NOT NULL,
    country      TEXT    NOT NULL,
    country_code TEXT,
    admin        TEXT,
    population   INTEGER,
    latitude     REAL,
    longitude    REAL,
    timezone     TEXT,
    description  TEXT
);
CREATE INDEX IF NOT EXISTS idx_cities_name_norm ON cities(name_norm);
CREATE INDEX IF NOT EXISTS idx_cities_population ON cities(population DESC);
"""

CITY_COLUMNS = (
    "name",
    "name_norm",
    "country",
    "country_code",
    "admin",
    "population",
    "latitude",
    "longitude",
    "timezone",
    "description",
)

INSERT_SQL = (
    f"INSERT INTO cities ({', '.join(CITY_COLUMNS)}) "
    f"VALUES ({', '.join('?' * len(CITY_COLUMNS))});"
)


def create_schema(conn: sqlite3.Connection) -> None:
    """Create the cities table and its indexes on an open connection."""
    conn.executescript(SCHEMA_SQL)


def normalise(text: str) -> str:
    """Case-fold, strip accents and punctuation, collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    stripped = _PUNCT.sub(" ", stripped.casefold())
    return _WS.sub(" ", stripped).strip()


def strip_filler(text: str) -> str:
    """Remove a leading carrier phrase such as "tell me about"."""
    result = text
    changed = True
    while changed:
        changed = False
        for phrase in LEADING_FILLER:
            if result.startswith(phrase + " "):
                result = result[len(phrase) + 1 :]
                changed = True
                break
    return result.strip()


def candidate_names(text: str, max_words: int = 4) -> list[str]:
    """Every plausible place-name n-gram, longest and leftmost first.

    Longest-first matters: "new york" must be tried before "york", or
    Yorkshire's county town answers a question about Manhattan.
    """
    words = strip_filler(normalise(text)).split()
    if not words:
        return []

    seen: set[str] = set()
    candidates: list[str] = []
    for size in range(min(max_words, len(words)), 0, -1):
        for start in range(len(words) - size + 1):
            window = words[start : start + size]
            gram = " ".join(window)
            if gram in seen:
                continue
            # A run made only of filler ("the a of") can never be a place name.
            # Dropping it here saves a database round trip per n-gram, which on
            # an SD card is the difference between a snappy and a sluggish reply.
            if all(word in NOISE_WORDS for word in window):
                continue
            seen.add(gram)
            candidates.append(gram)
    return candidates


@dataclass(frozen=True)
class City:
    name: str
    country: str
    population: int | None = None
    admin: str | None = None
    description: str | None = None
    latitude: float | None = None
    longitude: float | None = None

    @property
    def has_location(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    def summary(self) -> str:
        """A sentence suitable for handing straight to text-to-speech."""
        where = f"{self.name} is in {self.country}"
        if self.admin and self.admin.casefold() != self.name.casefold():
            where = f"{self.name} is in {self.admin}, {self.country}"

        parts = [where + "."]
        if self.population:
            parts.append(f"It has a population of roughly {self.population:,} people.")
        if self.description:
            parts.append(self.description.strip().rstrip(".") + ".")
        return " ".join(parts)


class CityNotFound(LookupError):
    """No row matched any candidate drawn from the utterance."""


class CityDatabase:
    """Read-only accessor for ``world_cities.db``.

    Opened with ``mode=ro`` so a corrupt query can never write to, or create,
    the database file on the SD card.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"city database not found: {self.path}")
        self._conn = sqlite3.connect(
            f"file:{self.path}?mode=ro", uri=True, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._columns = self._detect_columns()

    def _detect_columns(self) -> set[str]:
        rows = self._conn.execute("PRAGMA table_info(cities);").fetchall()
        columns = {row["name"] for row in rows}
        if not columns:
            raise sqlite3.DatabaseError(
                f"{self.path} has no 'cities' table; rebuild it with "
                "scripts/build_city_db.py"
            )
        required = {"name", "country", "name_norm"}
        if missing := required - columns:
            raise sqlite3.DatabaseError(
                f"{self.path} is missing column(s) {sorted(missing)}; rebuild it "
                "with scripts/build_city_db.py"
            )
        return columns

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "CityDatabase":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def lookup(self, name: str) -> City | None:
        """Find one city by normalised name, preferring the most populous."""
        selected = ["name", "country"]
        for optional in ("population", "admin", "description", "latitude", "longitude"):
            if optional in self._columns:
                selected.append(optional)

        order = (
            "ORDER BY population IS NULL, population DESC"
            if "population" in self._columns
            else ""
        )
        query = (
            f"SELECT {', '.join(selected)} FROM cities "  # noqa: S608 - names are literals
            f"WHERE name_norm = ? {order} LIMIT 1;"
        )
        try:
            row = self._conn.execute(query, (normalise(name),)).fetchone()
        except sqlite3.Error as exc:
            log.error("city lookup failed for %r: %s", name, exc)
            return None
        if row is None:
            return None

        keys = row.keys()
        return City(
            name=row["name"],
            country=row["country"],
            population=row["population"] if "population" in keys else None,
            admin=row["admin"] if "admin" in keys else None,
            description=row["description"] if "description" in keys else None,
            latitude=row["latitude"] if "latitude" in keys else None,
            longitude=row["longitude"] if "longitude" in keys else None,
        )

    def random_city(self, min_population: int = 250_000) -> City | None:
        """A populous city at random, for offline small talk.

        Restricted to somewhere sizeable because "here is something
        interesting" followed by a hamlet of 400 people is not interesting.
        """
        if "population" not in self._columns:
            return None
        try:
            row = self._conn.execute(
                "SELECT name FROM cities WHERE population >= ? "
                "ORDER BY RANDOM() LIMIT 1;",
                (min_population,),
            ).fetchone()
        except sqlite3.Error as exc:
            log.error("random city lookup failed: %s", exc)
            return None
        return self.lookup(row["name"]) if row else None

    def find_in_utterance(self, text: str, max_words: int = 4) -> City:
        """Scan an utterance for a place name and return the first match."""
        for candidate in candidate_names(text, max_words=max_words):
            city = self.lookup(candidate)
            if city is not None:
                return city
        raise CityNotFound(text)
