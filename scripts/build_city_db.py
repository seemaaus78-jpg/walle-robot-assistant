#!/usr/bin/env python3
"""Build ``world_cities.db`` from a GeoNames dump.

GeoNames publishes free, tab-separated city extracts under CC BY 4.0:

    https://download.geonames.org/export/dump/cities500.zip    (~200k places)
    https://download.geonames.org/export/dump/cities1000.zip   (~140k places)
    https://download.geonames.org/export/dump/cities15000.zip  (~26k places)

Two optional lookup files turn codes into words, and are worth downloading:

    https://download.geonames.org/export/dump/countryInfo.txt
    https://download.geonames.org/export/dump/admin1CodesASCII.txt

Usage::

    python3 scripts/build_city_db.py cities500.txt \
        --countries countryInfo.txt \
        --admin1 admin1CodesASCII.txt \
        --output world_cities.db

Run this on a laptop, not on the robot: building the indexes on an SD card is
slow, and the finished file copies across in seconds.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from walle.cities import INSERT_SQL, create_schema, normalise  # noqa: E402

# Column offsets in the GeoNames "cities" dump, per its readme.txt.
GEO_NAME = 1
GEO_ASCII = 2
GEO_ALTERNATES = 3
GEO_LAT = 4
GEO_LON = 5
GEO_COUNTRY = 8
GEO_ADMIN1 = 10
GEO_POPULATION = 14
GEO_TIMEZONE = 17
GEO_FIELDS = 19


def load_countries(path: Path | None) -> dict[str, str]:
    """ISO 3166-1 alpha-2 -> country name, from GeoNames countryInfo.txt."""
    if path is None:
        return {}
    mapping: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) > 4:
                mapping[fields[0]] = fields[4]
    return mapping


def load_admin1(path: Path | None) -> dict[str, str]:
    """"US.CA" -> "California", from GeoNames admin1CodesASCII.txt."""
    if path is None:
        return {}
    mapping: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 2:
                mapping[fields[0]] = fields[1]
    return mapping


def read_geonames(path: Path) -> Iterator[list[str]]:
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            if len(row) >= GEO_FIELDS:
                yield row


def _int_or_none(value: str) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    # GeoNames writes 0 for "unknown", which would otherwise be spoken as
    # "a population of roughly 0 people".
    return number or None


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ascii_aliases(field: str, limit: int = 4) -> list[str]:
    """Pick usable alternate names ("Bombay" for Mumbai) out of the dump.

    The alternatenames field mixes scripts, airport codes and URLs, so only
    plain ASCII multi-character entries are kept.
    """
    aliases: list[str] = []
    for raw in field.split(","):
        candidate = raw.strip()
        if len(candidate) < 3 or not candidate.isascii():
            continue
        if not all(ch.isalpha() or ch in " -'." for ch in candidate):
            continue
        aliases.append(candidate)
        if len(aliases) >= limit:
            break
    return aliases


def build(
    source: Path,
    output: Path,
    countries: dict[str, str],
    admin1: dict[str, str],
    min_population: int,
    with_aliases: bool,
) -> tuple[int, int]:
    if output.exists():
        output.unlink()

    conn = sqlite3.connect(output)
    try:
        create_schema(conn)
        rows = 0
        aliases_added = 0
        batch: list[tuple[object, ...]] = []

        for record in read_geonames(source):
            population = _int_or_none(record[GEO_POPULATION])
            if min_population and (population or 0) < min_population:
                continue

            name = record[GEO_NAME] or record[GEO_ASCII]
            if not name:
                continue

            code = record[GEO_COUNTRY]
            country = countries.get(code, code)
            admin = admin1.get(f"{code}.{record[GEO_ADMIN1]}")
            common = (
                country,
                code,
                admin,
                population,
                _float_or_none(record[GEO_LAT]),
                _float_or_none(record[GEO_LON]),
                record[GEO_TIMEZONE] or None,
                None,
            )

            names = {normalise(name): name}
            if with_aliases:
                for alias in ascii_aliases(record[GEO_ALTERNATES]):
                    key = normalise(alias)
                    if key and key not in names:
                        names[key] = name
                        aliases_added += 1

            for norm, display in names.items():
                if not norm:
                    continue
                batch.append((display, norm, *common))
                rows += 1

            if len(batch) >= 5000:
                conn.executemany(INSERT_SQL, batch)
                batch.clear()

        if batch:
            conn.executemany(INSERT_SQL, batch)
        conn.commit()
        conn.execute("ANALYZE;")
        conn.commit()
        conn.execute("VACUUM;")
        return rows, aliases_added
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="GeoNames cities*.txt dump")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("world_cities.db")
    )
    parser.add_argument("--countries", type=Path, help="countryInfo.txt")
    parser.add_argument("--admin1", type=Path, help="admin1CodesASCII.txt")
    parser.add_argument(
        "--min-population",
        type=int,
        default=0,
        help="skip places smaller than this (0 keeps everything)",
    )
    parser.add_argument(
        "--aliases",
        action="store_true",
        help="also index ASCII alternate names, so 'bombay' finds Mumbai",
    )
    args = parser.parse_args(argv)

    if not args.source.is_file():
        parser.error(f"no such file: {args.source}")

    rows, aliases = build(
        args.source,
        args.output,
        load_countries(args.countries),
        load_admin1(args.admin1),
        args.min_population,
        args.aliases,
    )
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(
        f"wrote {rows:,} rows ({aliases:,} aliases) to {args.output} "
        f"[{size_mb:.1f} MB]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
