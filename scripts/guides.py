#!/usr/bin/env python3
"""Manage the travel guides saved on the card.

Everything here can also be done by voice, but a keyboard is easier when you
want to see exactly what is stored or clear a lot at once.

    python3 scripts/guides.py list
    python3 scripts/guides.py show kyoto
    python3 scripts/guides.py delete kyoto
    python3 scripts/guides.py clear
    python3 scripts/guides.py size

Guides live in their own database file, so the nuclear option is simply
deleting it - nothing else depends on that file, and the robot recreates an
empty one on its next start.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from walle.config import load_config  # noqa: E402
from walle.guides import SECTION_BY_KEY, GuideStore  # noqa: E402


def cmd_list(store: GuideStore, _args) -> int:
    saved = store.list_cities()
    if not saved:
        print("No guides saved.")
        return 0
    width = max(len(name) for name, _ in saved)
    for name, age in saved:
        print(f"  {name:<{width}}  {age}")
    print(f"\n{len(saved)} guide(s).")
    return 0


def cmd_show(store: GuideStore, args) -> int:
    guide = store.get(args.city)
    if guide is None:
        print(f"Nothing saved for {args.city!r}.")
        return 1

    where = f"{guide.city_name}, {guide.country}" if guide.country else guide.city_name
    print(f"{where}  —  {guide.age_phrase()}", end="")
    print("  (out of date)" if guide.is_stale else "")
    print(f"source: {guide.source}" + (f" / {guide.model}" if guide.model else ""))

    for key, section in SECTION_BY_KEY.items():
        value = guide.sections.get(key)
        if not value:
            continue
        print(f"\n{section.label.upper()}")
        if isinstance(value, str):
            print(f"  {value}")
        else:
            for item in value:
                print(f"  - {item}")
    return 0


def cmd_delete(store: GuideStore, args) -> int:
    if store.delete(args.city):
        print(f"Deleted the guide for {args.city}.")
        return 0
    print(f"Nothing saved for {args.city!r}.")
    return 1


def cmd_clear(store: GuideStore, args) -> int:
    count = store.count()
    if count == 0:
        print("Nothing saved.")
        return 0
    if not args.yes:
        # Deleting someone's collected travel notes is not undoable, so it is
        # not something to do on a mistyped command.
        answer = input(f"Delete all {count} saved guide(s)? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Cancelled.")
            return 1
    print(f"Deleted {store.clear()} guide(s).")
    return 0


def cmd_size(store: GuideStore, _args) -> int:
    path = Path(store.path)
    size = path.stat().st_size if path.is_file() else 0
    print(f"{store.count()} guide(s), {size / 1024:.1f} KB")
    print(f"file: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, help="path to config.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="show every saved city and how old it is")

    show = sub.add_parser("show", help="print one guide in full")
    show.add_argument("city")

    delete = sub.add_parser("delete", help="forget one city")
    delete.add_argument("city")

    clear = sub.add_parser("clear", help="forget everything")
    clear.add_argument("-y", "--yes", action="store_true", help="skip the prompt")

    sub.add_parser("size", help="how much space the guides take up")

    args = parser.parse_args(argv)
    config = load_config(args.config)

    try:
        store = GuideStore(config.guides.database, config.guides.max_guides)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not open {config.guides.database}: {exc}", file=sys.stderr)
        return 1

    handlers = {
        "list": cmd_list,
        "show": cmd_show,
        "delete": cmd_delete,
        "clear": cmd_clear,
        "size": cmd_size,
    }
    try:
        return handlers[args.command](store, args)
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
