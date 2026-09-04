"""Saved travel guides: storing, ageing, reading back and deleting."""

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from walle.guides import (
    EMERGENCY_CAUTION,
    SECTIONS,
    Guide,
    GuideStore,
    match_section,
)

SAMPLE = {
    "summary": "Kyoto is the old capital, full of temples and quiet streets.",
    "areas": ["Gion, the old geisha district", "Arashiyama, bamboo and river"],
    "food": ["Nishiki Market for street food", "Pontocho Alley for dinner"],
    "transport": ["Kyoto Station is the main hub", "Buses reach most temples"],
    "stay": ["Downtown near Karasuma is central"],
    "emergency": ["Emergency number is 119", "Kyoto University Hospital"],
}


def store(tmp: Path, **kwargs) -> GuideStore:
    return GuideStore(tmp / "guides.db", **kwargs)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.store = store(self.tmp)

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_saves_and_reads_back(self):
        self.store.save("Kyoto", SAMPLE, country="Japan")
        guide = self.store.get("Kyoto")
        self.assertIsNotNone(guide)
        self.assertEqual(guide.city_name, "Kyoto")
        self.assertEqual(guide.country, "Japan")
        self.assertEqual(guide.sections["areas"], SAMPLE["areas"])

    def test_lookup_ignores_case_and_accents(self):
        self.store.save("Bogotá", SAMPLE)
        self.assertIsNotNone(self.store.get("bogota"))
        self.assertIsNotNone(self.store.get("BOGOTA"))

    def test_saving_again_replaces_rather_than_duplicates(self):
        self.store.save("Kyoto", SAMPLE)
        self.store.save("Kyoto", {"summary": "Updated."})
        self.assertEqual(self.store.count(), 1)
        self.assertEqual(self.store.get("Kyoto").sections["summary"], "Updated.")

    def test_missing_city(self):
        self.assertIsNone(self.store.get("Atlantis"))

    def test_creates_its_own_database_file(self):
        self.assertTrue((self.tmp / "guides.db").is_file())

    def test_survives_a_directory_that_does_not_exist_yet(self):
        deep = self.tmp / "a" / "b" / "guides.db"
        with GuideStore(deep) as nested:
            nested.save("Rome", SAMPLE)
            self.assertIsNotNone(nested.get("Rome"))

    def test_unreadable_sections_do_not_crash_the_read(self):
        # A truncated write on a card pulled mid-save should cost one guide,
        # not the whole store.
        self.store.save("Kyoto", SAMPLE)
        conn = sqlite3.connect(self.tmp / "guides.db")
        conn.execute("UPDATE guides SET sections = 'not json';")
        conn.commit()
        conn.close()
        guide = self.store.get("Kyoto")
        self.assertEqual(guide.sections, {})


class DeletionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = store(Path(self._tmp.name))
        for city in ("Kyoto", "Rome", "Lima"):
            self.store.save(city, SAMPLE)

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_delete_one(self):
        self.assertTrue(self.store.delete("Rome"))
        self.assertIsNone(self.store.get("Rome"))
        self.assertEqual(self.store.count(), 2)

    def test_delete_something_not_saved(self):
        self.assertFalse(self.store.delete("Atlantis"))
        self.assertEqual(self.store.count(), 3)

    def test_delete_is_accent_and_case_insensitive(self):
        self.store.save("Bogotá", SAMPLE)
        self.assertTrue(self.store.delete("bogota"))

    def test_clear_removes_everything_and_reports_the_count(self):
        self.assertEqual(self.store.clear(), 3)
        self.assertEqual(self.store.count(), 0)
        self.assertEqual(self.store.list_cities(), [])

    def test_clear_on_an_empty_store(self):
        self.store.clear()
        self.assertEqual(self.store.clear(), 0)

    def test_store_is_usable_after_clearing(self):
        self.store.clear()
        self.store.save("Oslo", SAMPLE)
        self.assertIsNotNone(self.store.get("Oslo"))


class EvictionTests(unittest.TestCase):
    def test_oldest_guides_are_dropped_past_the_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with store(Path(tmp), max_guides=3) as s:
                for city in ("A", "B", "C", "D", "E"):
                    s.save(city, SAMPLE)
                self.assertEqual(s.count(), 3)
                # The two oldest went; the newest stayed.
                self.assertIsNone(s.get("A"))
                self.assertIsNotNone(s.get("E"))

    def test_zero_limit_means_no_eviction(self):
        with tempfile.TemporaryDirectory() as tmp:
            with store(Path(tmp), max_guides=0) as s:
                for i in range(12):
                    s.save(f"City{i}", SAMPLE)
                self.assertEqual(s.count(), 12)


def aged(days: int) -> Guide:
    when = datetime.now(timezone.utc) - timedelta(days=days)
    return Guide("Kyoto", "Japan", when.isoformat(timespec="seconds"), "gemini", SAMPLE)


class AgeTests(unittest.TestCase):
    """A restaurant list is a snapshot, not a fact. Age has to be visible."""

    def test_age_in_days(self):
        self.assertEqual(aged(0).age_days, 0)
        self.assertEqual(aged(45).age_days, 45)

    def test_age_phrases_read_naturally(self):
        self.assertEqual(aged(0).age_phrase(), "saved today")
        self.assertEqual(aged(1).age_phrase(), "saved yesterday")
        self.assertEqual(aged(5).age_phrase(), "saved 5 days ago")
        self.assertEqual(aged(21).age_phrase(), "saved 3 weeks ago")
        self.assertEqual(aged(120).age_phrase(), "saved 4 months ago")

    def test_staleness_threshold(self):
        self.assertFalse(aged(30).is_stale)
        self.assertTrue(aged(120).is_stale)

    def test_a_broken_timestamp_does_not_crash(self):
        guide = Guide("Kyoto", "Japan", "not-a-date", "gemini", SAMPLE)
        self.assertEqual(guide.age_days, 0)


class ReadingTests(unittest.TestCase):
    def setUp(self):
        self.guide = Guide("Kyoto", "Japan", datetime.now(timezone.utc).isoformat(),
                           "gemini", SAMPLE)

    def test_list_sections_are_joined_into_speech(self):
        text = self.guide.section_text("food")
        self.assertIn("Nishiki Market", text)
        self.assertIn("Pontocho Alley", text)

    def test_string_sections_pass_through(self):
        self.assertEqual(self.guide.section_text("summary"), SAMPLE["summary"])

    def test_emergency_always_carries_the_caution(self):
        """A model can invent a hospital address as easily as recall one."""
        text = self.guide.section_text("emergency")
        self.assertIn("119", text)
        self.assertIn(EMERGENCY_CAUTION, text)

    def test_other_sections_do_not_carry_the_caution(self):
        self.assertNotIn(EMERGENCY_CAUTION, self.guide.section_text("food"))

    def test_missing_section(self):
        self.assertIsNone(Guide("X", None, "", "", {}).section_text("food"))

    def test_empty_section(self):
        self.assertIsNone(Guide("X", None, "", "", {"food": []}).section_text("food"))

    def test_summary_lists_what_else_is_available(self):
        spoken = self.guide.spoken_summary()
        self.assertIn("Kyoto is the old capital", spoken)
        self.assertIn("food and restaurants", spoken)

    def test_summary_without_sections(self):
        bare = Guide("Lima", None, "", "", {})
        self.assertIn("Lima", bare.spoken_summary())


class SectionMatchTests(unittest.TestCase):
    def test_common_phrasings(self):
        cases = {
            "restaurants in kyoto": "food",
            "where should i eat": "food",
            "how do i get around": "transport",
            "which train station": "transport",
            "where is the hospital": "emergency",
            "nearest police station": "emergency",
            "where should i stay": "stay",
            "best areas to visit": "areas",
        }
        for phrase, expected in cases.items():
            found = match_section(phrase)
            self.assertIsNotNone(found, phrase)
            self.assertEqual(found.key, expected, phrase)

    def test_longest_alias_wins(self):
        # "places to visit" must beat a bare "visit" landing elsewhere.
        self.assertEqual(match_section("famous places to visit").key, "areas")

    def test_no_match(self):
        self.assertIsNone(match_section("what is the weather like"))
        self.assertIsNone(match_section(""))

    def test_every_section_is_reachable(self):
        for section in SECTIONS:
            self.assertIsNotNone(match_section(section.aliases[0]), section.key)


if __name__ == "__main__":
    unittest.main()
