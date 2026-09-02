"""City lookup, including the specific failures the draft implementation had."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from walle.cities import (
    INSERT_SQL,
    City,
    CityDatabase,
    CityNotFound,
    candidate_names,
    create_schema,
    normalise,
    strip_filler,
)

ROWS = [
    # name, country, code, admin, population
    ("Tokyo", "Japan", "JP", "Tokyo", 8336599),
    ("New York City", "United States", "US", "New York", 8804190),
    ("York", "United Kingdom", "GB", "England", 153717),
    ("Bogotá", "Colombia", "CO", "Bogota D.C.", 7674366),
    ("Springfield", "United States", "US", "Massachusetts", 153606),
    ("Springfield", "United States", "US", "Missouri", None),
    ("Nuuk", "Greenland", "GL", None, None),
]


def make_db(path: Path, rows=ROWS) -> None:
    conn = sqlite3.connect(path)
    create_schema(conn)
    conn.executemany(
        INSERT_SQL,
        [
            (name, normalise(name), country, code, admin, pop, None, None, None, None)
            for name, country, code, admin, pop in rows
        ],
    )
    conn.commit()
    conn.close()


class NormaliseTests(unittest.TestCase):
    def test_folds_accents_case_and_punctuation(self):
        self.assertEqual(normalise("Bogotá!"), "bogota")
        self.assertEqual(normalise("  SÃO   Paulo "), "sao paulo")
        self.assertEqual(normalise("Saint-Étienne"), "saint etienne")

    def test_strip_filler_removes_stacked_carrier_phrases(self):
        self.assertEqual(strip_filler("tell me about paris"), "paris")
        self.assertEqual(strip_filler("what is about paris"), "paris")
        self.assertEqual(strip_filler("paris"), "paris")


class CandidateTests(unittest.TestCase):
    def test_longest_ngram_comes_first(self):
        # The draft took words[-1], which turned this into a search for "york".
        candidates = candidate_names("tell me about new york city")
        self.assertEqual(candidates[0], "new york city")
        self.assertLess(candidates.index("new york"), candidates.index("york"))

    def test_noise_words_are_not_single_word_candidates(self):
        self.assertNotIn("the", candidate_names("where is the city"))

    def test_empty_input(self):
        self.assertEqual(candidate_names(""), [])
        self.assertEqual(candidate_names("the a of"), [])

    def test_window_is_bounded(self):
        longest = candidate_names("a b c d e f", max_words=2)[0]
        self.assertEqual(len(longest.split()), 2)


class LookupTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "cities.db"
        make_db(self.path)
        self.db = CityDatabase(self.path)

    def tearDown(self):
        self.db.close()
        self._tmp.cleanup()

    def test_multi_word_city_wins_over_its_suffix(self):
        city = self.db.find_in_utterance("what about new york city")
        self.assertEqual(city.name, "New York City")
        self.assertEqual(city.country, "United States")

    def test_accent_folded_lookup(self):
        self.assertEqual(self.db.find_in_utterance("tell me about bogota").name, "Bogotá")

    def test_ambiguous_name_prefers_the_larger_city(self):
        city = self.db.find_in_utterance("where is springfield")
        self.assertEqual(city.admin, "Massachusetts")

    def test_null_population_does_not_crash(self):
        # f"{None:,}" raises TypeError; the draft did exactly that.
        city = self.db.find_in_utterance("tell me about nuuk")
        summary = city.summary()
        self.assertIn("Nuuk is in Greenland", summary)
        self.assertNotIn("population", summary)

    def test_missing_city_raises(self):
        with self.assertRaises(CityNotFound):
            self.db.find_in_utterance("tell me about atlantis")

    def test_database_is_opened_read_only(self):
        with self.assertRaises(sqlite3.OperationalError):
            self.db._conn.execute("DELETE FROM cities;")

    def test_quote_in_utterance_is_not_a_sql_problem(self):
        # Parameterised queries; the apostrophe is just a character.
        with self.assertRaises(CityNotFound):
            self.db.find_in_utterance("tell me about o'; DROP TABLE cities; --")
        self.assertEqual(
            self.db._conn.execute("SELECT COUNT(*) FROM cities;").fetchone()[0],
            len(ROWS),
        )


class SchemaTolerganceTests(unittest.TestCase):
    def test_missing_optional_columns_are_handled(self):
        # The draft SELECTed a `description` column unconditionally.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "minimal.db"
            conn = sqlite3.connect(path)
            conn.execute(
                "CREATE TABLE cities (name TEXT, name_norm TEXT, country TEXT);"
            )
            conn.execute(
                "INSERT INTO cities VALUES ('Reykjavik', 'reykjavik', 'Iceland');"
            )
            conn.commit()
            conn.close()

            with CityDatabase(path) as db:
                city = db.find_in_utterance("tell me about reykjavik")
            self.assertEqual(city.summary(), "Reykjavik is in Iceland.")

    def test_table_without_required_columns_is_rejected_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wrong.db"
            conn = sqlite3.connect(path)
            conn.execute("CREATE TABLE cities (name TEXT);")
            conn.commit()
            conn.close()
            with self.assertRaises(sqlite3.DatabaseError):
                CityDatabase(path)

    def test_missing_file_raises_filenotfound(self):
        with self.assertRaises(FileNotFoundError):
            CityDatabase(Path("/nonexistent/world_cities.db"))


class SummaryTests(unittest.TestCase):
    def test_admin_is_omitted_when_it_repeats_the_city_name(self):
        city = City("Tokyo", "Japan", 8336599, admin="Tokyo")
        self.assertEqual(
            city.summary(),
            "Tokyo is in Japan. It has a population of roughly 8,336,599 people.",
        )

    def test_description_is_appended_with_one_full_stop(self):
        city = City("Hoi An", "Vietnam", 120000, description="A lantern-lit old town.")
        self.assertTrue(city.summary().endswith("A lantern-lit old town."))
        self.assertNotIn("..", city.summary())


if __name__ == "__main__":
    unittest.main()
