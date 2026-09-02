"""Slippy-map arithmetic and tile fetching, with no network."""

import tempfile
import unittest
import urllib.error
from pathlib import Path

from walle.maps import (
    TILE_SIZE,
    MapRequest,
    TileSource,
    clamp_latitude,
    tile_coords,
    tile_range,
    wrap_tile_x,
    zoom_for_population,
)


class ProjectionTests(unittest.TestCase):
    def test_known_tile_for_london(self):
        # London at zoom 12 is tile 2046/1362 in the standard slippy scheme.
        x, y = tile_coords(51.5074, -0.1278, 12)
        self.assertEqual((int(x), int(y)), (2046, 1362))

    def test_known_tile_for_new_york(self):
        x, y = tile_coords(40.7128, -74.0060, 10)
        self.assertEqual((int(x), int(y)), (301, 385))

    def test_null_island_is_the_centre_of_the_world(self):
        self.assertEqual(tile_coords(0.0, 0.0, 1), (1.0, 1.0))

    def test_zoom_zero_is_a_single_tile(self):
        x, y = tile_coords(51.5, -0.12, 0)
        self.assertEqual((int(x), int(y)), (0, 0))

    def test_latitude_is_clamped_to_the_mercator_limit(self):
        # Web Mercator is undefined at the poles; without clamping this is a
        # division blow-up rather than a map.
        self.assertLess(clamp_latitude(90.0), 85.06)
        self.assertGreater(clamp_latitude(-90.0), -85.06)
        x, y = tile_coords(90.0, 0.0, 5)
        self.assertGreaterEqual(y, 0.0)
        self.assertLessEqual(y, 2**5)

    def test_longitude_wraps_at_the_antimeridian(self):
        # Otherwise a map of Fiji is blank.
        self.assertEqual(wrap_tile_x(-1, 3), 7)
        self.assertEqual(wrap_tile_x(8, 3), 0)
        self.assertEqual(wrap_tile_x(3, 3), 3)


class TileRangeTests(unittest.TestCase):
    def test_centre_offset_lands_inside_the_grid(self):
        for grid in (1, 2, 3, 4):
            _, _, ox, oy = tile_range(48.8566, 2.3522, 12, grid)
            self.assertGreaterEqual(ox, 0)
            self.assertGreaterEqual(oy, 0)
            self.assertLessEqual(ox, grid * TILE_SIZE)
            self.assertLessEqual(oy, grid * TILE_SIZE)

    def test_point_sits_near_the_middle_of_the_grid(self):
        grid = 2
        _, _, ox, oy = tile_range(48.8566, 2.3522, 12, grid)
        middle = grid * TILE_SIZE / 2
        self.assertLess(abs(ox - middle), TILE_SIZE)
        self.assertLess(abs(oy - middle), TILE_SIZE)


class ZoomTests(unittest.TestCase):
    def test_bigger_cities_get_wider_framing(self):
        self.assertLess(
            zoom_for_population(9_000_000), zoom_for_population(30_000)
        )

    def test_unknown_population_gets_a_sensible_default(self):
        self.assertEqual(zoom_for_population(None), 11)
        self.assertEqual(zoom_for_population(0), 11)

    def test_zoom_is_monotonic_in_population(self):
        sizes = [10_000, 100_000, 500_000, 2_000_000, 9_000_000]
        zooms = [zoom_for_population(p) for p in sizes]
        self.assertEqual(zooms, sorted(zooms, reverse=True))


class FakeOpener:
    """Stands in for urllib.request.urlopen."""

    def __init__(self, payload=b"tile-bytes", error=None):
        self.payload = payload
        self.error = error
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        if self.error:
            raise self.error
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.payload


class TileSourceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def source(self, opener=None, offline=False):
        src = TileSource(cache_dir=self.cache, offline=offline)
        if opener is not None:
            import walle.maps as maps

            maps.urllib.request.urlopen = opener
        return src

    def test_fetch_writes_through_to_the_cache(self):
        opener = FakeOpener()
        src = self.source(opener)
        self.assertEqual(src.fetch(5, 1, 2), b"tile-bytes")
        self.assertTrue((self.cache / "5" / "1" / "2.png").is_file())

    def test_second_fetch_uses_the_cache(self):
        # OpenStreetMap's tile policy asks that clients cache; this proves we do.
        opener = FakeOpener()
        src = self.source(opener)
        src.fetch(5, 1, 2)
        src.fetch(5, 1, 2)
        self.assertEqual(len(opener.requests), 1)

    def test_offline_mode_serves_the_cache_only(self):
        opener = FakeOpener()
        self.source(opener).fetch(5, 1, 2)

        blocked = FakeOpener(error=AssertionError("must not hit the network"))
        offline = self.source(blocked, offline=True)
        self.assertEqual(offline.fetch(5, 1, 2), b"tile-bytes")
        self.assertEqual(len(blocked.requests), 0)

    def test_offline_miss_returns_none(self):
        blocked = FakeOpener(error=AssertionError("must not hit the network"))
        self.assertIsNone(self.source(blocked, offline=True).fetch(9, 9, 9))

    def test_network_failure_returns_none(self):
        opener = FakeOpener(error=urllib.error.URLError("down"))
        self.assertIsNone(self.source(opener).fetch(5, 1, 2))

    def test_user_agent_identifies_the_project(self):
        # OSM blocks anonymous default agents outright.
        opener = FakeOpener()
        self.source(opener).fetch(5, 1, 2)
        agent = opener.requests[0].get_header("User-agent")
        self.assertIn("walle-robot-assistant", agent)

    def test_request_defaults(self):
        self.assertEqual(MapRequest(1.0, 2.0).grid, 2)
        self.assertEqual(MapRequest(1.0, 2.0).zoom, 11)


if __name__ == "__main__":
    unittest.main()
