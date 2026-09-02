"""Maps for the display.

Online, tiles come from OpenStreetMap and are stitched into one image centred on
the place being discussed. Offline, the same code reads from a tile pack you
downloaded in advance; if there is no pack, the assistant falls back to a text
card with the city's facts on it rather than showing an empty screen.

The slippy-map arithmetic is plain maths with no dependencies, so it is fully
testable here. Only the stitching needs Pillow, and that is imported lazily.

OpenStreetMap's tile servers are donated infrastructure with a published usage
policy: identify yourself with a real User-Agent, do not bulk download, and
cache what you fetch. This module does all three - one small grid per request,
written through to an on-disk cache. If you intend to hammer it, run your own
tile server or buy a commercial key.
"""

from __future__ import annotations

import logging
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

TILE_SIZE = 256
DEFAULT_USER_AGENT = (
    "walle-robot-assistant/0.1 (+https://github.com/seemaaus78-jpg/walle-robot-assistant)"
)


def clamp_latitude(lat: float) -> float:
    """Web Mercator is undefined at the poles; clamp to its usual limit."""
    return max(-85.05112878, min(85.05112878, lat))


def tile_coords(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    """Fractional slippy-map tile coordinates for a latitude and longitude."""
    lat = clamp_latitude(lat)
    n = 2.0**zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    # At the clamped Mercator limit the arithmetic lands a hair either side of
    # the grid (y = -2e-10 at the north edge). Left alone that is a negative
    # tile index and a 404 rather than a map.
    return x, min(max(y, 0.0), n)


def tile_range(
    lat: float, lon: float, zoom: int, grid: int
) -> tuple[int, int, int, int]:
    """The top-left tile and the pixel offset of the centre within the grid.

    Returns (x0, y0, offset_x, offset_y): which tile to start downloading at,
    and where the requested point lands once the grid is stitched, so the
    centre can be cropped to the panel accurately.
    """
    x, y = tile_coords(lat, lon, zoom)
    half = grid / 2.0
    x0 = int(math.floor(x - half + 0.5))
    y0 = int(math.floor(y - half + 0.5))
    offset_x = int(round((x - x0) * TILE_SIZE))
    offset_y = int(round((y - y0) * TILE_SIZE))
    return x0, y0, offset_x, offset_y


def wrap_tile_x(x: int, zoom: int) -> int:
    """Wrap longitude at the antimeridian so a map of Fiji is not blank."""
    n = 2**zoom
    return x % n


def zoom_for_population(population: int | None) -> int:
    """Pick a sensible zoom from a city's size.

    A capital of ten million and a village of two thousand need very different
    framing; showing both at the same zoom wastes the screen on one and
    overshoots the other.
    """
    if not population:
        return 11
    if population >= 5_000_000:
        return 9
    if population >= 1_000_000:
        return 10
    if population >= 250_000:
        return 11
    if population >= 50_000:
        return 12
    return 13


@dataclass(frozen=True)
class MapRequest:
    latitude: float
    longitude: float
    zoom: int = 11
    grid: int = 2
    """Tiles per side. 2 gives a 512x512 stitch, which downsamples cleanly to a
    240 or 320 pixel panel."""


class TileSource:
    """Fetches tiles from OpenStreetMap, caching every one to disk."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        url_template: str = "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        timeout_s: float = 5.0,
        offline: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.user_agent = user_agent
        self.url_template = url_template
        self.timeout_s = timeout_s
        self.offline = offline
        """When true, only the cache is consulted - which is what makes a
        pre-downloaded tile pack work with no connection."""

    def _cache_path(self, z: int, x: int, y: int) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / str(z) / str(x) / f"{y}.png"

    def fetch(self, z: int, x: int, y: int) -> bytes | None:
        """One tile, from cache if possible."""
        path = self._cache_path(z, x, y)
        if path is not None and path.is_file():
            try:
                return path.read_bytes()
            except OSError as exc:
                log.debug("tile cache read failed: %s", exc)

        if self.offline:
            return None

        url = self.url_template.format(z=z, x=x, y=y)
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                data = response.read()
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            log.warning("tile fetch failed (%s): %s", url, exc)
            return None

        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            except OSError as exc:
                log.debug("tile cache write failed: %s", exc)
        return data


class MapRenderer:
    """Stitches a tile grid into one image centred on the requested point."""

    def __init__(self, source: TileSource) -> None:
        self._source = source

    def render(self, request: MapRequest, size: tuple[int, int]):
        """Return a Pillow image, or None if the map could not be built."""
        try:
            from PIL import Image  # noqa: PLC0415 - optional dep
        except ImportError:
            log.warning("Pillow is not installed; cannot render maps")
            return None

        grid = max(1, request.grid)
        x0, y0, offset_x, offset_y = tile_range(
            request.latitude, request.longitude, request.zoom, grid
        )

        canvas = Image.new("RGB", (grid * TILE_SIZE, grid * TILE_SIZE), (40, 44, 52))
        fetched = 0
        for row in range(grid):
            for col in range(grid):
                data = self._source.fetch(
                    request.zoom, wrap_tile_x(x0 + col, request.zoom), y0 + row
                )
                if data is None:
                    continue
                try:
                    tile = Image.open(__import__("io").BytesIO(data))
                    tile.load()
                except Exception as exc:  # noqa: BLE001 - a bad tile is not fatal
                    log.debug("undecodable tile: %s", exc)
                    continue
                canvas.paste(tile.convert("RGB"), (col * TILE_SIZE, row * TILE_SIZE))
                fetched += 1

        # A grid where nothing arrived is a grey square, which looks like a
        # rendering bug. Say we failed and let the caller show a text card.
        if fetched == 0:
            return None

        return self._crop_and_mark(canvas, offset_x, offset_y, size, Image)

    @staticmethod
    def _crop_and_mark(canvas, offset_x: int, offset_y: int, size, Image):
        """Crop to the panel around the point, then mark it."""
        from PIL import ImageDraw  # noqa: PLC0415

        width, height = size
        left = max(0, min(canvas.width - width, offset_x - width // 2))
        top = max(0, min(canvas.height - height, offset_y - height // 2))
        cropped = canvas.crop((left, top, left + width, top + height))

        pen = ImageDraw.Draw(cropped)
        cx, cy = offset_x - left, offset_y - top
        radius = max(4, min(width, height) // 28)
        pen.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=(230, 60, 70),
            outline=(255, 255, 255),
            width=2,
        )
        return cropped
