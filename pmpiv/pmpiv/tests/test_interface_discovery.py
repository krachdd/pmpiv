#!/usr/bin/env python3
"""Tests for pmpiv.interface_discovery."""
import numpy as np
import pytest

import pmpiv
from pmpiv import interface_discovery as disc
from conftest import _synthetic_channel_image, _arc_y


WALLS = ((100, 112), (400, 412), (700, 712))
ARC = {"cx": 556, "cy": 150, "r": 200, "thickness": 20}
APEX_Y = ARC["cy"] + ARC["r"]


@pytest.fixture()
def channel_img():
    return _synthetic_channel_image(walls=WALLS, arcs=(ARC,), beads_in=(1,))


class TestFindChannelWalls:
    def test_finds_all_walls(self, channel_img):
        found = disc.find_channel_walls(channel_img)
        assert len(found) == len(WALLS)
        for (e0, e1), (f0, f1) in zip(WALLS, found):
            assert abs(f0 - e0) <= 2
            assert abs(f1 - e1) <= 2

    def test_thin_line_ignored(self, channel_img):
        img = channel_img.copy()
        img[:, 250:252] = 40
        found = disc.find_channel_walls(img)
        assert len(found) == len(WALLS)

    def test_wide_region_ignored(self, channel_img):
        img = channel_img.copy()
        img[:, 200:260] = 40
        found = disc.find_channel_walls(img)
        assert all(not (200 <= f0 < 260) for f0, _ in found)


class TestFindTopWall:
    def test_found(self):
        img = _synthetic_channel_image(walls=WALLS, arcs=(ARC,), top_wall_y=50)
        y = disc.find_top_wall(img)
        assert y is not None
        assert abs(y - 62) <= 3

    def test_none_without_wall(self, channel_img):
        assert disc.find_top_wall(channel_img) is None

    def test_from_wall_tops(self):
        # walls start at y=120 (channel mouth); a border line at the very top
        # must not be mistaken for it
        img = _synthetic_channel_image(walls=WALLS, arcs=(ARC,), wall_y_from=120, top_wall_y=10)
        walls = disc.find_channel_walls(img)
        y = disc.find_top_wall(img, walls=walls)
        assert abs(y - 120) <= 3

    def test_full_height_walls_fall_back(self):
        img = _synthetic_channel_image(walls=WALLS, arcs=(ARC,), top_wall_y=50)
        walls = disc.find_channel_walls(img)
        assert abs(disc.find_top_wall(img, walls=walls) - 62) <= 3


class TestChannelsFromWalls:
    def test_liquid_classification(self, channel_img):
        walls = disc.find_channel_walls(channel_img)
        channels = disc.channels_from_walls(channel_img, walls, y_from=0, verbose=False)
        assert len(channels) == 2
        assert channels[0]["liquid"] is False
        assert channels[1]["liquid"] is True
        assert channels[1]["texture"] > channels[0]["texture"]

    def test_border_included_on_request(self, channel_img):
        walls = disc.find_channel_walls(channel_img)
        channels = disc.channels_from_walls(channel_img, walls, y_from=0, include_border=True, verbose=False)
        assert len(channels) == 4
        assert channels[0]["x0"] == 0


class TestFindMenisciHough:
    def test_finds_arc(self, channel_img):
        circles = disc.find_menisci_hough(channel_img, min_radius=120, max_radius=300)
        assert circles, "no circle found"
        best = min(circles, key=lambda c: np.hypot(c[0] - ARC["cx"], c[1] - ARC["cy"]))
        assert abs(best[0] - ARC["cx"]) <= 10
        assert abs(best[1] - ARC["cy"]) <= 10
        assert abs(best[2] - ARC["r"]) <= 0.1 * ARC["r"]

    def test_blank_image(self):
        img = np.full((600, 800), 200, dtype=np.uint8)
        assert disc.find_menisci_hough(img) == []


class TestScanChannelForBand:
    def test_band_contains_arc(self, channel_img):
        y0, y1, score = disc.scan_channel_for_band(channel_img, (420, 692), 0, 600, 150)
        mid = np.nanmean(_arc_y(np.arange(420, 692), ARC["cx"], ARC["cy"], ARC["r"]))
        assert y0 <= mid <= y1
        assert score > 0


class TestDiscoverInterfaces:
    def test_walls_method(self, channel_img):
        seeds = disc.discover_interfaces(channel_img, method="walls", band_height=150, verbose=False)
        assert len(seeds) == 1
        s = seeds[0]
        assert s["seed_type"] == "scan"
        assert 412 <= s["x_search"][0] < s["x_search"][1] <= 700
        assert s["wall_x"] == (412.0, 700.0)
        mid = np.nanmean(_arc_y(np.arange(*s["x_search"]), ARC["cx"], ARC["cy"], ARC["r"]))
        assert s["y_search"][0] <= mid <= s["y_search"][1]

    def test_both_method(self, channel_img):
        seeds = disc.discover_interfaces(channel_img, method="both", band_height=150, verbose=False,
                                         hough_kwargs=dict(min_radius=120, max_radius=300))
        assert len(seeds) == 1
        s = seeds[0]
        assert s["seed_type"] in ("hough", "scan")
        mid = np.nanmean(_arc_y(np.arange(*s["x_search"]), ARC["cx"], ARC["cy"], ARC["r"]))
        assert s["y_search"][0] <= mid <= s["y_search"][1]

    def test_hough_method(self, channel_img):
        seeds = disc.discover_interfaces(channel_img, method="hough", band_height=150, verbose=False,
                                         hough_kwargs=dict(min_radius=120, max_radius=300))
        assert len(seeds) >= 1
        s = seeds[0]
        assert s["seed_type"] == "hough"
        assert s["wall_x"] is None
        mid = np.nanmean(_arc_y(np.arange(*s["x_search"]), ARC["cx"], ARC["cy"], ARC["r"]))
        assert s["y_search"][0] <= mid <= s["y_search"][1]

    def test_two_channels(self):
        arcs = ({"cx": 256, "cy": 150, "r": 200, "thickness": 20}, ARC)
        img = _synthetic_channel_image(walls=WALLS, arcs=arcs, beads_in=(0, 1))
        seeds = disc.discover_interfaces(img, method="walls", band_height=150, verbose=False)
        assert [s["label"] for s in seeds] == ["ch0", "ch1"]
        assert seeds[0]["x_search"][1] <= seeds[1]["x_search"][0]

    def test_bad_method_raises(self, channel_img):
        with pytest.raises(ValueError):
            disc.discover_interfaces(channel_img, method="magic")
