#!/usr/bin/env python3
"""
Shared fixtures for pmpiv pytest suite.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# sys.path: make `import pmpiv` work regardless of how pytest is invoked
# ---------------------------------------------------------------------------
_PKG_ROOT = Path(__file__).parent.parent          # …/pmpiv/pmpiv/pmpiv/
_PIMS_ROOT = _PKG_ROOT.parent / "pims"            # …/pmpiv/pmpiv/pims/
_TRACKPY_ROOT = _PKG_ROOT.parent / "trackpy"      # …/pmpiv/pmpiv/trackpy/

for _p in (_PKG_ROOT, _PIMS_ROOT, _TRACKPY_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pmpiv  # noqa: E402 – must come after path manipulation

# ---------------------------------------------------------------------------
# Paths to bundled test data
# ---------------------------------------------------------------------------
_TEST_DATA = Path(__file__).parents[3] / "tests" / "test1"
BIG_JSON = _TEST_DATA / "annotation" / "big_capillary.json"
SMALL_JSON = _TEST_DATA / "annotation" / "small_capillary.json"
TEST_IMAGE_TIF = _TEST_DATA / "annotation" / "big" / "images" / "15-14-11.000-0001.tif"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_metadata(directory: Path, **overrides) -> Path:
    """Write a minimal valid metadata .txt file into *directory* and return its path."""
    defaults = dict(
        IN_PATH="/tmp/pmpiv_test_images",
        WORKING_DIR=str(directory),
        IN_FORMAT="tif",
        PIXELSIZE="1.3e-06",
        HEIGHT="100.0e-06",
        START_FRAME="0",
        END_FRAME="0",
        RATE="1",
        FEATURE_SIZE="11",
        FEATURE_MIN_SIZE="500",
        FEATURES_ARE_DARK="True",
        FPS="70.0",
        MAX_PARTICLE_SPEED="4",
        MEMORY="6",
        REMOVE_STATIC="True",
        CHECK_STATIC="5",
        STATIC_DEV_PARAMETER="1.0",
        DURATION="5",
        JSON_PATH=str(directory),
        REMOVAL="",
        EXTRACTION="",
    )
    defaults.update(overrides)
    lines = [f"{k} {v}" for k, v in defaults.items()]
    cfg = directory / "test_metadata.txt"
    cfg.write_text("\n".join(lines) + "\n")
    return cfg


def _make_trajectory_df(
    n_particles: int = 4,
    n_frames: int = 10,
    dx: float = 3.0,
    dy: float = 2.0,
    x0: float = 100.0,
    y0: float = 200.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic moving-particle trajectory DataFrame (trackpy output format)."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_particles):
        for f in range(n_frames):
            rows.append(
                {
                    "x": x0 + p * 50 + f * dx + rng.uniform(-0.05, 0.05),
                    "y": y0 + p * 30 + f * dy + rng.uniform(-0.05, 0.05),
                    "frame": f,
                    "particle": p,
                    "mass": 1000.0 + rng.uniform(-50, 50),
                    "size": 3.0 + rng.uniform(-0.2, 0.2),
                    "ecc": 0.05 + rng.uniform(0, 0.05),
                }
            )
    df = pd.DataFrame(rows)
    df = df.set_index("frame", drop=False)
    return df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def metadata_file(tmp_path):
    """Path to a minimal valid metadata config file."""
    return _write_metadata(tmp_path)


@pytest.fixture()
def md(metadata_file):
    """Parsed Metadata object."""
    return pmpiv.metadata.Metadata(str(metadata_file))


@pytest.fixture()
def trajectory_df():
    """Synthetic moving-particle DataFrame (4 particles × 10 frames)."""
    return _make_trajectory_df()


@pytest.fixture()
def mixed_df():
    """DataFrame with one static and one clearly-moving particle."""
    rows = []
    for f in range(15):
        # particle 0: static (range 0 < CHECK_STATIC=5 → should be removed)
        rows.append(
            {"x": 100.0, "y": 200.0, "frame": f, "particle": 0,
             "mass": 1000.0, "size": 3.0, "ecc": 0.1}
        )
        # particle 1: moving 10 px/frame in x, 5 px/frame in y
        rows.append(
            {"x": 300.0 + f * 10, "y": 400.0 + f * 5, "frame": f, "particle": 1,
             "mass": 1000.0, "size": 3.0, "ecc": 0.1}
        )
    df = pd.DataFrame(rows)
    return df.set_index("frame", drop=False)


@pytest.fixture()
def stubs_df():
    """DataFrame with a 3-frame stub (removed) and a 10-frame trajectory (kept)."""
    rows = []
    for f in range(3):  # particle 0: too short (< DURATION=5)
        rows.append({"x": float(f), "y": float(f), "frame": f, "particle": 0,
                     "mass": 1000.0, "size": 3.0, "ecc": 0.1})
    for f in range(10):  # particle 1: long enough
        rows.append({"x": 100.0 + f, "y": 200.0 + f, "frame": f, "particle": 1,
                     "mass": 1000.0, "size": 3.0, "ecc": 0.1})
    df = pd.DataFrame(rows)
    return df.set_index("frame", drop=False)


@pytest.fixture()
def annotation_region_df():
    """Particles split into 'inside' and 'outside' a rectangular annotation region.

    The annotation region covers x ∈ [20, 40], y ∈ [20, 40] (pixel coordinates).
    Particle 0 lives entirely inside; particle 1 entirely outside.
    """
    rows = []
    for f in range(5):
        rows.append({"x": 30.0, "y": 30.0, "frame": f, "particle": 0,
                     "mass": 1000.0, "size": 3.0, "ecc": 0.1})
        rows.append({"x": 80.0, "y": 80.0, "frame": f, "particle": 1,
                     "mass": 1000.0, "size": 3.0, "ecc": 0.1})
    df = pd.DataFrame(rows)
    return df.set_index("frame", drop=False)


@pytest.fixture()
def annotation_filter_df():
    """COCO_range-format filter DataFrame: x ∈ [20, 40], y ∈ [20, 40]."""
    xs = list(range(20, 41))
    return pd.DataFrame({"x": xs, "y": [[(20, 40)]] * len(xs)})


@pytest.fixture()
def synthetic_coco_json(tmp_path):
    """Minimal COCO JSON with a 20×20 rectangular annotation in a 100×100 image."""
    data = {
        "images": [{"id": 1, "file_name": "test.png", "height": 100, "width": 100}],
        "categories": [{"id": 1, "name": "region"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "area": 400.0,
                "iscrowd": 0,
                # Rectangle x∈[20,40], y∈[20,40]
                "segmentation": [[20, 20, 40, 20, 40, 40, 20, 40]],
                "bbox": [20, 20, 20, 20],
            }
        ],
    }
    p = tmp_path / "synthetic.json"
    p.write_text(json.dumps(data))
    return str(p)


@pytest.fixture()
def big_coco_json():
    """Real big_capillary.json from the bundled test data."""
    if not BIG_JSON.exists():
        pytest.skip(f"Test data not found: {BIG_JSON}")
    return str(BIG_JSON)


@pytest.fixture()
def test_image_tif():
    """Path to a real TIFF image from bundled test data."""
    if not TEST_IMAGE_TIF.exists():
        pytest.skip(f"Test image not found: {TEST_IMAGE_TIF}")
    return str(TEST_IMAGE_TIF)
