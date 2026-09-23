#!/usr/bin/env python3
"""Tests for pmpiv.interface.Multi_Interface_Tracking."""
import numpy as np
import pandas as pd
import pytest

import pmpiv
from pmpiv import interface, df_io
from conftest import _synthetic_channel_image


WALLS = ((100, 112), (400, 412), (700, 712))
ARCS = ({"cx": 256, "cy": 150, "r": 200, "thickness": 20},
        {"cx": 556, "cy": 150, "r": 200, "thickness": 20})


class _CountingSequence:
    def __init__(self, frames):
        self.frames = frames
        self.reads = 0

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        self.reads += 1
        return self.frames[i]


def _moving_sequence(n_frames=8, dy=1.0):
    frames = []
    for f in range(n_frames):
        arcs = tuple(dict(a, cy=a["cy"] + dy * f) for a in ARCS)
        frames.append(_synthetic_channel_image(walls=WALLS, arcs=arcs, beads_in=(0, 1), seed=f))
    return frames


@pytest.fixture()
def tracked(md):
    frames = _moving_sequence()
    seq = _CountingSequence(frames)
    mt = interface.Multi_Interface_Tracking(m_metadata=md, discovery="walls", band_height=150,
                                            verbose=False, savgol_window=5)
    mt.discover(frames[0])
    dfs = mt.track(seq)
    return mt, seq, dfs


class TestMultiInterfaceTracking:
    def test_discovers_two_trackers(self, tracked):
        mt, _, _ = tracked
        assert set(mt.trackers) == {"ch0", "ch1"}

    def test_frames_read_once(self, tracked):
        _, seq, _ = tracked
        assert seq.reads == len(seq)

    def test_all_frames_ok(self, tracked):
        _, seq, dfs = tracked
        for label, df in dfs.items():
            assert len(df) == len(seq)
            assert (df["status"] == "ok").all(), df[["frame", "status", "n_valid", "snr", "residual_px"]]

    def test_to_dataframe_long_format(self, tracked):
        mt, _, _ = tracked
        mt.compute_velocity()
        long = mt.to_dataframe()
        assert long.columns[0] == "interface"
        assert set(long["interface"]) == {"ch0", "ch1"}
        assert "velocity_m_s" in long.columns

    def test_write_csv_roundtrip(self, tracked, tmp_path):
        mt, _, _ = tracked
        mt.compute_velocity()
        mt.write_csv(str(tmp_path), "multi.csv")
        back = df_io.read_csv(str(tmp_path), "multi.csv")
        assert set(back["interface"]) == {"ch0", "ch1"}
        assert len(back) == len(mt.to_dataframe())

    def test_plot_annotated_tiffs(self, tracked, tmp_path):
        mt, seq, _ = tracked
        out = tmp_path / "tiffs"
        mt.plot_annotated_tiffs(str(out), seq, parallel=False)
        assert len(list(out.glob("frame_*.tif"))) == len(seq)

    def test_track_before_discover_raises(self, md):
        mt = interface.Multi_Interface_Tracking(m_metadata=md, verbose=False)
        with pytest.raises(ValueError):
            mt.track([np.zeros((10, 10), dtype=np.uint8)])

    def test_add_interface_manual(self, md):
        frames = _moving_sequence(n_frames=3)
        mt = interface.Multi_Interface_Tracking(m_metadata=md, verbose=False)
        mt.add_interface("manual", (420, 692), (250, 400), wall_x=(412, 700))
        dfs = mt.track(frames)
        assert list(dfs) == ["manual"]
        assert (dfs["manual"]["status"] == "ok").all()
