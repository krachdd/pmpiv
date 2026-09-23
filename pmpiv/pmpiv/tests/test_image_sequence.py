#!/usr/bin/env python3
"""Tests for pmpiv.image_sequence.Image_Sequence."""
import numpy as np
from PIL import Image
import pytest

import pmpiv
from pmpiv import image_sequence
from conftest import _write_metadata


def _write_tif_sequence(directory, n_frames=10, size=(20, 20)):
    """Write n_frames distinguishable single-page TIFFs into directory."""
    for f in range(n_frames):
        arr = np.full(size, fill_value=f, dtype=np.uint8)
        Image.fromarray(arr).save(directory / f'frame_{f:04d}.tif')


class TestSubsectionRange:
    def test_end_frame_zero_uses_full_length(self, tmp_path):
        _write_tif_sequence(tmp_path, n_frames=10)
        cfg = _write_metadata(tmp_path, IN_PATH=str(tmp_path), START_FRAME='0', END_FRAME='0')
        md = pmpiv.metadata.Metadata(str(cfg))

        seq = image_sequence.Image_Sequence(m_metadata=md)
        result = seq.subsection_range()

        assert len(result) == 10

    def test_start_frame_applied_when_end_frame_zero(self, tmp_path):
        # Regression test: START_FRAME must not be silently ignored just
        # because END_FRAME == 0 ("use to the end").
        _write_tif_sequence(tmp_path, n_frames=10)
        cfg = _write_metadata(tmp_path, IN_PATH=str(tmp_path), START_FRAME='3', END_FRAME='0')
        md = pmpiv.metadata.Metadata(str(cfg))

        seq = image_sequence.Image_Sequence(m_metadata=md)
        result = seq.subsection_range()

        assert len(result) == 7
        assert np.asarray(result[0])[0, 0] == 3

    def test_start_and_end_frame_applied(self, tmp_path):
        _write_tif_sequence(tmp_path, n_frames=10)
        cfg = _write_metadata(tmp_path, IN_PATH=str(tmp_path), START_FRAME='2', END_FRAME='5')
        md = pmpiv.metadata.Metadata(str(cfg))

        seq = image_sequence.Image_Sequence(m_metadata=md)
        result = seq.subsection_range()

        assert len(result) == 3
        assert np.asarray(result[0])[0, 0] == 2
