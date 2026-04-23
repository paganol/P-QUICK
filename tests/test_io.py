from pathlib import Path

import numpy as np

from pquick.io import load_beam_alm


def test_load_planck_beam_table_with_crop():
    beam = Path("inputs/beams/blm_100-1a.fits")
    alm = load_beam_alm(beam, lmax=16, kmax=6)

    expected_size = (6 + 1) * (16 + 1) - (6 * 7) // 2
    assert alm.shape == (1, expected_size)
    assert np.iscomplexobj(alm)
    assert np.count_nonzero(np.abs(alm[0]) > 0) > 0


def test_load_planck_beam_table_rejects_too_large_kmax():
    beam = Path("inputs/beams/blm_100-1a.fits")

    try:
        load_beam_alm(beam, lmax=16, kmax=200)
    except ValueError as exc:
        assert "exceeds beam kmax" in str(exc)
    else:
        raise AssertionError("Expected ValueError for oversized kmax")