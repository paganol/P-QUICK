import numpy as np

from pquick.mapmaking import accumulate_simple_iqu, finalize_simple_iqu


def test_weighted_accumulation_changes_denominator():
    theta = np.array([0.5, 0.5])
    phi = np.array([1.0, 1.0])
    psi = np.array([0.0, 0.0])
    tod = np.array([1.0, 3.0])
    flags = np.array([0, 0], dtype=np.int8)

    a1 = accumulate_simple_iqu(theta, phi, psi, tod, flags, nside=1, det_weight=2.0)
    maps = finalize_simple_iqu(a1)

    # With constant psi=0 and same pixel, weighted I should be arithmetic mean of TOD.
    assert np.nanmax(maps["I"]) == 2.0
    assert np.nanmax(maps["hits"]) == 2
