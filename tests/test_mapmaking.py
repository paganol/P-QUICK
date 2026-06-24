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


def test_accumulate_tqu_local_matches_serial():
    # per-thread parallel accumulator, summed, must equal the serial single-matrix one.
    import numba
    from pquick.mapmaking import accumulate_tqu_local, accumulate_tqu_matrix, init_map_matrix

    nside = 8
    npix = 12 * nside * nside
    rng = np.random.default_rng(0)
    n = 5000
    pix = rng.integers(0, npix, n).astype(np.int64)
    psi = rng.uniform(0, np.pi, n)
    tod = rng.standard_normal(n)

    serial = init_map_matrix(nside)
    accumulate_tqu_matrix(serial, pix, psi, tod, 1.3, rho=0.9)

    local = np.zeros((numba.get_num_threads(), npix, 3, 3))
    accumulate_tqu_local(local, pix, psi, tod, 1.3, rho=0.9)

    assert np.allclose(local.sum(axis=0), serial)
