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


def test_temperature_only_solve_recovers_I_where_3x3_rejects():
    # An unpolarized detector (rho=0) gives a singular Q/U block: the 3x3 solve masks
    # every pixel, while the temperature-only solve recovers I = weighted mean of d.
    from pquick.mapmaking import accumulate_tqu_matrix, init_map_matrix, solve_tqu_from_matrix

    rng = np.random.default_rng(0)
    n, npix = 5000, 12
    pix = rng.integers(0, npix, n).astype(np.int64)
    psi = rng.uniform(0, np.pi, n)
    I = 4.0
    tod = np.full(n, I)                      # pure temperature signal, rho=0
    m = init_map_matrix(1)                   # nside=1 -> npix=12
    accumulate_tqu_matrix(m, pix, psi, tod, det_weight=2.0, rho=0.0)

    t3, q3, u3 = solve_tqu_from_matrix(m)                       # 3x3: all rejected
    assert np.all(t3 < -1e29)
    t1, q1, u1 = solve_tqu_from_matrix(m, temperature_only=True)
    hit = m[:, 0, 0] > 0
    assert np.allclose(t1[hit], I, atol=1e-9)                  # I recovered
    assert np.all(q1 < -1e29) and np.all(u1 < -1e29)           # Q/U left UNSEEN


def test_add_hits_matches_bincount():
    # persistent serial scatter must equal np.bincount, including accumulation across calls.
    from pquick.mapmaking import add_hits

    npix = 50
    rng = np.random.default_rng(1)
    hits = np.zeros(npix, np.int64)
    ref = np.zeros(npix, np.int64)
    for _ in range(3):
        pix = rng.integers(0, npix, 200).astype(np.int64)
        add_hits(hits, pix)
        ref += np.bincount(pix, minlength=npix)
    assert np.array_equal(hits, ref)
