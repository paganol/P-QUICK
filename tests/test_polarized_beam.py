import healpy as hp
import numpy as np

from pquick.io import build_polarized_beam_alm


def _analytic_gauss_teb(lmax: int, mmax: int, fwhm_rad: float) -> np.ndarray:
    """Analytic [T, E, B] alm of a circular Gaussian beam (litebird_sim convention).

    T lives at m=0, the polarised response at m=2 with the Challinor exp(2 sigma^2)
    spin-2 factor.  Used as an independent ground truth for the map-based spin-2
    construction in :func:`build_polarized_beam_alm`.
    """
    sig2 = fwhm_rad**2 / (8.0 * np.log(2.0))
    alm = np.zeros((3, hp.Alm.getsize(lmax, mmax)), dtype=np.complex128)
    ell = np.arange(lmax + 1)
    alm[0, hp.Alm.getidx(lmax, ell, 0)] = np.sqrt((2 * ell + 1) / (4 * np.pi)) * np.exp(
        -0.5 * sig2 * ell * (ell + 1)
    )
    ellp = np.arange(2, lmax + 1)
    bpol = np.sqrt((2 * ellp + 1) / (32 * np.pi)) * np.exp(-0.5 * sig2 * ellp * (ellp + 1))
    idx2 = hp.Alm.getidx(lmax, ellp, 2)
    alm[1, idx2] = bpol
    alm[2, idx2] = bpol * 1j
    alm[1:] *= np.exp(2.0 * sig2) * -np.sqrt(2.0)
    return alm


def test_polarized_beam_matches_analytic_gaussian():
    lmax, mmax = 256, 6
    fwhm = np.radians(0.5)

    ref = _analytic_gauss_teb(lmax, mmax, fwhm)
    scalar = ref[0:1].copy()  # circular Gaussian intensity beam as scalar input

    # nside well above lmax so the intermediate pixelisation is negligible.
    teb = build_polarized_beam_alm(scalar, psi_pol_rad=0.0, lmax=lmax, mmax=mmax, nside=512)

    # T is taken directly from the input alm, so it is preserved exactly.
    assert np.max(np.abs(teb[0] - ref[0])) < 1e-12

    # E/B profiles match the analytic spin-2 beam to <0.5% and flat in ell
    # (residual is the analytic constant exp(2 sigma^2) vs the exact transform).
    ellp = np.arange(2, lmax + 1)
    idx2 = hp.Alm.getidx(lmax, ellp, 2)
    ratio_e = teb[1, idx2] / ref[1, idx2]
    # ponytail: 3e-3 not 1e-3 -- inherent scatter is ~1e-3, threaded ducc0 SHT tips a
    # 1e-3 bound over run-to-run; tighten only if the spin-2 build gets deterministic.
    assert np.std(ratio_e.real) < 3e-3
    assert abs(ratio_e.real.mean() - 1.0) < 5e-3
    assert np.max(np.abs(ratio_e.imag)) < 1e-3

    # B beam = i * E beam (ideal co-polar, no V) to pixelisation accuracy.
    rel = np.max(np.abs(teb[2, idx2] - 1j * teb[1, idx2])) / np.max(np.abs(teb[1, idx2]))
    assert rel < 2e-3


def test_intensity_only_when_axis_drops_out():
    # m=2 pol power is ~1/4 of T power for an ideal co-polar beam, independent of
    # psi_pol (a frame rotation), confirming the construction is well-normalised.
    lmax, mmax = 128, 6
    ref = _analytic_gauss_teb(lmax, mmax, np.radians(0.8))
    scalar = ref[0:1].copy()
    p0 = build_polarized_beam_alm(scalar, 0.0, lmax, mmax)
    p30 = build_polarized_beam_alm(scalar, np.radians(30.0), lmax, mmax)
    r0 = (np.abs(p0[1]) ** 2).sum() / (np.abs(p0[0]) ** 2).sum()
    r30 = (np.abs(p30[1]) ** 2).sum() / (np.abs(p30[0]) ** 2).sum()
    assert abs(r0 - 0.25) < 0.02
    assert abs(r0 - r30) < 1e-3


def test_accumulate_tqu_rho_leaves_temperature_unchanged():
    """rho (cross-pol) must not change the recovered I/temperature, only Q/U."""
    import numpy as np
    from pquick.mapmaking import accumulate_tqu_matrix, solve_tqu_from_matrix

    rng = np.random.default_rng(0)
    n = 20000
    npix = 12
    pix = rng.integers(0, npix, size=n)
    psi = rng.uniform(0, np.pi, size=n)          # full angle coverage per pixel
    I, Q, U = 5.0, 1.3, -0.7
    rho = 0.9
    # Simulate an ideal-pointing TOD of a polarised sky seen through efficiency rho.
    tod = I + rho * (Q * np.cos(2 * psi) + U * np.sin(2 * psi))

    m = np.zeros((npix, 3, 3))
    accumulate_tqu_matrix(m, pix, psi, tod, det_weight=1.0, rho=rho)
    t, q, u = solve_tqu_from_matrix(m)
    good = t > -1e29
    assert np.allclose(t[good], I, atol=1e-6)
    assert np.allclose(q[good], Q, atol=1e-6)
    assert np.allclose(u[good], U, atol=1e-6)

    # With rho=1 (ideal) but the same polarised TOD, I is still recovered exactly
    # (temperature is rho-independent), while Q/U are biased by 1/rho.
    m1 = np.zeros((npix, 3, 3))
    accumulate_tqu_matrix(m1, pix, psi, tod, det_weight=1.0, rho=1.0)
    t1, q1, u1 = solve_tqu_from_matrix(m1)
    assert np.allclose(t1[t1 > -1e29], I, atol=1e-6)


def test_polarized_beam_recovers_e_mode_and_psb_arms_agree():
    """A pure-E sky convolved with the synthesised beam recovers Q/U, and the two
    PSB arms of a horn (psi_uv ~90 deg apart) recover identical polarization."""
    import healpy as hp
    import numpy as np
    from pquick.convolution import build_convolution_interpolator, evaluate_convolution
    from pquick.io import build_polarized_beam_alm, normalize_beam_alm
    from pquick.mapmaking import accumulate_tqu_matrix, solve_tqu_from_matrix

    lmax, mmax, nside = 192, 4, 128
    # A near-Gaussian scalar intensity beam (m=0) as the input blm.
    sig = np.radians(0.5) / np.sqrt(8 * np.log(2))
    ell = np.arange(lmax + 1)
    bl = np.exp(-0.5 * ell * (ell + 1) * sig**2)
    scal = np.zeros((1, hp.Alm.getsize(lmax, mmax)), dtype=np.complex128)
    scal[0, hp.Alm.getidx(lmax, ell, 0)] = bl * np.sqrt((2 * ell + 1) / (4 * np.pi))
    scal = normalize_beam_alm(scal)

    # pure-E sky (seeded for determinism)
    np.random.seed(0)
    cle = np.zeros(lmax + 1)
    cle[2:] = (np.arange(2, lmax + 1) / 50.0) ** -2.0
    almE = hp.synalm(cle, lmax=lmax)
    z = np.zeros_like(almE)
    sky = np.array([z, almE, z])
    # Beam-smoothed input Q/U: the convolved-then-solved map recovers the *smoothed*
    # sky, so compare against that (not the un-smoothed input) to avoid a single-pixel
    # smoothed-vs-unsmoothed mismatch.
    Im, Qm, Um = hp.alm2map([sky[0], sky[1], sky[2]], nside, lmax=lmax, pol=True)
    _, Qm, Um = hp.smoothing(np.array([Im, Qm, Um]), fwhm=np.radians(0.5), pol=True)

    pix = hp.ang2pix(nside, np.pi / 2, 1.0)
    th, ph = hp.pix2ang(nside, pix)
    psis = np.linspace(0, np.pi, 48, endpoint=False)

    def recover(psi_uv_deg, psi_pol_deg):
        beam = normalize_beam_alm(
            build_polarized_beam_alm(
                scal, psi_pol_rad=np.radians(psi_pol_deg), lmax=lmax, mmax=mmax,
                psi_uv_rad=np.radians(psi_uv_deg),
            )
        )
        M = np.zeros((hp.nside2npix(nside), 3, 3))
        for ps in psis:
            tod = evaluate_convolution(
                build_convolution_interpolator(sky, beam, lmax, mmax, npoints=1),
                np.array([[th, ph, ps]]),
            )
            accumulate_tqu_matrix(M, np.array([pix]), np.array([ps]), np.asarray(tod), 1.0)
        I, Q, U = solve_tqu_from_matrix(M)
        return I[pix], Q[pix], U[pix]

    Ia, Qa, Ua = recover(23.1, 0.22)    # 100-1a
    Ib, Qb, Ub = recover(-68.2, 0.13)   # 100-1b (psi_uv ~90 deg from a)

    # recovered Q/U match the beam-smoothed input sky (apples-to-apples), not cancelled
    assert np.sign(Qa) == np.sign(Qm[pix]) and np.sign(Ua) == np.sign(Um[pix])
    assert abs(Qa / Qm[pix] - 1.0) < 0.02 and abs(Ua / Um[pix] - 1.0) < 0.02
    # the two arms agree (the bug made them cancel / disagree)
    assert abs(Qa - Qb) < 1e-3 * abs(Qa) and abs(Ua - Ub) < 1e-3 * abs(Ua)
