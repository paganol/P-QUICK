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
    assert np.std(ratio_e.real) < 1e-3
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
