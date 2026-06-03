from pathlib import Path

import numpy as np

from pquick.config import DetectorSelection
from pquick.io import load_beam_alm, load_pointing_npz, normalize_beam_alm
from pquick.io import select_detectors


def test_load_planck_beam_table_with_crop():
    beam = Path("inputs/beams/blm_100-1a.fits")
    alm = load_beam_alm(beam, lmax=16, mmax=6)

    expected_size = (6 + 1) * (16 + 1) - (6 * 7) // 2
    assert alm.shape == (1, expected_size)
    assert np.iscomplexobj(alm)
    assert np.count_nonzero(np.abs(alm[0]) > 0) > 0


def test_load_planck_beam_table_rejects_too_large_mmax():
    beam = Path("inputs/beams/blm_100-1a.fits")

    try:
        load_beam_alm(beam, lmax=16, mmax=200)
    except ValueError as exc:
        assert "exceeds beam mmax" in str(exc)
    else:
        raise AssertionError("Expected ValueError for oversized mmax")


def test_normalize_beam_alm_unit_integral_sets_b00_to_standard_constant_sky_response():
    beam = Path("inputs/beams/blm_100-1a.fits")
    alm = load_beam_alm(beam, lmax=16, mmax=6)

    norm = normalize_beam_alm(alm, mode="unit_integral")

    assert np.isclose(norm[0, 0].real, 1.0 / np.sqrt(4.0 * np.pi), rtol=0.0, atol=1e-12)
    assert np.isclose(norm[0, 0].imag, 0.0, rtol=0.0, atol=1e-15)


def test_normalize_beam_alm_raw_leaves_coefficients_unchanged():
    alm = np.array([[2.0 + 0.0j, 1.0 - 3.0j]], dtype=np.complex128)

    out = normalize_beam_alm(alm, mode="raw")

    np.testing.assert_array_equal(out, alm)


def test_load_pointing_npz_accepts_single_flag_schema(tmp_path: Path):
    f = tmp_path / "pointing_single_flag.npz"
    np.savez_compressed(
        f,
        t0_ns=np.array([0.0], dtype=np.float64),
        qx=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        qy=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        qz=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        qs=np.array([1.0, 1.0, 1.0], dtype=np.float64),
        flag=np.array([0, 1, 0], dtype=np.int8),
        sampling_rate_hz=np.array([10.0], dtype=np.float64),
        idx_first=np.array([0], dtype=np.int64),
        idx_last=np.array([2], dtype=np.int64),
        idx_step=np.array([1], dtype=np.int64),
    )

    p = load_pointing_npz(f)
    np.testing.assert_array_equal(p.flag, np.array([0, 1, 0], dtype=np.int8))


def test_load_pointing_npz_defaults_to_zero_flag_without_flag_key(tmp_path: Path):
    f = tmp_path / "pointing_missing_flag_key.npz"
    np.savez_compressed(
        f,
        t0_ns=np.array([0.0], dtype=np.float64),
        qx=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        qy=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        qz=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        qs=np.array([1.0, 1.0, 1.0], dtype=np.float64),
        sampling_rate_hz=np.array([10.0], dtype=np.float64),
        idx_first=np.array([0], dtype=np.int64),
        idx_last=np.array([2], dtype=np.int64),
        idx_step=np.array([1], dtype=np.int64),
    )

    p = load_pointing_npz(f)
    np.testing.assert_array_equal(p.flag, np.array([0, 0, 0], dtype=np.int8))


def test_select_detectors_rejects_channel_and_detectors_together():
    all_detectors = ["100-1a", "100-1b", "100-2a"]
    selection = DetectorSelection(channel="100ghz", detectors=["100-1a"])

    try:
        select_detectors(all_detectors, selection)
    except ValueError as exc:
        assert "only one" in str(exc)
    else:
        raise AssertionError("Expected ValueError when both channel and detectors are set")
