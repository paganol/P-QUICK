import numpy as np
from astropy.coordinates import BarycentricMeanEcliptic, Galactic, SkyCoord
from astropy import units as u

from pquick.pointing import PointingData, reconstruct_native_pointing
from pquick.quaternion import quat_rotate_vec


def test_reconstruct_native_pointing_basic():
    time_us = np.array([0.0, 1_000_000_000.0], dtype=np.float64)
    quat_us = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)

    p = PointingData(
        time_us=time_us,
        quat_us=quat_us,
        flag=np.array([0, 1, 0], dtype=np.int8),
        sampling_rate_hz=2.0,
    )
    n = reconstruct_native_pointing(p)

    assert n.time_native.size == 3
    assert n.quat_native.shape == (3, 4)
    assert n.flag_native.shape == (3,)


def test_reconstruct_native_pointing_uses_original_indices_for_flag_and_length():
    time_us = np.array([10.0, 14.0, 18.0], dtype=np.float64)
    quat_us = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    p = PointingData(
        time_us=time_us,
        quat_us=quat_us,
        flag=np.array([0, 0, 1, 0, 1], dtype=np.int8),
        sampling_rate_hz=2.0,
        original_indices=np.array([0, 2, 4], dtype=np.int64),
    )
    n = reconstruct_native_pointing(p)

    assert n.time_native.size == 5
    np.testing.assert_array_equal(n.flag_native, np.array([0, 0, 1, 0, 1], dtype=np.int8))


def test_reconstruct_native_pointing_can_convert_to_galactic():
    time_us = np.array([0.0, 1_000_000_000.0], dtype=np.float64)
    quat_us = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    p = PointingData(
        time_us=time_us,
        quat_us=quat_us,
        flag=np.array([0, 0, 0], dtype=np.int8),
        sampling_rate_hz=2.0,
    )

    n = reconstruct_native_pointing(p, coordinate_system="galactic")
    bore = quat_rotate_vec(n.quat_native[0], np.array([0.0, 0.0, 1.0], dtype=np.float64))

    expected = (
        SkyCoord(
            lon=0.0 * u.deg,
            lat=90.0 * u.deg,
            frame=BarycentricMeanEcliptic(),
        )
        .transform_to(Galactic())
        .cartesian.xyz.to_value(u.one)
    )

    np.testing.assert_allclose(bore, expected, atol=1e-12)
