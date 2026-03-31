import numpy as np

from pquick.pointing import PointingData, reconstruct_native_pointing


def test_reconstruct_native_pointing_basic():
    time_us = np.array([0.0, 1_000_000_000.0], dtype=np.float64)
    quat_us = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)

    p = PointingData(
        time_us=time_us,
        quat_us=quat_us,
        flag_ext1=np.array([0, 1], dtype=np.int8),
        flag_ext3=np.array([0, 0], dtype=np.int8),
        sampling_rate_hz=2.0,
    )
    n = reconstruct_native_pointing(p)

    assert n.time_native.size == 3
    assert n.quat_native.shape == (3, 4)
    assert n.flag_native.shape == (3,)
