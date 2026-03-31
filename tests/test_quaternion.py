import numpy as np

from pquick.quaternion import normalize_quaternion, slerp, upsample_quaternions


def test_normalize_quaternion_unit_norm():
    q = np.array([[0.0, 0.0, 0.0, 2.0], [1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    out = normalize_quaternion(q)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0)


def test_slerp_endpoints():
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = np.array([0.0, 0.0, 1.0, 0.0])

    a = slerp(q0, q1, np.array(0.0))
    b = slerp(q0, q1, np.array(1.0))

    assert np.allclose(a, q0)
    assert np.allclose(b, q1)


def test_upsample_shape_and_norm():
    t_coarse = np.array([0.0, 10.0])
    q_coarse = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]])
    t_fine = np.linspace(0.0, 10.0, 21)

    q_fine = upsample_quaternions(t_coarse, q_coarse, t_fine)
    assert q_fine.shape == (21, 4)
    assert np.allclose(np.linalg.norm(q_fine, axis=1), 1.0)
