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


def test_bore_det_to_ptg_masked_matches_plain_on_subset():
    # masked(q, idx) must equal plain(q[idx]) — the flagged-sample fast path.
    from pquick.quaternion import bore_det_to_ptg, bore_det_to_ptg_masked

    rng = np.random.default_rng(0)
    q = normalize_quaternion(rng.standard_normal((50, 4)))
    dq = normalize_quaternion(rng.standard_normal(4))
    idx = np.array([1, 4, 5, 9, 30, 49], dtype=np.int64)

    ptg_m = np.empty((idx.size, 3)); psi_m = np.empty(idx.size)
    bore_det_to_ptg_masked(q, dq, idx, ptg_m, psi_m)

    ptg_p = np.empty((idx.size, 3)); psi_p = np.empty(idx.size)
    bore_det_to_ptg(q[idx], dq, ptg_p, psi_p)

    assert np.allclose(ptg_m, ptg_p) and np.allclose(psi_m, psi_p)
