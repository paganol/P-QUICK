import numpy as np

from pquick.mapmaking import accumulate_tqu_matrix, init_map_matrix, solve_tqu_from_matrix


def test_tqu_solver_recovers_signal_on_single_pixel():
    nside = 1
    mat = init_map_matrix(nside)

    psi = np.array([0.0, np.pi / 8.0, np.pi / 4.0, 3.0 * np.pi / 8.0, np.pi / 2.0])
    pix = np.zeros_like(psi, dtype=np.int64)

    t_true, q_true, u_true = 2.0, 0.5, -0.2
    tod = t_true + q_true * np.cos(2.0 * psi) + u_true * np.sin(2.0 * psi)

    accumulate_tqu_matrix(mat, pix, psi, tod, det_weight=3.0)
    t_map, q_map, u_map = solve_tqu_from_matrix(mat)

    assert np.isclose(t_map[0], t_true, atol=1e-10)
    assert np.isclose(q_map[0], q_true, atol=1e-10)
    assert np.isclose(u_map[0], u_true, atol=1e-10)
