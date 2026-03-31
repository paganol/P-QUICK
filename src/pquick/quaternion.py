from __future__ import annotations

import numpy as np


def normalize_quaternion(q: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    nrm = np.linalg.norm(q, axis=-1, keepdims=True)
    nrm = np.maximum(nrm, eps)
    return q / nrm


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.moveaxis(q1, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    qq = np.asarray(q, dtype=np.float64).copy()
    qq[..., :3] *= -1.0
    return qq


def quat_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    q = normalize_quaternion(np.asarray(q, dtype=np.float64))
    v_quat = np.concatenate([v, np.zeros(v.shape[:-1] + (1,), dtype=np.float64)], axis=-1)
    return quat_mul(quat_mul(q, v_quat), quat_conj(q))[..., :3]


def slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)

    flip = dot < 0
    q1 = np.where(flip, -q1, q1)
    dot = np.abs(dot)

    linear = dot > (1.0 - eps)
    omega = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_omega = np.sin(omega)

    t = t[..., None]
    w0 = np.sin((1.0 - t) * omega) / np.where(sin_omega == 0, 1.0, sin_omega)
    w1 = np.sin(t * omega) / np.where(sin_omega == 0, 1.0, sin_omega)
    out = w0 * q0 + w1 * q1

    lin_out = (1.0 - t) * q0 + t * q1
    out = np.where(linear, lin_out, out)
    return normalize_quaternion(out)


def upsample_quaternions(
    coarse_t: np.ndarray,
    coarse_q: np.ndarray,
    fine_t: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    coarse_t = np.asarray(coarse_t, dtype=np.float64)
    coarse_q = normalize_quaternion(np.asarray(coarse_q, dtype=np.float64))
    fine_t = np.asarray(fine_t, dtype=np.float64)

    if coarse_t.ndim != 1 or fine_t.ndim != 1:
        raise ValueError("time arrays must be 1D")
    if coarse_q.shape != (coarse_t.size, 4):
        raise ValueError("coarse_q must have shape (N, 4)")
    if np.any(np.diff(coarse_t) <= 0):
        raise ValueError("coarse_t must be strictly increasing")

    idx_right = np.searchsorted(coarse_t, fine_t, side="right")
    idx_right = np.clip(idx_right, 1, coarse_t.size - 1)
    idx_left = idx_right - 1

    t0 = coarse_t[idx_left]
    t1 = coarse_t[idx_right]
    alpha = np.clip((fine_t - t0) / np.maximum(t1 - t0, 1e-30), 0.0, 1.0)

    return slerp(coarse_q[idx_left], coarse_q[idx_right], alpha, eps=eps)


def quaternion_to_thetaphipsi(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = normalize_quaternion(np.asarray(q, dtype=np.float64))

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    bore = quat_rotate_vec(q, np.broadcast_to(z_axis, q.shape[:-1] + (3,)))
    x_rot = quat_rotate_vec(q, np.broadcast_to(x_axis, q.shape[:-1] + (3,)))

    theta = np.arccos(np.clip(bore[..., 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(bore[..., 1], bore[..., 0]), 2.0 * np.pi)

    e_theta = np.stack(
        [
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi),
            -np.sin(theta),
        ],
        axis=-1,
    )
    e_phi = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=-1)

    x_theta = np.sum(x_rot * e_theta, axis=-1)
    x_phi = np.sum(x_rot * e_phi, axis=-1)
    psi = np.arctan2(x_phi, x_theta)
    return theta, phi, psi
