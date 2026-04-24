from __future__ import annotations

import numpy as np


def normalize_quaternion(q: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Divide each quaternion by its norm, clamping the denominator to *eps*.

    Supports both single quaternions ``(4,)`` and batches ``(..., 4)``.

    Args:
        q: Input quaternion(s) in ``(x, y, z, w)`` order.
        eps: Minimum norm value to avoid division by zero.

    Returns:
        Unit quaternion(s) with the same shape as *q*.
    """
    q = np.asarray(q, dtype=np.float64)
    nrm = np.linalg.norm(q, axis=-1, keepdims=True)
    nrm = np.maximum(nrm, eps)
    return q / nrm


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Compute the Hamilton product of two (batches of) quaternions.

    Args:
        q1: First quaternion(s), shape ``(..., 4)`` in ``(x, y, z, w)`` order.
        q2: Second quaternion(s), same shape as *q1*.

    Returns:
        Product quaternion(s), shape ``(..., 4)``.
    """
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
    """Return the conjugate of a quaternion (or batch) by negating the vector part.

    Args:
        q: Quaternion(s), shape ``(..., 4)`` in ``(x, y, z, w)`` order.

    Returns:
        Conjugate quaternion(s) with the same shape.
    """
    qq = np.asarray(q, dtype=np.float64).copy()
    qq[..., :3] *= -1.0
    return qq


def quat_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3-vector(s) *v* by quaternion(s) *q* using the sandwich product ``q v q*``.

    Args:
        q: Rotation quaternion(s), shape ``(..., 4)``.
        v: 3-vector(s) to rotate, shape ``(..., 3)``.

    Returns:
        Rotated 3-vector(s), shape ``(..., 3)``.
    """
    v = np.asarray(v, dtype=np.float64)
    q = normalize_quaternion(np.asarray(q, dtype=np.float64))
    v_quat = np.concatenate([v, np.zeros(v.shape[:-1] + (1,), dtype=np.float64)], axis=-1)
    return quat_mul(quat_mul(q, v_quat), quat_conj(q))[..., :3]


def slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Spherical-linear interpolation between pairs of unit quaternions.

    Falls back to normalised linear interpolation when the two quaternions are
    nearly identical (dot product close to 1).

    Args:
        q0: Start quaternion(s), shape ``(..., 4)``.
        q1: End quaternion(s), shape ``(..., 4)``.
        t: Interpolation parameter(s) in ``[0, 1]``, shape ``(...,)``.
        eps: Threshold below which quaternions are treated as identical.

    Returns:
        Interpolated unit quaternion(s), shape ``(..., 4)``.
    """
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
    """Resample a coarse quaternion time series onto a fine time grid via SLERP.

    For each fine time stamp, the two nearest coarse neighbours are located by
    binary search and the quaternion is interpolated with :func:`slerp`.

    Args:
        coarse_t: Coarse time grid, strictly increasing 1-D array.
        coarse_q: Coarse quaternions, shape ``(N, 4)``.
        fine_t: Target time grid, 1-D array.
        eps: SLERP near-identity threshold.

    Returns:
        Unit quaternions at the fine grid points, shape ``(M, 4)``.

    Raises:
        ValueError: If input arrays have inconsistent shapes or non-monotone time grids.
    """
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
    """Convert unit quaternions to HEALPix sky angles ``(theta, phi, psi)``.

    Uses a direct closed-form expression in the quaternion components instead of
    repeated ``quat_mul`` calls, avoiding all intermediate quaternion temporaries.

    For unit quaternion ``q = (qx, qy, qz, qw)`` the rotation matrix column vectors are:

    * boresight ``R @ z``: ``(2(qx*qz + qy*qw),  2(qy*qz - qx*qw),  1 - 2(qx² + qy²))``
    * x-axis    ``R @ x``: ``(1 - 2(qy² + qz²),  2(qx*qy + qz*qw),  2(qx*qz - qy*qw))``

    Args:
        q: Unit quaternions, shape ``(N, 4)`` in ``(x, y, z, w)`` order.

    Returns:
        Tuple ``(theta, phi, psi)`` of float64 arrays, each of shape ``(N,)``, in radians.
    """
    q = normalize_quaternion(np.asarray(q, dtype=np.float64))
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Boresight direction: R(q) @ [0, 0, 1]
    bx = 2.0 * (qx * qz + qy * qw)
    by = 2.0 * (qy * qz - qx * qw)
    bz = 1.0 - 2.0 * (qx * qx + qy * qy)

    theta = np.arccos(np.clip(bz, -1.0, 1.0))
    phi = np.mod(np.arctan2(by, bx), 2.0 * np.pi)

    # X-axis direction: R(q) @ [1, 0, 0]
    xx = 1.0 - 2.0 * (qy * qy + qz * qz)
    xy = 2.0 * (qx * qy + qz * qw)
    xz = 2.0 * (qx * qz - qy * qw)

    # Local frame at (theta, phi); use boresight components to avoid extra trig calls.
    # sin(theta) = sqrt(bx² + by²); guard against the pole (sin_th → 0).
    sin_th = np.sqrt(np.maximum(bx * bx + by * by, 0.0))
    safe = sin_th > 0.0
    inv_sin_th = np.where(safe, 1.0 / np.where(safe, sin_th, 1.0), 0.0)

    # cos(phi) = bx / sin_th,  sin(phi) = by / sin_th
    cos_ph = bx * inv_sin_th
    sin_ph = by * inv_sin_th

    # e_theta = [bz*cos_ph, bz*sin_ph, -sin_th]
    # e_phi   = [-sin_ph, cos_ph, 0]
    x_theta = xx * bz * cos_ph + xy * bz * sin_ph - xz * sin_th
    x_phi = -xx * sin_ph + xy * cos_ph

    psi = np.arctan2(x_phi, x_theta)
    return theta, phi, psi
