from __future__ import annotations

import healpy as hp
import numpy as np


def init_map_matrix(nside: int) -> np.ndarray:
    """Allocate a zero-filled ``(npix, 3, 3)`` normal-equation matrix accumulator.

    Args:
        nside: HEALPix resolution parameter.

    Returns:
        Float64 array of shape ``(hp.nside2npix(nside), 3, 3)`` initialised to zero.
    """
    npix = hp.nside2npix(nside)
    return np.zeros((npix, 3, 3), dtype=np.float64)


def accumulate_tqu_matrix(
    matrix: np.ndarray,
    pix: np.ndarray,
    psi: np.ndarray,
    tod: np.ndarray,
    det_weight: float,
) -> None:
    """Accumulate a detector's TOD into the polarised normal-equation matrix in-place.

    The upper 2×2 block stores the weighted pointing matrix ``A^T N^{-1} A`` and the
    lower-left column stores the weighted RHS ``A^T N^{-1} d``, all indexed by pixel.

    Args:
        matrix: Accumulator array of shape ``(npix, 3, 3)`` from :func:`init_map_matrix`.
        pix: HEALPix pixel indices for each sample, shape ``(N,)``.
        psi: Polarisation angles in radians, shape ``(N,)``.
        tod: Convolved signal samples, shape ``(N,)``.
        det_weight: Inverse-noise weight for this detector.
    """
    if pix.size == 0:
        return

    w = float(det_weight)
    c2 = np.cos(2.0 * psi)
    s2 = np.sin(2.0 * psi)
    wy = w * tod

    # Upper triangle: normal matrix A^T N^-1 A
    np.add.at(matrix[:, 0, 0], pix, w)
    np.add.at(matrix[:, 0, 1], pix, w * c2)
    np.add.at(matrix[:, 0, 2], pix, w * s2)
    np.add.at(matrix[:, 1, 1], pix, w * c2 * c2)
    np.add.at(matrix[:, 1, 2], pix, w * c2 * s2)
    np.add.at(matrix[:, 2, 2], pix, w * s2 * s2)

    # Lower triangle: RHS A^T N^-1 d
    np.add.at(matrix[:, 1, 0], pix, wy)
    np.add.at(matrix[:, 2, 0], pix, wy * c2)
    np.add.at(matrix[:, 2, 1], pix, wy * s2)


def solve_tqu_from_matrix(
    matrix: np.ndarray,
    cond_threshold: float = 1e10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the per-pixel 3×3 polarised map-making equation to recover T, Q, U maps.

    Pixels with no hits or a poorly conditioned normal matrix are set to ``hp.UNSEEN``.

    Args:
        matrix: Accumulated normal-equation array of shape ``(npix, 3, 3)`` from
            :func:`accumulate_tqu_matrix`.
        cond_threshold: Pixels whose matrix condition number exceeds this value are masked.

    Returns:
        Tuple ``(t_map, q_map, u_map)`` of float64 HEALPix maps.
    """
    npix = matrix.shape[0]
    t_map = np.full(npix, hp.UNSEEN, dtype=np.float64)
    q_map = np.full(npix, hp.UNSEEN, dtype=np.float64)
    u_map = np.full(npix, hp.UNSEEN, dtype=np.float64)

    hit_idx = np.where(matrix[:, 0, 0] > 0)[0]
    if hit_idx.size == 0:
        return t_map, q_map, u_map

    # Work on a contiguous copy of only the hit pixels.
    A = matrix[hit_idx].copy()  # (n_hit, 3, 3)

    # Extract the RHS from the lower-left triangle before symmetrising.
    rhs = np.stack([A[:, 1, 0], A[:, 2, 0], A[:, 2, 1]], axis=1)  # (n_hit, 3)

    # Symmetrize the normal matrices in-place.
    A[:, 1, 0] = A[:, 0, 1]
    A[:, 2, 0] = A[:, 0, 2]
    A[:, 2, 1] = A[:, 1, 2]

    # Vectorised condition check; reject ill-conditioned pixels.
    cond = np.linalg.cond(A)  # (n_hit,)
    good = cond < cond_threshold
    if not np.any(good):
        return t_map, q_map, u_map

    sol = np.linalg.solve(A[good], rhs[good, :, np.newaxis]).squeeze(-1)  # (n_good, 3)
    idx_good = hit_idx[good]
    t_map[idx_good] = sol[:, 0]
    q_map[idx_good] = sol[:, 1]
    u_map[idx_good] = sol[:, 2]

    return t_map, q_map, u_map


def accumulate_simple_iqu(
    theta: np.ndarray,
    phi: np.ndarray,
    psi: np.ndarray,
    tod: np.ndarray,
    flags: np.ndarray,
    nside: int,
    det_weight: float,
    nest: bool = False,
) -> dict[str, np.ndarray]:
    """Accumulate weighted I/Q/U numerators and denominators into per-pixel bins.

    Flagged samples (``flags != 0``) are skipped.  Returns a dict of running
    accumulators that can be combined across detectors and passed to
    :func:`finalize_simple_iqu`.

    Args:
        theta: Co-latitude angles in radians, shape ``(N,)``.
        phi: Longitude angles in radians, shape ``(N,)``.
        psi: Polarisation angles in radians, shape ``(N,)``.
        tod: Signal samples, shape ``(N,)``.
        flags: Quality flags; samples with ``flags != 0`` are ignored.
        nside: HEALPix resolution parameter.
        det_weight: Inverse-noise weight for this detector.
        nest: If ``True``, use NESTED pixel ordering; otherwise RING.

    Returns:
        Dict with keys ``i_num``, ``q_num``, ``u_num``, ``i_den``, ``hits``, ``wpol``.
    """
    npix = hp.nside2npix(nside)
    i_num = np.zeros(npix, dtype=np.float64)
    q_num = np.zeros(npix, dtype=np.float64)
    u_num = np.zeros(npix, dtype=np.float64)
    i_den = np.zeros(npix, dtype=np.float64)
    w_pol = np.zeros(npix, dtype=np.float64)
    hits = np.zeros(npix, dtype=np.int64)

    good = np.asarray(flags) == 0
    if not np.any(good):
        return {
            "i_num": i_num,
            "q_num": q_num,
            "u_num": u_num,
            "i_den": i_den,
            "hits": hits,
            "wpol": w_pol,
        }

    th = np.asarray(theta, dtype=np.float64)[good]
    ph = np.asarray(phi, dtype=np.float64)[good]
    ps = np.asarray(psi, dtype=np.float64)[good]
    y = np.asarray(tod, dtype=np.float64)[good]

    pix = hp.ang2pix(nside, th, ph, nest=nest)
    c2 = np.cos(2.0 * ps)
    s2 = np.sin(2.0 * ps)
    w = float(det_weight)

    np.add.at(i_num, pix, w * y)
    np.add.at(q_num, pix, w * y * c2)
    np.add.at(u_num, pix, w * y * s2)
    np.add.at(i_den, pix, w)
    np.add.at(w_pol, pix, w * (c2 * c2 + s2 * s2))
    np.add.at(hits, pix, 1)

    return {
        "i_num": i_num,
        "q_num": q_num,
        "u_num": u_num,
        "i_den": i_den,
        "hits": hits,
        "wpol": w_pol,
    }


def finalize_simple_iqu(acc: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Divide accumulated numerators by denominators to produce final I, Q, U maps.

    Args:
        acc: Accumulator dict returned by (possibly multiple calls to)
            :func:`accumulate_simple_iqu`, with keys ``i_num``, ``q_num``, ``u_num``,
            ``i_den``, ``hits``, and ``wpol``.

    Returns:
        Dict with keys ``I``, ``Q``, ``U``, ``i_den``, ``hits``, and ``wpol``;
        unseen pixels are left at zero.
    """
    i_num = acc["i_num"]
    q_num = acc["q_num"]
    u_num = acc["u_num"]
    i_den = acc["i_den"]
    w_pol = acc["wpol"]
    hits = acc["hits"]

    i_map = np.zeros_like(i_num)
    q_map = np.zeros_like(q_num)
    u_map = np.zeros_like(u_num)

    m_i = i_den > 0
    m_p = w_pol > 0

    i_map[m_i] = i_num[m_i] / i_den[m_i]
    q_map[m_p] = q_num[m_p] / w_pol[m_p]
    u_map[m_p] = u_num[m_p] / w_pol[m_p]

    return {
        "I": i_map,
        "Q": q_map,
        "U": u_map,
        "i_den": i_den,
        "hits": hits,
        "wpol": w_pol,
    }
