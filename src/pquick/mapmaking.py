from __future__ import annotations

import numpy as np
from ducc0.healpix import Healpix_Base
from numba import njit as _njit

# Standard HEALPix sentinel for unobserved pixels (same value used by healpy).
_UNSEEN: float = -1.6375e30


@_njit(fastmath=True, cache=True)
def _accumulate_tqu_jit(
    matrix: np.ndarray,
    pix: np.ndarray,
    ac2: np.ndarray,
    as2: np.ndarray,
    w: float,
    wy: np.ndarray,
) -> None:
    """Single-pass scatter-accumulate of the 9 normal-equation entries per sample.

    Replaces ten separate ``np.add.at`` scatter passes (each re-reading ``pix``) with
    one fused loop, reading ``pix`` once per sample. ``ac2``/``as2`` already carry the
    polarisation efficiency ``rho`` and ``wy = det_weight * tod``; the per-sample writes
    target distinct pixels' entries, so no cross-iteration races exist.
    """
    for i in range(pix.shape[0]):
        p = pix[i]
        a = ac2[i]
        b = as2[i]
        y = wy[i]
        matrix[p, 0, 0] += w
        matrix[p, 0, 1] += w * a
        matrix[p, 0, 2] += w * b
        matrix[p, 1, 1] += w * a * a
        matrix[p, 1, 2] += w * a * b
        matrix[p, 2, 2] += w * b * b
        matrix[p, 1, 0] += y
        matrix[p, 2, 0] += y * a
        matrix[p, 2, 1] += y * b


def init_map_matrix(nside: int) -> np.ndarray:
    """Allocate a zero-filled ``(npix, 3, 3)`` normal-equation matrix accumulator.

    Args:
        nside: HEALPix resolution parameter.

    Returns:
        Float64 array of shape ``(npix, 3, 3)`` initialised to zero.
    """
    npix = Healpix_Base(nside, "RING").npix()
    return np.zeros((npix, 3, 3), dtype=np.float64)


def accumulate_tqu_matrix(
    matrix: np.ndarray,
    pix: np.ndarray,
    psi: np.ndarray,
    tod: np.ndarray,
    det_weight: float,
    rho: float = 1.0,
) -> None:
    """Accumulate a detector's TOD into the polarised normal-equation matrix in-place.

    The upper 2×2 block stores the weighted pointing matrix ``A^T N^{-1} A`` and the
    lower-left column stores the weighted RHS ``A^T N^{-1} d``, all indexed by pixel.

    The detector response model is ``d = I + rho (Q cos2psi + U sin2psi)`` where
    ``rho = (1 - eps)/(1 + eps)`` is the polarisation efficiency (cross-polar
    leakage ``eps``); ``rho = 1`` is an ideal polarisation-sensitive detector.
    This matches qp_planck's ``rhohit`` weighting (``Ideal`` = 1, ``IMO`` = RIMO
    epsilon). Note ``rho`` does not affect the recovered ``I``/temperature (the
    I-I element is ``rho``-independent), only Q/U.

    Args:
        matrix: Accumulator array of shape ``(npix, 3, 3)``.
        pix: HEALPix pixel indices for each sample, shape ``(N,)``.
        psi: Polarisation angles in radians, shape ``(N,)``.
        tod: Convolved signal samples, shape ``(N,)``.
        det_weight: Inverse-noise weight for this detector.
        rho: Polarisation efficiency ``(1 - eps)/(1 + eps)``. Default ``1.0`` (ideal).
    """
    if pix.size == 0:
        return

    w = float(det_weight)
    r = float(rho)
    # Polarised response columns carry rho; the intensity column does not. ac2/as2 and
    # wy are built vectorised, then a single fused numba pass does all 9 scatter-adds
    # (A = [1, rho*cos2psi, rho*sin2psi]; upper triangle = A^T N^-1 A, lower = A^T N^-1 d).
    ac2 = r * np.cos(2.0 * psi)
    as2 = r * np.sin(2.0 * psi)
    wy = w * np.asarray(tod, dtype=np.float64)
    pix64 = np.ascontiguousarray(pix, dtype=np.int64)
    _accumulate_tqu_jit(matrix, pix64, ac2, as2, w, wy)


def solve_tqu_from_matrix(
    matrix: np.ndarray,
    cond_threshold: float = 1e10,
    batch_size: int = 1_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the per-pixel 3x3 polarised map-making equation to recover T, Q, U maps.

    Pixels with no hits or a poorly conditioned normal matrix are set to ``_UNSEEN``.
    Pixels are processed in batches of ``batch_size`` to keep the memory footprint small
    (avoids allocating a full-sky copy of the matrix during the solve step).

    Args:
        matrix: Accumulated normal-equation array of shape ``(npix, 3, 3)`` from
            :func:`accumulate_tqu_matrix`.
        cond_threshold: Pixels whose matrix condition number exceeds this value are masked.
        batch_size: Number of hit pixels to process per batch. Reduce this to lower peak
            memory at the cost of slightly more Python overhead.

    Returns:
        Tuple ``(t_map, q_map, u_map)`` of float64 HEALPix maps.
    """
    npix = matrix.shape[0]
    t_map = np.full(npix, _UNSEEN, dtype=np.float64)
    q_map = np.full(npix, _UNSEEN, dtype=np.float64)
    u_map = np.full(npix, _UNSEEN, dtype=np.float64)

    hit_idx = np.where(matrix[:, 0, 0] > 0)[0]
    if hit_idx.size == 0:
        return t_map, q_map, u_map

    for start in range(0, hit_idx.size, batch_size):
        batch_idx = hit_idx[start : start + batch_size]

        # Fancy-indexed copy of this batch only (small, ~72 MB for 1 M pixels).
        A = matrix[batch_idx].copy()  # (batch, 3, 3)

        # Extract the RHS from the lower-left triangle before symmetrising.
        rhs = np.stack([A[:, 1, 0], A[:, 2, 0], A[:, 2, 1]], axis=1)  # (batch, 3)

        # Symmetrize the normal matrices in-place.
        A[:, 1, 0] = A[:, 0, 1]
        A[:, 2, 0] = A[:, 0, 2]
        A[:, 2, 1] = A[:, 1, 2]

        # Condition check via eigvalsh (symmetric matrices): faster than SVD-based cond.
        eigs = np.linalg.eigvalsh(A)  # (batch, 3), ascending order
        min_eig = eigs[:, 0]
        max_eig = eigs[:, -1]
        with np.errstate(divide="ignore", invalid="ignore"):
            cond = np.where(min_eig > 0, max_eig / min_eig, np.inf)
        good = cond < cond_threshold
        if not np.any(good):
            continue

        sol = np.linalg.solve(A[good], rhs[good, :, np.newaxis]).squeeze(-1)  # (n_good, 3)
        idx_good = batch_idx[good]
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
    hpx = Healpix_Base(nside, "NEST" if nest else "RING")

    npix = hpx.npix()
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

    pix = hpx.ang2pix(
        np.stack([th, ph], axis=-1)
    )
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
