from __future__ import annotations

import math

import numpy as np
from ducc0.healpix import Healpix_Base
from numba import njit as _njit
from numba import prange as _prange
from numba import get_thread_id as _get_thread_id

# Standard HEALPix sentinel for unobserved pixels (same value used by healpy).
_UNSEEN: float = -1.6375e30


@_njit(fastmath=True, cache=True, parallel=True)
def _accumulate_tqu_jit(
    matrix: np.ndarray,
    pix: np.ndarray,
    psi: np.ndarray,
    tod: np.ndarray,
    w: float,
    rho: float,
) -> None:
    """Accumulate the 9 normal-equation entries per sample into *matrix* in place.

    First a thread-parallel ``prange`` pass builds the per-sample response columns
    ``ac2 = rho*cos2psi``, ``as2 = rho*sin2psi`` and ``wy = det_weight*tod`` (the trig
    is the dominant per-sample cost and is embarrassingly parallel). Then a single
    serial scatter pass reads ``pix`` once and does all 9 adds — kept serial because
    different samples may target the same pixel (a parallel scatter would race).
    """
    n = pix.shape[0]
    ac2 = np.empty(n)
    as2 = np.empty(n)
    wy = np.empty(n)
    for i in _prange(n):
        two = 2.0 * psi[i]
        ac2[i] = rho * math.cos(two)
        as2[i] = rho * math.sin(two)
        wy[i] = w * tod[i]
    for i in range(n):
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

    # Response columns ac2 = rho*cos2psi, as2 = rho*sin2psi (rho on the polarised
    # columns only) and wy = det_weight*tod are built in a parallel pass inside the
    # kernel; the scatter (A = [1, ac2, as2]; upper triangle = A^T N^-1 A, lower =
    # A^T N^-1 d) is then done in a single serial pass.
    pix64 = np.ascontiguousarray(pix, dtype=np.int64)
    psi64 = np.ascontiguousarray(psi, dtype=np.float64)
    tod64 = np.ascontiguousarray(tod, dtype=np.float64)
    _accumulate_tqu_jit(matrix, pix64, psi64, tod64, float(det_weight), float(rho))


@_njit(fastmath=True, cache=True, parallel=True)
def _accumulate_tqu_local_jit(local, pix, psi, tod, w, rho):
    """Scatter the 9 normal-equation entries into per-thread accumulators.

    ``local`` has shape ``(nthreads, npix, 3, 3)``; each sample writes only into its
    own thread's slice, so the scatter runs fully in ``prange`` with no race (unlike
    :func:`_accumulate_tqu_jit`, whose scatter must be serial). Sum over axis 0 after
    all samples to get the ``(npix, 3, 3)`` matrix.
    """
    for i in _prange(pix.shape[0]):
        t = _get_thread_id()
        two = 2.0 * psi[i]
        a = rho * math.cos(two)
        b = rho * math.sin(two)
        y = w * tod[i]
        p = pix[i]
        local[t, p, 0, 0] += w
        local[t, p, 0, 1] += w * a
        local[t, p, 0, 2] += w * b
        local[t, p, 1, 1] += w * a * a
        local[t, p, 1, 2] += w * a * b
        local[t, p, 2, 2] += w * b * b
        local[t, p, 1, 0] += y
        local[t, p, 2, 0] += y * a
        local[t, p, 2, 1] += y * b


def accumulate_tqu_local(local, pix, psi, tod, det_weight, rho=1.0):
    """Parallel variant of :func:`accumulate_tqu_matrix` writing into per-thread
    accumulators ``local`` of shape ``(nthreads, npix, 3, 3)``. Reduce with
    ``local.sum(axis=0)`` once after the OD loop. Trades ``nthreads x`` the matrix
    memory for a parallel scatter."""
    if pix.size == 0:
        return
    pix64 = np.ascontiguousarray(pix, dtype=np.int64)
    psi64 = np.ascontiguousarray(psi, dtype=np.float64)
    tod64 = np.ascontiguousarray(tod, dtype=np.float64)
    _accumulate_tqu_local_jit(local, pix64, psi64, tod64, float(det_weight), float(rho))


@_njit(cache=True)
def _add_hits_jit(hits: np.ndarray, pix: np.ndarray) -> None:
    # ponytail: serial scatter into the persistent hits buffer. O(ngood), no per-call
    # npix alloc/zero -- np.bincount(minlength=npix) costs ~npix per call (50 ms at
    # nside 2048) regardless of ngood. Serial (not prange) to avoid a same-pixel race.
    for i in range(pix.shape[0]):
        hits[pix[i]] += 1


def add_hits(hits: np.ndarray, pix: np.ndarray) -> None:
    """Add per-sample hit counts into the persistent ``hits`` accumulator in place."""
    if pix.size == 0:
        return
    _add_hits_jit(hits, np.ascontiguousarray(pix, dtype=np.int64))


@_njit(fastmath=True, cache=True, parallel=True)
def _solve_tqu_jit(
    matrix: np.ndarray,
    t_map: np.ndarray,
    q_map: np.ndarray,
    u_map: np.ndarray,
    cond_threshold: float,
) -> None:
    """Per-pixel 3x3 polarised solve, in parallel over pixels.

    For each hit pixel (``matrix[p,0,0] > 0``): build the symmetric normal matrix from
    the upper triangle, reject it if its condition number (``lambda_max/lambda_min`` via
    the closed-form symmetric-3x3 eigenvalues) is >= ``cond_threshold``, otherwise solve
    ``A x = rhs`` (rhs in the lower triangle) by cofactor inversion and write T/Q/U.
    Unwritten pixels keep their pre-filled ``_UNSEEN`` value. Each pixel writes distinct
    output entries, so the ``prange`` loop is race-free.
    """
    two_pi_3 = 2.0 * math.pi / 3.0
    for p in _prange(matrix.shape[0]):
        a00 = matrix[p, 0, 0]
        if a00 <= 0.0:
            continue
        a01 = matrix[p, 0, 1]; a02 = matrix[p, 0, 2]
        a11 = matrix[p, 1, 1]; a12 = matrix[p, 1, 2]; a22 = matrix[p, 2, 2]

        # Eigenvalues of the symmetric 3x3 (Smith's closed form) for the condition number.
        p1 = a01 * a01 + a02 * a02 + a12 * a12
        if p1 == 0.0:
            lo = min(a00, a11, a22)
            hi = max(a00, a11, a22)
        else:
            q = (a00 + a11 + a22) / 3.0
            p2 = (a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2.0 * p1
            pp = math.sqrt(p2 / 6.0)
            b00 = (a00 - q) / pp; b11 = (a11 - q) / pp; b22 = (a22 - q) / pp
            b01 = a01 / pp; b02 = a02 / pp; b12 = a12 / pp
            detb = (
                b00 * (b11 * b22 - b12 * b12)
                - b01 * (b01 * b22 - b12 * b02)
                + b02 * (b01 * b12 - b11 * b02)
            )
            r = detb / 2.0
            if r <= -1.0:
                phi = math.pi / 3.0
            elif r >= 1.0:
                phi = 0.0
            else:
                phi = math.acos(r) / 3.0
            hi = q + 2.0 * pp * math.cos(phi)
            lo = q + 2.0 * pp * math.cos(phi + two_pi_3)

        if lo <= 0.0 or hi / lo >= cond_threshold:
            continue

        # Solve A x = rhs (rhs = lower triangle) via cofactor inverse of the symmetric A.
        r0 = matrix[p, 1, 0]; r1 = matrix[p, 2, 0]; r2 = matrix[p, 2, 1]
        c00 = a11 * a22 - a12 * a12
        c01 = a02 * a12 - a01 * a22
        c02 = a01 * a12 - a02 * a11
        det = a00 * c00 + a01 * c01 + a02 * c02
        if det == 0.0:
            continue
        c11 = a00 * a22 - a02 * a02
        c12 = a01 * a02 - a00 * a12
        c22 = a00 * a11 - a01 * a01
        inv = 1.0 / det
        t_map[p] = (c00 * r0 + c01 * r1 + c02 * r2) * inv
        q_map[p] = (c01 * r0 + c11 * r1 + c12 * r2) * inv
        u_map[p] = (c02 * r0 + c12 * r1 + c22 * r2) * inv


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
        batch_size: Unused; retained for backward compatibility. The solve now runs as a
            single in-place thread-parallel pass over pixels (no per-batch matrix copies).

    Returns:
        Tuple ``(t_map, q_map, u_map)`` of float64 HEALPix maps.
    """
    npix = matrix.shape[0]
    t_map = np.full(npix, _UNSEEN, dtype=np.float64)
    q_map = np.full(npix, _UNSEEN, dtype=np.float64)
    u_map = np.full(npix, _UNSEEN, dtype=np.float64)

    _solve_tqu_jit(np.ascontiguousarray(matrix, dtype=np.float64), t_map, q_map, u_map, float(cond_threshold))

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
