from __future__ import annotations

from typing import Any, Callable

import numpy as np


def _match_component_count(sky: np.ndarray, beam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align sky/beam component counts by promoting or slicing beam components."""
    if sky.shape[0] == beam.shape[0]:
        return sky, beam

    if sky.shape[0] == 3 and beam.shape[0] == 1:
        # Scalar (1-component) beam with a [T,E,B] sky: put it in the T slot and zero
        # the E/B slots (ducc0's other two components are the spin-2 response, not Q/U).
        # The pipeline always builds [T,E,B] via io.build_polarized_beam_alm, so this is
        # only a safety promotion for callers passing a bare intensity beam.
        promoted = np.zeros((3, beam.shape[1]), dtype=np.complex128)
        promoted[0] = beam[0]
        return sky, promoted

    if sky.shape[0] == 1 and beam.shape[0] == 3:
        return sky, beam[:1]

    raise ValueError(
        "sky_alm and beam_alm have incompatible component counts: "
        f"{sky.shape[0]} vs {beam.shape[0]}"
    )


def convolve_timeline(
    sky_alm: np.ndarray,
    beam_alm: np.ndarray,
    ptg_thetaphipsi: np.ndarray,
    lmax: int,
    mmax: int,
    nthreads: int = 0,
    epsilon: float = 1e-5,
    separate: bool = False,
    interpolator_cache: dict[int, Any] | None = None,
    interpolator_factory: Callable[..., Any] | None = None,
) -> np.ndarray:
    """Convolve sky and beam ALMs along a pointing timeline via ``ducc0`` total-convolution.

    Builds a ``ducc0.totalconvolve.Interpolator`` for the given sample count, evaluates
    the convolution at each pointing, and returns a 1-D float64 TOD array.  When
    *interpolator_cache* is provided the ``Interpolator`` object is stored and reused for
    subsequent calls with the same number of points.

    Args:
        sky_alm: Sky spherical harmonics, shape ``(ncomp, nalm_sky)``, healpy m-major order.
        beam_alm: Beam spherical harmonics, shape ``(ncomp, nalm_beam)``, healpy m-major
            order truncated at *mmax*.
        ptg_thetaphipsi: Pointing array of shape ``(N, 3)`` with columns
            ``(theta, phi, psi)`` in radians.
        lmax: Maximum multipole ℓ; both ALM arrays must be consistent with this value.
        mmax: Maximum azimuthal order of the beam (``kmax`` in ducc0 notation).
        nthreads: Number of OpenMP threads for ducc0 (0 = auto).
        epsilon: Accuracy target for the ducc0 gridder.
        separate: Passed directly to ``Interpolator``; keep ``False`` for summed output.
        interpolator_cache: Optional dict keyed by ``npoints``; allows reuse of the
            pre-computed ``Interpolator`` grid across calls with the same sample count.
        interpolator_factory: Replacement for ``ducc0.totalconvolve.Interpolator``,
            useful for testing.

    Returns:
        1-D float64 array of length *N* containing the convolved TOD samples.
    """
    if interpolator_factory is None:
        try:
            from ducc0.totalconvolve import Interpolator
        except Exception as exc:  # pragma: no cover
            raise ImportError("ducc0 is required for convolution; install ducc0>=0.41") from exc
        interpolator_factory = Interpolator

    ptg = np.asarray(ptg_thetaphipsi, dtype=np.float64)
    if ptg.ndim != 2 or ptg.shape[1] != 3:
        raise ValueError("ptg_thetaphipsi must have shape (N, 3)")

    if int(mmax) > int(lmax) - 4:
        raise ValueError(
            f"ducc0 requires mmax <= lmax - 4; got mmax={mmax}, lmax={lmax}"
        )

    sky = np.asarray(sky_alm, dtype=np.complex128)
    beam = np.asarray(beam_alm, dtype=np.complex128)
    if sky.ndim != 2 or beam.ndim != 2:
        raise ValueError("sky_alm and beam_alm must have shape (ncomp, nalm)")
    sky, beam = _match_component_count(sky, beam)

    npoints = int(ptg.shape[0])
    interp = interpolator_cache.get(npoints) if interpolator_cache is not None else None
    if interp is None:
        interp = interpolator_factory(
            sky=sky,
            beam=beam,
            separate=separate,
            lmax=int(lmax),
            kmax=int(mmax),
            npoints=npoints,
            epsilon=float(epsilon),
            nthreads=int(nthreads),
        )
        if interpolator_cache is not None:
            interpolator_cache[npoints] = interp

    data = np.asarray(interp.interpol(ptg), dtype=np.float64)

    if data.ndim == 2 and data.shape[0] == 1:
        return data[0]
    if data.ndim == 2:
        return np.sum(data, axis=0)
    if data.ndim == 1:
        return data
    raise ValueError("Unexpected interpolator output shape")
