from __future__ import annotations

import numpy as np


def convolve_timeline(
    sky_alm: np.ndarray,
    beam_alm: np.ndarray,
    ptg_thetaphipsi: np.ndarray,
    lmax: int,
    kmax: int,
    nthreads: int = 0,
    separate: bool = False,
    epsilon: float = 1e-5,
) -> np.ndarray:
    try:
        from ducc0.totalconvolve import Interpolator
    except Exception as exc:  # pragma: no cover
        raise ImportError("ducc0 is required for convolution; install ducc0>=0.41") from exc

    ptg = np.asarray(ptg_thetaphipsi, dtype=np.float64)
    if ptg.ndim != 2 or ptg.shape[1] != 3:
        raise ValueError("ptg_thetaphipsi must have shape (N, 3)")

    sky = np.asarray(sky_alm, dtype=np.complex128)
    beam = np.asarray(beam_alm, dtype=np.complex128)
    if sky.ndim != 2 or beam.ndim != 2:
        raise ValueError("sky_alm and beam_alm must have shape (ncomp, nalm)")

    interp = Interpolator(
        sky=sky,
        beam=beam,
        separate=separate,
        lmax=int(lmax),
        kmax=int(kmax),
        npoints=int(ptg.shape[0]),
        epsilon=float(epsilon),
        nthreads=int(nthreads),
    )
    data = np.asarray(interp.interpol(ptg), dtype=np.float64)

    if data.ndim == 2 and data.shape[0] == 1:
        return data[0]
    if data.ndim == 2:
        return np.sum(data, axis=0)
    if data.ndim == 1:
        return data
    raise ValueError("Unexpected interpolator output shape")
