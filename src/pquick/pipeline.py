from __future__ import annotations

import argparse
import time as _time
import warnings
from pathlib import Path
from typing import Any, cast

import healpy as hp
import numpy as np
from ducc0.healpix import Healpix_Base

from .config import PipelineConfig, load_config
from .convolution import build_convolution_interpolator, evaluate_convolution
from .io import (
    build_polarized_beam_alm,
    detector_to_beam_file,
    infer_lmax_from_alm,
    load_beam_alm,
    normalize_beam_alm,
    load_horn_flag_npz,
    load_pointing_npz,
    load_rimo_detectors,
    load_sky_alm,
    select_detectors,
    truncate_alm,
)
from .mapmaking import accumulate_tqu_local, add_hits, solve_tqu_from_matrix
from .pointing import build_pointing_interpolator
from .quaternion import bore_det_to_angles, bore_det_to_ptg, bore_det_to_ptg_masked, normalize_quaternion, quat_mul
from .utilities import build_pointing_file_paths, detector_map_weight, estimate_memory_per_rank_mb, extract_od_from_pointing_filename, is_psb, parse_mission_length, print_mpi_distribution, resolve_nthreads


def _load_bad_ring_intervals(path: str | Path) -> dict[str, np.ndarray]:
    """Load a TOAST/NPIPE-style bad-ring interval file.

    The file format is plain text with rows:
        ``<det_or_ALL> <tstart_s> <tstop_s>``
    and optional comment lines starting with ``#``.

    Returns one ``(M, 2)`` float64 array of ``[tstart, tstop]`` rows per key, so
    :func:`_chunk_bad_ring_mask` can filter overlaps with vectorised numpy ops instead
    of a Python loop over every interval (files can hold >100k intervals).
    """
    raw: dict[str, list[tuple[float, float]]] = {}
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for iline, line in enumerate(f, start=1):
            row = line.strip()
            if not row or row.startswith("#"):
                continue
            parts = row.split()
            if len(parts) != 3:
                raise ValueError(f"Invalid bad_rings_file row {iline} in {p}: {row!r}")
            det_key = parts[0].upper()
            tstart = float(parts[1])
            tstop = float(parts[2])
            if tstop < tstart:
                tstart, tstop = tstop, tstart
            raw.setdefault(det_key, []).append((tstart, tstop))
    return {k: np.asarray(v, dtype=np.float64).reshape(-1, 2) for k, v in raw.items()}


def _chunk_bad_ring_mask(
    intervals: dict[str, np.ndarray] | None,
    det_name: str,
    coarse_t0_ns: float,
    native_rate_hz: float,
    chunk_start: int,
    chunk_len: int,
) -> np.ndarray:
    """Return a boolean mask of bad samples for one chunk and detector.

    ``times[i] = t0_s + i*dt_s`` is monotonic, so each interval maps to a contiguous
    index range ``[ceil((tstart-t0)/dt), floor((tstop-t0)/dt)]`` — computed analytically
    with no per-sample comparison. The overlap test and index conversion are vectorised
    over the whole interval table (>100k rows) at once; only the few intervals that
    actually fall in this chunk are written.
    """
    if not intervals:
        return np.zeros(chunk_len, dtype=bool)

    parts = [a for a in (intervals.get(det_name.upper()), intervals.get("ALL")) if a is not None]
    out = np.zeros(chunk_len, dtype=bool)
    if not parts:
        return out
    iv = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

    dt_s = 1.0 / float(native_rate_hz)
    t0_s = float(coarse_t0_ns) * 1.0e-9 + float(chunk_start) * dt_s
    t_last = t0_s + (chunk_len - 1) * dt_s

    starts = iv[:, 0]
    stops = iv[:, 1]
    over = (starts <= t_last) & (stops >= t0_s)
    if not np.any(over):
        return out

    inv_dt = float(native_rate_hz)
    i0 = np.ceil((starts[over] - t0_s) * inv_dt).astype(np.int64)
    i1 = np.floor((stops[over] - t0_s) * inv_dt).astype(np.int64) + 1
    np.clip(i0, 0, chunk_len, out=i0)
    np.clip(i1, 0, chunk_len, out=i1)
    for a, b in zip(i0.tolist(), i1.tolist()):
        if b > a:
            out[a:b] = True
    return out


def _det_to_horn(detector: str) -> str:
    """Map detector arm names to horn names used in packed flag files."""
    if detector and detector[-1] in "abMS":
        return detector[:-1]
    return detector


def _detector_channel_ghz(detector: str) -> int | None:
    """Extract channel frequency in GHz from a detector string like ``100-1a``."""
    try:
        head = detector.split("-", 1)[0]
        return int(head)
    except Exception:
        return None


def _get_mpi():
    """Return ``(comm, rank, size)`` when MPI is available, else serial defaults."""
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        return comm, comm.rank, comm.size
    except Exception:
        return None, 0, 1


def _local_slice(items: list[Path], rank: int, size: int) -> list[Path]:
    """Return the subset of items assigned to a rank via round-robin partition."""
    return [x for i, x in enumerate(items) if i % size == rank]


def _lpt_slice(items: list[Path], rank: int, size: int) -> list[Path]:
    """Assign items to ranks using the LPT (Longest Processing Time) heuristic.

    Items are sorted by file size in descending order before round-robin
    partitioning.  This ensures that large ODs are spread across different ranks
    rather than accumulating on the same one, minimising the load imbalance that
    causes ranks to wait at the MPI barrier.
    """
    sorted_items = sorted(items, key=lambda p: p.stat().st_size, reverse=True)
    return [x for i, x in enumerate(sorted_items) if i % size == rank]


def _sum_reduce(comm, arr: np.ndarray, rank: int = 0) -> np.ndarray | None:
    """Sum-reduce *arr* across all MPI ranks; only rank 0 receives the result.

    Using ``Reduce`` instead of ``Allreduce`` halves network traffic because the
    aggregated matrix is *not* broadcast back to every rank — only rank 0 needs
    it to solve and write the output maps.

    Returns:
        The reduced array on rank 0; ``None`` on all other ranks.
    """
    if comm is None:
        return arr
    from mpi4py import MPI

    if rank == 0:
        comm.Reduce(MPI.IN_PLACE, arr, op=MPI.SUM, root=0)
        return arr
    else:
        comm.Reduce(arr, None, op=MPI.SUM, root=0)
        return None


def _vprint(enabled: bool, rank: int, msg: str) -> None:
    """Print a message only on rank 0 when verbose output is enabled."""
    if enabled and rank == 0:
        print(msg, flush=True)


def run_pipeline(config: PipelineConfig) -> Path | None:
    """Execute the full beam-convolution map-making pipeline.

    Loads sky ALMs and beam ALMs, iterates over operational days and detectors,
    accumulates the polarised normal-equation matrix, solves for T/Q/U, and writes
    FITS output maps. MPI-aware: ODs are distributed across ranks and the
    accumulated arrays are reduced to rank 0 with ``Reduce`` before writing.

    Args:
        config: Fully populated :class:`~pquick.config.PipelineConfig`.

    Returns:
        Path to the IQU FITS map on MPI rank 0; ``None`` on all other ranks.

    Raises:
        ValueError: If the configured lmax exceeds the sky ALM lmax.
    """
    comm, rank, size = _get_mpi()
    verbose = bool(config.verbose)

    import numba
    nthreads = resolve_nthreads(config.nthreads)
    numba.set_num_threads(nthreads)
    _vprint(verbose, rank, f"[Threads] nthreads={nthreads} (ducc0 + numba)")

    # Wall-clock anchor for the whole run (everything after MPI/import startup). The
    # per-section timers below (resamp/conv/macc/reduce/solve) only instrument specific
    # blocks; t_wall0 lets the summary report the true elapsed time and an "other"
    # bucket (setup, flag I/O, masking, boresight copies, output write) so the printed
    # accounting reconciles with a stopwatch around the process.
    t_wall0 = _time.perf_counter()

    sky_alm = load_sky_alm(config.inputs.sky_alm)
    lmax_alm = infer_lmax_from_alm(sky_alm)
    if config.convolution.lmax > lmax_alm:
        raise ValueError(f"Configured lmax={config.convolution.lmax} exceeds sky alm lmax={lmax_alm}")
    sky_alm = truncate_alm(sky_alm, lmax_alm, config.convolution.lmax)

    # Per-component (T,E,B) input rescale (debug knob, e.g. [1,0,0] for T-only).
    rescale = config.inputs.rescale
    if rescale != (1.0, 1.0, 1.0):
        sky_alm = np.array(sky_alm, dtype=np.complex128, copy=True)
        for i in range(min(3, sky_alm.shape[0])):
            if rescale[i] != 1.0:
                sky_alm[i] *= rescale[i]
        _vprint(verbose, rank, f"[Inputs] sky alm rescaled by (T,E,B) = {tuple(rescale)}")

    det_meta = load_rimo_detectors(config.inputs.rimo_file)
    detectors = select_detectors(list(det_meta.keys()), config.detector_selection)

    det_info: list[dict[str, object]] = []
    for det in detectors:
        # A detector can be in the RIMO but have no beam file (e.g. the 143-8 SWB,
        # excluded from the standard HFI set) — skip it instead of crashing a whole
        # channel selection. qp_planck likewise drops such detectors.
        try:
            beam_file = detector_to_beam_file(config.inputs.beams_dir, det)
        except FileNotFoundError:
            warnings.warn(
                f"No beam file for detector {det!r} in {config.inputs.beams_dir} — skipping it",
                UserWarning,
                stacklevel=2,
            )
            continue
        beam_alm = load_beam_alm(
            beam_file,
            lmax=config.convolution.lmax,
            mmax=config.convolution.mmax,
        )
        dmeta = det_meta.get(det, {})
        psi_pol_rad = float(dmeta.get("psi_pol_rad", 0.0))
        # Polarisation efficiency used by both the beam E/B scaling and the map-making
        # weight. use_cross_pol=True -> RIMO rho=(1-eps)/(1+eps) (qp_planck rhohit=IMO);
        # False -> ideal, but ideal still means rho=0 for an unpolarised SWB, not 1
        # (qp_planck rhohit=Ideal uses the PSB flag: 1 for PSB, 0 for SWB).
        rho_eff = (
            float(dmeta.get("rho_pol", 1.0))
            if config.map.use_cross_pol
            else (1.0 if is_psb(det) else 0.0)
        )
        # Scalar Planck blm (Dxx) -> spin-2 polarised [T, E, B] beam in the Pxx frame.
        beam_alm = build_polarized_beam_alm(
            beam_alm,
            psi_pol_rad=psi_pol_rad,
            lmax=config.convolution.lmax,
            mmax=config.convolution.mmax,
            psi_uv_rad=float(dmeta.get("psi_uv_rad", 0.0)),
            # Match the map-making polarisation efficiency so EE/BB are not
            # inflated by 1/rho^2.
            rho_pol=rho_eff,
            nthreads=nthreads,
        )
        _vprint(
            verbose,
            rank,
            f"  [beam] {det}: spin-2 polarised [T,E,B] beam built "
            f"(ncomp={beam_alm.shape[0]}, psi_pol={np.degrees(psi_pol_rad):.3f} deg)",
        )
        beam_alm = normalize_beam_alm(
            beam_alm,
            mode=config.convolution.beam_normalization,
        )
        dquat = normalize_quaternion(
            np.asarray(dmeta.get("quat", np.array([0.0, 0.0, 0.0, 1.0])), dtype=np.float64)
        )
        det_info.append(
            {
                "name": det,
                "beam_alm": beam_alm,
                "quat": dquat,
                "weight": detector_map_weight(det),
                "psi_pol_rad": psi_pol_rad,
                "psi_uv_rad": float(dmeta.get("psi_uv_rad", 0.0)),
                "rho_pol": rho_eff,
            }
        )

    if not det_info:
        raise FileNotFoundError(
            f"No detectors have beam files in {config.inputs.beams_dir} for the current "
            f"selection ({len(detectors)} detector(s) requested) — check beams_dir."
        )

    mission = config.inputs.mission_length or "full"
    od_start, od_end = parse_mission_length(mission)
    all_pointing = build_pointing_file_paths(config.inputs.pointings, od_start, od_end)

    bad_ring_intervals: dict[str, np.ndarray] | None = None
    if config.inputs.bad_rings_file is not None:
        bad_ring_intervals = _load_bad_ring_intervals(config.inputs.bad_rings_file)
        _vprint(
            verbose,
            rank,
            f"[Flags] Loaded bad ring intervals from {config.inputs.bad_rings_file}",
        )

    # Validate all pointing files exist before doing any work.
    missing_pointing = [p for p in all_pointing if not p.exists()]
    if missing_pointing:
        missing_list = "\n  ".join(str(p) for p in missing_pointing)
        raise FileNotFoundError(
            f"{len(missing_pointing)} pointing file(s) not found:\n  {missing_list}"
        )

    local_pointing = _lpt_slice(all_pointing, rank, size)

    # Build the HEALPix base object once, reused for every ang2pix call.
    hpx = Healpix_Base(config.map.nside, "NEST" if config.map.nest else "RING")
    npix = hpx.npix()
    center_pointing = bool(config.resampling.center_pointing)
    hpx_center = hpx if center_pointing else None

    if verbose:
        local_ods = [extract_od_from_pointing_filename(p) for p in local_pointing]
        print_mpi_distribution(comm, rank, size, local_ods)
        map_mb = estimate_memory_per_rank_mb(config.map.nside)
        interp_mb = estimate_memory_per_rank_mb(
            config.map.nside, lmax=config.convolution.lmax, mmax=config.convolution.mmax
        ) - map_mb
        alm_mb = (
            sky_alm.nbytes + sum(d["beam_alm"].nbytes for d in det_info)
        ) / 1024**2
        n_chunks = max(1, int(config.convolution.chunks))
        # ptg_buf (chunk_samples, 3) + psi_buf (chunk_samples,) + tod (≤chunk_samples,)
        tl_bytes_per_sample = (3 + 1 + 1) * 8
        est_mb = map_mb + interp_mb + alm_mb
        _vprint(
            verbose,
            rank,
            f"[Memory] Estimated peak per rank: {est_mb / 1024:.2f} GB"
            f" (for {size} ranks: {size * est_mb / 1024:.1f} GB total)\n"
            f"         nside={config.map.nside}  npix={npix:,}  lmax={config.convolution.lmax}  mmax={config.convolution.mmax}\n"
            f"           maps/matrix: {map_mb / 1024:.2f} GB\n"
            f"           Convolver  : {interp_mb / 1024:.2f} GB  [lower bound, actual can be 1.5–2×]\n"
            f"           ALMs       : {alm_mb:.0f} MB\n"
            f"           timeline   : ~{tl_bytes_per_sample} B/sample × chunk_samples × {n_chunks} chunk(s)",
        )
        if center_pointing:
            _vprint(
                verbose,
                rank,
                f"[Resampling] point-centering enabled (nside={config.map.nside}, order={'NEST' if config.map.nest else 'RING'})",
            )

    _vprint(verbose, rank, f"Starting pipeline: {len(local_pointing)} ODs on rank {rank}/{size}")

    # Per-thread accumulators so the scatter parallelises; summed after the OD loop.
    # Costs nthreads x (npix,3,3) = nthreads x ~3.6GB at nside 2048, but on real
    # (spatially local) scan data the parallel scatter is ~4x faster than a single
    # serial-scatter matrix -- worth the memory. Lower nthreads if a rank is RAM-bound.
    local_acc = np.zeros((numba.get_num_threads(), npix, 3, 3), dtype=np.float64)
    hits_acc = np.zeros(npix, dtype=np.int64)
    n_chunks_cfg = int(max(1, config.convolution.chunks))


    # Everything from sky load through beam build / MPI distribution / allocation is
    # one-time setup, untimed by the per-section timers.
    t_setup = _time.perf_counter() - t_wall0

    t_resamp_total = t_conv_total = t_macc_total = 0.0
    t_flag_total = t_prep_total = t_pix_total = 0.0
    t_od_wall_total = 0.0

    # The ducc0 convolution cube depends only on (sky, beam, lmax, mmax, epsilon) — not on
    # the pointing — so (when config.convolution.cache_interpolator) build it once per
    # detector and reuse across every OD/chunk on this rank. Building the cube is the
    # dominant convolution cost; reuse keeps one cube resident per detector (~0.4 GB at
    # lmax=1024/mmax=6, ~1–2 GB at lmax=2048). Disable for lower memory at the cost of a
    # per-OD rebuild.
    cache_interp = bool(config.convolution.cache_interpolator)
    beam_interp_cache: dict[str, Any] = {}
    _vprint(verbose, rank, f"[Convolution] cache_interpolator={cache_interp}")

    for od_idx, npz_path in enumerate(local_pointing, start=1):
        _vprint(verbose, rank, f"[OD {od_idx}/{len(local_pointing)}] {npz_path.name}")
        t_resamp_od = t_conv_od = t_macc_od = t_flag_od = t_prep_od = t_pix_od = 0.0
        _od_wall0 = _time.perf_counter()

        _t0 = _time.perf_counter()
        point_us = load_pointing_npz(npz_path)
        interp = build_pointing_interpolator(
            point_us,
            coordinate_system=config.resampling.coordinate_system,
        )
        del point_us
        t_resamp_od += _time.perf_counter() - _t0

        _t0 = _time.perf_counter()
        detector_flags: dict[str, np.ndarray] = {}
        use_flag_od = config.inputs.flags is not None
        if use_flag_od:
            od = extract_od_from_pointing_filename(npz_path)
            channel_cache: dict[int, Path] = {}
            for dinfo in det_info:
                det_name = str(dinfo["name"])
                ch = _detector_channel_ghz(det_name)
                if ch is None:
                    raise ValueError(f"Cannot infer channel from detector '{det_name}' for flags")
                if ch not in channel_cache:
                    flag_path = Path(f"{config.inputs.flags}{ch:03d}ghz_od_{od:04d}.npz")
                    channel_cache[ch] = flag_path if flag_path.exists() else None  # type: ignore[assignment]

                if channel_cache[ch] is None:
                    warnings.warn(
                        f"Flag file not found for ch={ch} GHz OD {od:04d}: "
                        f"{config.inputs.flags}{ch:03d}ghz_od_{od:04d}.npz — skipping flags for this OD",
                        UserWarning,
                        stacklevel=2,
                    )
                    use_flag_od = False
                    detector_flags.clear()
                    break

                horn = _det_to_horn(det_name)
                hflag = load_horn_flag_npz(channel_cache[ch], horn, n_samples=interp.n_native)
                if hflag.size != interp.n_native:
                    raise ValueError(
                        f"Flag length mismatch for {det_name} at OD {od:04d}: "
                        f"{hflag.size} != {interp.n_native}"
                    )
                detector_flags[det_name] = hflag
        t_flag_od += _time.perf_counter() - _t0

        # Whole-OD early-out: skip the chunk loop (and its per-chunk boresight
        # interpolation) when no detector has a single good sample. Computes the real
        # good mask per detector exactly as the loop does (common flag | horn flag |
        # bad-ring), over the full OD, short-circuiting on the first detector with any
        # good sample. Falls through to the OD timing accounting below.
        od_all_flagged = False
        if use_flag_od or bad_ring_intervals is not None:
            common_bad = (interp.flag_native != 0) if use_flag_od else None
            od_has_good = False
            for dinfo in det_info:
                dname = str(dinfo["name"])
                bad = _chunk_bad_ring_mask(
                    bad_ring_intervals, dname, interp.coarse_t0_ns,
                    interp.native_rate_hz, 0, interp.n_native,
                )
                if use_flag_od:
                    bad = bad | common_bad | (detector_flags[dname] != 0)
                if not bad.all():
                    od_has_good = True
                    break
            od_all_flagged = not od_has_good
            if od_all_flagged:
                _vprint(verbose, rank, f"  {npz_path.name}: no good samples — skipping OD")

        chunk_samples = max(1, (interp.n_native + n_chunks_cfg - 1) // n_chunks_cfg)
        n_chunks = (interp.n_native + chunk_samples - 1) // chunk_samples
        # Pre-allocate reusable buffers for the pointing array and psi (mapmaking).
        # These are sized for the largest possible chunk and reused across all
        # chunks and detectors, eliminating ~7 × chunk_samples × 8-byte allocations
        # (theta, phi, psi, psi_conv, column_stack) per detector per chunk.
        ptg_buf = np.empty((chunk_samples, 3), dtype=np.float64)
        psi_buf = np.empty(chunk_samples, dtype=np.float64)

        chunk_starts = () if od_all_flagged else range(0, interp.n_native, chunk_samples)
        for chunk_idx, chunk_start in enumerate(chunk_starts, start=1):
            chunk_end = min(chunk_start + chunk_samples, interp.n_native)
            chunk_len = chunk_end - chunk_start
            # Interpolate boresight once per chunk; skip the [good] boolean-index copy
            # when all samples are unflagged (avoids an unnecessary (chunk_len, 4) copy).
            _t0 = _time.perf_counter()
            q_bore_all = interp.get_boresight_quaternions(chunk_start, chunk_len)
            t_resamp_od += _time.perf_counter() - _t0

            # Compute ring_bad for the first detector once — reused in the first
            # detector-loop iteration to avoid a redundant call.
            _first_det_name = str(det_info[0]["name"])
            if bad_ring_intervals is not None:
                _ring_bad_first = _chunk_bad_ring_mask(
                    bad_ring_intervals,
                    _first_det_name,
                    interp.coarse_t0_ns,
                    interp.native_rate_hz,
                    chunk_start,
                    chunk_len,
                )
            else:
                _ring_bad_first = np.zeros(chunk_len, dtype=bool)
            # common_bad is detector-independent; build once here and reuse in the det loop.
            _common_bad = interp.flag_native[chunk_start:chunk_end] != 0 if use_flag_od else None
            if verbose:  # the good-count is log-only — don't pay for it in quiet runs
                _gf = (
                    ~(_common_bad | (detector_flags[_first_det_name][chunk_start:chunk_end] != 0) | _ring_bad_first)
                    if use_flag_od
                    else ~_ring_bad_first
                )
                _vprint(
                    verbose,
                    rank,
                    f"  [chunk {chunk_idx}/{n_chunks}] samples {chunk_start}:{chunk_end}"
                    f" | good={int(np.count_nonzero(_gf))}/{chunk_len} | n_native={interp.n_native}",
                )

            for det_idx, dinfo in enumerate(det_info, start=1):
                det_quat = np.asarray(dinfo["quat"], dtype=np.float64)
                beam_alm = np.asarray(dinfo["beam_alm"], dtype=np.complex128)
                det_weight = cast(float, dinfo["weight"])
                det_name = str(dinfo["name"])
                rho_pol = cast(float, dinfo["rho_pol"])

                _vprint(
                    verbose,
                    rank,
                    f"    [DET {det_idx}/{len(det_info)}] {det_name}",
                )

                _t0 = _time.perf_counter()
                if det_name == _first_det_name:
                    ring_bad = _ring_bad_first
                elif bad_ring_intervals is not None:
                    ring_bad = _chunk_bad_ring_mask(
                        bad_ring_intervals,
                        det_name,
                        interp.coarse_t0_ns,
                        interp.native_rate_hz,
                        chunk_start,
                        chunk_len,
                    )
                else:
                    ring_bad = _ring_bad_first  # all-zeros, safe to share (never mutated)
                if use_flag_od:
                    det_flag_chunk = detector_flags[det_name][chunk_start:chunk_end]
                    good = ~(_common_bad | (det_flag_chunk != 0) | ring_bad)
                else:
                    good = ~ring_bad

                ngood = int(np.count_nonzero(good))
                if ngood == 0:
                    continue

                # idx of good samples (cheap int array) instead of the 4-wide q_bore[good] copy.
                good_idx = None if ngood == chunk_len else np.flatnonzero(good)
                t_prep_od += _time.perf_counter() - _t0

                # Fill pre-allocated buffers directly — no temporary arrays. det_quat is
                # built from psi_uv only, so psi_buf is the Pxx (polarisation-frame)
                # angle used directly for mapmaking. The [T,E,B] beam is already rotated
                # to Pxx in build_polarized_beam_alm, so convolve at psi_pxx = psi_buf
                # (offset 0) and the beam, convolution and map-making share that frame.
                _t0 = _time.perf_counter()
                if good_idx is None:
                    bore_det_to_ptg(q_bore_all, det_quat, ptg_buf[:ngood], psi_buf[:ngood])
                else:
                    bore_det_to_ptg_masked(q_bore_all, det_quat, good_idx, ptg_buf[:ngood], psi_buf[:ngood])
                t_resamp_od += _time.perf_counter() - _t0

                pix_center = None
                if hpx_center is not None:
                    # Snap (theta, phi) to HEALPix pixel centers to suppress
                    # subpixel pointing variation before convolution.
                    _t0 = _time.perf_counter()
                    pix_center = hpx_center.ang2pix(
                        ptg_buf[:ngood, :2],
                        nthreads=nthreads,
                    )
                    ptg_buf[:ngood, :2] = hpx_center.pix2ang(pix_center, nthreads=nthreads)
                    t_pix_od += _time.perf_counter() - _t0

                _t0 = _time.perf_counter()
                conv_interp = beam_interp_cache.get(det_name) if cache_interp else None
                if conv_interp is None:
                    conv_interp = build_convolution_interpolator(
                        sky_alm,
                        beam_alm,
                        lmax=config.convolution.lmax,
                        mmax=config.convolution.mmax,
                        nthreads=nthreads,
                        epsilon=config.convolution.epsilon,
                        npoints=chunk_samples,
                    )
                    if cache_interp:
                        beam_interp_cache[det_name] = conv_interp
                tod = evaluate_convolution(conv_interp, ptg_buf[:ngood])
                t_conv_od += _time.perf_counter() - _t0

                _t0 = _time.perf_counter()
                if pix_center is not None:
                    pix = np.asarray(pix_center, dtype=np.int64)
                else:
                    pix = hpx.ang2pix(
                        ptg_buf[:ngood, :2],
                        nthreads=nthreads,
                    )
                accumulate_tqu_local(local_acc, pix, psi_buf[:ngood], np.asarray(tod, dtype=np.float64), det_weight, rho=rho_pol)
                # Scatter hits straight into the persistent buffer (O(ngood)); np.bincount
                # would alloc+zero a full npix array every call (~50 ms at nside 2048).
                add_hits(hits_acc, pix)
                del pix, tod
                t_macc_od += _time.perf_counter() - _t0

            del q_bore_all

        t_resamp_total += t_resamp_od
        t_conv_total += t_conv_od
        t_macc_total += t_macc_od
        t_flag_total += t_flag_od
        t_prep_total += t_prep_od
        t_pix_total += t_pix_od
        od_wall = _time.perf_counter() - _od_wall0
        t_od_wall_total += od_wall
        _od_other = od_wall - (t_resamp_od + t_conv_od + t_macc_od + t_flag_od + t_prep_od + t_pix_od)
        _pix_od = f"  pix={t_pix_od:.2f}s" if hpx_center is not None else ""
        _vprint(
            verbose,
            rank,
            f"  [OD timing] resamp={t_resamp_od:.2f}s  conv={t_conv_od:.2f}s  macc={t_macc_od:.2f}s"
            f"  flag={t_flag_od:.2f}s  prep={t_prep_od:.2f}s{_pix_od}"
            f"  other={_od_other:.2f}s  od_wall={od_wall:.2f}s",
        )

    _vprint(verbose, rank, f"OD loop done. Reducing matrices across {size} rank(s) …")
    _t0 = _time.perf_counter()
    matrix_acc = local_acc.sum(axis=0)  # reduce per-thread accumulators -> (npix,3,3)
    del local_acc
    matrix_all = _sum_reduce(comm, matrix_acc, rank)
    del matrix_acc
    hits_all = _sum_reduce(comm, hits_acc, rank)
    del hits_acc
    t_reduce = _time.perf_counter() - _t0
    _vprint(verbose, rank, f"Reduce done in {t_reduce:.2f}s. Solving T/Q/U …")

    if rank != 0:
        _od_other_total = t_od_wall_total - (
            t_resamp_total + t_conv_total + t_macc_total
            + t_flag_total + t_prep_total + t_pix_total
        )
        _wall = _time.perf_counter() - t_wall0
        # pix is its own bucket only when centering is on; otherwise pixelisation folds
        # into macc and t_pix_total is 0, so don't print a noise "pix=0.00s".
        _pix = f"  pix={t_pix_total:.2f}s" if hpx_center is not None else ""
        _vprint(
            verbose,
            rank,
            f"[Timing summary]"
            f"  setup={t_setup:.2f}s"
            f"  resamp={t_resamp_total:.2f}s"
            f"  conv={t_conv_total:.2f}s"
            f"  macc={t_macc_total:.2f}s"
            f"  flag={t_flag_total:.2f}s"
            f"  prep={t_prep_total:.2f}s"
            f"{_pix}"
            f"  od_other={_od_other_total:.2f}s"
            f"  reduce={t_reduce:.2f}s"
            f"  wall={_wall:.2f}s",
        )
        return None

    assert matrix_all is not None
    assert hits_all is not None
    _t0 = _time.perf_counter()
    t_map, q_map, u_map = solve_tqu_from_matrix(matrix_all)
    t_solve = _time.perf_counter() - _t0
    nobs00 = matrix_all[:, 0, 0]

    outdir = Path(config.output.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    prefix = config.output.output_prefix
    map_path = outdir / f"{prefix}_iqu.fits"
    hits_path = outdir / f"{prefix}_hits.fits"
    wpol_path = outdir / f"{prefix}_wpol.fits"
    nobs_path = outdir / f"{prefix}_nobs00.fits"

    _t0 = _time.perf_counter()
    hp.write_map(
        str(map_path),
        [t_map, q_map, u_map],
        overwrite=True,
        dtype=np.float64,
        nest=config.map.nest,
    )
    hp.write_map(str(hits_path), hits_all.astype(np.float64), overwrite=True, dtype=np.float64, nest=config.map.nest)
    hp.write_map(str(wpol_path), nobs00, overwrite=True, dtype=np.float64, nest=config.map.nest)
    hp.write_map(str(nobs_path), nobs00, overwrite=True, dtype=np.float64, nest=config.map.nest)
    t_write = _time.perf_counter() - _t0

    # Reconcile against the stopwatch: wall is the true elapsed time inside run_pipeline;
    # the section timers (resamp/conv/macc/flag/prep/pix/reduce/solve/write) plus setup and
    # od_other (residual per-OD work not in a named bucket) should sum to it, leaving only
    # a small "unaccounted" remainder (import-triggered lazy work, GC, etc.).
    od_other_total = t_od_wall_total - (
        t_resamp_total + t_conv_total + t_macc_total
        + t_flag_total + t_prep_total + t_pix_total
    )
    wall = _time.perf_counter() - t_wall0
    accounted = (
        t_setup + t_resamp_total + t_conv_total + t_macc_total
        + t_flag_total + t_prep_total + t_pix_total
        + od_other_total + t_reduce + t_solve + t_write
    )
    _pix = f"  pix={t_pix_total:.2f}s" if hpx_center is not None else ""
    _vprint(
        verbose,
        rank,
        f"[Timing summary]"
        f"  setup={t_setup:.2f}s"
        f"  resamp={t_resamp_total:.2f}s"
        f"  conv={t_conv_total:.2f}s"
        f"  macc={t_macc_total:.2f}s"
        f"  flag={t_flag_total:.2f}s"
        f"  prep={t_prep_total:.2f}s"
        f"{_pix}"
        f"  od_other={od_other_total:.2f}s"
        f"  reduce={t_reduce:.2f}s"
        f"  solve={t_solve:.2f}s"
        f"  write={t_write:.2f}s"
        f"  unaccounted={wall - accounted:.2f}s"
        f"  wall={wall:.2f}s",
    )

    return map_path


def main() -> None:
    """CLI entry point: parse ``--config``, run the pipeline, and print the output path."""
    parser = argparse.ArgumentParser(description="Run P-QUICK end-to-end pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML configuration")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = run_pipeline(cfg)
    if out is not None:
        print(f"Wrote map: {out}")


if __name__ == "__main__":
    main()
