from __future__ import annotations

import argparse
import time as _time
import warnings
from pathlib import Path
from typing import cast

import healpy as hp
import numpy as np
from ducc0.healpix import Healpix_Base

from .config import PipelineConfig, load_config
from .convolution import convolve_timeline
from .io import (
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
from .mapmaking import accumulate_tqu_matrix, solve_tqu_from_matrix
from .pointing import build_pointing_interpolator
from .quaternion import bore_det_to_angles, bore_det_to_ptg, normalize_quaternion, quat_mul
from .utilities import build_pointing_file_paths, detector_map_weight, estimate_memory_per_rank_mb, extract_od_from_pointing_filename, parse_mission_length, print_mpi_distribution, resolve_nthreads


def _load_bad_ring_intervals(path: str | Path) -> dict[str, list[tuple[float, float]]]:
    """Load a TOAST/NPIPE-style bad-ring interval file.

    The file format is plain text with rows:
        ``<det_or_ALL> <tstart_s> <tstop_s>``
    and optional comment lines starting with ``#``.
    """
    intervals: dict[str, list[tuple[float, float]]] = {}
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
            intervals.setdefault(det_key, []).append((tstart, tstop))
    return intervals


def _chunk_bad_ring_mask(
    intervals: dict[str, list[tuple[float, float]]] | None,
    det_name: str,
    coarse_t0_ns: float,
    native_rate_hz: float,
    chunk_start: int,
    chunk_len: int,
) -> np.ndarray:
    """Return a boolean mask of bad samples for one chunk and detector."""
    if not intervals:
        return np.zeros(chunk_len, dtype=bool)

    det_key = det_name.upper()
    det_intervals = intervals.get(det_key, []) + intervals.get("ALL", [])
    if not det_intervals:
        return np.zeros(chunk_len, dtype=bool)

    dt_s = 1.0 / float(native_rate_hz)
    t0_s = float(coarse_t0_ns) * 1.0e-9 + float(chunk_start) * dt_s
    times = t0_s + dt_s * np.arange(chunk_len, dtype=np.float64)

    out = np.zeros(chunk_len, dtype=bool)
    t_first = float(times[0])
    t_last = float(times[-1])
    for tstart, tstop in det_intervals:
        if tstart <= t_last and tstop >= t_first:
            out |= (times >= tstart) & (times <= tstop)
    return out


def _det_to_horn(detector: str) -> str:
    if detector and detector[-1] in "abMS":
        return detector[:-1]
    return detector


def _detector_channel_ghz(detector: str) -> int | None:
    try:
        head = detector.split("-", 1)[0]
        return int(head)
    except Exception:
        return None


def _get_mpi():
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        return comm, comm.rank, comm.size
    except Exception:
        return None, 0, 1


def _local_slice(items: list[Path], rank: int, size: int) -> list[Path]:
    return [x for i, x in enumerate(items) if i % size == rank]


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
    if enabled and rank == 0:
        print(msg, flush=True)


def run_pipeline(config: PipelineConfig) -> Path | None:
    """Execute the full beam-convolution map-making pipeline.

    Loads sky ALMs and beam ALMs, iterates over operational days and detectors,
    accumulates the polarised normal-equation matrix, solves for T/Q/U, and writes
    FITS output maps.  MPI-aware: ODs are distributed across ranks and results
    are reduced with ``Allreduce`` before writing.

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

    sky_alm = load_sky_alm(config.inputs.sky_alm)
    lmax_alm = infer_lmax_from_alm(sky_alm)
    if config.convolution.lmax > lmax_alm:
        raise ValueError(f"Configured lmax={config.convolution.lmax} exceeds sky alm lmax={lmax_alm}")
    sky_alm = truncate_alm(sky_alm, lmax_alm, config.convolution.lmax)

    det_meta = load_rimo_detectors(config.inputs.rimo_file)
    detectors = select_detectors(list(det_meta.keys()), config.detector_selection)

    det_info: list[dict[str, object]] = []
    for det in detectors:
        beam_file = detector_to_beam_file(config.inputs.beams_dir, det)
        beam_alm = load_beam_alm(
            beam_file,
            lmax=config.convolution.lmax,
            mmax=config.convolution.mmax,
        )
        beam_alm = normalize_beam_alm(
            beam_alm,
            mode=config.convolution.beam_normalization,
        )
        dmeta = det_meta.get(det, {})
        dquat = normalize_quaternion(
            np.asarray(dmeta.get("quat", np.array([0.0, 0.0, 0.0, 1.0])), dtype=np.float64)
        )
        det_info.append(
            {
                "name": det,
                "beam_alm": beam_alm,
                "quat": dquat,
                "weight": detector_map_weight(det),
            }
        )

    mission = config.inputs.mission_length or "full"
    od_start, od_end = parse_mission_length(mission)
    all_pointing = build_pointing_file_paths(config.inputs.pointings, od_start, od_end)

    bad_ring_intervals: dict[str, list[tuple[float, float]]] | None = None
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

    local_pointing = _local_slice(all_pointing, rank, size)

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

    matrix_acc = np.zeros((npix, 3, 3), dtype=np.float64)
    hits_acc = np.zeros(matrix_acc.shape[0], dtype=np.int64)
    n_chunks_cfg = int(max(1, config.convolution.chunks))


    t_resamp_total = t_conv_total = t_macc_total = 0.0

    for od_idx, npz_path in enumerate(local_pointing, start=1):
        _vprint(verbose, rank, f"[OD {od_idx}/{len(local_pointing)}] {npz_path.name}")
        t_resamp_od = t_conv_od = t_macc_od = 0.0

        _t0 = _time.perf_counter()
        point_us = load_pointing_npz(npz_path)
        interp = build_pointing_interpolator(
            point_us,
            coordinate_system=config.resampling.coordinate_system,
        )
        del point_us
        t_resamp_od += _time.perf_counter() - _t0

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

        chunk_samples = max(1, (interp.n_native + n_chunks_cfg - 1) // n_chunks_cfg)
        n_chunks = (interp.n_native + chunk_samples - 1) // chunk_samples
        # Pre-allocate reusable buffers for the pointing array and psi (mapmaking).
        # These are sized for the largest possible chunk and reused across all
        # chunks and detectors, eliminating ~7 × chunk_samples × 8-byte allocations
        # (theta, phi, psi, psi_conv, column_stack) per detector per chunk.
        ptg_buf = np.empty((chunk_samples, 3), dtype=np.float64)
        psi_buf = np.empty(chunk_samples, dtype=np.float64)

        for chunk_idx, chunk_start in enumerate(range(0, interp.n_native, chunk_samples), start=1):
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
            if use_flag_od:
                _first_flag = detector_flags[_first_det_name][chunk_start:chunk_end]
                _common_bad = interp.flag_native[chunk_start:chunk_end] != 0
                _good_first = ~(_common_bad | (_first_flag != 0) | _ring_bad_first)
            else:
                _good_first = ~_ring_bad_first
            _vprint(
                verbose,
                rank,
                f"  [chunk {chunk_idx}/{n_chunks}] samples {chunk_start}:{chunk_end}"
                f" | good={int(np.count_nonzero(_good_first))}/{chunk_len} | n_native={interp.n_native}",
            )

            for det_idx, dinfo in enumerate(det_info, start=1):
                det_quat = np.asarray(dinfo["quat"], dtype=np.float64)
                beam_alm = np.asarray(dinfo["beam_alm"], dtype=np.complex128)
                det_weight = cast(float, dinfo["weight"])
                det_name = str(dinfo["name"])

                _vprint(
                    verbose,
                    rank,
                    f"    [DET {det_idx}/{len(det_info)}] {det_name}",
                )

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
                    common_bad = interp.flag_native[chunk_start:chunk_end] != 0
                    good = ~(common_bad | (det_flag_chunk != 0) | ring_bad)
                else:
                    good = ~ring_bad

                ngood = int(np.count_nonzero(good))
                if ngood == 0:
                    continue

                q_bore_good = q_bore_all if ngood == chunk_len else q_bore_all[good]

                # Fill pre-allocated buffers directly — no temporary arrays.
                # ptg_buf[:, 0/1/2] = theta / phi / (psi - pi/2)  (ducc0 Ludwig-III offset)
                # psi_buf[:] = psi  (polarisation angle for mapmaking, without offset)
                _t0 = _time.perf_counter()
                bore_det_to_ptg(q_bore_good, det_quat, ptg_buf[:ngood], psi_buf[:ngood])
                t_resamp_od += _time.perf_counter() - _t0

                pix_center = None
                if hpx_center is not None:
                    # Snap (theta, phi) to HEALPix pixel centers to suppress
                    # subpixel pointing variation before convolution.
                    pix_center = hpx_center.ang2pix(
                        ptg_buf[:ngood, :2],
                        nthreads=nthreads,
                    )
                    ptg_buf[:ngood, :2] = hpx_center.pix2ang(pix_center)

                _t0 = _time.perf_counter()
                tod = convolve_timeline(
                    sky_alm=sky_alm,
                    beam_alm=beam_alm,
                    ptg_thetaphipsi=ptg_buf[:ngood],
                    lmax=config.convolution.lmax,
                    mmax=config.convolution.mmax,
                    nthreads=nthreads,
                    epsilon=config.convolution.epsilon,
                    interpolator_cache=None,
                )
                t_conv_od += _time.perf_counter() - _t0

                _t0 = _time.perf_counter()
                if pix_center is not None:
                    pix = np.asarray(pix_center, dtype=np.int64)
                else:
                    pix = hpx.ang2pix(
                        ptg_buf[:ngood, :2],
                        nthreads=nthreads,
                    )
                accumulate_tqu_matrix(matrix_acc, pix, psi_buf[:ngood], np.asarray(tod, dtype=np.float64), det_weight)
                np.add.at(hits_acc, pix, 1)
                del pix, tod
                t_macc_od += _time.perf_counter() - _t0

            del q_bore_all

        t_resamp_total += t_resamp_od
        t_conv_total += t_conv_od
        t_macc_total += t_macc_od
        _vprint(
            verbose,
            rank,
            f"  [OD timing] resamp={t_resamp_od:.2f}s  conv={t_conv_od:.2f}s  macc={t_macc_od:.2f}s"
            f"  od_total={t_resamp_od + t_conv_od + t_macc_od:.2f}s",
        )

    _vprint(verbose, rank, f"OD loop done. Reducing matrices across {size} rank(s) …")
    _t0 = _time.perf_counter()
    matrix_all = _sum_reduce(comm, matrix_acc, rank)
    del matrix_acc
    hits_all = _sum_reduce(comm, hits_acc, rank)
    del hits_acc
    t_reduce = _time.perf_counter() - _t0
    _vprint(verbose, rank, f"Reduce done in {t_reduce:.2f}s. Solving T/Q/U …")

    if rank != 0:
        _vprint(
            verbose,
            rank,
            f"[Timing summary]"
            f"  resamp={t_resamp_total:.2f}s"
            f"  conv={t_conv_total:.2f}s"
            f"  macc={t_macc_total:.2f}s"
            f"  reduce={t_reduce:.2f}s"
            f"  total={t_resamp_total + t_conv_total + t_macc_total + t_reduce:.2f}s",
        )
        return None

    assert matrix_all is not None
    assert hits_all is not None
    _t0 = _time.perf_counter()
    t_map, q_map, u_map = solve_tqu_from_matrix(matrix_all)
    t_solve = _time.perf_counter() - _t0
    _vprint(
        verbose,
        rank,
        f"[Timing summary]"
        f"  resamp={t_resamp_total:.2f}s"
        f"  conv={t_conv_total:.2f}s"
        f"  macc={t_macc_total:.2f}s"
        f"  reduce={t_reduce:.2f}s"
        f"  solve={t_solve:.2f}s"
        f"  total={t_resamp_total + t_conv_total + t_macc_total + t_reduce + t_solve:.2f}s",
    )
    nobs00 = matrix_all[:, 0, 0]

    outdir = Path(config.output.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    prefix = config.output.output_prefix
    map_path = outdir / f"{prefix}_iqu.fits"
    hits_path = outdir / f"{prefix}_hits.fits"
    wpol_path = outdir / f"{prefix}_wpol.fits"
    nobs_path = outdir / f"{prefix}_nobs00.fits"

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
