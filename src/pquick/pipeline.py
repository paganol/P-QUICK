from __future__ import annotations

import argparse
import time as _time
from pathlib import Path
from typing import cast

import healpy as hp
import numpy as np

from .config import PipelineConfig, load_config
from .convolution import convolve_timeline
from .io import (
    detector_to_beam_file,
    infer_lmax_from_alm,
    load_beam_alm,
    load_horn_flag_npz,
    load_pointing_npz,
    load_rimo_detectors,
    load_sky_alm,
    select_detectors,
    truncate_alm,
)
from .mapmaking import accumulate_tqu_matrix, init_map_matrix, solve_tqu_from_matrix
from .pointing import build_pointing_interpolator
from .quaternion import bore_det_to_angles, bore_det_to_ptg, normalize_quaternion, quat_mul
from .utilities import build_pointing_file_paths, detector_map_weight, estimate_memory_per_rank_mb, extract_od_from_pointing_filename, parse_mission_length, print_mpi_distribution, resolve_nthreads


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


def _sum_reduce(comm, arr: np.ndarray) -> np.ndarray:
    if comm is None:
        return arr
    from mpi4py import MPI

    comm.Allreduce(MPI.IN_PLACE, arr)
    return arr


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
    local_pointing = _local_slice(all_pointing, rank, size)

    if verbose:
        local_ods = [extract_od_from_pointing_filename(p) for p in local_pointing]
        print_mpi_distribution(comm, rank, size, local_ods)
        est_mb = estimate_memory_per_rank_mb(config.map.nside)
        _vprint(
            verbose,
            rank,
            f"[Memory] Estimated peak per rank: {est_mb / 1024:.2f} GB"
            f" (nside={config.map.nside}, npix={12 * config.map.nside**2:,})"
            f" | for {size} ranks: {size * est_mb / 1024:.1f} GB total",
        )

    _vprint(verbose, rank, f"Starting pipeline: {len(local_pointing)} ODs on rank {rank}/{size}")

    matrix_acc = init_map_matrix(config.map.nside)
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
                    _vprint(verbose, rank, f"  [flags] no flag file for ch={ch} OD {od:04d}, skipping flags")
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

            # Compute good-sample mask for the first detector (common to all detectors
            # when flags are absent; printed before entering the detector loop).
            if use_flag_od:
                _first_flag = detector_flags[str(det_info[0]["name"])][chunk_start:chunk_end]
                _common_bad = interp.flag_native[chunk_start:chunk_end] != 0
                _good_first = ~(_common_bad | (_first_flag != 0))
            else:
                _good_first = np.ones(chunk_len, dtype=bool)
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

                if use_flag_od:
                    det_flag_chunk = detector_flags[det_name][chunk_start:chunk_end]
                    # Keep compatibility with any legacy per-pointing bad samples.
                    common_bad = interp.flag_native[chunk_start:chunk_end] != 0
                    good = ~(common_bad | (det_flag_chunk != 0))
                else:
                    good = np.ones(chunk_len, dtype=bool)

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
                pix = hp.ang2pix(config.map.nside, ptg_buf[:ngood, 0], ptg_buf[:ngood, 1], nest=config.map.nest)
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
    matrix_all = _sum_reduce(comm, matrix_acc)
    del matrix_acc
    hits_all = _sum_reduce(comm, hits_acc)
    del hits_acc
    _vprint(verbose, rank, "Reduce done. Solving T/Q/U …")

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
        f"  solve={t_solve:.2f}s"
        f"  total={t_resamp_total + t_conv_total + t_macc_total + t_solve:.2f}s",
    )
    nobs00 = matrix_all[:, 0, 0]

    if rank != 0:
        return None

    outdir = Path(config.output.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    prefix = config.map.output_prefix
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
