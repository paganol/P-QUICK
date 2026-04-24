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
    load_pointing_npz,
    load_rimo_detectors,
    load_sky_alm,
    select_detectors,
    truncate_alm,
)
from .mapmaking import accumulate_tqu_matrix, init_map_matrix, solve_tqu_from_matrix
from .pointing import build_pointing_interpolator
from .quaternion import bore_det_to_angles, normalize_quaternion, quat_mul
from .utilities import build_pointing_file_paths, detector_map_weight, estimate_memory_per_rank_mb, extract_od_from_pointing_filename, parse_mission_length, print_mpi_distribution, resolve_nthreads


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

    mission = config.inputs.pointing.mission_length or "full"
    od_start, od_end = parse_mission_length(mission)
    all_pointing = build_pointing_file_paths(config.inputs.pointing.input_root, od_start, od_end)
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

        chunk_samples = max(1, (interp.n_native + n_chunks_cfg - 1) // n_chunks_cfg)
        n_chunks = (interp.n_native + chunk_samples - 1) // chunk_samples

        for chunk_idx, chunk_start in enumerate(range(0, interp.n_native, chunk_samples), start=1):
            chunk_end = min(chunk_start + chunk_samples, interp.n_native)
            chunk_len = chunk_end - chunk_start
            flag_chunk = interp.flag_native[chunk_start:chunk_end]
            good = (flag_chunk == 0) if config.inputs.pointing.use_flag else np.ones(chunk_len, dtype=bool)
            ngood = int(np.count_nonzero(good))
            _vprint(
                verbose,
                rank,
                f"  [chunk {chunk_idx}/{n_chunks}] samples {chunk_start}:{chunk_end}"
                f" | good={ngood}/{chunk_len} | n_native={interp.n_native}",
            )
            if not np.any(good):
                continue

            # Interpolate boresight once per chunk; apply per-detector offset in numpy.
            _t0 = _time.perf_counter()
            q_bore_good = interp.get_boresight_quaternions(chunk_start, chunk_len)[good]
            t_resamp_od += _time.perf_counter() - _t0

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

                _t0 = _time.perf_counter()
                theta, phi, psi = bore_det_to_angles(q_bore_good, det_quat)
                t_resamp_od += _time.perf_counter() - _t0

                _t0 = _time.perf_counter()
                # The Planck beam files (blm_*.fits) are defined in the Dxx frame
                # using the Ludwig III convention (PDD sec 5.5), where the co-polar
                # direction is along Y_Dxx, not X_Dxx.  ducc0 totalconvolve uses the
                # x-axis of the arriving frame as psi=0, so we must rotate psi by
                # -pi/2 to align the beam's co-polar axis with ducc0's reference.
                # The uncorrected psi (x-axis orientation) is still the correct
                # polarisation angle for the map-making normal equations below.
                psi_conv = psi - 0.5 * np.pi
                ptg = np.column_stack([theta, phi, psi_conv])
                tod = convolve_timeline(
                    sky_alm=sky_alm,
                    beam_alm=beam_alm,
                    ptg_thetaphipsi=ptg,
                    lmax=config.convolution.lmax,
                    mmax=config.convolution.mmax,
                    nthreads=nthreads,
                    epsilon=config.convolution.epsilon,
                    interpolator_cache=None,
                )
                del ptg, psi_conv
                t_conv_od += _time.perf_counter() - _t0

                _t0 = _time.perf_counter()
                pix = hp.ang2pix(config.map.nside, theta, phi, nest=config.map.nest)
                del theta, phi
                accumulate_tqu_matrix(matrix_acc, pix, psi, np.asarray(tod, dtype=np.float64), det_weight)
                np.add.at(hits_acc, pix, 1)
                del pix, psi, tod
                t_macc_od += _time.perf_counter() - _t0

            del q_bore_good

        t_resamp_total += t_resamp_od
        t_conv_total += t_conv_od
        t_macc_total += t_macc_od
        _vprint(
            verbose,
            rank,
            f"  [OD timing] resamp={t_resamp_od:.2f}s  conv={t_conv_od:.2f}s  macc={t_macc_od:.2f}s"
            f"  od_total={t_resamp_od + t_conv_od + t_macc_od:.2f}s",
        )

    matrix_all = _sum_reduce(comm, matrix_acc)
    del matrix_acc
    hits_all = _sum_reduce(comm, hits_acc)
    del hits_acc

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
