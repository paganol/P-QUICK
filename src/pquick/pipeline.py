from __future__ import annotations

import argparse
from pathlib import Path

import healpy as hp
import numpy as np

from .config import PipelineConfig, load_config
from .convolution import convolve_timeline
from .io import (
    detector_to_beam_file,
    discover_pointing_files,
    infer_lmax_from_alm,
    load_beam_alm,
    load_pointing_npz,
    load_rimo_detectors,
    load_sky_alm,
    select_detectors,
)
from .mapmaking import accumulate_simple_iqu, finalize_simple_iqu
from .pointing import reconstruct_native_pointing
from .quaternion import normalize_quaternion, quat_mul, quaternion_to_thetaphipsi
from .weights import detector_map_weight


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
    out = np.zeros_like(arr)
    comm.Allreduce(arr, out)
    return out


def run_pipeline(config: PipelineConfig) -> Path | None:
    comm, rank, size = _get_mpi()

    sky_alm = load_sky_alm(config.inputs.sky_alm)
    lmax_alm = infer_lmax_from_alm(sky_alm)
    if config.convolution.lmax > lmax_alm:
        raise ValueError(f"Configured lmax={config.convolution.lmax} exceeds sky alm lmax={lmax_alm}")

    det_meta = load_rimo_detectors(config.inputs.rimo_files)
    detectors = select_detectors(list(det_meta.keys()), config.detector_selection)

    all_pointing = discover_pointing_files(config.inputs.pointing.npz_glob)
    local_pointing = _local_slice(all_pointing, rank, size)

    npix = hp.nside2npix(config.map.nside)
    i_num_acc = np.zeros(npix, dtype=np.float64)
    q_num_acc = np.zeros(npix, dtype=np.float64)
    u_num_acc = np.zeros(npix, dtype=np.float64)
    i_den_acc = np.zeros(npix, dtype=np.float64)
    hits_acc = np.zeros(npix, dtype=np.int64)
    wpol_acc = np.zeros(npix, dtype=np.float64)

    for npz_path in local_pointing:
        point_us = load_pointing_npz(npz_path)
        native = reconstruct_native_pointing(point_us, angular_eps=config.resampling.angular_eps)
        for det in detectors:
            dmeta = det_meta.get(det, {})
            dquat = np.asarray(dmeta.get("quat", np.array([0.0, 0.0, 0.0, 1.0])), dtype=np.float64)
            dquat = normalize_quaternion(dquat)

            q_det = quat_mul(native.quat_native, np.broadcast_to(dquat, native.quat_native.shape))
            theta, phi, psi = quaternion_to_thetaphipsi(q_det)
            ptg = np.column_stack([theta, phi, psi])

            beam_file = detector_to_beam_file(config.inputs.beams_dir, det)
            beam_alm = load_beam_alm(beam_file)

            tod = convolve_timeline(
                sky_alm=sky_alm,
                beam_alm=beam_alm,
                ptg_thetaphipsi=ptg,
                lmax=config.convolution.lmax,
                kmax=config.convolution.kmax,
                nthreads=config.convolution.nthreads,
                separate=config.convolution.separate,
                epsilon=config.convolution.epsilon,
            )
            tod = np.where(native.flag_native == 0, tod, np.nan)

            binned = accumulate_simple_iqu(
                theta=theta,
                phi=phi,
                psi=psi,
                tod=np.nan_to_num(tod, nan=0.0),
                flags=native.flag_native,
                nside=config.map.nside,
                det_weight=detector_map_weight(det),
                nest=config.map.nest,
            )
            i_num_acc += binned["i_num"]
            q_num_acc += binned["q_num"]
            u_num_acc += binned["u_num"]
            i_den_acc += binned["i_den"]
            hits_acc += binned["hits"]
            wpol_acc += binned["wpol"]

    i_num_all = _sum_reduce(comm, i_num_acc)
    q_num_all = _sum_reduce(comm, q_num_acc)
    u_num_all = _sum_reduce(comm, u_num_acc)
    i_den_all = _sum_reduce(comm, i_den_acc)
    hits_all = _sum_reduce(comm, hits_acc)
    wpol_all = _sum_reduce(comm, wpol_acc)

    maps = finalize_simple_iqu(
        {
            "i_num": i_num_all,
            "q_num": q_num_all,
            "u_num": u_num_all,
            "i_den": i_den_all,
            "hits": hits_all,
            "wpol": wpol_all,
        }
    )

    if rank != 0:
        return None

    outdir = Path(config.output.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    prefix = config.map.output_prefix
    map_path = outdir / f"{prefix}_iqu.fits"
    hits_path = outdir / f"{prefix}_hits.fits"
    wpol_path = outdir / f"{prefix}_wpol.fits"

    hp.write_map(
        str(map_path),
        [maps["I"], maps["Q"], maps["U"]],
        overwrite=True,
        dtype=np.float64,
        nest=config.map.nest,
    )
    hp.write_map(str(hits_path), hits_all.astype(np.float64), overwrite=True, dtype=np.float64, nest=config.map.nest)
    hp.write_map(str(wpol_path), wpol_all, overwrite=True, dtype=np.float64, nest=config.map.nest)

    return map_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P-QUICK end-to-end pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML configuration")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = run_pipeline(cfg)
    if out is not None:
        print(f"Wrote map: {out}")


if __name__ == "__main__":
    main()
