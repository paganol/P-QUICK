#!/usr/bin/env python3
"""Build per-detector **polmoments** files in qp_planck's format from the P-QUICK scan.

qp_planck (QuickPol) builds its effective window / transfer functions from per-detector
orientation "polmoments": for every detector it stores, per HEALPix pixel, the
hit-weighted scan orientation moments

    C_k(p) = sum_{i in p} cos(k psi_i)      S_k(p) = sum_{i in p} sin(k psi_i)

for k = 1..6 (12 columns), plus the hit count in a separate file. Computing them from
P-QUICK's actual pointing produces drop-in inputs for qp_planck's transfer-function
build, and lets the two codes' scan sampling be diffed directly.

Output, per detector ``<det>`` (qp_planck filenames), under ``--out-dir`` (one
sub-directory per mission when several are requested):
  * ``polmoments_<det>.fits``       -- 12 cols [C1,S1,...,C6,S6], RING, raw sums
  * ``polmoments_<det>_hits.fits``  -- 1 col "T" = hit count, RING

Everything is supplied either by a **YAML** (`--config`) with the blocks below, or by
CLI flags named like the YAML leaves (each overrides the YAML); unset input paths fall
back to the configs/default.yaml layout. The YAML structure is::

    inputs:            {rimo_file, pointings, flags, bad_rings_file, mission_length}
    detector_selection: {channel, detectors}
    resampling:        {coordinate_system}
    output:            {nside, output_dir}

CLI flags: --rimo_file --pointings --flags --bad_rings_file --mission_length
           --channel | --detectors  --coordinate_system  --nside  --output_dir
(`--channel` accepts a comma-separated list; `--mission_length` is a single value.)

    # from a YAML (inputs + selection + output all inside)
    python scripts/build_polmoments.py --config configs/polmoments_full_100ghz.yaml

    # pure CLI (default.yaml input paths), one or more channels
    python scripts/build_polmoments.py --channel 100ghz,143ghz \
        --mission_length full --nside 256 --output_dir out/pm

    # YAML for inputs, CLI to override the output nside/dir
    python scripts/build_polmoments.py --config configs/polmoments_full_100ghz.yaml \
        --nside 512 --output_dir out/pm_ns512

psi is the same per-sample orientation the map-making uses (`bore_det_to_angles`), and
the same flags the pipeline applies (common + bad-ring + per-horn) are honoured so the
hit counts line up with the maps.

**MPI**: detectors are processed one at a time (outer loop); for each, the ODs are
partitioned round-robin across ranks and its maps are sum-reduced to rank 0, which
writes the files — so a full-mission build scales ~linearly with ranks. Run e.g.
``srun -n 64 python scripts/build_polmoments.py ...`` (or ``mpirun -n 64 ...``); without
MPI it runs serially. Only one detector's 13 maps are resident per rank, so the
footprint is independent of the detector count: ~4.9 GB at nside 2048, ~0.08 GB at
nside 256 (the qp_planck reference).
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from types import SimpleNamespace

import healpy as hp
import numpy as np
import yaml

from pquick.config import DetectorSelection
from pquick.io import (
    load_horn_flag_npz,
    load_pointing_npz,
    load_rimo_detectors,
    select_detectors,
)
from pquick.pipeline import (
    _chunk_bad_ring_mask,
    _det_to_horn,
    _detector_channel_ghz,
    _get_mpi,
    _load_bad_ring_intervals,
    _local_slice,
    _sum_reduce,
)
from pquick.pointing import build_pointing_interpolator
from pquick.quaternion import bore_det_to_angles, normalize_quaternion
from pquick.utilities import (
    build_pointing_file_paths,
    extract_od_from_pointing_filename,
    parse_mission_length,
)

KMAX = 6  # qp_planck stores k = 1..6  -> 12 columns

# Feedback levels: 0 quiet (errors + final line only), 1 normal (headers + per-detector
# writes), 2 verbose (+ per-OD progress), 3 debug (+ per-OD timing).
DEFAULT_FEEDBACK = 1


def _vlog(fb: int, level: int, msg: str) -> None:
    """Print *msg* when the active feedback level is at least *level*."""
    if fb >= level:
        print(msg, flush=True)


# Standard inputs / resampling, matching configs/default.yaml. Used when --config is
# not given (or to fill any field a config leaves unset); individual --pointings /
# --rimo / --flags / --bad-rings / --coord override them.
DEFAULT_INPUTS = SimpleNamespace(
    pointings="inputs/pointings/pointing_",
    rimo_file="inputs/RIMOs/RIMO_HFI_npipe5v16_symmetrized.fits",
    flags="inputs/flags/flags_",
    bad_rings_file=None,
    mission_length="full",
)
DEFAULT_COORD = "galactic"
DEFAULT_NSIDE = 256

# coordinate_system -> FITS COORDSYS letter (matches build_pointing_interpolator frames).
_COORDSYS = {"galactic": "G", "ecliptic": "E", "equatorial": "C", "celestial": "C"}


def _build_cfg(args: argparse.Namespace):
    """Assemble the (inputs, resampling, detector_selection, output) the builder needs.

    Sources, in increasing precedence: configs/default.yaml-style defaults, the
    optional --config YAML, then explicit CLI flags (named like the YAML leaves).
    """
    inputs = SimpleNamespace(**vars(DEFAULT_INPUTS))
    coord = DEFAULT_COORD
    det_sel = DetectorSelection()
    nside = DEFAULT_NSIDE
    output_dir = None
    feedback = DEFAULT_FEEDBACK
    if args.config is not None:
        # Parse only the blocks polmoments needs; a full P-QUICK YAML works too.
        with open(args.config) as fh:
            data = yaml.safe_load(fh) or {}
        ic = data.get("inputs", {}) or {}
        for k in ("pointings", "rimo_file", "flags", "bad_rings_file", "mission_length"):
            if ic.get(k) is not None:
                setattr(inputs, k, str(ic[k]))
        rc = data.get("resampling", {}) or {}
        if rc.get("coordinate_system") is not None:
            coord = str(rc["coordinate_system"])
        ds = data.get("detector_selection", {}) or {}
        det_sel = DetectorSelection(channel=ds.get("channel"), detectors=list(ds.get("detectors") or []))
        oc = data.get("output", {}) or {}
        if oc.get("nside") is not None:
            nside = int(oc["nside"])
        if oc.get("output_dir") is not None:
            output_dir = str(oc["output_dir"])
        if data.get("feedback") is not None:
            feedback = int(data["feedback"])
        elif data.get("verbose") is not None:  # accept P-QUICK's bool 'verbose'
            feedback = 2 if bool(data["verbose"]) else 0
    # CLI overrides (flag names match the YAML leaves).
    if args.rimo_file is not None:
        inputs.rimo_file = args.rimo_file
    if args.pointings is not None:
        inputs.pointings = args.pointings
    if args.flags is not None:
        inputs.flags = args.flags
    if args.bad_rings_file is not None:
        inputs.bad_rings_file = args.bad_rings_file
    if args.mission_length is not None:
        inputs.mission_length = args.mission_length
    if args.coordinate_system is not None:
        coord = args.coordinate_system
    if args.nside is not None:
        nside = int(args.nside)
    if args.output_dir is not None:
        output_dir = str(args.output_dir)
    if args.feedback is not None:
        feedback = int(args.feedback)
    return SimpleNamespace(
        inputs=inputs,
        resampling=SimpleNamespace(coordinate_system=coord),
        detector_selection=det_sel,
        output=SimpleNamespace(nside=nside, output_dir=output_dir),
        feedback=feedback,
    )


def _resolve_detectors(args: argparse.Namespace, cfg, all_dets: list[str]) -> list[str]:
    """Pick the detector list from --detectors, --channel, or the config selection."""
    if args.detectors:
        explicit = {d.strip() for d in args.detectors.split(",") if d.strip()}
        sel = select_detectors(all_dets, DetectorSelection(detectors=sorted(explicit)))
    elif args.channel:
        sel_set: set[str] = set()
        for ch in (c.strip() for c in args.channel.split(",") if c.strip()):
            sel_set.update(select_detectors(all_dets, DetectorSelection(channel=ch)))
        sel = sorted(sel_set)
    elif cfg.detector_selection.channel or cfg.detector_selection.detectors:
        sel = select_detectors(all_dets, cfg.detector_selection)
    else:
        sel = []
    return sel


def build_detector(cfg, det, quat, od_paths, nside, out_dir, chunk, use_flags, bad_rings,
                   fb=DEFAULT_FEEDBACK, comm=None, rank=0, size=1):
    """Accumulate and write polmoments for ONE detector.

    ODs are partitioned round-robin across MPI ranks; each rank accumulates its
    subset, the maps are sum-reduced to rank 0, which writes them. Only this one
    detector's 13 maps are in memory at a time, so the footprint is independent of
    the number of detectors (~4.9 GB at nside 2048, ~0.08 GB at nside 256).
    """
    npix = hp.nside2npix(nside)
    fbe = fb if rank == 0 else -1  # only rank 0 prints
    horn = _det_to_horn(det)
    ch = _detector_channel_ghz(det)
    cos = np.zeros((KMAX, npix))
    sin = np.zeros((KMAX, npix))
    hits = np.zeros(npix)

    for oi, npz in enumerate(od_paths, 1):
        if (oi - 1) % size != rank:  # round-robin partition of ODs across ranks
            continue
        _t0 = time.perf_counter()
        if not Path(npz).exists():
            _vlog(fbe, 1, f"    [{det}] [OD {oi}/{len(od_paths)}] MISSING {Path(npz).name} (skip)")
            continue
        interp = build_pointing_interpolator(load_pointing_npz(npz), coordinate_system=cfg.resampling.coordinate_system)
        n = interp.n_native
        common_good = interp.flag_native == 0
        hflag = None
        if use_flags:
            od = extract_od_from_pointing_filename(npz)
            fp = Path(f"{cfg.inputs.flags}{ch:03d}ghz_od_{od:04d}.npz")
            hflag = load_horn_flag_npz(fp, horn, n_samples=n) if fp.exists() else np.zeros(n, np.int8)

        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            good = common_good[s:e].copy()
            if use_flags:
                good &= hflag[s:e] == 0
                if bad_rings is not None:
                    good &= ~_chunk_bad_ring_mask(bad_rings, det, interp.coarse_t0_ns, interp.native_rate_hz, s, e - s)
            if not np.any(good):
                continue
            q_bore = interp.get_boresight_quaternions(s, e - s)[good]
            _theta, _phi, psi = bore_det_to_angles(q_bore, quat)
            # qp_planck's orientation convention is offset by pi from P-QUICK's
            # bore_det_to_angles psi. This is invisible to the even (polarised) moments
            # but flips the odd ones; add pi so all k match qp_planck's polmoments.
            psi = psi + np.pi
            pix = hp.ang2pix(nside, _theta, _phi).astype(np.int64)  # RING
            hits += np.bincount(pix, minlength=npix)
            for k in range(1, KMAX + 1):
                cos[k - 1] += np.bincount(pix, weights=np.cos(k * psi), minlength=npix)
                sin[k - 1] += np.bincount(pix, weights=np.sin(k * psi), minlength=npix)
        _dt = time.perf_counter() - _t0
        _vlog(fbe, 2, f"    [{det}] [OD {oi}/{len(od_paths)}] {Path(npz).name} done"
                      + (f"  ({_dt:.1f}s)" if fb >= 3 else ""))
        # The interpolator is rebuilt once per OD per detector; drop it (and its
        # native-rate buffers) eagerly so nothing accumulates across the OD loop.
        del interp, common_good, hflag
        gc.collect()

    # Sum-reduce this detector's maps to rank 0 (no-op when serial).
    hits = _sum_reduce(comm, hits, rank)
    cos = _sum_reduce(comm, cos, rank)
    sin = _sum_reduce(comm, sin, rank)
    if rank != 0:
        del cos, sin, hits
        gc.collect()
        return
    coordsys = _COORDSYS.get(cfg.resampling.coordinate_system.strip().lower(), "G")
    cols = []
    names = []
    for k in range(KMAX):
        cols.append(cos[k])
        cols.append(sin[k])
        names += [f"SUM_COS_{k+1}_PSI", f"SUM_SIN_{k+1}_PSI"]  # toast_planck.polmoments names
    hp.write_map(str(out_dir / f"polmoments_{det}.fits"), cols, nest=False, overwrite=True,
                 column_names=names, coord=coordsys, dtype=[np.float64] * (2 * KMAX))
    hp.write_map(str(out_dir / f"polmoments_{det}_hits.fits"), hits, nest=False, overwrite=True,
                 column_names=["hits"], coord=coordsys, dtype=np.float64)
    nhit = int(np.count_nonzero(hits))
    _vlog(fbe, 1, f"  wrote polmoments_{det}.fits (+_hits)  hit pixels={nhit}  fsky={nhit/npix:.4f}")
    del cos, sin, hits, cols
    gc.collect()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None,
                   help="Optional YAML with inputs/detector_selection/resampling/output blocks (see module "
                        "docstring). Any field can be overridden by the flags below; configs/default.yaml-style "
                        "input paths are used for anything left unset.")
    # inputs.*
    p.add_argument("--rimo_file", type=str, default=None, help="RIMO FITS path.")
    p.add_argument("--pointings", type=str, default=None, help="Pointing file prefix (default: inputs/pointings/pointing_).")
    p.add_argument("--flags", type=str, default=None, help="Flag file prefix (default: inputs/flags/flags_).")
    p.add_argument("--bad_rings_file", type=str, default=None, help="Bad-ring intervals file (default: none).")
    p.add_argument("--mission_length", type=str, default=None,
                   help="Mission length, e.g. 'full' or 'survey 1' (single value).")
    # detector_selection.*
    p.add_argument("--channel", type=str, default=None, help="Channel(s), comma-separated, e.g. '100ghz,143ghz'.")
    p.add_argument("--detectors", type=str, default=None, help="Detector names, comma-separated (overrides --channel).")
    # resampling.*
    p.add_argument("--coordinate_system", type=str, default=None, help="Sky frame (default: galactic).")
    # output.*
    p.add_argument("--nside", type=int, default=None, help="Output nside (default: 256).")
    p.add_argument("--output_dir", type=str, default=None, help="Output directory for the polmoments files.")
    # utility
    p.add_argument("--feedback", type=int, default=None, choices=range(0, 4), metavar="0-3",
                   help="Verbosity: 0 quiet, 1 normal (default), 2 per-OD, 3 +timing. "
                        "Also from YAML top-level 'feedback' (or 'verbose').")
    p.add_argument("--chunk", type=int, default=2_000_000, help="Native samples per chunk.")
    p.add_argument("--no-flags", action="store_true", help="Ignore per-horn/bad-ring flags (common flags only).")
    args = p.parse_args()

    cfg = _build_cfg(args)
    if cfg.output.output_dir is None:
        p.error("output_dir is required (set output.output_dir in --config or pass --output_dir)")

    det_meta = load_rimo_detectors(cfg.inputs.rimo_file)
    detectors = _resolve_detectors(args, cfg, list(det_meta.keys()))
    if not detectors:
        p.error("no detectors selected (set detector_selection in --config or pass --channel/--detectors)")
    quats = {
        d: normalize_quaternion(np.asarray(det_meta[d].get("quat", np.array([0.0, 0.0, 0.0, 1.0])), dtype=np.float64))
        for d in detectors
    }
    mission = cfg.inputs.mission_length or "full"
    use_flags = (cfg.inputs.flags is not None) and (not args.no_flags)
    out_dir = Path(cfg.output.output_dir)
    nside = int(cfg.output.nside)
    comm, rank, size = _get_mpi()
    fb = cfg.feedback
    fbe = fb if rank == 0 else -1  # only rank 0 prints

    od_start, od_end = parse_mission_length(mission)
    od_paths = build_pointing_file_paths(cfg.inputs.pointings, od_start, od_end)
    bad_rings = (
        _load_bad_ring_intervals(cfg.inputs.bad_rings_file)
        if (cfg.inputs.bad_rings_file is not None and use_flags)
        else None
    )
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    _vlog(fbe, 1, f"detectors ({len(detectors)}): {detectors}")
    _vlog(fbe, 1, f"mission '{mission}'  ODs {od_start}..{od_end} ({len(od_paths)}, "
                  f"{len(_local_slice(od_paths, rank, size))}/rank over {size} rank(s))  "
                  f"nside={nside}  coord={cfg.resampling.coordinate_system}  flags={use_flags}  -> {out_dir}")

    # Detector-outer / OD-parallel: only one detector's maps are resident at a time.
    for det in detectors:
        build_detector(cfg, det, quats[det], od_paths, nside, out_dir, int(args.chunk),
                       use_flags, bad_rings, fb, comm, rank, size)
        gc.collect()  # reclaim this detector's maps before the next one allocates
    _vlog(fbe, 0, f"Done -> {out_dir}")


if __name__ == "__main__":
    main()
