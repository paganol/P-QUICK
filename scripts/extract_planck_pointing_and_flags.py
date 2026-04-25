#!/usr/bin/env python3
"""
extract_planck_pointing_and_flags.py
------------------------------------
NERSC-only extractor for Planck NPIPE data.

Produces two sets of compressed NPZ files:

1) Pointing file (one per OD):
     pointing_od_{OD:04d}.npz
   Keys: t0_ns, qx, qy, qz, qs, sampling_rate_hz, idx_first, idx_last, idx_step

2) Per-channel flag file (one per channel per OD):
     flags_{FREQ:03d}ghz_od_{OD:04d}.npz
   Keys: {horn_name} -> uint8 packed-bit array
         horn_names, n_samples, sampling_rate_hz

Flag recipe (NPIPE defaults):
   common_bad       = (ptg_obt_flag & 0x01) | (ptg_att_flag & 0x02) | (ch_obt_flag & 0x01)
   horn_bad         = OR over arms: (det_flag & 0x01)
   final_flag[horn] = (common_bad | horn_bad) != 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

# ---------------------------------------------------------------------------
# NERSC paths
# ---------------------------------------------------------------------------
MISS03_DIR = Path("/global/cfs/cdirs/cmb/data/planck2020/npipe/hfi_miss03")
REPROCD_DIR = Path("/global/cfs/cdirs/cmb/data/planck2020/npipe/hfi_toi_reprocessed")
OUTPUT_DIR = Path("./planck_extracted_data")

UNDERSAMPLE_FACTOR = 1000

OBT_MASK = np.uint8(0x01)
ATT_MASK = np.uint8(0x02)
DET_MASK = np.uint8(0x01)

# ---------------------------------------------------------------------------
# Horn groupings: ordered list of (horn_name, [det, ...])
# ---------------------------------------------------------------------------
CHANNEL_HORNS: dict[int, list[tuple[str, list[str]]]] = {
    100: [
        ("100-1", ["100-1a", "100-1b"]),
        ("100-2", ["100-2a", "100-2b"]),
        ("100-3", ["100-3a", "100-3b"]),
        ("100-4", ["100-4a", "100-4b"]),
    ],
    143: [
        ("143-1", ["143-1a", "143-1b"]),
        ("143-2", ["143-2a", "143-2b"]),
        ("143-3", ["143-3a", "143-3b"]),
        ("143-4", ["143-4a", "143-4b"]),
        ("143-5", ["143-5"]),
        ("143-6", ["143-6"]),
        ("143-7", ["143-7"]),
    ],
    217: [
        ("217-1", ["217-1"]),
        ("217-2", ["217-2"]),
        ("217-3", ["217-3"]),
        ("217-4", ["217-4"]),
        ("217-5", ["217-5a", "217-5b"]),
        ("217-6", ["217-6a", "217-6b"]),
        ("217-7", ["217-7a", "217-7b"]),
        ("217-8", ["217-8a", "217-8b"]),
    ],
    353: [
        ("353-1", ["353-1"]),
        ("353-2", ["353-2"]),
        ("353-3", ["353-3a", "353-3b"]),
        ("353-4", ["353-4a", "353-4b"]),
        ("353-5", ["353-5a", "353-5b"]),
        ("353-6", ["353-6a", "353-6b"]),
        ("353-7", ["353-7"]),
        ("353-8", ["353-8"]),
    ],
    545: [
        ("545-1", ["545-1"]),
        ("545-2", ["545-2"]),
        ("545-4", ["545-4"]),
    ],
    857: [
        ("857-1", ["857-1"]),
        ("857-2", ["857-2"]),
        ("857-3", ["857-3"]),
        ("857-4", ["857-4"]),
    ],
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _find_pointing_fits(miss03_dir: Path, od_str: str) -> Path | None:
    p = miss03_dir / od_str / f"pointing-{od_str}.fits"
    return p if p.exists() else None


def _find_channel_fits(base_dir: Path, od_str: str, freq: int) -> Path | None:
    for p in sorted((base_dir / od_str).glob(f"H{freq:03d}_{od_str}_R*.fits")):
        return p
    return None


def _sampling_rate_hz(obt_ns: np.ndarray) -> float:
    if obt_ns.size < 2:
        return np.nan
    return 1.0 / (float(np.median(np.diff(obt_ns))) * 1e-9)


# ---------------------------------------------------------------------------
# Inspect helper
# ---------------------------------------------------------------------------

def inspect_fits(path: Path) -> None:
    print(f"\n{'=' * 60}\nFile: {path}")
    with fits.open(path) as hdul:
        hdul.info()
        for i, hdu in enumerate(hdul):
            if hasattr(hdu, "columns") and hdu.columns:
                print(f"\n  HDU {i}  name={hdu.name!r}")
                for col in hdu.columns:
                    print(f"    {col.name:20s} {col.format}")


# ---------------------------------------------------------------------------
# Pointing extraction
# ---------------------------------------------------------------------------

def extract_pointing(
    pointing_fits: Path,
    od_str: str,
    output_dir: Path,
    undersample: int,
) -> tuple[Path, np.ndarray]:
    """
    Undersample boresight quaternions, save pointing NPZ.
    Returns (out_path, common_flag_native_uint8).
    """
    with fits.open(pointing_fits) as hdul:
        obt_time = np.asarray(hdul[1].data["OBT"], dtype=np.int64)
        obt_flag = np.asarray(hdul[1].data["FLAG"], dtype=np.uint8)
        qx = np.asarray(hdul[3].data["QUATERNION_X"], dtype=np.float64)
        qy = np.asarray(hdul[3].data["QUATERNION_Y"], dtype=np.float64)
        qz = np.asarray(hdul[3].data["QUATERNION_Z"], dtype=np.float64)
        qs = np.asarray(hdul[3].data["QUATERNION_S"], dtype=np.float64)
        att_flag = np.asarray(hdul[3].data["FLAG"], dtype=np.uint8)

    n = obt_time.size
    if n == 0:
        raise ValueError(f"No pointing samples in {pointing_fits}")

    indices = np.arange(0, n, undersample, dtype=np.int64)
    if indices[-1] != n - 1:
        indices = np.append(indices, n - 1)

    common_flag = (obt_flag & OBT_MASK) | (att_flag & ATT_MASK)

    out_path = output_dir / f"pointing_od_{od_str}.npz"
    np.savez_compressed(
        out_path,
        t0_ns=np.array([float(obt_time[indices[0]])], dtype=np.float64),
        qx=qx[indices],
        qy=qy[indices],
        qz=qz[indices],
        qs=qs[indices],
        sampling_rate_hz=_sampling_rate_hz(obt_time),
        idx_first=np.array([indices[0]], dtype=np.int64),
        idx_last=np.array([indices[-1]], dtype=np.int64),
        idx_step=np.array([undersample], dtype=np.int64),
    )
    return out_path, common_flag


# ---------------------------------------------------------------------------
# Channel flag extraction (packed bits per horn)
# ---------------------------------------------------------------------------

def extract_channel_flags_packed(
    channel_fits: Path,
    od_str: str,
    freq: int,
    common_flag: np.ndarray,
    output_dir: Path,
) -> Path:
    """
    Build per-horn packed-bit bad flags for one channel and save as NPZ.
    """
    horn_defs = CHANNEL_HORNS[freq]
    payload: dict[str, np.ndarray] = {}

    with fits.open(channel_fits) as hdul:
        ch_obt_flag = np.asarray(hdul[1].data["FLAG"], dtype=np.uint8)
        obt_ch = np.asarray(hdul[1].data["OBT"], dtype=np.int64)

        n_samples = min(common_flag.size, ch_obt_flag.size)
        if n_samples == 0:
            raise ValueError(f"No channel samples in {channel_fits}")

        ch_common = (common_flag[:n_samples] | (ch_obt_flag[:n_samples] & OBT_MASK)).astype(np.uint8)

        for horn, dets in horn_defs:
            horn_bad = np.zeros(n_samples, dtype=np.uint8)
            for det in dets:
                try:
                    det_flag = np.asarray(hdul[det].data["FLAG"], dtype=np.uint8)
                except KeyError:
                    print(
                        f"    [WARN] Missing HDU '{det}' in {channel_fits.name} -- marking '{horn}' fully bad"
                    )
                    det_flag = np.ones(n_samples, dtype=np.uint8)

                if det_flag.size < n_samples:
                    pad = np.ones(n_samples, dtype=np.uint8)
                    pad[: det_flag.size] = det_flag
                    det_flag = pad

                horn_bad |= (det_flag[:n_samples] & DET_MASK)

            bad = (ch_common | horn_bad) != 0
            payload[horn] = np.packbits(bad, bitorder="little")

    payload["horn_names"] = np.asarray([h for h, _ in horn_defs], dtype="U16")
    payload["n_samples"] = np.asarray(n_samples, dtype=np.uint32)
    payload["sampling_rate_hz"] = np.asarray(_sampling_rate_hz(obt_ch[:n_samples]), dtype=np.float64)

    out_path = output_dir / f"flags_{freq:03d}ghz_od_{od_str}.npz"
    np.savez_compressed(out_path, **payload)
    return out_path


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_horn_flag(npz_path: str | Path, horn: str) -> np.ndarray:
    """
    Return bool array (True = bad) for requested horn.
    """
    with np.load(npz_path, allow_pickle=False) as data:
        horn_names = data["horn_names"].astype(str)
        if horn not in horn_names:
            raise KeyError(f"Horn '{horn}' not in {npz_path}. Available: {list(horn_names)}")
        n_samples = int(data["n_samples"])
        packed = np.asarray(data[horn], dtype=np.uint8)

    return np.unpackbits(packed, bitorder="little")[:n_samples].astype(bool)


def det_to_horn(det: str) -> str:
    """Map detector name to horn name: '100-1a' -> '100-1'."""
    return det[:-1] if det[-1] in "ab" else det


# ---------------------------------------------------------------------------
# Per-OD driver
# ---------------------------------------------------------------------------

def process_od(
    od_str: str,
    channels: list[int],
    miss03_dir: Path,
    reprocd_dir: Path,
    output_dir: Path,
    undersample: int,
    overwrite: bool,
) -> None:
    pointing_fits = _find_pointing_fits(miss03_dir, od_str)
    if pointing_fits is None:
        print(f"OD {od_str}: [SKIP] pointing FITS not found")
        return

    ptg_out = output_dir / f"pointing_od_{od_str}.npz"
    if ptg_out.exists() and not overwrite:
        print(f"OD {od_str}: pointing exists, re-reading common flags from FITS")
        with fits.open(pointing_fits) as hdul:
            obt_flag = np.asarray(hdul[1].data["FLAG"], dtype=np.uint8)
            att_flag = np.asarray(hdul[3].data["FLAG"], dtype=np.uint8)
        common_flag = (obt_flag & OBT_MASK) | (att_flag & ATT_MASK)
    else:
        print(f"OD {od_str}: extracting pointing ...", end=" ", flush=True)
        ptg_out, common_flag = extract_pointing(pointing_fits, od_str, output_dir, undersample)
        print(f"ok [{ptg_out.name}]")

    for freq in channels:
        if freq not in CHANNEL_HORNS:
            print(f"  {freq} GHz: [SKIP] unsupported channel")
            continue

        flag_out = output_dir / f"flags_{freq:03d}ghz_od_{od_str}.npz"
        if flag_out.exists() and not overwrite:
            print(f"  {freq} GHz: flag file exists, skipping")
            continue

        ch_fits = _find_channel_fits(reprocd_dir, od_str, freq)
        if ch_fits is None:
            print(f"  {freq} GHz: [SKIP] channel FITS not found in reprocessed dir")
            continue

        print(f"  {freq} GHz: packed-bit flags from {ch_fits.name} ...", end=" ", flush=True)
        result = extract_channel_flags_packed(ch_fits, od_str, freq, common_flag, output_dir)
        print(f"ok [{result.name}]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Planck pointing + packed-bit horn flags (NERSC)")
    parser.add_argument("--miss03-dir", default=str(MISS03_DIR))
    parser.add_argument("--reprocd-dir", default=str(REPROCD_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--od-range", nargs=2, type=int, metavar=("START", "END"), default=[91, 974])
    parser.add_argument("--channels", nargs="+", type=int, default=list(CHANNEL_HORNS.keys()))
    parser.add_argument("--undersample", type=int, default=UNDERSAMPLE_FACTOR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    miss03_dir = Path(args.miss03_dir)
    reprocd_dir = Path(args.reprocd_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ods = [f"{od:04d}" for od in range(args.od_range[0], args.od_range[1] + 1)]

    if args.inspect:
        for od_str in ods:
            ptg = _find_pointing_fits(miss03_dir, od_str)
            if ptg:
                inspect_fits(ptg)
                for freq in args.channels:
                    ch = _find_channel_fits(reprocd_dir, od_str, freq)
                    if ch:
                        inspect_fits(ch)
                return
        print("No files found for inspect.")
        return

    print(f"Processing ODs {args.od_range[0]}-{args.od_range[1]}")
    print(f"Channels: {args.channels}")
    print(f"Output:   {output_dir}")
    print("-" * 60)

    for od_str in ods:
        try:
            process_od(
                od_str=od_str,
                channels=args.channels,
                miss03_dir=miss03_dir,
                reprocd_dir=reprocd_dir,
                output_dir=output_dir,
                undersample=args.undersample,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            print(f"OD {od_str}: [ERROR] {exc}")


if __name__ == "__main__":
    main()
