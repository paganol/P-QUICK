from __future__ import annotations

import glob
import re
from pathlib import Path

import healpy as hp
import numpy as np
from astropy.io import fits

from .config import DetectorSelection
from .pointing import PointingData
from .quaternion import normalize_quaternion


def _required_npz_keys() -> tuple[str, ...]:
    return ("time", "qx", "qy", "qz", "qs", "flag_ext1", "flag_ext3", "sampling_rate_hz")


def load_pointing_npz(path: str | Path) -> PointingData:
    p = Path(path)
    with np.load(p, allow_pickle=False) as data:
        missing = [k for k in _required_npz_keys() if k not in data]
        if missing:
            raise ValueError(f"{p} missing keys: {missing}")

        time_us = np.asarray(data["time"], dtype=np.float64)
        qx = np.asarray(data["qx"], dtype=np.float64)
        qy = np.asarray(data["qy"], dtype=np.float64)
        qz = np.asarray(data["qz"], dtype=np.float64)
        qs = np.asarray(data["qs"], dtype=np.float64)
        flag_ext1 = np.asarray(data["flag_ext1"], dtype=np.int8)
        flag_ext3 = np.asarray(data["flag_ext3"], dtype=np.int8)
        sampling_rate_hz = float(np.asarray(data["sampling_rate_hz"]).reshape(-1)[0])

    if not (time_us.size == qx.size == qy.size == qz.size == qs.size):
        raise ValueError("undersampled quaternion arrays must have identical length")
    quat_us = normalize_quaternion(np.stack([qx, qy, qz, qs], axis=-1))
    return PointingData(
        time_us=time_us,
        quat_us=quat_us,
        flag_ext1=flag_ext1,
        flag_ext3=flag_ext3,
        sampling_rate_hz=sampling_rate_hz,
    )


def discover_pointing_files(npz_glob: str) -> list[Path]:
    paths = [Path(p) for p in sorted(glob.glob(npz_glob))]
    if not paths:
        raise FileNotFoundError(f"No pointing files found for glob: {npz_glob}")
    return paths


def load_sky_alm(path: str | Path) -> np.ndarray:
    p = Path(path)
    suf = p.suffix.lower()

    if suf in {".fits", ".fit"}:
        alms = hp.read_alm(str(p), hdu=(1, 2, 3))
        return np.asarray(alms, dtype=np.complex128)

    if suf == ".npy":
        arr = np.load(p)
        return _coerce_alm_shape(arr)

    if suf == ".npz":
        with np.load(p, allow_pickle=False) as d:
            if "alm" in d:
                arr = d["alm"]
            elif "alms" in d:
                arr = d["alms"]
            else:
                raise ValueError(f"{p} must contain key 'alm' or 'alms'")
        return _coerce_alm_shape(arr)

    raise ValueError(f"Unsupported sky alm format: {p}")


def _coerce_alm_shape(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim == 1:
        out = out[None, :]
    if out.ndim != 2:
        raise ValueError("alm array must have shape (ncomp, nalm) or (nalm,)")
    if out.shape[0] not in (1, 3):
        raise ValueError("ncomp must be 1 or 3")
    return np.asarray(out, dtype=np.complex128)


def infer_lmax_from_alm(alm: np.ndarray) -> int:
    nalm = int(alm.shape[1])
    return hp.Alm.getlmax(nalm)


def _to_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value).strip()


def _rimo_detector_quat(phi_uv_deg: float, theta_uv_deg: float, psi_uv_deg: float) -> np.ndarray:
    degree = np.pi / 180.0
    phi = phi_uv_deg * degree
    theta = theta_uv_deg * degree
    psi = (psi_uv_deg) * degree - phi

    quat = np.zeros(4, dtype=np.float64)
    quat[3] = np.cos(0.5 * theta) * np.cos(0.5 * (phi + psi))
    quat[0] = -np.sin(0.5 * theta) * np.sin(0.5 * (phi - psi))
    quat[1] = np.sin(0.5 * theta) * np.cos(0.5 * (phi - psi))
    quat[2] = np.cos(0.5 * theta) * np.sin(0.5 * (phi + psi))
    return normalize_quaternion(quat)


def load_rimo_detectors(rimo_paths: list[str]) -> dict[str, dict[str, np.ndarray | float]]:
    out: dict[str, dict[str, np.ndarray | float]] = {}
    for rimo in rimo_paths:
        with fits.open(rimo) as hdul:
            tab = hdul[1].data
            names = [n.upper() for n in tab.columns.names]

            det_col = "DETECTOR" if "DETECTOR" in names else names[0]
            has_phi = "PHI_UV" in names
            has_theta = "THETA_UV" in names
            has_psi = "PSI_UV" in names

            for row in tab:
                det = _to_text(row[det_col])
                rec: dict[str, np.ndarray | float] = {}
                if has_phi and has_theta and has_psi:
                    phi_uv = float(row["PHI_UV"])
                    theta_uv = float(row["THETA_UV"])
                    psi_uv = float(row["PSI_UV"])
                    rec["phi_uv"] = phi_uv
                    rec["theta_uv"] = theta_uv
                    rec["psi_uv"] = psi_uv
                    rec["quat"] = _rimo_detector_quat(phi_uv, theta_uv, psi_uv)
                out[det] = rec
    if not out:
        raise ValueError("No detectors loaded from RIMO files")
    return out


def select_detectors(all_detectors: list[str], selection: DetectorSelection) -> list[str]:
    selected = list(all_detectors)

    if selection.channel:
        tag = selection.channel.strip().lower()
        selected = [d for d in selected if d.lower().startswith(tag)]

    if selection.detectors:
        allowed = {d.strip() for d in selection.detectors}
        selected = [d for d in selected if d in allowed]

    for pat in selection.include_regex:
        rx = re.compile(pat)
        selected = [d for d in selected if rx.search(d)]

    for pat in selection.exclude_regex:
        rx = re.compile(pat)
        selected = [d for d in selected if not rx.search(d)]

    if not selected:
        raise ValueError("Detector selection is empty")
    return sorted(selected)


def detector_to_beam_file(beams_dir: str | Path, detector: str) -> Path:
    bdir = Path(beams_dir)

    cand = [
        bdir / f"blm_{detector}.fits",
        bdir / f"blm_{detector.replace('_', '-')}.fits",
        bdir / f"blm_{detector.lower()}.fits",
        bdir / f"blm_{detector.lower().replace('_', '-')}.fits",
    ]
    for c in cand:
        if c.exists():
            return c

    # Fallback: try suffix match against inventory.
    inv = list(bdir.glob("blm_*.fits"))
    norm = detector.lower().replace("_", "-")
    for p in inv:
        key = p.stem.replace("blm_", "").lower()
        if key == norm or key.endswith(norm):
            return p

    raise FileNotFoundError(f"No beam file found for detector '{detector}' in {bdir}")


def load_beam_alm(path: str | Path) -> np.ndarray:
    # Beam files can be scalar or T/E/B-like; normalize to (ncomp, nalm)
    alm = hp.read_alm(str(path), hdu=(1, 2, 3))
    return _coerce_alm_shape(np.asarray(alm, dtype=np.complex128))
