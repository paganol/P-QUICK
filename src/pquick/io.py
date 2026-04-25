from __future__ import annotations

from pathlib import Path

import healpy as hp
import numpy as np
from astropy.io import fits

from .config import DetectorSelection
from .pointing import PointingData
from .quaternion import normalize_quaternion
from .utilities import DETSETS


def _required_npz_keys() -> tuple[str, ...]:
    return ("time", "qx", "qy", "qz", "qs", "flag", "sampling_rate_hz")


def load_pointing_npz(path: str | Path) -> PointingData:
    """Load a compressed NPZ pointing file and return a :class:`~pquick.pointing.PointingData`.

    Validates required keys, normalises the stacked quaternion array, and optionally
    reads ``original_indices`` for non-uniform undersampling.

    Expects a single native-rate ``flag`` field.

    Args:
        path: Path to the ``.npz`` pointing file.

    Returns:
        A :class:`~pquick.pointing.PointingData` instance at the undersampled rate.

    Raises:
        ValueError: If required keys are missing or array lengths are inconsistent.
    """
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
        flag = np.asarray(data["flag"], dtype=np.int8)
        sampling_rate_hz = float(np.asarray(data["sampling_rate_hz"]).reshape(-1)[0])
        original_indices = (
            np.asarray(data["original_indices"], dtype=np.int64)
            if "original_indices" in data
            else None
        )

    if not (time_us.size == qx.size == qy.size == qz.size == qs.size):
        raise ValueError("undersampled quaternion arrays must have identical length")
    quat_us = normalize_quaternion(np.stack([qx, qy, qz, qs], axis=-1))
    return PointingData(
        time_us=time_us,
        quat_us=quat_us,
        flag=flag,
        sampling_rate_hz=sampling_rate_hz,
        original_indices=original_indices,
    )


def load_sky_alm(path: str | Path) -> np.ndarray:
    """Read sky spherical-harmonic coefficients from a FITS, ``.npy``, or ``.npz`` file.

    Returns a ``(ncomp, nalm)`` complex128 array in healpy m-major order, where
    ``ncomp`` is 1 or 3 (T-only or T/Q/U).

    Args:
        path: Path to the sky ALM file.

    Returns:
        Complex128 array of shape ``(ncomp, nalm)``.

    Raises:
        ValueError: If the file format is unsupported or the array shape is invalid.
    """
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
    """Infer the maximum multipole ``lmax`` from a healpy-ordered ALM array.

    Args:
        alm: Array of shape ``(ncomp, nalm)`` in healpy m-major order.

    Returns:
        The integer ``lmax`` corresponding to the second-axis length.
    """
    nalm = int(alm.shape[1])
    return hp.Alm.getlmax(nalm)


def truncate_alm(alm: np.ndarray, lmax_src: int, lmax_dst: int) -> np.ndarray:
    """Truncate a healpy-ordered alm array from lmax_src down to lmax_dst."""
    if lmax_dst >= lmax_src:
        return alm
    nalm_dst = (lmax_dst + 1) * (lmax_dst + 2) // 2
    out = np.zeros((alm.shape[0], nalm_dst), dtype=np.complex128)
    for m in range(lmax_dst + 1):
        src_start = m * (2 * lmax_src + 1 - m) // 2
        dst_start = m * (2 * lmax_dst + 1 - m) // 2
        count = lmax_dst - m + 1
        out[:, dst_start : dst_start + count] = alm[:, src_start : src_start + count]
    return out


def _nalm_mmajor(lmax: int, mmax: int) -> int:
    return (mmax + 1) * (lmax + 1) - (mmax * (mmax + 1)) // 2


def _beam_index_to_lm(index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx0 = np.asarray(index, dtype=np.int64) - 1
    ell = np.floor((np.sqrt(4.0 * idx0 + 1.0) - 1.0) / 2.0).astype(np.int64)
    emm = idx0 - ell * (ell + 1)
    return ell, emm


def _pack_truncated_alm(coeff: np.ndarray, ell: np.ndarray, emm: np.ndarray, lmax: int, kmax: int) -> np.ndarray:
    out = np.zeros(_nalm_mmajor(lmax, kmax), dtype=np.complex128)
    start = emm * (lmax + 1) - (emm * (emm - 1)) // 2
    out[start + (ell - emm)] = coeff
    return out


def _to_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value).strip()


def _rimo_detector_quat(
    phi_uv_deg: float,
    theta_uv_deg: float,
    psi_uv_deg: float,
    psi_pol_deg: float = 0.0,
) -> np.ndarray:
    """Build the ZYZ-convention detector quaternion for the Dxx beam frame.

    The Dxx (beam) frame requires the full rotation angle ``psi_uv + psi_pol``.
    Following toast-npipe utilities.load_RIMO, the ZYZ quaternion is left-multiplied
    by SPINROT = rotation(Y, pi/2 - 85deg) to account for the 85° Planck spin angle:
    in the pointing-file frame (X=spin axis, Z≈LOS), the nominal boresight sits at
    5° from Z, so SPINROT corrects the reference direction for all detectors.
    """
    degree = np.pi / 180.0
    phi = phi_uv_deg * degree
    theta = theta_uv_deg * degree
    # psi_uv + psi_pol gives the Dxx orientation; subtract phi per ZYZ convention
    psi = (psi_uv_deg + psi_pol_deg) * degree - phi

    quat = np.zeros(4, dtype=np.float64)
    quat[3] = np.cos(0.5 * theta) * np.cos(0.5 * (phi + psi))
    quat[0] = -np.sin(0.5 * theta) * np.sin(0.5 * (phi - psi))
    quat[1] = np.sin(0.5 * theta) * np.cos(0.5 * (phi - psi))
    quat[2] = np.cos(0.5 * theta) * np.sin(0.5 * (phi + psi))

    # SPINROT = rotation(Y, pi/2 - 85deg = 5deg):  q = (0, sin(2.5deg), 0, cos(2.5deg))
    _spin_half = 0.5 * (np.pi / 2.0 - np.radians(85.0))  # = 2.5 deg in radians
    sy = np.sin(_spin_half)
    cy = np.cos(_spin_half)
    # SPINROT ⊗ quat  (left-multiply), convention (x,y,z,w)
    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
    out = np.array([
         cy * qx + sy * qz,
         cy * qy + sy * qw,  # w1*y2 - x1*z2 + y1*w2 + z1*x2 with x1=z1=0
         cy * qz - sy * qx,
         cy * qw - sy * qy,
    ])
    return normalize_quaternion(out)


def load_rimo_detectors(rimo_path: str | Path) -> dict[str, dict[str, np.ndarray | float]]:
    """Read a Planck RIMO FITS table and return per-detector orientation metadata.

    For each detector, the UV-frame angles ``(phi_uv, theta_uv, psi_uv)`` are read
    (when present) and converted to a unit quaternion stored under the ``"quat"`` key.

    Args:
        rimo_path: Path to the RIMO FITS file.

    Returns:
        Dict mapping detector name to a metadata dict with keys
        ``phi_uv``, ``theta_uv``, ``psi_uv``, and ``quat``.

    Raises:
        ValueError: If no detectors were loaded from the file.
    """
    out: dict[str, dict[str, np.ndarray | float]] = {}
    with fits.open(rimo_path) as hdul:
        tab = hdul[1].data
        names = [n.upper() for n in tab.columns.names]

        det_col = "DETECTOR" if "DETECTOR" in names else names[0]
        has_phi = "PHI_UV" in names
        has_theta = "THETA_UV" in names
        has_psi = "PSI_UV" in names
        has_psi_pol = "PSI_POL" in names

        for row in tab:
            det = _to_text(row[det_col])
            rec: dict[str, np.ndarray | float] = {}
            if has_phi and has_theta and has_psi:
                phi_uv = float(row["PHI_UV"])
                theta_uv = float(row["THETA_UV"])
                psi_uv = float(row["PSI_UV"])
                psi_pol = float(row["PSI_POL"]) if has_psi_pol else 0.0
                rec["phi_uv"] = phi_uv
                rec["theta_uv"] = theta_uv
                rec["psi_uv"] = psi_uv
                rec["psi_pol"] = psi_pol
                rec["quat"] = _rimo_detector_quat(phi_uv, theta_uv, psi_uv, psi_pol)
            out[det] = rec
    if not out:
        raise ValueError(f"No detectors loaded from RIMO file: {rimo_path}")
    return out


def select_detectors(all_detectors: list[str], selection: DetectorSelection) -> list[str]:
    """Filter a detector list according to a :class:`~pquick.config.DetectorSelection`.

    Applies, in order: channel/detset filter and explicit allowlist.

    Args:
        all_detectors: Full list of detector names to filter.
        selection: Selection rules from the pipeline configuration.

    Returns:
        Sorted list of detectors that pass all active filters.

    Raises:
        ValueError: If both ``channel`` and ``detectors`` are provided, or if
            the resulting selection is empty.
    """
    selected = list(all_detectors)

    if selection.channel and selection.detectors:
        raise ValueError("Specify only one of detector_selection.channel or detector_selection.detectors")

    if selection.channel:
        tag = selection.channel.strip().lower()
        if tag in DETSETS:
            allowed = set(DETSETS[tag])
            selected = [d for d in selected if d in allowed]
        else:
            selected = [d for d in selected if d.lower().startswith(tag)]

    if selection.detectors:
        allowed = {d.strip() for d in selection.detectors}
        selected = [d for d in selected if d in allowed]

    if not selected:
        raise ValueError("Detector selection is empty")
    return sorted(selected)


def detector_to_beam_file(beams_dir: str | Path, detector: str) -> Path:
    """Resolve the beam FITS file for a detector by name.

    Tries canonical filename variants (with and without lowercase / ``_``→``-``
    substitution) first, then falls back to a suffix-match scan of the directory.

    Args:
        beams_dir: Directory containing ``blm_*.fits`` beam files.
        detector: Detector name (e.g. ``"100-1a"`` or ``"LFI27M"``).

    Returns:
        :class:`pathlib.Path` pointing to the resolved beam file.

    Raises:
        FileNotFoundError: If no matching file is found.
    """
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


def load_beam_alm(path: str | Path, lmax: int | None = None, mmax: int | None = None) -> np.ndarray:
    """Read beam spherical-harmonic coefficients from a Planck-format FITS file.

    The FITS table must contain ``index``, ``real``, and ``imag`` columns where
    ``index`` encodes *(ℓ, m)* using the Planck sequential convention.  The
    coefficients are repacked into a healpy m-major array truncated at *lmax*
    and *mmax*.

    Args:
        path: Path to the beam FITS file (``blm_*.fits``).
        lmax: Maximum ℓ to retain; defaults to the value found in the file.
        mmax: Maximum azimuthal order *m* to retain; defaults to the file value.

    Returns:
        Complex128 array of shape ``(1, nalm_beam)`` in healpy m-major order.

    Raises:
        ValueError: If the requested *lmax* or *mmax* exceeds what the file provides.
    """
    p = Path(path)
    with fits.open(p) as hdul:
        if len(hdul) > 1 and getattr(hdul[1], "columns", None) is not None:
            names = {name.lower() for name in hdul[1].columns.names}
            if {"index", "real", "imag"}.issubset(names):
                data = hdul[1].data
                coeff = np.asarray(data["real"], dtype=np.float64) + 1j * np.asarray(data["imag"], dtype=np.float64)
                ell, emm = _beam_index_to_lm(np.asarray(data["index"], dtype=np.int64))

                src_lmax = int(ell.max())
                src_mmax = int(emm.max())
                use_lmax = src_lmax if lmax is None else int(lmax)
                use_mmax = src_mmax if mmax is None else int(mmax)

                if use_lmax > src_lmax:
                    raise ValueError(f"Requested lmax={use_lmax} exceeds beam lmax={src_lmax} for {p}")
                if use_mmax > src_mmax:
                    raise ValueError(f"Requested mmax={use_mmax} exceeds beam mmax={src_mmax} for {p}")

                mask = (ell <= use_lmax) & (emm <= use_mmax)
                packed = _pack_truncated_alm(coeff[mask], ell[mask], emm[mask], use_lmax, use_mmax)
                return packed[None, :]

    # Fallback for more standard FITS ALM storage.
    try:
        alm = hp.read_alm(str(p), hdu=(1, 2, 3))
    except Exception:
        alm = hp.read_alm(str(p), hdu=1)
    return _coerce_alm_shape(np.asarray(alm, dtype=np.complex128))
