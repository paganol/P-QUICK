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
    """Return required NPZ keys for undersampled pointing input files."""
    return ("t0_ns", "qx", "qy", "qz", "qs", "sampling_rate_hz", "idx_first", "idx_last", "idx_step")


def load_pointing_npz(path: str | Path) -> PointingData:
    """Load a compressed NPZ pointing file and return a :class:`~pquick.pointing.PointingData`.

    Validates required keys, normalises the stacked quaternion array, and reads
    ``original_indices`` for undersampling reconstruction.

    The ``flag`` field is optional. If absent, a zero-valued native-rate flag
    array is synthesised from ``original_indices``.

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

        t0_ns = float(np.asarray(data["t0_ns"]).reshape(-1)[0])
        qx = np.asarray(data["qx"], dtype=np.float64)
        qy = np.asarray(data["qy"], dtype=np.float64)
        qz = np.asarray(data["qz"], dtype=np.float64)
        qs = np.asarray(data["qs"], dtype=np.float64)
        flag = np.asarray(data["flag"], dtype=np.int8) if "flag" in data else None
        sampling_rate_hz = float(np.asarray(data["sampling_rate_hz"]).reshape(-1)[0])
        idx_first = int(np.asarray(data["idx_first"]).reshape(-1)[0])
        idx_last = int(np.asarray(data["idx_last"]).reshape(-1)[0])
        idx_step = int(np.asarray(data["idx_step"]).reshape(-1)[0])

    # Reconstruct original_indices from the 3 compact scalars.
    original_indices = np.arange(idx_first, idx_last, idx_step, dtype=np.int64)
    if original_indices.size == 0 or original_indices[-1] != idx_last:
        original_indices = np.append(original_indices, idx_last)

    if not (qx.size == qy.size == qz.size == qs.size == original_indices.size):
        raise ValueError("undersampled quaternion arrays must have identical length")
    if flag is None:
        n_native = int(original_indices[-1]) + 1
        flag = np.zeros(n_native, dtype=np.int8)

    quat_us = normalize_quaternion(np.stack([qx, qy, qz, qs], axis=-1))
    return PointingData(
        t0_ns=t0_ns,
        quat_us=quat_us,
        flag=flag,
        sampling_rate_hz=sampling_rate_hz,
        original_indices=original_indices,
    )


def load_horn_flag_npz(path: str | Path, horn: str, n_samples: int | None = None) -> np.ndarray:
    """Load and unpack a per-horn packed flag stream from ``flags_*.npz``.

    Args:
        path: Path to the channel flag NPZ file.
        horn: Horn key, e.g. ``"100-1"``.
        n_samples: Optional explicit output length. If omitted, uses the
            ``n_samples`` metadata from the NPZ.

    Returns:
        Int8 array where 0=good, 1=bad.
    """
    p = Path(path)
    with np.load(p, allow_pickle=False) as data:
        if horn not in data:
            available = [k for k in data.files if "-" in k]
            raise KeyError(f"Horn '{horn}' not found in {p}. Available: {available}")
        packed = np.asarray(data[horn], dtype=np.uint8)
        n = int(n_samples) if n_samples is not None else int(np.asarray(data["n_samples"]).reshape(-1)[0])
    unpacked = np.unpackbits(packed, bitorder="little")[:n]
    return unpacked.astype(np.int8)


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
    """Coerce ALM input to a complex array with shape ``(ncomp, nalm)``."""
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
    """Return ``nalm`` for healpy m-major storage truncated at ``mmax``."""
    return (mmax + 1) * (lmax + 1) - (mmax * (mmax + 1)) // 2


def _beam_index_to_lm(index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode Planck sequential beam indices into ``(ell, m)`` arrays."""
    # Planck sequential convention: index = ell*(ell+1) + m + 1, m in [-ell, +ell]
    # Inverting: ell = floor(sqrt(index - 1)), m = index - ell*(ell+1) - 1
    # (same formula used by qp_planck / get_blm_det)
    idx0 = np.asarray(index, dtype=np.int64) - 1
    ell = np.floor(np.sqrt(idx0.astype(np.float64))).astype(np.int64)
    emm = idx0 - ell * (ell + 1)
    return ell, emm


def _pack_truncated_alm(coeff: np.ndarray, ell: np.ndarray, emm: np.ndarray, lmax: int, mmax: int) -> np.ndarray:
    """Pack selected ``(ell, m)`` coefficients into healpy m-major order."""
    out = np.zeros(_nalm_mmajor(lmax, mmax), dtype=np.complex128)
    start = emm * (lmax + 1) - (emm * (emm - 1)) // 2
    out[start + (ell - emm)] = coeff
    return out


def _to_text(value: object) -> str:
    """Convert a FITS scalar value to stripped text."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value).strip()


def _rimo_detector_quat(
    phi_uv_deg: float,
    theta_uv_deg: float,
    psi_uv_deg: float,
    psi_pol_deg: float = 0.0,
) -> np.ndarray:
    """Build the ZYZ-convention detector quaternion for the Pxx polarization frame.

    Uses ``psi_uv`` only (without ``psi_pol``) so that extracted psi is the
    polarization-frame (Pxx) angle used for mapmaking I/Q/U weights.  The Dxx
    beam-frame angle for convolution is obtained by adding ``psi_pol`` as a
    separate offset in the timeline pointing construction.

    Following toast-npipe utilities.load_RIMO, the ZYZ quaternion is left-multiplied
    by SPINROT = rotation(Y, pi/2 - 85deg) to account for the 85° Planck spin angle:
    in the pointing-file frame (X=spin axis, Z≈LOS), the nominal boresight sits at
    5° from Z, so SPINROT corrects the reference direction for all detectors.
    """
    degree = np.pi / 180.0
    phi = phi_uv_deg * degree
    theta = theta_uv_deg * degree
    # Pxx frame: psi_uv only, subtract phi per ZYZ convention
    psi = psi_uv_deg * degree - phi

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
    out: np.ndarray[tuple[Any, ...], np.dtype[Unknown]] = np.array([
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
        has_eps = "EPSILON" in names

        for row in tab:
            det = _to_text(row[det_col])
            rec: dict[str, np.ndarray | float] = {}
            if has_phi and has_theta and has_psi:
                phi_uv = float(row["PHI_UV"])
                theta_uv = float(row["THETA_UV"])
                psi_uv = float(row["PSI_UV"])
                psi_pol = float(row["PSI_POL"]) if has_psi_pol else 0.0
                eps = float(row["EPSILON"]) if has_eps else 0.0
                rec["phi_uv"] = phi_uv
                rec["theta_uv"] = theta_uv
                rec["psi_uv"] = psi_uv
                rec["psi_pol"] = psi_pol
                rec["psi_pol_rad"] = psi_pol * (np.pi / 180.0)
                rec["psi_uv_rad"] = psi_uv * (np.pi / 180.0)
                rec["epsilon"] = eps
                # Polarisation efficiency rho = (1 - eps)/(1 + eps); 1.0 for ideal.
                rec["rho_pol"] = (1.0 - eps) / (1.0 + eps)
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


def load_beam_alm(
    path: str | Path,
    lmax: int | None = None,
    mmax: int | None = None,
) -> np.ndarray:
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

                # Keep only non-negative m: Planck beam files store all m in [-ℓ, ℓ]
                # using the sequential index ℓ(ℓ+1)+m, so negative-m entries are
                # present but redundant (for real beams b_{ℓ,-m} = (-1)^m b_{ℓm}*).
                # _pack_truncated_alm uses the healpy m-major layout which only holds
                # m ≥ 0; passing negative m produces wrong (wrapped) numpy indices that
                # silently corrupt the m=0 coefficients.
                mask = (ell <= use_lmax) & (emm >= 0) & (emm <= use_mmax)
                packed = _pack_truncated_alm(coeff[mask], ell[mask], emm[mask], use_lmax, use_mmax)
                return packed[None, :]

    # Fallback for more standard FITS ALM storage.
    try:
        alm = hp.read_alm(str(p), hdu=(1, 2, 3))
    except Exception:
        alm = hp.read_alm(str(p), hdu=1)
    return _coerce_alm_shape(np.asarray(alm, dtype=np.complex128))


def build_polarized_beam_alm(
    beam_alm_scalar: np.ndarray,
    psi_pol_rad: float,
    lmax: int,
    mmax: int,
    nside: int | None = None,
    psi_uv_rad: float = 0.0,
    rho_pol: float = 1.0,
) -> np.ndarray:
    """Build an ideal co-polar polarised beam ``[T, E, B]`` from a scalar Planck blm.

    Planck ``blm_*`` files store only the **scalar (spin-0) intensity** beam, measured
    on planets, in the **Dxx** (beam-geometric) frame.  ducc0 total-convolution needs a
    3-component ``[T, E, B]`` beam where the E/B components are the **spin-2** polarised
    response — *not* a copy of the spin-0 beam.  Copying the intensity alm into the E/B
    slots (e.g. ``np.repeat``) injects a mis-normalised polarisation response that leaks
    E into the T solve and produces an oscillating transfer function.

    For an ideal polarisation-sensitive detector the polarised beam is the **same**
    intensity pattern, but the detector's fixed focal-plane polarisation direction, when
    expressed in the local ``(e_theta, e_phi)`` basis across the beam, rotates with the
    azimuth ``phi`` — giving the spin-2 ``e^{±2i phi}`` structure.  Concretely the beam
    Stokes map is ``(I, Q, U) = B (1, cos 2(psi_pol - phi), sin 2(psi_pol - phi))`` and
    its ``map2alm(pol=True)`` yields ``[T, E, B]``.  The scalar beam is first rotated
    Dxx → Pxx via ``b_lm -> b_lm e^{i m psi_pol}`` so the returned beam is in the **Pxx**
    polarisation frame (pol axis = beam x-axis); it must therefore be convolved with the
    pointing psi in the Pxx frame (``psi_offset = 0``).

    Validated against ``litebird_sim.beam_synthesis`` analytic Gaussian beams: the E/B
    profiles match to <0.3 % (flat in ℓ; the residual is litebird's constant
    ``exp(2 sigma^2)`` approximation vs the exact spin-2 transform used here).

    Args:
        beam_alm_scalar: Scalar intensity beam, shape ``(1, nalm)`` or ``(nalm,)``,
            healpy m-major order at ``(lmax, mmax)``, in the Dxx frame.
        psi_pol_rad: Detector polarisation angle ``psi_pol`` (radians) relative to the
            Dxx beam x-axis.
        lmax: Maximum multipole of the beam alm.
        mmax: Maximum azimuthal order to retain in the output beam.
        nside: HEALPix resolution for the intermediate map; defaults to the smallest
            power of two with ``2 * nside >= lmax``.
        psi_uv_rad: Detector ``psi_uv`` (radians). The beam *shape* of all three
            (T, E, B) components is pre-rotated by ``-psi_uv`` so that, convolved at
            ``psi_pxx``, it stays co-oriented across a horn's two PSB arms (their
            identical ellipses would otherwise land 90 deg apart and cancel). For
            E/B the spin-2 phase ``chi`` carries a compensating ``+psi_uv`` so the
            polarisation axis is still restored to ``psi_pxx``; only the shape, not
            the polarisation direction, follows psi_uv.

    Returns:
        Complex128 array of shape ``(3, nalm)`` holding ``[T, E, B]`` beam alm at
        ``(lmax, mmax)``, in the Pxx frame. Convolve at ``psi_pxx = psi_buf``.
    """
    bb = beam_alm_scalar[0] if np.ndim(beam_alm_scalar) == 2 else np.asarray(beam_alm_scalar)
    bb = np.ascontiguousarray(bb, dtype=np.complex128)

    if nside is None:
        nside = 1
        while 2 * nside < lmax:
            nside *= 2

    # Dxx -> Pxx frame rotation about the boresight: b_lm -> b_lm e^{i m psi_pol}
    # (same sense as qp_planck's e^{i m (psi_uv+psi_pol)}; psi_uv is carried by the
    # pointing quaternion, so only psi_pol remains here).
    ell, emm = hp.Alm.getlm(lmax, np.arange(bb.size))

    # E/B (spin-2) must NOT be psi_uv-de-rotated the way the scalar T beam is. For the
    # polarised response, psi_uv carries the per-arm polarisation orthogonality (the
    # ~90 deg difference between a horn's two PSB arms). Removing psi_uv from the E/B
    # shape destroys that orthogonality, the two arms' polarised beams cancel, and EE
    # collapses to ~zero (confirmed on the full-sky transfer function). So apply only
    # psi_pol here and keep psi_uv: it is supplied by the convolution (pointing
    # quaternion) as the Pxx polarisation axis. Only the scalar T shape is
    # psi_uv-de-rotated (bb_T below), which is harmless for an intensity pattern.
    bb_pol = bb * np.exp(1j * emm * float(psi_pol_rad))

    # Synthesise the intensity beam map and build the ideal co-polar Stokes pattern.
    # Pol axis = Pxx x-axis (psi_pol applied above), so psi_pol drops out of chi here.
    # The spin-2 (E/B) response is obtained by map2alm(pol=True).
    beam_map = hp.alm2map(bb_pol, nside, lmax=lmax, mmax=mmax)
    phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))[1]
    chi = -2.0 * phi
    iqu = np.array([beam_map, beam_map * np.cos(chi), beam_map * np.sin(chi)])
    _, e_alm, b_alm = hp.map2alm(iqu, lmax=lmax, mmax=mmax, pol=True, iter=3)

    # Polarisation efficiency: scale the E/B (polarised) response by rho so the
    # simulated TOD matches the rho-weighted map-making (map.use_cross_pol). Without
    # this the ideal (rho=1) beam over-drives the rho-weighted solve and EE/BB come
    # out inflated by 1/rho^2. qp_planck carries rho in both beam and hit matrix.
    e_alm = e_alm * float(rho_pol)
    b_alm = b_alm * float(rho_pol)

    # The whole beam is convolved at the Pxx polarisation angle psi_pxx = psi_buf
    # (so E/B and map-making share the same polarisation frame). But the *intensity*
    # beam shape must stay co-oriented across a horn's two PSB arms (psi_uv removed),
    # otherwise their identical ellipses land 90 deg apart and cancel. So pre-rotate
    # only the T component by -psi_uv: convolved at psi_pxx its shape lands at the
    # boresight (psi_uv-free) frame, while E/B keep the polarisation axis at psi_pxx.
    # T is taken from the (rotated) input alm so it is exact (no pixelisation).
    bb_T = bb * np.exp(1j * emm * (float(psi_pol_rad) - float(psi_uv_rad)))
    return np.ascontiguousarray(np.array([bb_T, e_alm, b_alm], dtype=np.complex128))


def normalize_beam_alm(beam_alm: np.ndarray, mode: str = "unit_integral") -> np.ndarray:
    """Normalize beam ALMs for the requested convolution convention.

    Args:
        beam_alm: Complex beam ALMs of shape ``(ncomp, nalm)``.
        mode: Normalization mode. ``"unit_integral"`` divides all components by
            ``sqrt(4 pi) b_00`` of the temperature beam so a constant sky remains
            constant after convolution. ``"raw"`` returns the input unchanged.

    Returns:
        Complex128 beam ALMs with the requested normalization applied.

    Raises:
        ValueError: If *mode* is unsupported or if the beam monopole cannot be used
            for unit-integral normalization.
    """
    beam = np.asarray(beam_alm, dtype=np.complex128)
    normalized = mode.strip().lower()
    if normalized == "raw":
        return beam
    if normalized != "unit_integral":
        raise ValueError(
            f"Unsupported beam_normalization={mode!r}; expected 'unit_integral' or 'raw'"
        )
    if beam.ndim != 2 or beam.shape[0] < 1 or beam.shape[1] < 1:
        raise ValueError("beam_alm must have shape (ncomp, nalm) with at least one coefficient")

    scale = np.sqrt(4.0 * np.pi) * beam[0, 0]
    if abs(scale) == 0.0:
        raise ValueError("Cannot apply unit-integral beam normalization: b_00 is zero")
    if abs(np.imag(scale)) > 1e-12 * max(1.0, abs(np.real(scale))):
        raise ValueError(
            "Cannot apply unit-integral beam normalization: sqrt(4 pi) * b_00 is not real"
        )
    return beam / float(np.real(scale))
