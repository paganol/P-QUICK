from __future__ import annotations

import os
import re
import socket
from pathlib import Path

# Detector sets: base detector groups by frequency, then derived aliases below.
DETSETS: dict[str, tuple[str, ...]] = {
    # 100 GHz
    "100ds1": ("100-1a", "100-1b", "100-4a", "100-4b"),
    "100ds2": ("100-2a", "100-2b", "100-3a", "100-3b"),
    # 143 GHz
    "143ds1": ("143-1a", "143-1b", "143-3a", "143-3b"),
    "143ds2": ("143-2a", "143-2b", "143-4a", "143-4b"),
    "143swb": ("143-5", "143-6", "143-7"),
    # 217 GHz
    "217ds1": ("217-5a", "217-5b", "217-7a", "217-7b"),
    "217ds2": ("217-6a", "217-6b", "217-8a", "217-8b"),
    "217swb": ("217-1", "217-2", "217-3", "217-4"),
    # 353 GHz
    "353ds1": ("353-3a", "353-3b", "353-5a", "353-5b"),
    "353ds2": ("353-4a", "353-4b", "353-6a", "353-6b"),
    "353swb": ("353-1", "353-2", "353-7", "353-8"),
    # 545 GHz
    "545dsA": ("545-1",),
    "545dsB": ("545-2", "545-4"),
    "545ghz": ("545-1", "545-2", "545-4"),
    # 857 GHz
    "857dsA": ("857-1", "857-3"),
    "857dsB": ("857-2", "857-4"),
    "857ghz": ("857-1", "857-2", "857-3", "857-4"),
}

# Derived aliases and channel-wide groupings.
DETSETS["100dsA"] = DETSETS["100ds1"]
DETSETS["100dsB"] = DETSETS["100ds2"]
DETSETS["100psb"] = DETSETS["100ds1"] + DETSETS["100ds2"]
DETSETS["100ghz"] = DETSETS["100psb"]

DETSETS["143dsA"] = DETSETS["143ds1"] + ("143-5", "143-7")
DETSETS["143dsB"] = DETSETS["143ds2"] + ("143-6",)
DETSETS["143psb"] = DETSETS["143ds1"] + DETSETS["143ds2"]
DETSETS["143ghz"] = DETSETS["143psb"] + DETSETS["143swb"]

DETSETS["217dsA"] = DETSETS["217ds1"] + ("217-5", "217-7")
DETSETS["217dsB"] = DETSETS["217ds2"] + ("217-6", "217-8")
DETSETS["217psb"] = DETSETS["217ds1"] + DETSETS["217ds2"]
DETSETS["217ghz"] = DETSETS["217psb"] + DETSETS["217swb"]

DETSETS["353dsA"] = DETSETS["353ds1"] + ("353-1", "353-7")
DETSETS["353dsB"] = DETSETS["353ds2"] + ("353-2", "353-8")
DETSETS["353psb"] = DETSETS["353ds1"] + DETSETS["353ds2"]
DETSETS["353ghz"] = DETSETS["353psb"] + DETSETS["353swb"]

# NPIPE per-detector map weights: both arms of a horn share the horn weight
# (ported from qp_planck/qp_planck/utilities.py detector_weights). Non-working
# bolometers (143-8, 545-3) are absent, so they get skipped.
NPIPE_DETECTOR_WEIGHTS: dict[str, float] = {
    "100-1a": 763430.0, "100-1b": 763430.0,
    "100-2a": 1266100.0, "100-2b": 1266100.0,
    "100-3a": 1063100.0, "100-3b": 1063100.0,
    "100-4a": 1053200.0, "100-4b": 1053200.0,
    "143-1a": 1640700.0, "143-1b": 1640700.0,
    "143-2a": 1857700.0, "143-2b": 1857700.0,
    "143-3a": 1643900.0, "143-3b": 1643900.0,
    "143-4a": 1445800.0, "143-4b": 1445800.0,
    "143-5": 2763000.0, "143-6": 2694200.0, "143-7": 2859900.0,
    "217-1": 1105800.0, "217-2": 1026100.0, "217-3": 1095800.0, "217-4": 1059300.0,
    "217-5a": 673180.0, "217-5b": 673180.0,
    "217-6a": 710920.0, "217-6b": 710920.0,
    "217-7a": 765760.0, "217-7b": 765760.0,
    "217-8a": 712260.0, "217-8b": 712260.0,
    "353-1": 128290.0, "353-2": 134750.0,
    "353-3a": 48067.0, "353-3b": 48067.0,
    "353-4a": 42187.0, "353-4b": 42187.0,
    "353-5a": 56914.0, "353-5b": 56914.0,
    "353-6a": 25293.0, "353-6b": 25293.0,
    "353-7": 87730.0, "353-8": 74453.0,
    "545-1": 4475.5, "545-2": 5540.3, "545-4": 4321.0,
    "857-1": 6.8895, "857-2": 6.3108, "857-3": 6.5964, "857-4": 3.6785,
    "LFI18M": 53650.0, "LFI18S": 53650.0,
    "LFI19M": 42141.0, "LFI19S": 42141.0,
    "LFI20M": 36579.0, "LFI20S": 36579.0,
    "LFI21M": 50355.0, "LFI21S": 50355.0,
    "LFI22M": 49363.0, "LFI22S": 49363.0,
    "LFI23M": 47966.0, "LFI23S": 47966.0,
    "LFI24M": 123720.0, "LFI24S": 123720.0,
    "LFI25M": 140490.0, "LFI25S": 140490.0,
    "LFI26M": 112330.0, "LFI26S": 112330.0,
    "LFI27M": 401640.0, "LFI27S": 401640.0,
    "LFI28M": 369000.0, "LFI28S": 369000.0,
}

MISSION_LENGTH_RANGES: dict[str, tuple[int, int]] = {
    "full": (91, 974),
    "hm1": (91, 563),
    "hm2": (564, 974),
    "survey1": (91, 270),
    "survey2": (271, 456),
    "survey3": (457, 636),
    "survey4": (637, 807),
    "survey5": (808, 974),
}


# PR3 per-detector map weights.
#  - HFI: SRoll per-detector (calib/NEP)^2 (DX11 calib / RD12 NEP) — differs between
#    a horn's a/b arms, unlike NPIPE.
#  - LFI: Planck-2018 (aa33293-18) per-horn Eq. 7 weight 2/(sigma_M^2 + sigma_S^2)
#    from the Table 4 white-noise levels [uK^2/Hz]; shared by a horn's M/S arms, e.g.
#    LFI27 = 2/(281.5 + 302.8). 143-8/545-3 absent (dead). Absolute scale is arbitrary
#    (it cancels in the per-pixel solve); only the within-channel ratios matter.
PR3_DETECTOR_WEIGHTS: dict[str, float] = {
    "100-1a": 162673.0, "100-1b": 227632.3,
    "100-2a": 674779.9, "100-2b": 346375.2,
    "100-3a": 903876.6, "100-3b": 610547.8,
    "100-4a": 416489.2, "100-4b": 226073.2,
    "143-1a": 1702989.0, "143-1b": 704902.4,
    "143-2a": 1740084.0, "143-2b": 1509531.0,
    "143-3a": 1435034.0, "143-3b": 1530800.0,
    "143-4a": 1276859.0, "143-4b": 1069561.0,
    "143-5": 2115344.0, "143-6": 2045240.0, "143-7": 2669092.0,
    "217-1": 1200040.0, "217-2": 1120057.0, "217-3": 1249481.0, "217-4": 1408729.0,
    "217-5a": 459494.2, "217-5b": 608457.2,
    "217-6a": 480378.1, "217-6b": 528185.9,
    "217-7a": 634943.2, "217-7b": 654692.2,
    "217-8a": 492636.1, "217-8b": 464691.9,
    "353-1": 166463.3, "353-2": 158481.0,
    "353-3a": 30471.63, "353-3b": 42703.47,
    "353-4a": 41951.33, "353-4b": 37735.2,
    "353-5a": 44020.63, "353-5b": 42320.42,
    "353-6a": 20896.82, "353-6b": 22874.54,
    "353-7": 109668.2, "353-8": 91233.83,
    "545-1": 2.572538, "545-2": 3.164496, "545-4": 2.7508,
    "857-1": 2.494976, "857-2": 2.561947, "857-3": 2.430096, "857-4": 1.28278,
    "LFI18M": 0.00204248, "LFI18S": 0.00204248,
    "LFI19M": 0.00176507, "LFI19S": 0.00176507,
    "LFI20M": 0.00165714, "LFI20S": 0.00165714,
    "LFI21M": 0.00197981, "LFI21S": 0.00197981,
    "LFI22M": 0.00195886, "LFI22S": 0.00195886,
    "LFI23M": 0.00191755, "LFI23S": 0.00191755,
    "LFI24M": 0.00231669, "LFI24S": 0.00231669,
    "LFI25M": 0.00246853, "LFI25S": 0.00246853,
    "LFI26M": 0.00220872, "LFI26S": 0.00220872,
    "LFI27M": 0.00342290, "LFI27S": 0.00342290,
    "LFI28M": 0.00331126, "LFI28S": 0.00331126,
}

_WEIGHT_SETS: dict[str, dict[str, float]] = {
    "NPIPE": NPIPE_DETECTOR_WEIGHTS,
    "PR3": PR3_DETECTOR_WEIGHTS,
}


def detector_map_weight(detector: str, weights: str = "NPIPE", default: float = 1.0) -> float:
    """Per-detector inverse-noise map weight for the chosen weight set (``NPIPE``/``PR3``)."""
    return _WEIGHT_SETS[weights.upper()].get(detector.strip(), default)


def has_detector_weight(detector: str, weights: str = "NPIPE") -> bool:
    """True if the detector is in the chosen weight set, i.e. a working Planck detector.

    The set is the canonical good-detector list (cf qp_planck's ``list_planck(good=True)``):
    non-working bolometers — Planck HFI 143-8 and 545-3, the RTS-noise detectors — are
    deliberately absent.
    """
    return detector.strip() in _WEIGHT_SETS[weights.upper()]


def is_psb(detector: str) -> bool:
    """True for a polarization-sensitive detector, False for an unpolarized SWB.

    Matches qp_planck: the name ends in ``a``/``b`` (HFI PSB arm) or ``M``/``S``
    (LFI radiometer arm); spider-web bolometers (e.g. ``143-5``) do not.
    """
    return detector.strip()[-1:] in "abMS"


def parse_mission_length(value: str) -> tuple[int, int]:
    """Parse a mission-length selector into an inclusive OD range.

    Supported values:
    - Named ranges: ``full``, ``survey1`` ... ``survey5``, ``hm1``, ``hm2``
    - Explicit range: ``91-99`` (optionally with ``OD`` prefixes)
    """
    raw = value.strip()
    normalized = re.sub(r"[\s_\-]+", " ", raw.lower()).strip()
    if normalized in MISSION_LENGTH_RANGES:
        return MISSION_LENGTH_RANGES[normalized]

    m = re.fullmatch(r"(?:od)?\s*(\d+)\s*-\s*(?:od)?\s*(\d+)", raw, flags=re.IGNORECASE)
    if m is None:
        known = ", ".join(sorted(MISSION_LENGTH_RANGES.keys()))
        raise ValueError(
            f"Unsupported mission_length={value!r}. Use one of [{known}] or an explicit range like '91-99'."
        )

    od_start = int(m.group(1))
    od_end = int(m.group(2))
    if od_start > od_end:
        raise ValueError(f"Invalid mission_length={value!r}: start OD must be <= end OD")
    return od_start, od_end


def extract_od_from_pointing_filename(path: Path) -> int:
    """Extract OD number from a pointing filename stem.

    Uses the last contiguous digit block in the stem, e.g.:
    - ``processed_od_0091`` -> 91
    - ``pointing-0092`` -> 92
    """
    matches = re.findall(r"(\d+)", path.stem)
    if not matches:
        raise ValueError(f"Cannot infer OD from filename: {path.name}")
    return int(matches[-1])


def filter_pointing_files_by_mission_length(files: list[Path], mission_length: str | None) -> list[Path]:
    """Filter discovered pointing files to the requested mission-length range."""
    if mission_length is None or mission_length.strip() == "":
        return files

    od_start, od_end = parse_mission_length(mission_length)
    out: list[Path] = []
    for p in files:
        od = extract_od_from_pointing_filename(p)
        if od_start <= od <= od_end:
            out.append(p)
    return out


def build_pointing_file_paths(pointings_prefix: str, od_start: int, od_end: int) -> list[Path]:
    """Build the list of pointing NPZ paths for an OD range.

    Constructs paths of the form ``{pointings_prefix}od_{od:04d}.npz`` for each
    OD in ``[od_start, od_end]`` (inclusive), returning only paths that exist.

    Args:
        pointings_prefix: Prefix for pointing files, e.g.
            ``"inputs/pointings/pointing_"``.
        od_start: First operational day (inclusive).
        od_end: Last operational day (inclusive).

    Returns:
        Sorted list of existing :class:`pathlib.Path` objects.

    Raises:
        FileNotFoundError: If no files exist for the requested range.
    """
    candidates = [Path(f"{pointings_prefix}od_{od:04d}.npz") for od in range(od_start, od_end + 1)]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        raise FileNotFoundError(
            f"No pointing files found for pointings={pointings_prefix!r}, OD{od_start}-OD{od_end}"
        )
    return existing


def _format_od_ranges(ods: list[int]) -> str:
    """Format a sorted list of OD numbers as compact ranges.

    Consecutive ODs are collapsed with ``-``; gaps are separated by ``,``.
    E.g. ``[91,92,93,95,98,99]`` → ``"91-93,95,98-99"``.
    """
    if not ods:
        return ""
    parts: list[str] = []
    start = prev = ods[0]
    for od in ods[1:]:
        if od == prev + 1:
            prev = od
        else:
            parts.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = od
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def print_mpi_distribution(
    comm,
    rank: int,
    size: int,
    local_ods: list[int] | None = None,
) -> None:
    """Print the MPI distribution at the start of a run.

    Gathers the hostname and assigned OD list from every rank and prints a
    summary on rank 0.  If the run is serial (``size == 1``) a single
    informational line is printed instead.

    Args:
        comm: The ``mpi4py`` communicator, or ``None`` for a serial run.
        rank: MPI rank of the calling process.
        size: Total number of MPI ranks.
        local_ods: OD numbers assigned to this rank (optional).
    """
    hostname = socket.gethostname()

    if size == 1:
        msg = f"[MPI] Serial run on host {hostname}"
        if local_ods:
            ods_sorted = sorted(local_ods)
            od_info = f"ODs {_format_od_ranges(ods_sorted)} ({len(ods_sorted)} ODs)"
            msg += f" | {od_info}"
        print(msg, flush=True)
        return

    hostnames: list[str] = comm.gather(hostname, root=0)
    all_ods: list[list[int] | None] = comm.gather(local_ods, root=0)
    if rank == 0:
        width = len(str(size - 1))
        print(f"[MPI] Parallel run: {size} ranks", flush=True)
        for r, h in enumerate(hostnames):
            ods = all_ods[r] if all_ods is not None else None
            if ods:
                ods_sorted = sorted(ods)
                od_info = f"ODs {_format_od_ranges(ods_sorted)} ({len(ods_sorted)} ODs)"
            else:
                od_info = "no ODs assigned"
            print(f"  rank {r:>{width}} : {h} | {od_info}", flush=True)


def estimate_memory_per_rank_mb(nside: int, lmax: int = 0, mmax: int = 0) -> float:
    """Estimate the peak memory per MPI rank in MB.

    Accounts for the full-sky normal-equation matrix, hit map, output maps, and —
    when *lmax* > 0 — one ``ducc0.totalconvolve.Interpolator`` instance (the dominant
    term at high *lmax* / *mmax*).

    The Interpolator estimate is a lower bound assuming complex64 (float32) internal
    storage and no NUFFT oversampling: ``(lmax+1) × 2(lmax+1) × (2*mmax+1) × 8 B``.
    The actual size can be 1.5–2× larger depending on the epsilon target.

    Args:
        nside: HEALPix resolution parameter.
        lmax: Maximum multipole (0 → Interpolator term omitted).
        mmax: Maximum beam azimuthal order (beam kmax).

    Returns:
        Estimated peak memory in MB.
    """
    npix = 12 * nside * nside
    matrix_mb = npix * 9 * 8 / 1024**2   # (npix, 3, 3) float64
    hits_mb   = npix * 8 / 1024**2        # (npix,) int64
    maps_mb   = npix * 3 * 8 / 1024**2   # t, q, u output maps
    interp_mb = 0.0
    if lmax > 0:
        # One ducc0 Interpolator: internal grid (lmax+1) × 2*(lmax+1) × (2*mmax+1)
        # stored as complex64 (8 B) — conservative lower bound, no oversampling.
        interp_mb = (lmax + 1) * 2 * (lmax + 1) * (2 * mmax + 1) * 8 / 1024**2
    return matrix_mb + hits_mb + maps_mb + interp_mb


def suggest_tasks_per_node(
    nside: int,
    node_memory_mb: float,
    cores_per_node: int,
    lmax: int = 0,
    mmax: int = 0,
) -> int:
    """Suggest the maximum number of MPI tasks per node for a given nside.

    Args:
        nside: HEALPix resolution parameter.
        node_memory_mb: Total physical memory on one compute node in MB.
        cores_per_node: Number of cores (and therefore maximum tasks) per node.
        lmax: Maximum multipole (passed to :func:`estimate_memory_per_rank_mb`).
        mmax: Maximum beam azimuthal order (passed to :func:`estimate_memory_per_rank_mb`).

    Returns:
        Recommended number of MPI tasks per node (capped at ``cores_per_node``).
    """
    mem_per_rank = estimate_memory_per_rank_mb(nside, lmax=lmax, mmax=mmax)
    max_tasks = int(node_memory_mb / mem_per_rank)
    return min(max_tasks, cores_per_node)


def resolve_nthreads(nthreads: int) -> int:
    """Resolve the effective number of threads to use for both ducc0 and numba.

    Convention:
    - ``nthreads == 0``: read ``OMP_NUM_THREADS`` from the environment;
      fall back to 1 if the variable is unset or invalid.
    - ``nthreads > 0``: use the given value as-is.

    Args:
        nthreads: Value from the ``convolution.nthreads`` config key.

    Returns:
        Resolved thread count (always >= 1).
    """
    if nthreads != 0:
        return max(1, int(nthreads))
    raw = os.environ.get("OMP_NUM_THREADS", "")
    try:
        return max(1, int(raw))
    except ValueError:
        return 1
