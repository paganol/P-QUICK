from __future__ import annotations

import re
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

# Weights ported from qp_planck/qp_planck/utilities.py detector_weights.
DETECTOR_WEIGHTS: dict[str, float] = {
    "LFI27": 0.40164e06,
    "LFI28": 0.36900e06,
    "LFI24": 0.12372e06,
    "LFI25": 0.14049e06,
    "LFI26": 0.11233e06,
    "LFI18": 53650.0,
    "LFI19": 42141.0,
    "LFI20": 36579.0,
    "LFI21": 50355.0,
    "LFI22": 49363.0,
    "LFI23": 47966.0,
    "100-1": 0.76343e06,
    "100-2": 0.12661e07,
    "100-3": 0.10631e07,
    "100-4": 0.10532e07,
    "143-1": 0.16407e07,
    "143-2": 0.18577e07,
    "143-3": 0.16439e07,
    "143-4": 0.14458e07,
    "143-5": 0.27630e07,
    "143-6": 0.26942e07,
    "143-7": 0.28599e07,
    "217-1": 0.11058e07,
    "217-2": 0.10261e07,
    "217-3": 0.10958e07,
    "217-4": 0.10593e07,
    "217-5": 0.67318e06,
    "217-6": 0.71092e06,
    "217-7": 0.76576e06,
    "217-8": 0.71226e06,
    "353-1": 0.12829e06,
    "353-2": 0.13475e06,
    "353-3": 48067.0,
    "353-4": 42187.0,
    "353-5": 56914.0,
    "353-6": 25293.0,
    "353-7": 87730.0,
    "353-8": 74453.0,
    "545-1": 4475.5,
    "545-2": 5540.3,
    "545-4": 4321.0,
    "857-1": 6.8895,
    "857-2": 6.3108,
    "857-3": 6.5964,
    "857-4": 3.6785,
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


def _weight_key(detector: str) -> str:
    det = detector.strip()
    if det.startswith("LFI") and det[-1:] in {"M", "S"}:
        return det[:-1]
    if "-" in det and det[-1:].lower() in {"a", "b"}:
        return det[:-1]
    return det


def detector_map_weight(detector: str, default: float = 1.0) -> float:
    """Return the inverse-noise map weight for a Planck detector."""
    return float(DETECTOR_WEIGHTS.get(_weight_key(detector), default))


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


def build_pointing_file_paths(input_root: str, od_start: int, od_end: int) -> list[Path]:
    """Build the list of pointing NPZ paths for an OD range.

    Constructs paths of the form ``{input_root}{od:04d}.npz`` for each OD in
    ``[od_start, od_end]`` (inclusive), returning only paths that exist on disk.

    Args:
        input_root: Path prefix shared by all pointing files, e.g.
            ``"inputs/pointings/processed_od_"``.
        od_start: First operational day (inclusive).
        od_end: Last operational day (inclusive).

    Returns:
        Sorted list of existing :class:`pathlib.Path` objects.

    Raises:
        FileNotFoundError: If no files exist for the requested range.
    """
    candidates = [Path(f"{input_root}{od:04d}.npz") for od in range(od_start, od_end + 1)]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        raise FileNotFoundError(
            f"No pointing files found for input_root={input_root!r}, OD{od_start}-OD{od_end}"
        )
    return existing
