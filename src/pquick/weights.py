from __future__ import annotations

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


def _weight_key(detector: str) -> str:
    det = detector.strip()
    if det.startswith("LFI") and det[-1:] in {"M", "S"}:
        return det[:-1]
    if "-" in det and det[-1:].lower() in {"a", "b"}:
        return det[:-1]
    return det


def detector_map_weight(detector: str, default: float = 1.0) -> float:
    """Return the inverse-noise map weight for a Planck detector.

    The detector name is normalised by stripping polarisation suffixes (``M``/``S`` for
    LFI, ``a``/``b`` for HFI) before looking up the value in :data:`DETECTOR_WEIGHTS`.

    Args:
        detector: Detector name (e.g. ``"100-1a"``, ``"LFI27M"``).
        default: Value returned when the detector is not in the table.

    Returns:
        Float map weight, or *default* if the detector is unknown.
    """
    return float(DETECTOR_WEIGHTS.get(_weight_key(detector), default))
