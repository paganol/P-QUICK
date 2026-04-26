from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DetectorSelection:
    """Filters for selecting a subset of detectors from the full focal plane.

    Attributes:
        channel: Optional channel prefix (e.g. ``"100"``); keeps only detectors whose
            name starts with this string.
        detectors: Explicit allowlist of detector names; if non-empty, only these pass.
    """

    channel: str | None = None
    detectors: list[str] = field(default_factory=list)


@dataclass
class InputsConfig:
    """Paths to all pipeline inputs.

    Attributes:
        sky_alm: Sky ALM file path.
        beams_dir: Directory with detector beam ALM FITS files.
        rimo_file: RIMO FITS path.
        mission_length: OD selector (e.g. ``"full"``, ``"survey 2"``,
            or explicit ``"91-99"``). Defaults to ``"full"`` when omitted.
        pointings: Prefix for pointing NPZ files. The pipeline appends
            ``od_{OD:04d}.npz``.
            Example: ``inputs/pointings/pointing_`` ->
            ``inputs/pointings/pointing_od_0091.npz``.
        flags: Prefix for flags NPZ files. The pipeline appends
            ``{FREQ:03d}ghz_od_{OD:04d}.npz``.
            Example: ``inputs/flags/flags_`` ->
            ``inputs/flags/flags_100ghz_od_0091.npz``.
            When ``None`` (default) or when the per-OD file is absent, flags
            are not applied and all samples are treated as good.
        bad_rings_file: Optional path to a TOAST/NPIPE-style bad-ring interval
            text file with rows ``<det_or_ALL> <tstart_s> <tstop_s>``.
            Intervals are applied on top of existing flags.
    """

    sky_alm: str
    beams_dir: str
    rimo_file: str
    mission_length: str | None = None
    pointings: str = "inputs/pointings/pointing_"
    flags: str | None = None
    bad_rings_file: str | None = None


@dataclass
class ResamplingConfig:
    """Parameters controlling native-rate pointing reconstruction.

    Attributes:
        coordinate_system: Output sky frame for reconstructed pointing. Supported
            values are ``"ecliptic"`` and ``"galactic"``.
    """

    coordinate_system: str = "galactic"


@dataclass
class ConvolutionConfig:
    """Parameters controlling the ``ducc0`` total-convolution step.

    Attributes:
        lmax: Maximum multipole ℓ for both sky and beam ALMs.
        mmax: Maximum azimuthal order *m* of the beam (``kmax`` in ducc0 notation).
        epsilon: Interpolation accuracy target for the ducc0 gridder.
        chunks: Number of equal-length interpolation calls per OD (1 = whole OD at once).
    """

    lmax: int
    mmax: int
    epsilon: float = 1e-5
    chunks: int = 1


@dataclass
class MapConfig:
    """HEALPix output map parameters: resolution, pixel ordering, and filename prefix."""

    nside: int
    nest: bool = False
    output_prefix: str = "pquick"


@dataclass
class OutputConfig:
    """Specifies the directory where output FITS maps are written."""

    output_dir: str = "outputs"


@dataclass
class PipelineConfig:
    """Top-level configuration object aggregating all sub-configs for a pipeline run.

    Attributes:
        nthreads: Number of threads for ducc0 and numba.
            ``0`` = read ``OMP_NUM_THREADS`` from the environment (fallback 1).
    """

    inputs: InputsConfig
    detector_selection: DetectorSelection
    resampling: ResamplingConfig
    convolution: ConvolutionConfig
    map: MapConfig
    output: OutputConfig
    verbose: bool = False
    nthreads: int = 0


def _to_dataclass(data: dict[str, Any]) -> PipelineConfig:
    detsel = data.get("detector_selection") or {}
    channel = detsel.get("channel")
    detectors = list(detsel.get("detectors") or [])
    if channel and detectors:
        raise ValueError("Specify only one of detector_selection.channel or detector_selection.detectors")

    return PipelineConfig(
        inputs=InputsConfig(
            sky_alm=data["inputs"]["sky_alm"],
            beams_dir=data["inputs"]["beams_dir"],
            rimo_file=str(data["inputs"]["rimo_file"]),
            mission_length=(str(data["inputs"].get("mission_length")) if data["inputs"].get("mission_length") is not None else None),
            pointings=str(data["inputs"].get("pointings", "inputs/pointings/pointing_")),
            flags=(str(data["inputs"]["flags"]) if data["inputs"].get("flags") is not None else None),
            bad_rings_file=(
                str(data["inputs"]["bad_rings_file"])
                if data["inputs"].get("bad_rings_file") is not None
                else None
            ),
        ),
        detector_selection=DetectorSelection(
            channel=channel,
            detectors=detectors,
        ),
        resampling=ResamplingConfig(
            coordinate_system=str(data.get("resampling", {}).get("coordinate_system", "galactic")),
        ),
        convolution=ConvolutionConfig(
            lmax=int(data["convolution"]["lmax"]),
            mmax=int(data["convolution"].get("mmax", data["convolution"].get("kmax"))),
            epsilon=float(data.get("convolution", {}).get("epsilon", 1e-5)),
            chunks=int(data.get("convolution", {}).get("chunks", 1)),
        ),
        map=MapConfig(
            nside=int(data["map"]["nside"]),
            nest=bool(data.get("map", {}).get("nest", False)),
            output_prefix=str(data.get("map", {}).get("output_prefix", "pquick")),
        ),
        output=OutputConfig(output_dir=str(data.get("output", {}).get("output_dir", "outputs"))),
        verbose=bool(data.get("verbose", False)),
        nthreads=int(data.get("nthreads", 0)),
    )


def load_config(path: str | Path) -> PipelineConfig:
    """Read a YAML file and deserialise it into a :class:`PipelineConfig`.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A fully populated :class:`PipelineConfig` instance.

    Raises:
        ValueError: If the YAML root element is not a mapping.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return _to_dataclass(raw)
