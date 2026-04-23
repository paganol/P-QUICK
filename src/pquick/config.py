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
        include_regex: Additional regex patterns; a detector must match at least one.
        exclude_regex: Regex patterns; a detector matching any of these is dropped.
    """

    channel: str | None = None
    detectors: list[str] = field(default_factory=list)
    include_regex: list[str] = field(default_factory=list)
    exclude_regex: list[str] = field(default_factory=list)


@dataclass
class PointingConfig:
    """Configuration for locating input pointing NPZ files via a glob pattern."""

    npz_glob: str


@dataclass
class ResamplingConfig:
    """Parameters controlling native-rate pointing reconstruction.

    Attributes:
        angular_eps: Legacy interpolation tolerance retained for API compatibility.
        coordinate_system: Output sky frame for reconstructed pointing. Supported
            values are ``"ecliptic"`` and ``"galactic"``.
    """

    angular_eps: float = 1e-10
    coordinate_system: str = "galactic"


@dataclass
class ConvolutionConfig:
    """Parameters controlling the ``ducc0`` total-convolution step.

    Attributes:
        lmax: Maximum multipole ℓ for both sky and beam ALMs.
        mmax: Maximum azimuthal order *m* of the beam (``kmax`` in ducc0 notation).
        nthreads: Number of OpenMP threads passed to ducc0 (0 = auto-detect).
        separate: If ``True``, keep per-component TOD streams separate in ducc0.
        epsilon: Interpolation accuracy target for the ducc0 gridder.
        chunks: Number of equal-length interpolation calls per OD (1 = whole OD at once).
    """

    lmax: int
    mmax: int
    nthreads: int = 0
    separate: bool = False
    epsilon: float = 1e-5
    chunks: int = 1


@dataclass
class MapConfig:
    """HEALPix output map parameters: resolution, pixel ordering, and filename prefix."""

    nside: int
    nest: bool = False
    output_prefix: str = "pquick"


@dataclass
class InputsConfig:
    """Paths to all pipeline inputs: sky ALMs, beam directory, RIMOs, and pointings."""

    sky_alm: str
    beams_dir: str
    rimo_files: list[str]
    pointing: PointingConfig


@dataclass
class OutputConfig:
    """Specifies the directory where output FITS maps are written."""

    output_dir: str = "outputs"


@dataclass
class PipelineConfig:
    """Top-level configuration object aggregating all sub-configs for a pipeline run."""

    inputs: InputsConfig
    detector_selection: DetectorSelection
    resampling: ResamplingConfig
    convolution: ConvolutionConfig
    map: MapConfig
    output: OutputConfig
    verbose: bool = False


def _to_dataclass(data: dict[str, Any]) -> PipelineConfig:
    return PipelineConfig(
        inputs=InputsConfig(
            sky_alm=data["inputs"]["sky_alm"],
            beams_dir=data["inputs"]["beams_dir"],
            rimo_files=list(data["inputs"]["rimo_files"]),
            pointing=PointingConfig(npz_glob=data["inputs"]["pointing"]["npz_glob"]),
        ),
        detector_selection=DetectorSelection(
            channel=data.get("detector_selection", {}).get("channel"),
            detectors=list(data.get("detector_selection", {}).get("detectors", [])),
            include_regex=list(data.get("detector_selection", {}).get("include_regex", [])),
            exclude_regex=list(data.get("detector_selection", {}).get("exclude_regex", [])),
        ),
        resampling=ResamplingConfig(
            angular_eps=float(data.get("resampling", {}).get("angular_eps", 1e-10)),
            coordinate_system=str(data.get("resampling", {}).get("coordinate_system", "galactic")),
        ),
        convolution=ConvolutionConfig(
            lmax=int(data["convolution"]["lmax"]),
            mmax=int(data["convolution"].get("mmax", data["convolution"].get("kmax"))),
            nthreads=int(data.get("convolution", {}).get("nthreads", 0)),
            separate=bool(data.get("convolution", {}).get("separate", False)),
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
